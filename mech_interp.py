"""
mech_interp.py
==============
Mechanistic interpretability toolkit for PyTorch transformers.

Five self-contained modules:

  1. ModelGraph      — structure discovery, path navigation, named hooks
  2. Patcher         — activation patching / causal tracing
  3. LogitLens       — per-layer residual-stream → output projection
  4. Probes          — linear probes at every layer
  5. AttentionViz    — attention weight extraction and entropy analysis

All modules work on any nn.Module with transformer-like structure.
No external dependencies beyond PyTorch, NumPy, scikit-learn,
matplotlib and (optionally) NNsight.

Usage
-----
    from mech_interp import ModelGraph, Patcher, LogitLens, Probes, AttentionViz

    graph   = ModelGraph(model)
    patcher = Patcher(model)
    lens    = LogitLens(model, decoder=graph.find_decoder())
    probes  = Probes(model)
    viz     = AttentionViz(model)
"""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False
    warnings.warn("matplotlib not found — visualization methods disabled.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    _SKL = True
except ImportError:
    _SKL = False
    warnings.warn("scikit-learn not found — Probes module disabled.")

try:
    from nnsight import NNsight
    _NNSIGHT = True
except ImportError:
    _NNSIGHT = False


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _nav(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Navigate dotted path, e.g. 'encoder.layers.0.attn'."""
    obj = model
    for part in path.split("."):
        if obj is None:
            return None
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
    return obj


def _unwrap(proxy: Any) -> torch.Tensor:
    """Unwrap NNsight saved proxy or return tensor directly."""
    return proxy.value if hasattr(proxy, "value") else proxy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


@dataclass
class HookHandle:
    """Thin wrapper so users can call .remove() on any registered hook."""
    _handles: List[Any] = field(default_factory=list)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 1. ModelGraph
# ══════════════════════════════════════════════════════════════════════════════

class ModelGraph:
    """
    Discover and navigate the structure of any PyTorch transformer.

    Parameters
    ----------
    model : nn.Module
        The raw model (not a wrapper / pipeline object).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._cache: Dict[str, nn.Module] = {}

    # ── Path navigation ───────────────────────────────────────────────────────

    def get(self, path: str) -> Optional[nn.Module]:
        """Return sub-module at dotted path, or None."""
        if path not in self._cache:
            self._cache[path] = _nav(self.model, path)
        return self._cache[path]

    def get_tensor(self, path: str) -> Optional[torch.Tensor]:
        """Return parameter/buffer at dotted path."""
        obj = self.model
        parts = path.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
            if obj is None:
                return None
        return getattr(obj, parts[-1], None)

    # ── Structure discovery ───────────────────────────────────────────────────

    def find_layers(
        self,
        keywords: Sequence[str] = ("blocks", "layers", "encoder"),
        leaf_digits: bool = True,
    ) -> List[str]:
        """
        Return sorted list of transformer block paths.
        Matches paths that contain any keyword and end with a digit (leaf blocks).

        Example
        -------
        >>> graph.find_layers(keywords=["icl_predictor.tf_icl.blocks"])
        ['icl_predictor.tf_icl.blocks.0', ..., 'icl_predictor.tf_icl.blocks.11']
        """
        results = []
        for name, _ in self.model.named_modules():
            if any(kw in name for kw in keywords):
                if leaf_digits and name.split(".")[-1].isdigit():
                    results.append(name)
                elif not leaf_digits:
                    results.append(name)
        seen, ordered = set(), []
        for r in results:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return sorted(ordered, key=lambda x: int(x.split(".")[-1])
                      if x.split(".")[-1].isdigit() else 0)

    def find_decoder(
        self,
        keywords: Sequence[str] = ("decoder", "head", "classifier", "lm_head"),
    ) -> Optional[str]:
        """
        Return dotted path to the output projection module.
        Prefers Sequential or containers; falls back to last Linear.
        """
        for name, mod in self.model.named_modules():
            if any(name.endswith("." + kw) or name == kw for kw in keywords):
                if hasattr(mod, "forward"):
                    return name

        # Fallback: last Linear layer
        last_linear_path = None
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear_path = name
        return last_linear_path

    def find_attention_modules(
        self,
        layer_paths: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Return {layer_path: attn_subpath} for each layer.
        Looks for MultiheadAttention or modules named 'attn'/'self_attn'.
        """
        result = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                # Match to containing layer
                if layer_paths:
                    for lp in layer_paths:
                        if name.startswith(lp + ".") or name == lp:
                            result[lp] = name
                            break
                else:
                    result[name] = name
        return result

    def summary(self) -> str:
        """Print a readable summary of discovered structure."""
        layers = self.find_layers()
        decoder = self.find_decoder()
        attn = self.find_attention_modules(layers)

        lines = [
            "ModelGraph Summary",
            "=" * 48,
            f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}",
            f"  Layers found: {len(layers)}",
        ]
        if layers:
            lines.append(f"    {layers[0]}  →  {layers[-1]}")
        lines.append(f"  Decoder: {decoder}")
        lines.append(f"  Attention modules found: {len(attn)}")
        return "\n".join(lines)

    # ── Hook utilities ────────────────────────────────────────────────────────

    def register_hook(
        self,
        path: str,
        fn: Callable,
        mode: str = "forward",
    ) -> HookHandle:
        """
        Register a hook at the given path.

        Parameters
        ----------
        path : str
            Dotted path to target module.
        fn : Callable
            Hook function. For forward hooks: fn(module, input, output).
        mode : str
            'forward', 'backward', or 'forward_pre'.

        Returns
        -------
        HookHandle
            Call .remove() to clean up.
        """
        mod = self.get(path)
        if mod is None:
            raise ValueError(f"Module not found at path: {path}")
        if mode == "forward":
            h = mod.register_forward_hook(fn)
        elif mode == "backward":
            h = mod.register_full_backward_hook(fn)
        elif mode == "forward_pre":
            h = mod.register_forward_pre_hook(fn)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return HookHandle([h])

    @contextmanager
    def capture(self, paths: List[str], token_idx: int = -1):
        """
        Context manager that captures output activations at all given paths.

        Yields a dict that will be populated after the forward pass.
        The value at each key is the activation at position `token_idx`.

        Example
        -------
        >>> with graph.capture(["encoder.layers.0", "encoder.layers.6"]) as acts:
        ...     model(input)
        >>> acts["encoder.layers.0"]  # (d_model,) numpy array
        """
        store: Dict[str, np.ndarray] = {}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach().cpu().float()
                while t.dim() > 2:
                    t = t.squeeze(0)
                store[name] = t[token_idx].numpy()
            return hook

        for path in paths:
            mod = self.get(path)
            if mod is not None:
                handles.append(mod.register_forward_hook(make_hook(path)))

        try:
            yield store
        finally:
            for h in handles:
                h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Patcher
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatchResult:
    """Result of a single activation patching experiment."""
    path: str
    clean_pred: int
    corrupt_pred: int
    patched_pred: int
    restoration_rate: float     # 1.0 if patched_pred == clean_pred
    p_true_clean: float
    p_true_corrupt: float
    p_true_patched: float


class Patcher:
    """
    Activation patching (causal tracing) for any PyTorch transformer.

    Identifies which components are causally necessary for a prediction
    by patching clean activations into a corrupted forward pass.

    Parameters
    ----------
    model : nn.Module
    graph : ModelGraph, optional
        If None, creates one automatically.
    """

    def __init__(self, model: nn.Module, graph: Optional[ModelGraph] = None):
        self.model = model
        self.graph = graph or ModelGraph(model)

    def _run(
        self,
        forward_fn: Callable,
        inputs: Any,
    ) -> Tuple[torch.Tensor, int]:
        """Run forward pass and return (logits, pred)."""
        self.model.eval()
        with torch.no_grad():
            out = forward_fn(inputs)
        if isinstance(out, tuple):
            out = out[0]
        out = out.detach().cpu().float()
        while out.dim() > 1:
            out = out.squeeze(0)
        return out, int(out.argmax().item())

    def _capture_activation(
        self,
        path: str,
        forward_fn: Callable,
        inputs: Any,
        token_idx: int = -1,
    ) -> Optional[torch.Tensor]:
        """Run a forward pass and capture activation at path."""
        captured = {}

        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach().clone()
            while t.dim() > 2:
                t = t.squeeze(0)
            captured["act"] = t[token_idx]

        mod = self.graph.get(path)
        if mod is None:
            return None
        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                forward_fn(inputs)
        finally:
            h.remove()
        return captured.get("act", None)

    def _patch_and_run(
        self,
        path: str,
        clean_act: torch.Tensor,
        forward_fn: Callable,
        corrupt_inputs: Any,
        token_idx: int = -1,
    ) -> Tuple[torch.Tensor, int]:
        """
        Run forward pass on corrupt_inputs, replacing activation at path
        with clean_act at position token_idx.
        """
        def hook(module, inp, out):
            if isinstance(out, tuple):
                t = out[0].clone()
                t[..., token_idx, :] = clean_act.to(t.device)
                return (t,) + out[1:]
            else:
                t = out.clone()
                t[..., token_idx, :] = clean_act.to(t.device)
                return t

        mod = self.graph.get(path)
        if mod is None:
            return torch.zeros(2), 0
        h = mod.register_forward_hook(hook)
        try:
            logits, pred = self._run(forward_fn, corrupt_inputs)
        finally:
            h.remove()
        return logits, pred

    def patch_single(
        self,
        path: str,
        forward_fn: Callable,
        clean_inputs: Any,
        corrupt_inputs: Any,
        true_label: int,
        token_idx: int = -1,
    ) -> PatchResult:
        """
        Patch one path; return a PatchResult.

        Parameters
        ----------
        path : str
            Module path to patch.
        forward_fn : Callable
            fn(inputs) → model output. Handle batching inside this function.
        clean_inputs : Any
            Inputs that produce the correct prediction.
        corrupt_inputs : Any
            Inputs where prediction is wrong (e.g., poisoned context).
        true_label : int
        token_idx : int
            Position in sequence to patch (default: -1 = last / test token).
        """
        # Clean activation to save
        clean_act = self._capture_activation(path, forward_fn, clean_inputs, token_idx)
        if clean_act is None:
            raise RuntimeError(f"Could not capture activation at {path}")

        # Clean and corrupt predictions
        clean_logits, clean_pred = self._run(forward_fn, clean_inputs)
        corrupt_logits, corrupt_pred = self._run(forward_fn, corrupt_inputs)

        # Patched prediction
        patched_logits, patched_pred = self._patch_and_run(
            path, clean_act, forward_fn, corrupt_inputs, token_idx
        )

        probs_clean   = _softmax(_to_np(clean_logits))
        probs_corrupt = _softmax(_to_np(corrupt_logits))
        probs_patched = _softmax(_to_np(patched_logits))

        n = len(probs_clean)
        tl = min(true_label, n - 1)

        return PatchResult(
            path=path,
            clean_pred=clean_pred,
            corrupt_pred=corrupt_pred,
            patched_pred=patched_pred,
            restoration_rate=float(patched_pred == clean_pred),
            p_true_clean=float(probs_clean[tl]),
            p_true_corrupt=float(probs_corrupt[tl]),
            p_true_patched=float(probs_patched[tl]),
        )

    def scan(
        self,
        layer_paths: List[str],
        forward_fn: Callable,
        clean_inputs: Any,
        corrupt_inputs: Any,
        true_label: int,
        n_samples: int = 1,
        token_idx: int = -1,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Scan all layer_paths and return restoration rates.

        If n_samples > 1, average over multiple (clean_inputs, corrupt_inputs)
        pairs — pass lists of inputs in that case.

        Returns
        -------
        dict: {path: mean_restoration_rate}
        """
        if n_samples == 1:
            clean_list  = [clean_inputs]
            corrupt_list = [corrupt_inputs]
            label_list   = [true_label]
        else:
            clean_list   = clean_inputs
            corrupt_list = corrupt_inputs
            label_list   = true_label if hasattr(true_label, "__iter__") \
                           else [true_label] * n_samples

        rates = defaultdict(list)
        for ci, (cl, co, lbl) in enumerate(zip(clean_list, corrupt_list, label_list)):
            for path in layer_paths:
                try:
                    result = self.patch_single(path, forward_fn, cl, co, lbl, token_idx)
                    rates[path].append(result.restoration_rate)
                except Exception as e:
                    if verbose:
                        print(f"  [PATCHER] {path} sample {ci}: {e}")
                    rates[path].append(float("nan"))

        summary = {}
        for path in layer_paths:
            vals = [v for v in rates[path] if not np.isnan(v)]
            summary[path] = float(np.mean(vals)) if vals else float("nan")
            if verbose:
                rate_str = f"{summary[path]*100:.1f}%" if not np.isnan(summary[path]) else "n/a"
                print(f"  {path:<50s}  restoration={rate_str}")

        return summary

    def plot_circuit_atlas(
        self,
        restoration_rates: Dict[str, float],
        title: str = "Circuit Atlas — Restoration Rate per Layer",
        save_path: Optional[str] = None,
    ):
        """Bar chart of restoration rates across layers."""
        if not _MPL:
            warnings.warn("matplotlib not available.")
            return

        paths = list(restoration_rates.keys())
        rates = [restoration_rates[p] * 100 for p in paths]
        labels = [p.split(".")[-2] + "." + p.split(".")[-1] for p in paths]

        fig, ax = plt.subplots(figsize=(max(8, len(paths) * 0.7), 4))
        colors = ["#4C72B0" if r > 10 else "#BBBBBB" for r in rates]
        ax.bar(range(len(paths)), rates, color=colors, edgecolor="white")
        ax.set_xticks(range(len(paths)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Restoration Rate (%)")
        ax.set_title(title, fontweight="bold")
        ax.axhline(10, color="gray", linestyle="--", alpha=0.5, label="10% threshold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [PATCHER] Saved {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. LogitLens
# ══════════════════════════════════════════════════════════════════════════════

class LogitLens:
    """
    Project residual stream at each layer through the output decoder.

    Tracks how the predicted class evolves layer by layer.

    Parameters
    ----------
    model : nn.Module
    decoder : str or nn.Module
        Dotted path to decoder, or the module itself.
    graph : ModelGraph, optional
    """

    def __init__(
        self,
        model: nn.Module,
        decoder: Union[str, nn.Module],
        graph: Optional[ModelGraph] = None,
    ):
        self.model  = model
        self.graph  = graph or ModelGraph(model)
        if isinstance(decoder, str):
            mod = self.graph.get(decoder)
            if mod is None:
                raise ValueError(f"Decoder not found at path: {decoder}")
            self.decoder = mod
        else:
            self.decoder = decoder
        self.decoder.eval()

    def _project(self, repr_vec: np.ndarray) -> np.ndarray:
        """Project (d_model,) array through decoder → probabilities."""
        with torch.no_grad():
            t = torch.tensor(repr_vec, dtype=torch.float32)
            t = t.to(next(self.decoder.parameters()).device)
            logits = self.decoder(t)
        logits = logits.detach().cpu().float()
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        return _softmax(_to_np(logits))

    def trace(
        self,
        layer_paths: List[str],
        forward_fn: Callable,
        inputs: Any,
        token_idx: int = -1,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        One forward pass; return [(layer_idx, probs)] for each layer.

        Parameters
        ----------
        layer_paths : list of str
            Paths to ICL blocks in order.
        forward_fn : Callable
            fn(inputs) → model output.
        inputs : Any
        token_idx : int
            Sequence position for test token (default: -1).

        Returns
        -------
        list of (layer_idx, probs_array)
        """
        results = []
        with self.graph.capture(layer_paths, token_idx=token_idx) as acts:
            with torch.no_grad():
                forward_fn(inputs)

        for i, path in enumerate(layer_paths):
            if path not in acts:
                continue
            try:
                probs = self._project(acts[path])
                results.append((i, probs))
            except Exception:
                continue

        return results

    def compare(
        self,
        layer_paths: List[str],
        forward_fn: Callable,
        clean_inputs: Any,
        corrupt_inputs: List[Any],
        corrupt_labels: Optional[List[str]] = None,
        token_idx: int = -1,
        n_samples: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Average logit-lens trajectories over n_samples for each condition.

        Parameters
        ----------
        corrupt_inputs : list
            Each element is an inputs object for one corrupt condition.
            For n_samples > 1 each element should be a list of n_samples inputs.
        corrupt_labels : list of str, optional
            Names for each corrupt condition.

        Returns
        -------
        dict with keys 'clean' and each corrupt label → (n_layers, n_classes) array
        """
        if corrupt_labels is None:
            corrupt_labels = [f"corrupt_{i}" for i in range(len(corrupt_inputs))]

        def _avg_trajectory(inputs_list):
            trajs = []
            for inp in inputs_list:
                t = self.trace(layer_paths, forward_fn, inp, token_idx)
                if t:
                    trajs.append(np.array([p for _, p in t]))
            return np.mean(trajs, axis=0) if trajs else None

        clean_list = [clean_inputs] if n_samples == 1 else clean_inputs
        result = {"clean": _avg_trajectory(clean_list)}

        for label, cinputs in zip(corrupt_labels, corrupt_inputs):
            clist = [cinputs] if n_samples == 1 else cinputs
            result[label] = _avg_trajectory(clist)

        return result

    def kl_divergence(self, clean_traj: np.ndarray,
                       poison_traj: np.ndarray) -> np.ndarray:
        """Per-layer KL(clean || poisoned)."""
        kl = []
        for layer in range(len(clean_traj)):
            c = np.clip(clean_traj[layer], 1e-8, 1)
            p = np.clip(poison_traj[layer], 1e-8, 1)
            kl.append(float(np.sum(c * np.log(c / p))))
        return np.array(kl)

    def critical_layer(self, kl: np.ndarray) -> int:
        """Layer with the largest single-step KL increase."""
        diffs = np.diff(kl)
        return int(np.argmax(diffs)) + 1

    def plot(
        self,
        trajectories: Dict[str, np.ndarray],
        class_idx: int = 0,
        title: str = "Logit Lens — Prediction Trajectory",
        save_path: Optional[str] = None,
        highlight_layer: Optional[int] = None,
    ):
        """
        Plot P(class_idx) per layer for each condition.

        Parameters
        ----------
        trajectories : dict {label: (n_layers, n_classes) array}
        class_idx : int
            Which output class to track.
        highlight_layer : int, optional
            Draw a vertical line at this layer (e.g. critical layer).
        """
        if not _MPL:
            warnings.warn("matplotlib not available.")
            return

        colors = ["#4C72B0", "#E84040", "#55A868", "#DD8452", "#9467BD"]
        styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Panel left: probability trajectory
        ax = axes[0]
        for i, (label, traj) in enumerate(trajectories.items()):
            if traj is None:
                continue
            layers = np.arange(len(traj))
            c = colors[i % len(colors)]
            s = styles[i % len(styles)]
            ax.plot(layers, traj[:, class_idx], linestyle=s, marker="o",
                    markersize=4, linewidth=2, color=c, label=label)
        if highlight_layer is not None:
            ax.axvline(highlight_layer, color="gray", linestyle=":",
                       alpha=0.6, label=f"Layer {highlight_layer}")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"P(class {class_idx})")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

        # Panel right: KL divergence from clean
        ax = axes[1]
        clean_traj = trajectories.get("clean")
        if clean_traj is not None:
            for i, (label, traj) in enumerate(trajectories.items()):
                if label == "clean" or traj is None:
                    continue
                kl = self.kl_divergence(clean_traj, traj)
                cl = self.critical_layer(kl)
                c = colors[i % len(colors)]
                ax.plot(np.arange(len(kl)), kl, marker="s", linewidth=2,
                        color=c, label=f"{label} (critical={cl})")
                ax.axvline(cl, color=c, linestyle=":", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("KL(clean || condition)")
        ax.set_title("KL Divergence from Clean", fontweight="bold")
        ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [LOGITLENS] Saved {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Probes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbeResult:
    layer: int
    path: str
    accuracy: float
    auroc: float
    n_train: int
    n_test: int


class Probes:
    """
    Train linear probes at each layer to validate residual-stream representations.

    If probe accuracy tracks the logit-lens trajectory, the commitment
    is a genuine residual-stream phenomenon, not a decoder artifact.

    Parameters
    ----------
    model : nn.Module
    graph : ModelGraph, optional
    """

    def __init__(self, model: nn.Module, graph: Optional[ModelGraph] = None):
        if not _SKL:
            raise ImportError("scikit-learn required for Probes module.")
        self.model = model
        self.graph = graph or ModelGraph(model)

    def collect(
        self,
        layer_paths: List[str],
        forward_fn: Callable,
        inputs_list: List[Any],
        labels: List[int],
        token_idx: int = -1,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run forward passes and collect test-token representations.

        Returns
        -------
        dict {path: (X_repr, y_labels)} where X_repr is (n_samples, d_model).
        """
        storage: Dict[str, List] = {p: [] for p in layer_paths}

        for inp in inputs_list:
            self.model.eval()
            with self.graph.capture(layer_paths, token_idx=token_idx) as acts:
                with torch.no_grad():
                    forward_fn(inp)
            for path in layer_paths:
                if path in acts:
                    storage[path].append(acts[path])

        result = {}
        for path in layer_paths:
            if len(storage[path]) == 0:
                continue
            X = np.stack(storage[path])
            y = np.array(labels[: len(X)])
            result[path] = (X, y)
        return result

    def fit_and_score(
        self,
        reprs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        test_frac: float = 0.2,
        C: float = 1.0,
        max_iter: int = 500,
    ) -> List[ProbeResult]:
        """
        Fit one logistic regression probe per layer and return results.

        Parameters
        ----------
        reprs : output of .collect()
        test_frac : float
            Fraction of data held out for evaluation.
        """
        results = []
        for i, (path, (X, y)) in enumerate(reprs.items()):
            if len(np.unique(y)) < 2 or len(X) < 5:
                continue
            split = max(1, int(test_frac * len(X)))
            X_tr, X_te = X[split:], X[:split]
            y_tr, y_te = y[split:], y[:split]

            sc  = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            try:
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                proba = clf.predict_proba(X_te)[:, 1]
                acc   = accuracy_score(y_te, preds)
                try:
                    auroc = roc_auc_score(y_te, proba)
                except Exception:
                    auroc = float("nan")
            except Exception:
                acc, auroc = float("nan"), float("nan")

            results.append(ProbeResult(
                layer=i,
                path=path,
                accuracy=acc,
                auroc=auroc,
                n_train=len(X_tr),
                n_test=len(X_te),
            ))

        return results

    def run(
        self,
        layer_paths: List[str],
        forward_fn: Callable,
        clean_inputs: List[Any],
        clean_labels: List[int],
        corrupt_inputs: Optional[Dict[str, List[Any]]] = None,
        token_idx: int = -1,
        verbose: bool = True,
    ) -> Dict[str, List[ProbeResult]]:
        """
        Full pipeline: collect → fit → score for all conditions.

        Parameters
        ----------
        corrupt_inputs : dict {label: list_of_inputs}, optional

        Returns
        -------
        dict {condition_label: [ProbeResult, ...]}
        """
        all_results: Dict[str, List[ProbeResult]] = {}

        clean_reprs = self.collect(layer_paths, forward_fn,
                                    clean_inputs, clean_labels, token_idx)
        all_results["clean"] = self.fit_and_score(clean_reprs)

        if corrupt_inputs:
            for label, cinputs in corrupt_inputs.items():
                reprs = self.collect(layer_paths, forward_fn,
                                      cinputs, clean_labels, token_idx)
                all_results[label] = self.fit_and_score(reprs)

        if verbose:
            print(f"\n  {'Layer':>5s}  {'Path':<40s}", end="")
            for label in all_results:
                print(f"  {label[:12]:>12s}", end="")
            print()
            print("  " + "─" * (55 + 14 * len(all_results)))
            n_layers = max(len(v) for v in all_results.values())
            for i in range(n_layers):
                row_results = {k: (v[i] if i < len(v) else None)
                               for k, v in all_results.items()}
                first = next(r for r in row_results.values() if r is not None)
                print(f"  {first.layer:>5d}  {first.path:<40s}", end="")
                for label in all_results:
                    r = row_results.get(label)
                    if r and not np.isnan(r.accuracy):
                        print(f"  {r.accuracy:>12.3f}", end="")
                    else:
                        print(f"  {'n/a':>12s}", end="")
                print()

        return all_results

    def find_commitment_layer(self, results: List[ProbeResult]) -> int:
        """Return layer index with largest single-step accuracy increase."""
        accs = [r.accuracy for r in results]
        diffs = np.diff(accs)
        valid = np.where(~np.isnan(diffs), diffs, -999)
        return int(np.argmax(valid)) + 1

    def plot(
        self,
        all_results: Dict[str, List[ProbeResult]],
        title: str = "Linear Probe Accuracy per Layer",
        save_path: Optional[str] = None,
        logit_lens_commit: Optional[int] = None,
    ):
        """
        Plot probe accuracy curves for all conditions.

        Parameters
        ----------
        logit_lens_commit : int, optional
            Layer identified by logit lens, for comparison annotation.
        """
        if not _MPL:
            warnings.warn("matplotlib not available.")
            return

        colors = ["#4C72B0", "#E84040", "#55A868", "#DD8452"]
        styles = ["-", "--", ":", "-."]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        probe_commits = {}
        for i, (label, results) in enumerate(all_results.items()):
            layers = [r.layer for r in results]
            accs   = [r.accuracy for r in results]
            c = colors[i % len(colors)]
            s = styles[i % len(styles)]
            ax.plot(layers, accs, linestyle=s, marker="o",
                    markersize=4, linewidth=2, color=c, label=label)
            if label == "clean":
                commit = self.find_commitment_layer(results)
                probe_commits["probe"] = commit
                ax.axvline(commit, color=c, linestyle=":", alpha=0.5,
                           label=f"Probe commit ({commit})")

        if logit_lens_commit is not None:
            ax.axvline(logit_lens_commit, color="gray", linestyle="--",
                       alpha=0.6, label=f"Logit-lens commit ({logit_lens_commit})")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.05)

        # Panel right: AUROC
        ax = axes[1]
        for i, (label, results) in enumerate(all_results.items()):
            layers = [r.layer for r in results]
            aurocs = [r.auroc for r in results]
            c = colors[i % len(colors)]
            s = styles[i % len(styles)]
            ax.plot(layers, aurocs, linestyle=s, marker="s",
                    markersize=4, linewidth=2, color=c, label=label)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe AUROC")
        ax.set_title("Probe AUROC per Layer", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.05)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [PROBES] Saved {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5. AttentionViz
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionResult:
    layer_path: str
    mode: str                       # "weights" or "norm_proxy"
    test_token_attn: np.ndarray     # (ctx_len,) attention from test token
    entropy: float
    concentration_at: Dict[str, float]  # {label: mass at target positions}


class AttentionViz:
    """
    Extract and visualize attention patterns at any transformer layer.

    Falls back to activation-norm proxy if true attention weights are
    unavailable (e.g., fused kernels with need_weights=False).

    Parameters
    ----------
    model : nn.Module
    graph : ModelGraph, optional
    """

    def __init__(self, model: nn.Module, graph: Optional[ModelGraph] = None):
        self.model  = model
        self.graph  = graph or ModelGraph(model)

    # ── True attention weight extraction ─────────────────────────────────────

    def _patch_mha_for_weights(
        self,
        mha: nn.MultiheadAttention,
        forward_fn: Callable,
        inputs: Any,
    ) -> Optional[torch.Tensor]:
        """Monkey-patch MHA.forward to force need_weights=True."""
        captured = {}
        original = mha.forward

        def patched(query, key, value, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, weights = original(query, key, value, **kwargs)
            captured["w"] = weights
            return out, weights

        mha.forward = patched
        try:
            self.model.eval()
            with torch.no_grad():
                forward_fn(inputs)
        except Exception:
            pass
        finally:
            mha.forward = original

        return captured.get("w", None)

    def _get_weights(
        self,
        layer_path: str,
        forward_fn: Callable,
        inputs: Any,
        ctx_len: int,
        token_idx: int = -1,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Try to get real attention weights, fall back to norm proxy.
        Returns (test_token_attn_over_ctx, mode).
        """
        # 1. Find the MHA module inside this layer
        mha = None
        for name, mod in self.model.named_modules():
            if name.startswith(layer_path) and isinstance(mod, nn.MultiheadAttention):
                mha = mod
                break
        # Also try common submodule names
        if mha is None:
            for suffix in ["attn", "self_attn", "attention"]:
                cand = self.graph.get(f"{layer_path}.{suffix}")
                if isinstance(cand, nn.MultiheadAttention):
                    mha = cand
                    break

        if mha is not None:
            raw = self._patch_mha_for_weights(mha, forward_fn, inputs)
            if raw is not None:
                w = raw.detach().cpu().float()
                # shapes: (B, H, S, S) or (H, S, S) or (B, S, S) or (S, S)
                while w.dim() > 3:
                    w = w.squeeze(0)
                if w.dim() == 3:          # (H, S, S)
                    test_row = w[:, token_idx, :ctx_len]   # (H, ctx_len)
                    return test_row.mean(0).numpy(), "attention_weights"
                elif w.dim() == 2:        # (S, S)
                    return w[token_idx, :ctx_len].numpy(), "attention_weights"

        # 2. Activation-norm fallback
        with self.graph.capture([layer_path], token_idx=None) as acts_full:
            with torch.no_grad():
                forward_fn(inputs)

        if layer_path not in acts_full:
            # capture with token_idx=None doesn't work; try direct hook
            captured = {}

            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach().cpu().float()
                while t.dim() > 2:
                    t = t.squeeze(0)
                captured["seq"] = t  # (S, D)

            mod = self.graph.get(layer_path)
            if mod is None:
                return None, "unavailable"
            h = mod.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    forward_fn(inputs)
            finally:
                h.remove()
            if "seq" not in captured:
                return None, "unavailable"
            seq = captured["seq"]
        else:
            seq = acts_full[layer_path]
            if not isinstance(seq, np.ndarray):
                seq = _to_np(seq)
            seq = torch.tensor(seq)

        # seq: (S, D) — norm over D per position
        if isinstance(seq, np.ndarray):
            seq = torch.tensor(seq)
        norms = seq.norm(dim=-1).numpy()
        ctx_norms = norms[:ctx_len]
        total = ctx_norms.sum()
        if total < 1e-8:
            return None, "norm_proxy"
        return ctx_norms / total, "norm_proxy"

    def extract(
        self,
        layer_path: str,
        forward_fn: Callable,
        inputs: Any,
        ctx_len: int,
        token_idx: int = -1,
        target_positions: Optional[Dict[str, List[int]]] = None,
    ) -> Optional[AttentionResult]:
        """
        Extract attention from test token to context positions.

        Parameters
        ----------
        layer_path : str
            e.g. "encoder.layers.0"
        forward_fn : Callable
        inputs : Any
        ctx_len : int
            Number of context positions (excluding test token).
        token_idx : int
            Test token position in the full sequence (default: -1).
        target_positions : dict, optional
            {label: [pos1, pos2, ...]} — compute attention mass at those positions.
            e.g. {"poison": [3, 17, 42]}

        Returns
        -------
        AttentionResult or None
        """
        attn, mode = self._get_weights(layer_path, forward_fn, inputs,
                                        ctx_len, token_idx)
        if attn is None:
            return None

        ent = _entropy(attn)
        conc = {}
        if target_positions:
            for label, positions in target_positions.items():
                valid = [p for p in positions if 0 <= p < len(attn)]
                conc[label] = float(attn[valid].sum()) if valid else 0.0

        return AttentionResult(
            layer_path=layer_path,
            mode=mode,
            test_token_attn=attn,
            entropy=ent,
            concentration_at=conc,
        )

    def compare(
        self,
        layer_path: str,
        forward_fn: Callable,
        conditions: Dict[str, Any],
        ctx_len: int,
        token_idx: int = -1,
        target_positions: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, Optional[AttentionResult]]:
        """
        Extract attention for multiple input conditions.

        Parameters
        ----------
        conditions : dict {label: inputs}

        Returns
        -------
        dict {label: AttentionResult}
        """
        return {
            label: self.extract(layer_path, forward_fn, inputs,
                                 ctx_len, token_idx, target_positions)
            for label, inputs in conditions.items()
        }

    def aggregate(
        self,
        layer_path: str,
        forward_fn: Callable,
        conditions_list: Dict[str, List[Any]],
        ctx_len: int,
        token_idx: int = -1,
        target_positions_list: Optional[List[Dict[str, List[int]]]] = None,
    ) -> Dict[str, AttentionResult]:
        """
        Average AttentionResults over multiple samples per condition.

        Parameters
        ----------
        conditions_list : dict {label: [inputs1, inputs2, ...]}
        target_positions_list : list of per-sample target_positions dicts, optional
        """
        avg_attns: Dict[str, List[np.ndarray]] = defaultdict(list)
        avg_conc:  Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        modes: Dict[str, str] = {}
        entropies: Dict[str, List[float]] = defaultdict(list)

        n_samples = max(len(v) for v in conditions_list.values())
        for i in range(n_samples):
            tgt = target_positions_list[i] if target_positions_list else None
            for label, inputs_list in conditions_list.items():
                if i >= len(inputs_list):
                    continue
                r = self.extract(layer_path, forward_fn, inputs_list[i],
                                  ctx_len, token_idx, tgt)
                if r is None:
                    continue
                avg_attns[label].append(r.test_token_attn)
                entropies[label].append(r.entropy)
                modes[label] = r.mode
                for k, v in r.concentration_at.items():
                    avg_conc[label][k].append(v)

        results = {}
        for label in avg_attns:
            if not avg_attns[label]:
                continue
            mean_attn = np.mean(avg_attns[label], axis=0)
            mean_conc = {k: float(np.mean(v)) for k, v in avg_conc[label].items()}
            results[label] = AttentionResult(
                layer_path=layer_path,
                mode=modes.get(label, "unknown"),
                test_token_attn=mean_attn,
                entropy=float(np.mean(entropies[label])),
                concentration_at=mean_conc,
            )
        return results

    def plot(
        self,
        results: Dict[str, AttentionResult],
        ctx_len: Optional[int] = None,
        n_show: int = 40,
        title: str = "Attention at blocks.0 — Test Token → Context",
        save_path: Optional[str] = None,
        target_label: Optional[str] = None,
    ):
        """
        Three-panel visualization:
          Left:   attention weight curves per position
          Middle: entropy bar chart
          Right:  concentration at target positions (if available)

        Parameters
        ----------
        target_label : str, optional
            Key inside concentration_at to plot in right panel.
        """
        if not _MPL:
            warnings.warn("matplotlib not available.")
            return

        colors = ["#4C72B0", "#E84040", "#55A868", "#DD8452"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # ── Panel 1: attention curves ─────────────────────────────────────────
        ax = axes[0]
        max_pos = n_show
        for i, (label, r) in enumerate(results.items()):
            attn = r.test_token_attn[:max_pos]
            ax.plot(range(len(attn)), attn, color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.85, label=label)
        if results:
            first = next(iter(results.values()))
            n = min(len(first.test_token_attn), max_pos)
            ax.axhline(1 / n, color="gray", linestyle="--", alpha=0.5, label="Uniform")
        ax.set_xlabel("Context position")
        ax.set_ylabel("Attention weight (test token)")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

        # ── Panel 2: entropy ──────────────────────────────────────────────────
        ax = axes[1]
        labels_ = list(results.keys())
        ents    = [results[l].entropy for l in labels_]
        bars    = ax.bar(labels_, ents,
                          color=[colors[i % len(colors)]
                                 for i in range(len(labels_))],
                          alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, ents):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        if results:
            n = len(next(iter(results.values())).test_token_attn)
            ax.axhline(np.log(n), color="gray", linestyle="--",
                       alpha=0.5, label="Max entropy")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title("Attention Entropy\n(lower = more concentrated)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

        # ── Panel 3: concentration at target positions ────────────────────────
        ax = axes[2]
        keys_available = set()
        for r in results.values():
            keys_available.update(r.concentration_at.keys())

        if target_label and target_label in keys_available:
            concs  = [results[l].concentration_at.get(target_label, 0) for l in labels_]
            bars   = ax.bar(labels_, concs,
                             color=[colors[i % len(colors)]
                                    for i in range(len(labels_))],
                             alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, concs):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel(f"Attention mass @ '{target_label}' positions")
            ax.set_title(f"Retrieval Concentration\nat '{target_label}' positions",
                         fontweight="bold", fontsize=10)
        elif keys_available:
            # Plot all available concentration keys as a grouped bar
            x = np.arange(len(labels_))
            w = 0.8 / max(len(keys_available), 1)
            for j, key in enumerate(sorted(keys_available)):
                vals = [results[l].concentration_at.get(key, 0) for l in labels_]
                ax.bar(x + j * w - 0.4, vals, w, alpha=0.8, label=key)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_)
            ax.set_ylabel("Attention mass at positions")
            ax.set_title("Retrieval Concentration", fontweight="bold", fontsize=10)
            ax.legend(fontsize=8)
        else:
            mode_str = next(iter(results.values())).mode if results else "unknown"
            ax.text(0.5, 0.5, f"No target positions\nspecified\n\nMode: {mode_str}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title("Retrieval Concentration", fontweight="bold", fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [ATTNVIZ] Saved {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run the full suite in one call
# ══════════════════════════════════════════════════════════════════════════════

class MechInterpSuite:
    """
    Runs all five modules on a single model with a unified API.

    Parameters
    ----------
    model : nn.Module
        Raw PyTorch module.
    layer_keywords : list of str, optional
        Passed to ModelGraph.find_layers(). Auto-detected if None.
    decoder_path : str, optional
        Dotted path to output decoder. Auto-detected if None.

    Example
    -------
    >>> suite = MechInterpSuite(model)
    >>> suite.run(
    ...     forward_fn=lambda inp: model(inp["X"], inp["y"]),
    ...     clean_inputs=clean_inp,
    ...     corrupt_inputs={"Attack D": poisoned_inp},
    ...     true_label=0,
    ...     ctx_len=256,
    ...     n_samples=50,
    ...     save_dir="/outputs",
    ... )
    """

    def __init__(
        self,
        model: nn.Module,
        layer_keywords: Optional[List[str]] = None,
        decoder_path: Optional[str] = None,
    ):
        self.model  = model
        self.graph  = ModelGraph(model)
        kw = layer_keywords or ["blocks", "layers", "encoder"]
        self.layer_paths = self.graph.find_layers(kw)
        dec = decoder_path or self.graph.find_decoder()
        self.patcher    = Patcher(model, self.graph)
        self.lens       = LogitLens(model, dec, self.graph) if dec else None
        self.probes     = Probes(model, self.graph) if _SKL else None
        self.attn_viz   = AttentionViz(model, self.graph)

        print(self.graph.summary())

    def run(
        self,
        forward_fn: Callable,
        clean_inputs: Any,
        corrupt_inputs: Dict[str, Any],
        true_label: int,
        ctx_len: int,
        n_samples: int = 30,
        token_idx: int = -1,
        save_dir: Optional[str] = None,
        poison_positions: Optional[Dict[str, List[int]]] = None,
    ) -> Dict:
        """
        Run Patcher → LogitLens → Probes → AttentionViz and return all results.

        Parameters
        ----------
        clean_inputs : Any
            Single input or list of inputs (if n_samples > 1).
        corrupt_inputs : dict {label: inputs or list of inputs}
        poison_positions : dict {label: [pos, ...]}, optional
            For AttentionViz concentration analysis.
        save_dir : str, optional
            Directory to save all figures.
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        out = {}

        # Normalise to lists
        clean_list = clean_inputs if isinstance(clean_inputs, list) else [clean_inputs]
        corrupt_lists = {k: (v if isinstance(v, list) else [v])
                         for k, v in corrupt_inputs.items()}

        # ── Patcher ───────────────────────────────────────────────────────────
        print("\n[SUITE] Running Patcher (causal tracing)...")
        first_corrupt_label = next(iter(corrupt_lists))
        rates = self.patcher.scan(
            layer_paths=self.layer_paths,
            forward_fn=forward_fn,
            clean_inputs=clean_list,
            corrupt_inputs=corrupt_lists[first_corrupt_label],
            true_label=[true_label] * min(n_samples, len(clean_list)),
            n_samples=min(n_samples, len(clean_list)),
            token_idx=token_idx,
        )
        out["restoration_rates"] = rates
        if save_dir:
            self.patcher.plot_circuit_atlas(
                rates, save_path=f"{save_dir}/circuit_atlas.png"
            )

        # ── LogitLens ─────────────────────────────────────────────────────────
        if self.lens:
            print("\n[SUITE] Running LogitLens...")
            trajectories = self.lens.compare(
                layer_paths=self.layer_paths,
                forward_fn=forward_fn,
                clean_inputs=clean_list[0],
                corrupt_inputs=[corrupt_lists[k][0] for k in corrupt_lists],
                corrupt_labels=list(corrupt_lists.keys()),
                token_idx=token_idx,
            )
            out["trajectories"] = trajectories
            kls = {}
            for label, traj in trajectories.items():
                if label == "clean" or traj is None:
                    continue
                kl = self.lens.kl_divergence(trajectories["clean"], traj)
                kls[label] = {"kl": kl, "critical_layer": self.lens.critical_layer(kl)}
                print(f"  {label}: critical layer = {kls[label]['critical_layer']}")
            out["kl_divergences"] = kls
            if save_dir:
                self.lens.plot(trajectories,
                               save_path=f"{save_dir}/logit_lens.png")

        # ── Probes ────────────────────────────────────────────────────────────
        if self.probes:
            print("\n[SUITE] Running Linear Probes...")
            probe_results = self.probes.run(
                layer_paths=self.layer_paths,
                forward_fn=forward_fn,
                clean_inputs=clean_list[:n_samples],
                clean_labels=[true_label] * min(n_samples, len(clean_list)),
                corrupt_inputs={k: v[:n_samples] for k, v in corrupt_lists.items()},
                token_idx=token_idx,
            )
            out["probe_results"] = probe_results
            if "clean" in probe_results:
                commit = self.probes.find_commitment_layer(probe_results["clean"])
                out["probe_commitment_layer"] = commit
                print(f"  Probe commitment layer: {commit}")
            lens_commit = (kls[first_corrupt_label]["critical_layer"]
                           if kls else None)
            if save_dir:
                self.probes.plot(probe_results,
                                  logit_lens_commit=lens_commit,
                                  save_path=f"{save_dir}/probes.png")

        # ── AttentionViz ──────────────────────────────────────────────────────
        print("\n[SUITE] Running AttentionViz at first layer...")
        first_layer = self.layer_paths[0]
        target_pos = poison_positions or {}
        attn_results = self.attn_viz.aggregate(
            layer_path=first_layer,
            forward_fn=forward_fn,
            conditions_list={"clean": clean_list[:n_samples],
                              **{k: v[:n_samples] for k, v in corrupt_lists.items()}},
            ctx_len=ctx_len,
            token_idx=token_idx,
            target_positions_list=None,
        )
        out["attention"] = attn_results
        for label, r in attn_results.items():
            print(f"  {label}: entropy={r.entropy:.3f}  mode={r.mode}")
        if save_dir:
            first_target = next(iter(target_pos.keys())) if target_pos else None
            self.attn_viz.plot(attn_results,
                                target_label=first_target,
                                save_path=f"{save_dir}/attention_viz.png")

        print("\n[SUITE] Complete.")
        return out
