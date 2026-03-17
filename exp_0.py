# ============================================================
# EXPERIMENT 0 - Baseline Setup & NNsight Validation
# T4 GPU / Google Colab ready
# ============================================================
# Run this cell-by-cell in Colab, or as a script on any
# CUDA machine. T4 has 16GB VRAM - more than enough for
# ORION-BiX at context size 256.
#
# Cell order:
#   [INSTALL]  → pip installs
#   [SETUP]    → imports, device, seed
#   [0A]       → data
#   [0B]       → model
#   [0C]       → submodule map
#   [0D]       → clean metrics
#   [0E]       → NNsight lossless validation
#   [0F]       → hook shim fallback (if needed)
#   [MAIN]     → run everything, print Day 1 gate result
# ============================================================


# ── [INSTALL] ────────────────────────────────────────────────────────────────
# Run once per Colab session. Comment out if running locally with env already set.

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

INSTALLS = [
    "nnsight",
    "tabtune",          # Lexsi Labs - may need: pip install git+https://github.com/Lexsi-Labs/TabTune
    "scikit-learn",
    "pandas",
    "einops",
    "foolbox",          # needed in Experiment 1 - install now to catch issues early
]

for pkg in INSTALLS:
    try:
        install(pkg)
        print(f"[INSTALL] ✅ {pkg}")
    except Exception as e:
        print(f"[INSTALL] ⚠️  {pkg} failed: {e}")

# If tabtune pip install fails, try the GitHub source:
# !pip install -q git+https://github.com/Lexsi-Labs/TabTune.git


# ── [SETUP] ──────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Device ────────────────────────────────────────────────────────────────────
# On Colab: Runtime → Change runtime type → T4 GPU
# Verify with: !nvidia-smi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SETUP] Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[SETUP] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # T4 supports float16 - useful if ORION-BiX supports half precision
    # For now keep float32 for numerical stability in patching experiments
    DTYPE = torch.float32
else:
    print("[SETUP] ⚠️  No GPU found. Colab: Runtime → Change runtime type → T4 GPU")
    DTYPE = torch.float32

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    # Deterministic ops - slight speed cost but required for patching experiments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── NNsight ───────────────────────────────────────────────────────────────────
try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
    print("[SETUP] ✅ nnsight available")
except ImportError:
    NNSIGHT_AVAILABLE = False
    print("[SETUP] ⚠️  nnsight not available - HookShim fallback will be used")

# ── TabTune ───────────────────────────────────────────────────────────────────
# Correct import per TabTune README (v0.1.9):
#   from tabtune.TabularPipeline.pipeline import TabularPipeline
# Model names: "OrionBix", "OrionMSP", "TabPFN", "TabICL" etc.
# API: pipeline.fit(X, y) / pipeline.predict(X) / pipeline.evaluate(X, y)
# Note: TabularPipeline accepts pandas DataFrames or numpy arrays - NOT tensors.
#       Always call .cpu().numpy() before passing GPU tensors to TabTune.
try:
    from tabtune.TabularPipeline.pipeline import TabularPipeline as TabTunePipeline
    TABTUNE_AVAILABLE = True
    print("[SETUP] ✅ tabtune available (TabularPipeline)")
except ImportError:
    TABTUNE_AVAILABLE = False
    print("[SETUP] ⚠️  tabtune not available - structural fallback model will be used")


# ── [0A] DATA ────────────────────────────────────────────────────────────────

def load_adult_income(n_context: int = 256, n_test: int = 512, seed: int = SEED):
    """
    Load Adult Income (UCI) and return context/test splits as GPU tensors.

    T4 note: tensors are moved to DEVICE here so every downstream operation
    runs on GPU. The context window (256 rows × 14 features) is ~14KB -
    negligible VRAM.
    """
    print("\n[0A] Loading Adult Income dataset...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.dropna().reset_index(drop=True)

    X = df.drop(columns=["class"])
    y = (df["class"].str.strip() == ">50K").astype(int)

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.astype(float)
    feature_names = list(X.columns)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Split - no overlap between context and test
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_scaled))
    ctx_idx  = idx[:n_context]
    test_idx = idx[n_context : n_context + n_test]

    # Move to DEVICE immediately
    X_context = torch.tensor(X_scaled[ctx_idx],  dtype=DTYPE).to(DEVICE)
    y_context = torch.tensor(y.values[ctx_idx],   dtype=torch.long).to(DEVICE)
    X_test    = torch.tensor(X_scaled[test_idx],  dtype=DTYPE).to(DEVICE)
    y_test    = torch.tensor(y.values[test_idx],   dtype=torch.long).to(DEVICE)

    print(f"[0A] Context : {X_context.shape} on {X_context.device}")
    print(f"[0A] Test    : {X_test.shape}    on {X_test.device}")
    print(f"[0A] Features: {feature_names}")
    label_pct = y_test.sum().item() / len(y_test) * 100
    print(f"[0A] Label balance (test) - positive: {y_test.sum().item()} / {len(y_test)} ({label_pct:.1f}%)")

    return X_context, y_context, X_test, y_test, feature_names, scaler


# ── [0B] MODEL ───────────────────────────────────────────────────────────────

def load_model(model_name: str = "orion-bix"):
    """
    Load ORION-BiX or ORION-MSP via TabTune's TabularPipeline.

    Per TabTune README (v0.1.9):
        - model_name arg is "OrionBix" / "OrionMSP" (not HuggingFace IDs)
        - tuning_strategy="inference" for zero-shot ICL
        - device passed via tuning_params
        - pipeline.fit(X_train, y_train) sets the context window
        - pipeline.predict(X_test) returns class predictions
        - pipeline.evaluate(X_test, y_test) returns metrics dict

    Returns the TabularPipeline wrapper. Raw nn.Module extracted in [0C].
    """
    if TABTUNE_AVAILABLE:
        print(f"\n[0B] Loading {model_name} via TabTune TabularPipeline...")
        # Map our internal names to TabTune model_name strings from README
        tabtune_name = {
            "orion-bix": "OrionBix",
            "orion-msp": "OrionMSP",
        }[model_name]
        pipeline = TabTunePipeline(
            model_name=tabtune_name,
            task_type="classification",
            tuning_strategy="inference",
            tuning_params={"device": "cuda" if DEVICE.type == "cuda" else "cpu"}
        )
        print(f"[0B] ✅ TabularPipeline({tabtune_name}) ready")
        return pipeline
    else:
        print(f"\n[0B] TabTune unavailable - using structural fallback for {model_name}")
        return _load_fallback_model()


def _load_fallback_model():
    """
    Structural stand-in for ORION-BiX.
    Bi-axial attention (row + column) matching ORION-BiX's design pattern.
    Moved to DEVICE on instantiation.
    """
    import torch.nn as nn

    class BiAxialBlock(nn.Module):
        def __init__(self, d_model=64, n_heads=4):
            super().__init__()
            self.row_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.col_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            r, _ = self.row_attn(x, x, x)
            x = self.norm1(x + r)
            c, _ = self.col_attn(x, x, x)
            x = self.norm2(x + c)
            return x

    class FallbackBiX(nn.Module):
        def __init__(self, n_features=14, n_classes=2, d_model=64, n_layers=4):
            super().__init__()
            self.embed   = nn.Linear(n_features, d_model)
            self.encoder = nn.ModuleList([BiAxialBlock(d_model) for _ in range(n_layers)])
            self.head    = nn.Linear(d_model, n_classes)

        def forward(self, x_ctx, y_ctx, x_test):
            # Simple ICL: embed all tokens, encode, classify last
            x = torch.cat([x_ctx, x_test.unsqueeze(0)], dim=0)  # (ctx+1, features)
            h = self.embed(x).unsqueeze(0)                        # (1, ctx+1, d_model)
            for layer in self.encoder:
                h = layer(h)
            return self.head(h[0, -1])                            # (n_classes,)

    model = FallbackBiX().to(DEVICE)
    print(f"[0B] Fallback model on {next(model.parameters()).device}")
    return model


def extract_raw_module(wrapper) -> torch.nn.Module:
    """
    Extract the raw nn.Module from a TabTune TabularPipeline and move to DEVICE.

    TabularPipeline structure (confirmed from attribute dump):
        pipeline.model          -> OrionBixClassifier  (sklearn-style wrapper, NOT nn.Module)
        pipeline.model.model    -> nn.Module            (the actual PyTorch model)
        pipeline.tuner          -> TuningManager

    OrionBixClassifier is NOT a subclass of nn.Module, so a direct
    isinstance check on pipeline.model fails. We go one level deeper.
    """
    # Already a raw nn.Module (structural fallback path)
    if isinstance(wrapper, torch.nn.Module):
        return wrapper.to(DEVICE)

    # Step 1: get OrionBixClassifier at pipeline.model
    classifier = getattr(wrapper, "model", None)

    # Step 2: find nn.Module inside OrionBixClassifier
    if classifier is not None:
        for attr in ["model", "net", "backbone", "transformer", "base_model", "_model"]:
            raw = getattr(classifier, attr, None)
            if isinstance(raw, torch.nn.Module):
                raw = raw.to(DEVICE)
                print(f"[0C] Raw module at pipeline.model.{attr} "
                      f"-> {type(raw).__name__} on {DEVICE}")
                return raw

        # Step 3: broad scan including private _attrs (model likely stored as _model)
        for attr in sorted(dir(classifier)):
            if attr.startswith("__"):  # skip dunder only, not single underscore
                continue
            try:
                val = getattr(classifier, attr)
                if isinstance(val, torch.nn.Module):
                    val = val.to(DEVICE)
                    print(f"[0C] Raw module at pipeline.model.{attr} "
                          f"-> {type(val).__name__} on {DEVICE}")
                    return val
            except Exception:
                continue

    # Step 4: dump classifier internals to inform next fix
    print(f"[0C] pipeline.model type: {type(classifier).__name__}")
    print("[0C] pipeline.model attributes:")
    for attr in sorted(dir(classifier)):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(classifier, attr, None)
            print(f"     {attr}: {type(val).__name__}")
        except Exception:
            pass
    raise AttributeError("[0C] nn.Module not found. Check dump above and add path.")


# ── [0C] SUBMODULE MAP ───────────────────────────────────────────────────────

def print_submodules(raw_model: torch.nn.Module, max_depth: int = 4):
    """
    Print the full submodule tree. Use the output to build NNsight path strings.
    Run this BEFORE writing any Experiment 2 patching code.
    """
    print("\n[0C] Submodule map - use these paths in NNsight:")
    print("─" * 64)
    for name, module in raw_model.named_modules():
        depth = name.count(".")
        if depth > max_depth:
            continue
        indent = "  " * depth
        tag = " ◀ attention" if any(k in name.lower() for k in ["attn", "attention"]) else ""
        print(f"{indent}{name or '(root)'}: {type(module).__name__}{tag}")
    print("─" * 64)
    print("[0C] Paths marked ◀ attention are the primary NNsight patch targets.")


# ── [0D] CLEAN METRICS ───────────────────────────────────────────────────────

def compute_clean_metrics(wrapper, X_context, y_context, X_test, y_test,
                           model_name="orion-bix", feature_names=None) -> dict:
    """
    Accuracy, F1, AUC-ROC under clean ICL conditions.

    T4 note: inference is batched where possible. For TabTune this is handled
    internally. For the fallback model we loop per test point (acceptable at
    n_test=512 on T4 - ~seconds).

    VRAM monitor: prints peak allocated VRAM after inference.
    """
    print(f"\n[0D] Computing clean metrics for {model_name}...")

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if TABTUNE_AVAILABLE:
        # TabTune's DataProcessor requires pandas DataFrames - NOT numpy arrays.
        # Passing numpy causes: AttributeError: 'numpy.ndarray' has no attribute 'columns'
        # Always wrap with feature_names before passing to fit/predict/evaluate.
        X_ctx_df  = pd.DataFrame(X_context.cpu().numpy(), columns=feature_names)
        X_test_df = pd.DataFrame(X_test.cpu().numpy(),    columns=feature_names)
        y_ctx_s   = pd.Series(y_context.cpu().numpy())
        y_test_s  = pd.Series(y_test.cpu().numpy())

        wrapper.fit(X_ctx_df, y_ctx_s)

        if hasattr(wrapper, "predict_proba"):
            probs     = wrapper.predict_proba(X_test_df)   # (n_test, 2)
            preds     = np.argmax(probs, axis=1)
            pos_probs = probs[:, 1]
        else:
            preds     = wrapper.predict(X_test_df)
            eval_dict = wrapper.evaluate(X_test_df, y_test_s)
            pos_probs = preds.astype(float)
            print(f"[0D] TabTune evaluate() metrics: {eval_dict}")

    else:
        wrapper.eval()
        preds, pos_probs = [], []
        with torch.no_grad():
            for i in range(len(X_test)):
                logits = wrapper(X_context, y_context, X_test[i])
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                preds.append(int(np.argmax(prob)))
                pos_probs.append(float(prob[1]))

    if DEVICE.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[0D] Peak VRAM used: {peak_mb:.1f} MB / {torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")

    y_np = y_test.cpu().numpy()
    metrics = {
        "model":    model_name,
        "accuracy": round(accuracy_score(y_np, preds), 4),
        "f1":       round(f1_score(y_np, preds, zero_division=0), 4),
        "auc":      round(roc_auc_score(y_np, pos_probs), 4),
        "n_context": len(X_context),
        "n_test":    len(X_test),
        "device":    str(DEVICE),
    }

    print(f"[0D] {model_name.upper()} | "
          f"Acc: {metrics['accuracy']} | "
          f"F1: {metrics['f1']} | "
          f"AUC: {metrics['auc']}")
    return metrics


# ── [0E] NNSIGHT LOSSLESS VALIDATION ─────────────────────────────────────────

def validate_nnsight_lossless(raw_model: torch.nn.Module,
                               X_context: torch.Tensor,
                               y_context: torch.Tensor,
                               X_test: torch.Tensor,
                               target_submodule: str = None) -> bool:
    """
    Confirm NNsight identity patch is lossless on the T4.

    T4 caveats handled here
    -----------------------
    1. raw_model must already be on DEVICE before passing to NNsight().
    2. saved_val must be on the same device as the model - .to(DEVICE) enforced.
    3. torch.cuda.synchronize() called before diff comparison to ensure
       async GPU ops are complete.
    4. Tolerance raised slightly to 1e-4 for float32 on T4 (GPU ops are not
       bitwise identical to CPU due to non-deterministic CUDA kernels).
       Set DETERMINISTIC=True above to get 1e-6 tolerance.
    """
    if not NNSIGHT_AVAILABLE:
        print("[0E] NNsight unavailable - HookShim will be used. See [0F].")
        return False

    print("\n[0E] NNsight lossless validation on", DEVICE)

    # Auto-detect attention submodule from 0C output if not specified
    if target_submodule is None:
        target_submodule = _find_first_attention_submodule(raw_model)
        if target_submodule is None:
            print("[0E] ⚠️  No attention submodule found. Pass target_submodule explicitly.")
            return False
        print(f"[0E] Auto-detected patch target: '{target_submodule}'")

    raw_model.eval()
    nn_model = NNsight(raw_model)

    if type(raw_model).__name__ in ["OrionBix", "OrionMSP"]:
        X_seq = torch.cat([X_context, X_test[0].unsqueeze(0)], dim=0).unsqueeze(0)
        y_seq = y_context.unsqueeze(0)
        sample_input = (X_seq, y_seq)
    else:
        sample_input = (X_context, y_context, X_test[0])

    # ── Baseline (no NNsight) ─────────────────────────────────────────────
    with torch.no_grad():
        baseline_out = raw_model(*sample_input)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # ── Save activation ───────────────────────────────────────────────────
    submod = _get_submodule_by_path(nn_model, target_submodule)
    if submod is None:
        print(f"[0E] ❌ Submodule '{target_submodule}' not found.")
        print(f"[0E]    Use print_submodules() output to find the correct path.")
        return False

    # nn.MultiheadAttention returns (attn_output, attn_weights) - a tuple.
    # Always index output[0] to get the attention tensor; output[1] is weights.
    with nn_model.trace(*sample_input):
        saved_raw = submod.output[0].save()

    # NNsight version guard:
    #   ≤0.3: saved_raw is a proxy  → tensor lives at saved_raw.value
    #   ≥0.4: saved_raw IS the tensor after the context exits
    # hasattr handles both without version pinning.
    saved_tensor = saved_raw.value if hasattr(saved_raw, "value") else saved_raw
    saved_val = saved_tensor.detach().clone().to(DEVICE)

    # ── Identity patch ────────────────────────────────────────────────────
    # Patch output[0] in-place; leave attn_weights (output[1]) intact.
    # Replacing the full tuple breaks downstream LayerNorm.
    with nn_model.trace(*sample_input):
        submod.output[0][:] = saved_val
        patched_raw = nn_model.output.save()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    patched_tensor = patched_raw.value if hasattr(patched_raw, "value") else patched_raw
    patched_val = patched_tensor.detach()

    # ── Compare ───────────────────────────────────────────────────────────
    baseline_val = (baseline_out[0] if isinstance(baseline_out, tuple)
                    else baseline_out).detach()

    max_diff = (baseline_val.cpu() - patched_val.cpu()).abs().max().item()
    # Float32 on T4: tolerate up to 1e-4 (non-deterministic CUDA kernels)
    TOLERANCE = 1e-4
    is_lossless = max_diff < TOLERANCE

    if is_lossless:
        print(f"[0E] ✅ LOSSLESS  max_diff={max_diff:.2e}  (tolerance={TOLERANCE})")
        print(f"[0E] NNsight safe for Experiments 2–4 on {DEVICE}.")
    else:
        print(f"[0E] ❌ NOT LOSSLESS  max_diff={max_diff:.4f}")
        print(f"[0E] Triage checklist:")
        print(f"[0E]   1. Is raw_model in eval() mode?  → call raw_model.eval()")
        print(f"[0E]   2. Is the model on DEVICE?       → raw_model.to(DEVICE)")
        print(f"[0E]   3. TabTune wrapper interfering?  → re-extract raw module")
        print(f"[0E]   4. Dynamic control flow?         → use HookShim instead")
        print(f"[0E]   5. Tuple output?                 → patch submod.output[0]")

    return is_lossless


def _find_first_attention_submodule(model):
    for name, _ in model.named_modules():
        if any(k in name.lower() for k in ["row_attn", "col_attn", "attention", "attn"]):
            return name
    return None


def _get_submodule_by_path(nn_model, path):
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ── [0F] HOOK SHIM FALLBACK ───────────────────────────────────────────────────

class HookShim:
    """
    PyTorch hook-based save/patch fallback.
    Used when NNsight torch.fx tracing fails on ORION-BiX's dynamic control flow.

    T4 note: all saved tensors are kept on DEVICE. No .cpu() transfers
    during patching - this avoids PCIe round-trips that would corrupt
    timing measurements in Experiments 3–4.

    Usage
    -----
    shim = HookShim(raw_model)
    shim.register("encoder.0.row_attn")

    # Save
    raw_model.eval()
    with torch.no_grad():
        _ = raw_model(*inputs)
    clean_act = shim.saved["encoder.0.row_attn"]   # on DEVICE

    # Patch
    with shim.patching({"encoder.0.row_attn": clean_act}):
        with torch.no_grad():
            out = raw_model(*inputs)
    """

    def __init__(self, model: torch.nn.Module):
        self.model   = model
        self.saved   = {}
        self._hooks  = []
        self._patches = {}

    def register(self, *paths: str):
        for path in paths:
            mod = self._resolve(path)
            if mod is None:
                raise ValueError(f"[HookShim] Module '{path}' not found.")

            def make_hooks(p):
                def save_hook(module, inp, out):
                    val = out[0] if isinstance(out, tuple) else out
                    self.saved[p] = val.detach().clone()

                def patch_hook(module, inp, out):
                    if p not in self._patches:
                        return out
                    patch = self._patches[p].to(DEVICE)
                    if isinstance(out, tuple):
                        return (patch,) + out[1:]
                    return patch

                return save_hook, patch_hook

            sh, ph = make_hooks(path)
            self._hooks.append(mod.register_forward_hook(sh))
            self._hooks.append(mod.register_forward_hook(ph))
            print(f"[HookShim] Registered hooks on: {path}")

    def patching(self, patch_dict: dict):
        """Context manager for patching. Clears patches on exit."""
        self._patches = patch_dict
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._patches = {}

    def _resolve(self, path):
        mod = self.model
        for part in path.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part, None)
            if mod is None:
                return None
        return mod

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        print("[HookShim] All hooks removed.")


# ── [MAIN] ────────────────────────────────────────────────────────────────────

def run_experiment_0(models=("orion-bix",)):
    print("=" * 64)
    print("EXPERIMENT 0 - Baseline Setup & NNsight Validation")
    print(f"Device: {DEVICE} | Seed: {SEED}")
    print("=" * 64)

    # 0A
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    all_metrics = {}
    nnsight_ok = False
    raw_primary = None

    for model_name in models:
        print(f"\n{'─'*64}\nModel: {model_name.upper()}\n{'─'*64}")

        wrapper = load_model(model_name)

        # IMPORTANT: fit() must run before extract_raw_module().
        # OrionBixClassifier uses lazy loading - the internal nn.Module
        # is not instantiated until the first fit() call.
        metrics = compute_clean_metrics(wrapper, X_ctx, y_ctx, X_test, y_test, model_name, feat_names)
        all_metrics[model_name] = metrics

        # Now extract the raw module - weights are loaded after fit()
        raw = extract_raw_module(wrapper)
        print_submodules(raw)

        if model_name == models[0]:
            raw_primary = raw
            nnsight_ok = validate_nnsight_lossless(raw, X_ctx, y_ctx, X_test)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EXPERIMENT 0 - DAY 1 GATE SUMMARY")
    print("=" * 64)
    df = pd.DataFrame(all_metrics).T[["accuracy", "f1", "auc"]]
    print(df.to_string())
    print(f"\nNNsight : {'✅ PASS' if nnsight_ok else '❌ FAIL - activate HookShim'}")
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"VRAM    : {torch.cuda.memory_allocated()/1e6:.1f} MB allocated")

    if nnsight_ok:
        print("\n✅ All checks passed. Proceed to exp1_baselines.py")
    else:
        print("\n⚠️  NNsight failed. Instantiate HookShim before Experiment 2:")
        print("    shim = HookShim(raw_model)")
        print("    shim.register('YOUR_ATTN_PATH')  # from 0C output")

    return all_metrics, feat_names, scaler, raw_primary, nnsight_ok


# Entry point
if __name__ == "__main__":
    results, feat_names, scaler, raw_model, nnsight_ok = run_experiment_0(
        models=["orion-bix"]
        # Add "orion-msp" for transferability baseline (Experiment 5)
    )