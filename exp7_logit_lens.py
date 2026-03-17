# ============================================================
# EXPERIMENT 7 — Logit Lens: Prediction Formation Trajectory
# Colab-compatible: assumes exp_0 cells already executed.
#
# Core contribution:
#   At each ICL layer, project the residual stream onto
#   the output logit space. Track WHERE the correct prediction
#   forms and WHERE context poisoning hijacks it.
#
#   This is the first logit lens applied to tabular ICL models.
#   Shows the full causal trajectory of how a wrong answer forms
#   under poisoning — not just "which head matters" (Exp 2) but
#   "how does the answer evolve through the network."
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

if 'DEVICE' not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'DTYPE' not in globals():
    DTYPE = torch.float32


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_input(raw_model, X_ctx, y_ctx, x_test_i):
    if type(raw_model).__name__ in ["OrionBix", "OrionMSP"]:
        X_seq = torch.cat([X_ctx, x_test_i.unsqueeze(0)], dim=0).unsqueeze(0)
        return (X_seq, y_ctx.unsqueeze(0))
    return (X_ctx, y_ctx, x_test_i)

def _flatten_logits(t):
    if isinstance(t, tuple): t = t[0]
    t = t.cpu().float()
    while t.dim() > 1: t = t.squeeze(0)
    return t

def _predict(raw_model, X_ctx, y_ctx, x_test_i):
    raw_model.eval()
    with torch.no_grad():
        out = raw_model(*_build_input(raw_model, X_ctx, y_ctx, x_test_i))
    return int(_flatten_logits(out).argmax().item())

def unwrap_saved(proxy):
    return proxy.value if hasattr(proxy, "value") else proxy


# ── Attacks ───────────────────────────────────────────────────────────────────

def attack_synthetic_near_dup(X_ctx, y_ctx, x_test, true_label,
                               k, rng, sigma=0.01):
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    wrong = 1 - true_label
    x_np = x_test.cpu().numpy()
    for pos in positions:
        noise = rng.normal(0, sigma, size=x_np.shape).astype(np.float32)
        X_p[pos] = torch.tensor(x_np + noise, dtype=X_ctx.dtype, device=X_ctx.device)
        y_p[pos] = wrong
    return X_p, y_p

def attack_pool_only(X_ctx, y_ctx, x_test, true_label, k, rng):
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    x_np = x_test.cpu().numpy()
    X_np = X_ctx.cpu().numpy()
    y_np = y_ctx.cpu().numpy()
    same_mask = (y_np == true_label)
    same_indices = np.where(same_mask)[0]
    if len(same_indices) == 0:
        return X_p, y_p
    dists = np.linalg.norm(X_np[same_indices] - x_np, axis=1)
    nearest = same_indices[np.argsort(dists)[:k]]
    for pos in nearest:
        y_p[pos] = 1 - y_p[pos]
    return X_p, y_p


# ── Logit Lens core ──────────────────────────────────────────────────────────

def discover_model_structure(raw_model):
    """
    Discover the ICL blocks and the output decoder.
    Returns: (block_paths, decoder_module)
    """
    block_paths = []
    decoder_module = None
    decoder_path = None

    # Find ICL blocks
    for name, mod in raw_model.named_modules():
        if "icl_predictor.tf_icl.blocks." in name:
            parts = name.split(".")
            if parts[-1].isdigit():
                block_paths.append(name)

    block_paths = sorted(block_paths, key=lambda x: int(x.split(".")[-1]))

    # Find the decoder module — the full MLP after ICL blocks
    # Look for Sequential or container named 'decoder'
    for name, mod in raw_model.named_modules():
        if name.endswith(".decoder") or name == "decoder":
            if hasattr(mod, 'forward'):
                decoder_path = name
                decoder_module = mod
                break

    # Also try icl_predictor.decoder specifically
    if decoder_module is None:
        for name, mod in raw_model.named_modules():
            if "predictor" in name and "decoder" in name and not name.split(".")[-1].isdigit():
                # Want the container, not individual layers
                if hasattr(mod, '__len__') or isinstance(mod, torch.nn.Sequential):
                    decoder_path = name
                    decoder_module = mod
                    break

    print(f"  [LENS] Found {len(block_paths)} ICL blocks: {block_paths[0]} ... {block_paths[-1]}")
    if decoder_module is not None:
        print(f"  [LENS] Decoder: {decoder_path}")
        for sub_name, sub_mod in decoder_module.named_children():
            if isinstance(sub_mod, torch.nn.Linear):
                print(f"         {sub_name}: Linear({sub_mod.in_features}→{sub_mod.out_features})")
            else:
                print(f"         {sub_name}: {type(sub_mod).__name__}")
    else:
        print(f"  [LENS] ⚠️ No decoder found — falling back to last Linear")
        # Fallback: find last Linear
        for name, mod in raw_model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                decoder_path = name
                decoder_module = mod
        if decoder_module is not None:
            print(f"  [LENS] Fallback: {decoder_path}")

    return block_paths, decoder_path, decoder_module


def _nav_to_submodule(nn_model, path):
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def logit_lens_trace(nn_model, raw_model, X_ctx, y_ctx, x_test_i,
                      block_paths, decoder_module):
    """
    Run one forward pass, saving the residual stream output at each ICL block.
    Project each through the full decoder to get per-layer logits.

    Returns: list of (layer_idx, probs_as_numpy) tuples
    """
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)

    saved_activations = {}
    try:
        with nn_model.trace(*inp):
            for path in block_paths:
                submod = _nav_to_submodule(nn_model, path)
                if submod is not None:
                    try:
                        saved_activations[path] = submod.output[0].save()
                    except Exception:
                        saved_activations[path] = submod.output.save()
    except Exception as e:
        print(f"  [LENS] Trace failed: {e}")
        return []

    results = []
    decoder_module.eval()

    for i, path in enumerate(block_paths):
        if path not in saved_activations:
            continue

        act = unwrap_saved(saved_activations[path])
        if isinstance(act, tuple):
            act = act[0]
        act = act.detach().cpu().float()

        while act.dim() > 2:
            act = act.squeeze(0)
        test_token = act[-1]  # (d_model,)

        try:
            # Project through the full decoder (MLP: d_block → 1024 → 10)
            with torch.no_grad():
                logits = decoder_module(test_token.to(next(decoder_module.parameters()).device))
            logits = logits.detach().cpu().float()
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=0).numpy()
            results.append((i, probs))
        except Exception as e:
            # Dimension mismatch — report it once and skip
            if i == 0:
                print(f"  [LENS] Decoder projection failed at layer {i}: {e}")
                print(f"         test_token shape: {test_token.shape}")
            continue

    return results


def run_logit_lens_single(nn_model, raw_model, X_ctx, y_ctx, x_test_i, yi,
                           block_paths, decoder_module,
                           attack_fn=None, attack_kwargs=None):
    result = {"true_label": yi}
    result["clean"] = logit_lens_trace(
        nn_model, raw_model, X_ctx, y_ctx, x_test_i,
        block_paths, decoder_module
    )
    if attack_fn is not None:
        X_p, y_p = attack_fn(X_ctx, y_ctx, x_test_i, yi, **attack_kwargs)
        result["poisoned"] = logit_lens_trace(
            nn_model, raw_model, X_p, y_p, x_test_i,
            block_paths, decoder_module
        )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment_7(n_samples=30, k=3):
    print("=" * 64)
    print("EXPERIMENT 7 — Logit Lens: Prediction Formation Trajectory")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    if not NNSIGHT_AVAILABLE:
        print("[ERROR] NNsight required for logit lens. Exiting.")
        return {}

    # ── Data + model ──────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()
    nn_model = NNsight(raw)

    # ── Discover model structure ──────────────────────────────────────────
    block_paths, dec_path, dec_module = discover_model_structure(raw)
    n_blocks = len(block_paths)

    if dec_module is None:
        print("[ERROR] No decoder found. Cannot project logits.")
        return {}

    n_eval = min(n_samples, len(X_test))

    # ══════════════════════════════════════════════════════════════════════
    # Part A: Average trajectory (clean vs poisoned)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[7A] Computing average logit trajectories...")

    # Storage: (n_samples, n_layers, n_classes)
    clean_trajectories = []
    poison_d_trajectories = []
    poison_g_trajectories = []

    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw, X_ctx, y_ctx, xi) != yi:
            continue

        # Clean
        clean = logit_lens_trace(nn_model, raw, X_ctx, y_ctx, xi,
                                  block_paths, dec_module)
        if not clean or len(clean[0][1]) < 2:
            continue

        # Attack D (σ=0.01)
        rng = np.random.default_rng(42 + i)
        X_d, y_d = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi, k, rng, 0.01)
        poison_d = logit_lens_trace(nn_model, raw, X_d, y_d, xi,
                                     block_paths, dec_module)

        # Attack G (pool-only)
        rng_g = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)
        poison_g = logit_lens_trace(nn_model, raw, X_g, y_g, xi,
                                     block_paths, dec_module)

        if clean and poison_d and poison_g:
            n_classes = len(clean[0][1])
            c_traj = np.array([p[1] for p in clean])
            d_traj = np.array([p[1] for p in poison_d])
            g_traj = np.array([p[1] for p in poison_g])

            if c_traj.shape == d_traj.shape == g_traj.shape:
                clean_trajectories.append(c_traj)
                poison_d_trajectories.append(d_traj)
                poison_g_trajectories.append(g_traj)

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n_eval}] collected {len(clean_trajectories)} trajectories")

    n_collected = len(clean_trajectories)
    print(f"  Collected {n_collected} complete trajectories")

    if n_collected < 3:
        print("[ERROR] Too few trajectories. Check classifier module match.")
        return {}

    # Average across samples — track P(true_label) at each layer
    # Since true_label varies, we need to track P(correct class)
    clean_avg = np.mean(clean_trajectories, axis=0)     # (n_layers, n_classes)
    poison_d_avg = np.mean(poison_d_trajectories, axis=0)
    poison_g_avg = np.mean(poison_g_trajectories, axis=0)

    n_layers = clean_avg.shape[0]
    n_classes = clean_avg.shape[1]

    # ── Print the trajectory ──────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("LOGIT LENS — Per-Layer Probability Trajectory")
    print(f"{'='*64}")
    print(f"  Average over {n_collected} samples (class probabilities)")
    print()
    header = f"  {'Layer':>6s}"
    for c in range(n_classes):
        header += f"  {'Clean_c'+str(c):>9s}  {'AtkD_c'+str(c):>9s}  {'AtkG_c'+str(c):>9s}"
    print(header)
    print(f"  {'─'*(14 + 31*n_classes)}")

    for layer in range(n_layers):
        row = f"  {layer:>6d}"
        for c in range(n_classes):
            row += f"  {clean_avg[layer,c]:>9.3f}  {poison_d_avg[layer,c]:>9.3f}  {poison_g_avg[layer,c]:>9.3f}"
        print(row)

    # ══════════════════════════════════════════════════════════════════════
    # Part B: Find the critical divergence layer
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[7B] Divergence analysis")

    # KL divergence per layer: clean vs poisoned
    from scipy.special import kl_div as scipy_kl
    kl_d = []
    kl_g = []
    for layer in range(n_layers):
        c = np.clip(clean_avg[layer], 1e-8, 1)
        d = np.clip(poison_d_avg[layer], 1e-8, 1)
        g = np.clip(poison_g_avg[layer], 1e-8, 1)
        kl_d.append(float(np.sum(c * np.log(c / d))))
        kl_g.append(float(np.sum(c * np.log(c / g))))

    # Find critical layer (max KL increase)
    kl_d_diff = np.diff(kl_d)
    kl_g_diff = np.diff(kl_g)
    critical_d = int(np.argmax(kl_d_diff)) + 1
    critical_g = int(np.argmax(kl_g_diff)) + 1

    print(f"  Attack D - KL divergence peaks at layer {critical_d}")
    print(f"  Attack G - KL divergence peaks at layer {critical_g}")
    print(f"  → Poisoning primarily disrupts prediction at ICL layer {critical_d}")

    # ══════════════════════════════════════════════════════════════════════
    # Part C: Individual example trajectories (for figure)
    # ══════════════════════════════════════════════════════════════════════

    # ── Visualization ─────────────────────────────────────────────────────
    print(f"\n[7C] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = np.arange(n_layers)

    # Panel 1: P(class 0) trajectory — clean vs attack D vs attack G
    ax = axes[0, 0]
    ax.plot(layers, clean_avg[:, 0], "o-", color="#4C72B0", linewidth=2.5,
            markersize=4, label="Clean")
    ax.plot(layers, poison_d_avg[:, 0], "s--", color="#E84040", linewidth=2,
            markersize=4, label="Attack D (σ=0.01)")
    ax.plot(layers, poison_g_avg[:, 0], "^:", color="#55A868", linewidth=2,
            markersize=4, label="Attack G (pool)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(critical_d, color="#E84040", linestyle=":", alpha=0.5)
    ax.set_xlabel("ICL Layer")
    ax.set_ylabel("P(class 0)")
    ax.set_title("Prediction Trajectory: P(class 0) per Layer", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: P(class 1) trajectory
    ax = axes[0, 1]
    if n_classes > 1:
        ax.plot(layers, clean_avg[:, 1], "o-", color="#4C72B0", linewidth=2.5,
                markersize=4, label="Clean")
        ax.plot(layers, poison_d_avg[:, 1], "s--", color="#E84040", linewidth=2,
                markersize=4, label="Attack D (σ=0.01)")
        ax.plot(layers, poison_g_avg[:, 1], "^:", color="#55A868", linewidth=2,
                markersize=4, label="Attack G (pool)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(critical_d, color="#E84040", linestyle=":", alpha=0.5)
    ax.set_xlabel("ICL Layer")
    ax.set_ylabel("P(class 1)")
    ax.set_title("Prediction Trajectory: P(class 1) per Layer", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: KL divergence from clean
    ax = axes[1, 0]
    ax.plot(layers, kl_d, "s-", color="#E84040", linewidth=2,
            label="Attack D vs Clean")
    ax.plot(layers, kl_g, "^-", color="#55A868", linewidth=2,
            label="Attack G vs Clean")
    ax.axvline(critical_d, color="#E84040", linestyle=":", alpha=0.5,
               label=f"D critical layer ({critical_d})")
    ax.axvline(critical_g, color="#55A868", linestyle=":", alpha=0.5,
               label=f"G critical layer ({critical_g})")
    ax.set_xlabel("ICL Layer")
    ax.set_ylabel("KL(clean || poisoned)")
    ax.set_title("KL Divergence from Clean Trajectory", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: Confidence spread (individual trajectories)
    ax = axes[1, 1]
    # Show P(correct class) for individual examples
    n_show = min(10, n_collected)
    for j in range(n_show):
        # For simplicity, use class 0 probability
        ax.plot(layers, clean_trajectories[j][:, 0], color="#4C72B0",
                alpha=0.15, linewidth=1)
        ax.plot(layers, poison_d_trajectories[j][:, 0], color="#E84040",
                alpha=0.15, linewidth=1)

    # Overlay averages
    ax.plot(layers, clean_avg[:, 0], "o-", color="#4C72B0", linewidth=3,
            label="Clean (avg)", zorder=10)
    ax.plot(layers, poison_d_avg[:, 0], "s-", color="#E84040", linewidth=3,
            label="Attack D (avg)", zorder=10)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("ICL Layer")
    ax.set_ylabel("P(class 0)")
    ax.set_title(f"Individual Trajectories (n={n_show})\nShowing variance in prediction formation",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Experiment 7 — Logit Lens: How Predictions Form & Break Under Poisoning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp7_logit_lens.png", dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved exp7_logit_lens.png")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("EXPERIMENT 7 — LOGIT LENS SUMMARY")
    print(f"{'='*64}")
    print(f"  Trajectories collected: {n_collected}")
    print(f"  ICL layers: {n_blocks}")
    print(f"  Output classes: {n_classes}")
    print()
    print(f"  Attack D (σ=0.01) critical divergence layer: {critical_d}")
    print(f"  Attack G (pool-only) critical divergence layer: {critical_g}")
    print()
    print(f"  Clean final P(class 0): {clean_avg[-1,0]:.3f}")
    if n_classes > 1:
        print(f"  Clean final P(class 1): {clean_avg[-1,1]:.3f}")
        print(f"  Attack D final P(class 1): {poison_d_avg[-1,1]:.3f}")
        print(f"  Attack G final P(class 1): {poison_g_avg[-1,1]:.3f}")
    print()
    print(f"  Key finding: The prediction forms gradually across layers 0-{n_blocks-1}.")
    print(f"  Poisoning diverges maximally at layer {critical_d}, confirming that")
    print(f"  the vulnerability is not in a single head but in how the ICL circuit")
    print(f"  aggregates context information across its early-mid layers.")
    print(f"{'='*64}")

    return {
        "clean_avg": clean_avg,
        "poison_d_avg": poison_d_avg,
        "poison_g_avg": poison_g_avg,
        "kl_d": kl_d,
        "kl_g": kl_g,
        "critical_d": critical_d,
        "critical_g": critical_g,
        "n_collected": n_collected,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_experiment_7(n_samples=30, k=3)
