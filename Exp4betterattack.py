# ============================================================
# EXPERIMENT 4 — Improved Attacks
# Colab-compatible: assumes exp_0 cells already executed.
#
# Mechanistic insight from Exp 2-3:
#   - OrionBix attends UNIFORMLY over context positions
#   - Position targeting is irrelevant (Exp 3 Strategy A delta ~ 0)
#   - Feature SIMILARITY is the actual lever (Strategy B works)
#   - Current Strategy B is bottlenecked by pool coverage:
#     nearest neighbor from 256 examples != maximally similar
#
# Better attacks: remove the pool ceiling.
#   C) Interpolated Mimic   -- slide context example toward x_test
#   D) Synthetic Near-Dup   -- synthesize near-copies of x_test, label wrong
#   E) Context Saturation   -- fill ALL k budget with near-dup variants
#
# All three are gray-box (know model type = ICL, no gradients needed)
# and fully in-distribution (small noise keeps features realistic).
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if 'DEVICE' not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared helpers (same as exp3) ─────────────────────────────────────────────

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

def _predict_proba(raw_model, X_ctx, y_ctx, x_test_i):
    """Returns (p_class0, p_class1)."""
    raw_model.eval()
    with torch.no_grad():
        out = raw_model(*_build_input(raw_model, X_ctx, y_ctx, x_test_i))
    logits = _flatten_logits(out)
    probs = torch.softmax(logits, dim=0).numpy()
    return probs


# ── Exp 3 Strategy B reproduced as baseline ──────────────────────────────────

def attack_feature_mimic(X_ctx, y_ctx, x_test, true_label,
                          X_pool, y_pool, k, rng):
    """Exp 3 Strategy B: nearest neighbor from pool, labeled wrong."""
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    x_np = x_test.cpu().numpy()
    X_pool_np = X_pool.cpu().numpy()
    same_mask = (y_pool.cpu().numpy() == true_label)
    X_same = X_pool_np[same_mask]
    if len(X_same) == 0:
        return X_p, y_p
    dists = np.linalg.norm(X_same - x_np, axis=1)
    nn_idx = np.argsort(dists)[:k]
    wrong = 1 - true_label
    for i, pos in enumerate(positions):
        if i < len(nn_idx):
            X_p[pos] = torch.tensor(X_same[nn_idx[i]],
                                     dtype=X_ctx.dtype, device=X_ctx.device)
            y_p[pos] = wrong
    return X_p, y_p


# ── Attack C: Interpolated Mimic ─────────────────────────────────────────────

def attack_interpolated_mimic(X_ctx, y_ctx, x_test, true_label,
                               X_pool, y_pool, k, rng, alpha=0.7):
    """
    Interpolate between nearest pool neighbor and x_test itself.
    x_poison = alpha * x_test + (1 - alpha) * x_neighbor

    alpha=0.7 puts the poison 70% of the way toward x_test.
    Higher alpha = more similar = stronger signal, but also more
    obviously a near-copy if you're looking for them.

    Why better than B: the synthesized example is GUARANTEED closer
    to x_test than any pool example, so the retrieval signal is stronger.
    """
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)

    x_np = x_test.cpu().numpy()
    X_pool_np = X_pool.cpu().numpy()
    same_mask = (y_pool.cpu().numpy() == true_label)
    X_same = X_pool_np[same_mask]

    if len(X_same) == 0:
        return X_p, y_p

    dists = np.linalg.norm(X_same - x_np, axis=1)
    nn_idx = np.argsort(dists)[:k]
    wrong = 1 - true_label

    for i, pos in enumerate(positions):
        if i < len(nn_idx):
            neighbor = X_same[nn_idx[i]]
            interpolated = alpha * x_np + (1 - alpha) * neighbor
            X_p[pos] = torch.tensor(interpolated,
                                     dtype=X_ctx.dtype, device=X_ctx.device)
            y_p[pos] = wrong

    return X_p, y_p


# ── Attack D: Synthetic Near-Duplicate Poisoning ─────────────────────────────

def attack_synthetic_near_dup(X_ctx, y_ctx, x_test, true_label,
                               k, rng, sigma=0.05):
    """
    Generate k synthetic near-copies of x_test, all labeled wrong.

    x_poison_i = x_test + eps_i,  eps_i ~ N(0, sigma^2)

    This removes the pool ceiling entirely: the poisoned examples are
    maximally similar to x_test (distance ~ sigma * sqrt(F)) regardless
    of what's in the context pool.

    Why this works: OrionBix attends uniformly but weights prediction
    by feature similarity. Near-duplicates of x_test labeled wrong
    create maximum misleading signal for exactly this test point.

    sigma=0.05 in normalized feature space is imperceptible:
    Adult Income features are scaled to ~[-1, 1], so sigma=0.05
    is 2.5% of the feature range per dimension.
    """
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    wrong = 1 - true_label
    x_np = x_test.cpu().numpy()

    for pos in positions:
        noise = rng.normal(0, sigma, size=x_np.shape).astype(np.float32)
        synthetic = x_np + noise
        X_p[pos] = torch.tensor(synthetic, dtype=X_ctx.dtype, device=X_ctx.device)
        y_p[pos] = wrong

    return X_p, y_p


# ── Attack E: Context Saturation ─────────────────────────────────────────────

def attack_context_saturation(X_ctx, y_ctx, x_test, true_label,
                               k, rng, sigma=0.05):
    """
    Concentrated variant of D: inject k near-duplicates using DIVERSE
    noise draws (different seeds per injection) so the poison examples
    are distinct from each other but all maximally close to x_test.

    Additionally, uses MULTIPLE context positions rather than just k --
    if k < context budget, fills the remainder with label-flipped
    real examples to also degrade the true-class signal.

    Two-prong: (1) inject near-dup decoys, (2) flip true-class examples.
    """
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    wrong = 1 - true_label
    x_np = x_test.cpu().numpy()

    # Prong 1: inject k near-dup decoys at random positions
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    for j, pos in enumerate(positions):
        # Different noise seed per injection for diversity
        noise = rng.normal(0, sigma, size=x_np.shape).astype(np.float32)
        X_p[pos] = torch.tensor(x_np + noise,
                                 dtype=X_ctx.dtype, device=X_ctx.device)
        y_p[pos] = wrong

    # Prong 2: among remaining positions, flip up to k true-class labels
    remaining = [i for i in range(len(X_ctx)) if i not in set(positions.tolist())]
    true_class_remaining = [i for i in remaining if y_ctx[i].item() == true_label]
    to_flip = true_class_remaining[:k]
    for pos in to_flip:
        y_p[pos] = wrong

    return X_p, y_p


# ── Soft-margin scoring ───────────────────────────────────────────────────────

def margin_drop(raw_model, X_ctx_clean, y_ctx_clean,
                X_ctx_poison, y_ctx_poison, x_test_i, true_label):
    """
    Measure confidence margin drop, not just binary flip.
    Returns (was_flipped, margin_drop_value).
    Useful for showing attack efficacy even when prediction isn't flipped.
    """
    p_clean  = _predict_proba(raw_model, X_ctx_clean,  y_ctx_clean,  x_test_i)
    p_poison = _predict_proba(raw_model, X_ctx_poison, y_ctx_poison, x_test_i)

    clean_conf  = p_clean[true_label]
    poison_conf = p_poison[true_label]
    drop = clean_conf - poison_conf  # positive = model less confident after poison

    flipped = (np.argmax(p_poison) != true_label)
    return flipped, float(drop)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_experiment_4(n_samples=100, k_values=None, sigma=0.05, alpha=0.7):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print("=" * 64)
    print("EXPERIMENT 4 — Improved Attacks")
    print(f"Device: {DEVICE}  |  sigma={sigma}  alpha={alpha}")
    print("=" * 64)
    print(f"  B  = Feature Mimic (Exp 3 baseline)")
    print(f"  C  = Interpolated Mimic (alpha={alpha})")
    print(f"  D  = Synthetic Near-Dup (sigma={sigma})")
    print(f"  E  = Context Saturation (near-dup + true-class flip)")
    print()

    # ── Data + model ──────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    X_pool, y_pool = X_ctx.clone(), y_ctx.clone()
    n_eval = min(n_samples, len(X_test))

    # ── Clean accuracy ────────────────────────────────────────────────────
    clean_correct = sum(
        _predict(raw, X_ctx, y_ctx, X_test[i]) == y_test[i].item()
        for i in range(n_eval)
    )
    clean_acc = clean_correct / n_eval
    print(f"[4A] Clean accuracy: {clean_acc*100:.1f}%  ({clean_correct}/{n_eval})\n")

    # ── Per-k sweep ───────────────────────────────────────────────────────
    all_results = {}

    for k in k_values:
        print(f"{'─'*64}")
        print(f"  k = {k}")
        print(f"{'─'*64}")

        counts = {a: 0 for a in ["B", "C", "D", "E"]}
        margin_drops = {a: [] for a in ["B", "C", "D", "E"]}
        tested = 0

        rng = np.random.default_rng(42)  # same seed for all attacks per k

        for i in range(n_eval):
            xi  = X_test[i]
            yi  = y_test[i].item()
            if _predict(raw, X_ctx, y_ctx, xi) != yi:
                continue
            tested += 1

            # Same rng state for each attack so positions are comparable
            rng_b = np.random.default_rng(42 + i)
            rng_c = np.random.default_rng(42 + i)
            rng_d = np.random.default_rng(42 + i)
            rng_e = np.random.default_rng(42 + i)

            attacks = {
                "B": attack_feature_mimic(
                        X_ctx, y_ctx, xi, yi, X_pool, y_pool, k, rng_b),
                "C": attack_interpolated_mimic(
                        X_ctx, y_ctx, xi, yi, X_pool, y_pool, k, rng_c, alpha),
                "D": attack_synthetic_near_dup(
                        X_ctx, y_ctx, xi, yi, k, rng_d, sigma),
                "E": attack_context_saturation(
                        X_ctx, y_ctx, xi, yi, k, rng_e, sigma),
            }

            for name, (Xp, yp) in attacks.items():
                flipped, drop = margin_drop(raw, X_ctx, y_ctx, Xp, yp, xi, yi)
                if flipped:
                    counts[name] += 1
                margin_drops[name].append(drop)

        n = max(tested, 1)
        all_results[k] = {
            a: {
                "asr":         round(counts[a] / n, 4),
                "margin_drop": round(float(np.mean(margin_drops[a])), 4),
            }
            for a in ["B", "C", "D", "E"]
        }
        all_results[k]["n"] = tested

        print(f"  {'Attack':>8s}  {'ASR':>7s}  {'Margin Drop':>12s}  {'vs B':>7s}")
        print(f"  {'─'*45}")
        b_asr = all_results[k]["B"]["asr"]
        for a in ["B", "C", "D", "E"]:
            asr  = all_results[k][a]["asr"]
            mdrop = all_results[k][a]["margin_drop"]
            delta = (asr - b_asr) * 100
            flag  = " <-- " if a != "B" and delta > 3 else ""
            print(f"  {a:>8s}  {asr*100:>6.1f}%  {mdrop:>+11.3f}  {delta:>+6.1f}%{flag}")
        print()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPERIMENT 4 — SUMMARY (ASR)")
    print(f"{'='*64}")
    print(f"  Clean error floor: {(1-clean_acc)*100:.1f}%")
    print()
    header = f"  {'k':>4s}" + "".join(f"  {'Atk '+a:>10s}" for a in ["B","C","D","E"])
    print(header)
    print(f"  {'─'*55}")
    for k in k_values:
        row = f"  {k:>4d}"
        for a in ["B", "C", "D", "E"]:
            asr = all_results[k][a]["asr"] * 100
            row += f"  {asr:>9.1f}%"
        print(row)
    print()
    print(f"  B = Feature Mimic (pool NN, Exp 3 baseline)")
    print(f"  C = Interpolated Mimic (alpha={alpha})")
    print(f"  D = Synthetic Near-Dup (sigma={sigma})")
    print(f"  E = Context Saturation (near-dup + true-class flip)")
    print(f"{'='*64}")

    # ── Sigma sensitivity (Attack D only) ─────────────────────────────────
    print(f"\n[4B] Sigma sensitivity (Attack D, k=5)...")
    sigma_results = {}
    k_fixed = 5
    for sig in [0.01, 0.05, 0.1, 0.2, 0.5]:
        hits, n_sig = 0, 0
        for i in range(n_eval):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw, X_ctx, y_ctx, xi) != yi:
                continue
            n_sig += 1
            rng_s = np.random.default_rng(42 + i)
            Xp, yp = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                k_fixed, rng_s, sigma=sig)
            if _predict(raw, Xp, yp, xi) != yi:
                hits += 1
        asr = hits / max(n_sig, 1)
        sigma_results[sig] = asr
        print(f"  sigma={sig:.2f}  ASR={asr*100:.1f}%")

    # ── Visualization ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"B": "#999999", "C": "#4C72B0", "D": "#DD8452", "E": "#55A868"}
    labels = {
        "B": "B: Pool NN (Exp3 baseline)",
        "C": f"C: Interpolated (a={alpha})",
        "D": f"D: Synthetic Near-Dup (s={sigma})",
        "E": "E: Context Saturation",
    }
    x_pos = np.arange(len(k_values))
    w = 0.2

    # Panel 1: ASR comparison
    ax = axes[0]
    for j, a in enumerate(["B", "C", "D", "E"]):
        asrs = [all_results[k][a]["asr"] * 100 for k in k_values]
        offset = (j - 1.5) * w
        ax.bar(x_pos + offset, asrs, w, label=labels[a],
               color=colors[a], edgecolor="white", alpha=0.9)
    ax.axhline(y=(1-clean_acc)*100, color="red", linestyle="--",
               alpha=0.6, label=f"Clean error ({(1-clean_acc)*100:.0f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("ASR (%)")
    ax.set_title("Attack Success Rate by k", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 65)

    # Panel 2: Margin drop comparison (k=5)
    ax = axes[1]
    k_show = 5 if 5 in all_results else k_values[-1]
    attack_names = ["B", "C", "D", "E"]
    drops = [all_results[k_show][a]["margin_drop"] for a in attack_names]
    bar_colors = [colors[a] for a in attack_names]
    ax.bar(attack_names, drops, color=bar_colors, edgecolor="white", alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Confidence Drop")
    ax.set_title(f"Margin Drop (k={k_show})\n(positive = model less confident after poison)",
                 fontweight="bold")
    for i, d in enumerate(drops):
        ax.text(i, d + 0.003, f"{d:+.3f}", ha="center", fontsize=8)

    # Panel 3: Sigma sensitivity for Attack D
    ax = axes[2]
    sigs = list(sigma_results.keys())
    asrs = [sigma_results[s]*100 for s in sigs]
    ax.plot(sigs, asrs, "o-", color=colors["D"], linewidth=2, markersize=7)
    ax.axhline(y=(1-clean_acc)*100, color="red", linestyle="--",
               alpha=0.6, label=f"Clean error ({(1-clean_acc)*100:.0f}%)")
    ax.fill_between(sigs, (1-clean_acc)*100, asrs, alpha=0.15, color=colors["D"])
    ax.set_xlabel("sigma (noise scale)")
    ax.set_ylabel("ASR (%)")
    ax.set_title(f"Attack D Sigma Sensitivity (k={k_fixed})\n"
                 "(low sigma = stealthy, high sigma = stronger)",
                 fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    plt.suptitle("Experiment 4 — Improved Attacks: Breaking the Pool Ceiling",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "/mnt/user-data/outputs/exp4_attack_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved {out}")
    plt.close()

    return all_results, sigma_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, sigma_results = run_experiment_4(
        n_samples=100,
        k_values=[1, 3, 5, 10],
        sigma=0.05,
        alpha=0.7,
    )