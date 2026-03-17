# ============================================================
# EXPERIMENT 3 — Circuit-Guided Context Poisoning Attack
# Colab-compatible: assumes exp_0 cells already executed.
# Uses circuit atlas from Experiment 2.
# ============================================================
#
# Key insight from Exp 2:
#   ICL predictor is the ONLY causal stage (COL=0%, ROW=0%)
#   blocks.0.attn = 14% restoration (most critical head)
#   blocks.1-3.attn = 4-6% (secondary)
#
# Attack strategies:
#   A) Targeted Label Flip — flip labels at circuit-important
#      positions (tests: does WHERE you poison matter?)
#   B) Feature Mimic — insert test-similar examples with wrong
#      labels at important positions (tests: full circuit hijack)
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

if 'DEVICE' not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def unwrap_saved(proxy):
    return proxy.value if hasattr(proxy, "value") else proxy

def _nav_to_submodule(nn_model, path):
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj

def _build_input(raw_model, X_ctx, y_ctx, x_test_i):
    if type(raw_model).__name__ in ["OrionBix", "OrionMSP"]:
        X_seq = torch.cat([X_ctx, x_test_i.unsqueeze(0)], dim=0).unsqueeze(0)
        y_seq = y_ctx.unsqueeze(0)
        return (X_seq, y_seq)
    return (X_ctx, y_ctx, x_test_i)

def _flatten_logits(t):
    if isinstance(t, tuple): t = t[0]
    t = t.cpu().float()
    while t.dim() > 1: t = t.squeeze(0)
    return t

def _predict(raw_model, X_ctx, y_ctx, x_test_i):
    raw_model.eval()
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    with torch.no_grad():
        out = raw_model(*inp)
    return int(_flatten_logits(out).argmax().item())


# ── Circuit profiling ─────────────────────────────────────────────────────────

ICL_ATTN_TARGETS = [
    "icl_predictor.tf_icl.blocks.0",
    "icl_predictor.tf_icl.blocks.1",
    "icl_predictor.tf_icl.blocks.2",
    "icl_predictor.tf_icl.blocks.3",
]

def profile_context_importance(nn_model, raw_model, X_ctx, y_ctx, x_test_i,
                                target_paths=None):
    """
    Measure activation norms at ICL attn heads → importance per context position.
    Higher norm = the model relies more on that position for prediction.
    """
    if target_paths is None:
        target_paths = ICL_ATTN_TARGETS

    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    ctx_size = len(X_ctx)
    total_norms = np.zeros(ctx_size + 1)

    for path in target_paths:
        submod = _nav_to_submodule(nn_model, path)
        if submod is None:
            continue
        try:
            with nn_model.trace(*inp):
                saved = submod.output[0].save()
            act = unwrap_saved(saved).detach().cpu().float()
        except Exception:
            try:
                with nn_model.trace(*inp):
                    saved = submod.output.save()
                act = unwrap_saved(saved)
                if isinstance(act, tuple): act = act[0]
                act = act.detach().cpu().float()
            except Exception:
                continue

        while act.dim() > 2:
            act = act.squeeze(0)
        norms = act.norm(dim=-1).numpy()
        if len(norms) == len(total_norms):
            total_norms += norms

    importance = total_norms[:ctx_size]
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


# ── Attack Strategy A: Targeted Label Flip ────────────────────────────────────

def attack_targeted_label_flip(X_ctx, y_ctx, importance_scores, k=3):
    """
    Flip labels at the top-k most important context positions.
    Features stay the same — only labels change.

    This is the cleanest test of whether circuit-guided TARGETING
    matters: same corruption method as Exp 1, but at specific positions.
    """
    X_poisoned = X_ctx.clone()
    y_poisoned = y_ctx.clone()

    top_k = np.argsort(importance_scores)[-k:][::-1]
    for pos in top_k:
        y_poisoned[pos] = 1 - y_poisoned[pos]  # flip label

    return X_poisoned, y_poisoned, list(top_k)


def attack_random_label_flip(X_ctx, y_ctx, k=3, rng=None):
    """Random label flip baseline (same as Exp 1)."""
    if rng is None:
        rng = np.random.default_rng(42)

    X_poisoned = X_ctx.clone()
    y_poisoned = y_ctx.clone()

    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    for pos in positions:
        y_poisoned[pos] = 1 - y_poisoned[pos]

    return X_poisoned, y_poisoned


# ── Attack Strategy B: Feature Mimic ──────────────────────────────────────────

def attack_feature_mimic(X_ctx, y_ctx, x_test, true_label,
                          importance_scores, X_pool, y_pool, k=3):
    """
    At top-k important positions, insert examples that:
    - Are feature-similar to the TEST POINT (so ICL attention is drawn to them)
    - Carry the WRONG label (opposite of test point's true label)

    This hijacks the retrieval circuit: examples that look like x_test
    but say "opposite class" → model is misled.
    """
    X_poisoned = X_ctx.clone()
    y_poisoned = y_ctx.clone()

    top_k = np.argsort(importance_scores)[-k:][::-1]

    # Find same-class examples near x_test (feature-similar to test point)
    x_np = x_test.cpu().numpy()
    X_pool_np = X_pool.cpu().numpy()
    y_pool_np = y_pool.cpu().numpy()

    # Use SAME-class examples (feature-similar to test point)
    # but we'll label them as OPPOSITE class
    same_mask = (y_pool_np == true_label)
    X_same = X_pool_np[same_mask]

    if len(X_same) == 0:
        return X_poisoned, y_poisoned, []

    dists = np.linalg.norm(X_same - x_np, axis=1)
    nearest_idx = np.argsort(dists)[:k]

    wrong_label = 1 - true_label
    replaced = []
    for i, pos in enumerate(top_k):
        if i < len(nearest_idx):
            X_poisoned[pos] = torch.tensor(X_same[nearest_idx[i]],
                                            dtype=X_ctx.dtype, device=X_ctx.device)
            y_poisoned[pos] = wrong_label  # WRONG label for these features
            replaced.append(int(pos))

    return X_poisoned, y_poisoned, replaced


def attack_random_feature_mimic(X_ctx, y_ctx, x_test, true_label,
                                 X_pool, y_pool, k=3, rng=None):
    """Same feature-mimic poison but at RANDOM positions."""
    if rng is None:
        rng = np.random.default_rng(42)

    X_poisoned = X_ctx.clone()
    y_poisoned = y_ctx.clone()

    random_positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)

    x_np = x_test.cpu().numpy()
    X_pool_np = X_pool.cpu().numpy()
    y_pool_np = y_pool.cpu().numpy()

    same_mask = (y_pool_np == true_label)
    X_same = X_pool_np[same_mask]

    if len(X_same) == 0:
        return X_poisoned, y_poisoned

    dists = np.linalg.norm(X_same - x_np, axis=1)
    nearest_idx = np.argsort(dists)[:k]

    wrong_label = 1 - true_label
    for i, pos in enumerate(random_positions):
        if i < len(nearest_idx):
            X_poisoned[pos] = torch.tensor(X_same[nearest_idx[i]],
                                            dtype=X_ctx.dtype, device=X_ctx.device)
            y_poisoned[pos] = wrong_label

    return X_poisoned, y_poisoned


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_experiment_3(n_samples=100, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print("=" * 64)
    print("EXPERIMENT 3 — Circuit-Guided Context Poisoning Attack")
    print(f"Device: {DEVICE}  |  Seed: 42")
    print("=" * 64)

    # ── Data ──────────────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    # ── Model ─────────────────────────────────────────────────────────────
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    nn_model = NNsight(raw) if NNSIGHT_AVAILABLE else None

    # ── Clean accuracy ────────────────────────────────────────────────────
    n_eval = min(n_samples, len(X_test))
    clean_correct = 0
    for i in range(n_eval):
        pred = _predict(raw, X_ctx, y_ctx, X_test[i])
        if pred == y_test[i].item():
            clean_correct += 1
    clean_acc = clean_correct / n_eval
    print(f"\n[3A] Clean accuracy: {clean_acc*100:.1f}% ({clean_correct}/{n_eval})")

    X_pool = X_ctx.clone()
    y_pool = y_ctx.clone()

    # ── Results storage ───────────────────────────────────────────────────
    all_results = {}

    for k in k_values:
        print(f"\n{'═'*64}")
        print(f"  k = {k}")
        print(f"{'═'*64}")

        counts = {
            "A_circuit": 0, "A_random": 0,
            "B_circuit": 0, "B_random": 0,
            "tested": 0,
        }
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        for i in range(n_eval):
            xi = X_test[i]
            yi = y_test[i].item()
            clean_pred = _predict(raw, X_ctx, y_ctx, xi)
            if clean_pred != yi:
                continue

            counts["tested"] += 1

            # Get importance if NNsight available
            if NNSIGHT_AVAILABLE:
                importance = profile_context_importance(
                    nn_model, raw, X_ctx, y_ctx, xi, ICL_ATTN_TARGETS
                )
            else:
                importance = np.ones(len(X_ctx))

            # ── Strategy A: Label Flip ────────────────────────────────
            # Circuit-guided
            Xa, ya, _ = attack_targeted_label_flip(X_ctx, y_ctx, importance, k)
            if _predict(raw, Xa, ya, xi) != yi:
                counts["A_circuit"] += 1

            # Random
            Xa_r, ya_r = attack_random_label_flip(X_ctx, y_ctx, k, rng_a)
            if _predict(raw, Xa_r, ya_r, xi) != yi:
                counts["A_random"] += 1

            # ── Strategy B: Feature Mimic ─────────────────────────────
            # Circuit-guided
            Xb, yb, _ = attack_feature_mimic(
                X_ctx, y_ctx, xi, yi, importance, X_pool, y_pool, k
            )
            if _predict(raw, Xb, yb, xi) != yi:
                counts["B_circuit"] += 1

            # Random
            Xb_r, yb_r = attack_random_feature_mimic(
                X_ctx, y_ctx, xi, yi, X_pool, y_pool, k, rng_b
            )
            if _predict(raw, Xb_r, yb_r, xi) != yi:
                counts["B_random"] += 1

            if (i + 1) % 25 == 0:
                n = max(counts["tested"], 1)
                print(f"  [{i+1:>3d}/{n_eval}]"
                      f"  A: circ={counts['A_circuit']/n*100:.0f}% rand={counts['A_random']/n*100:.0f}%"
                      f"  B: circ={counts['B_circuit']/n*100:.0f}% rand={counts['B_random']/n*100:.0f}%")

        n = max(counts["tested"], 1)
        all_results[k] = {
            "A_circuit": round(counts["A_circuit"] / n, 4),
            "A_random":  round(counts["A_random"] / n, 4),
            "B_circuit": round(counts["B_circuit"] / n, 4),
            "B_random":  round(counts["B_random"] / n, 4),
            "n": counts["tested"],
        }

        print(f"\n  Strategy A (Label Flip):")
        print(f"    Circuit-guided: {counts['A_circuit']/n*100:.1f}%  Random: {counts['A_random']/n*100:.1f}%")
        print(f"  Strategy B (Feature Mimic):")
        print(f"    Circuit-guided: {counts['B_circuit']/n*100:.1f}%  Random: {counts['B_random']/n*100:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPERIMENT 3 — FULL RESULTS")
    print(f"{'='*64}")
    print(f"Clean accuracy: {clean_acc*100:.1f}%  |  Clean error: {(1-clean_acc)*100:.1f}%\n")

    print(f"{'k':>4s}  {'A-Circ':>7s}  {'A-Rand':>7s}  {'Δ-A':>6s}  │  {'B-Circ':>7s}  {'B-Rand':>7s}  {'Δ-B':>6s}")
    print(f"{'─'*60}")
    for k, r in all_results.items():
        da = (r["A_circuit"] - r["A_random"]) * 100
        db = (r["B_circuit"] - r["B_random"]) * 100
        print(f"{k:>4d}  {r['A_circuit']*100:>6.1f}%  {r['A_random']*100:>6.1f}%  {da:>+5.1f}  │  "
              f"{r['B_circuit']*100:>6.1f}%  {r['B_random']*100:>6.1f}%  {db:>+5.1f}")

    print(f"\nA = Targeted Label Flip (same features, wrong labels)")
    print(f"B = Feature Mimic (test-similar features, wrong labels)")
    print(f"Circ = circuit-guided position selection")
    print(f"Rand = random position selection")
    print(f"{'='*64}")

    # ── Visualization ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(len(k_values))
    w = 0.35

    # Strategy A
    a_circ = [all_results[k]["A_circuit"]*100 for k in k_values]
    a_rand = [all_results[k]["A_random"]*100 for k in k_values]
    ax1.bar(x_pos - w/2, a_rand, w, label="Random", color="#DD8452", edgecolor="white")
    ax1.bar(x_pos + w/2, a_circ, w, label="Circuit-Guided", color="#4C72B0", edgecolor="white")
    ax1.axhline(y=(1-clean_acc)*100, color='red', linestyle='--', alpha=0.5,
                label=f"Clean error ({(1-clean_acc)*100:.0f}%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"k={k}" for k in k_values])
    ax1.set_ylabel("ASR (%)")
    ax1.set_title("Strategy A: Targeted Label Flip", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(max(a_circ + a_rand) * 1.3, 40))

    # Strategy B
    b_circ = [all_results[k]["B_circuit"]*100 for k in k_values]
    b_rand = [all_results[k]["B_random"]*100 for k in k_values]
    ax2.bar(x_pos - w/2, b_rand, w, label="Random", color="#DD8452", edgecolor="white")
    ax2.bar(x_pos + w/2, b_circ, w, label="Circuit-Guided", color="#4C72B0", edgecolor="white")
    ax2.axhline(y=(1-clean_acc)*100, color='red', linestyle='--', alpha=0.5,
                label=f"Clean error ({(1-clean_acc)*100:.0f}%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"k={k}" for k in k_values])
    ax2.set_ylabel("ASR (%)")
    ax2.set_title("Strategy B: Feature Mimic", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(max(b_circ + b_rand) * 1.3, 40))

    plt.suptitle("Experiment 3 — Circuit-Guided vs Random Context Poisoning",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp3_attack_comparison.png", dpi=150, bbox_inches="tight")
    print("[VIZ] Saved exp3_attack_comparison.png")
    plt.close()

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_experiment_3(n_samples=100, k_values=[1, 3, 5, 10])
