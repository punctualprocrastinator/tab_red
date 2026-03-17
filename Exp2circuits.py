# ============================================================
# EXPERIMENT 2 - Activation Patching & Circuit Identification
# T4 GPU / Google Colab
# Depends on: exp0 cells already executed in session
# ============================================================
#
# Goal: identify which (layer, axis, head) tuples in ORION-BiX
# are causally responsible for correct predictions.
#
# Method: activation patching (causal tracing)
#   1. Clean run  -> save activations at every (layer, axis)
#   2. Corrupt run (flipped context labels) -> wrong prediction
#   3. Patch one (layer, axis) from clean into corrupt
#   4. Measure: does the correct prediction restore?
#
# Output: circuit atlas - ranked list of causally critical
#   (layer, axis) tuples with restoration scores and attention
#   pattern visualisations for the top-5.
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False
    print("[EXP2] NNsight not available - will use HookShim")


# ── NNsight version-safe save helper ─────────────────────────────────────────

def unwrap_saved(proxy):
    """
    NNsight version guard:
      old (<=0.3): proxy.value  -> tensor
      new (>=0.4): proxy IS the tensor after context exits
    """
    return proxy.value if hasattr(proxy, "value") else proxy


# ── Model interface ───────────────────────────────────────────────────────────

def get_raw_model_forward_args(raw_model, X_ctx, y_ctx, x_test_single):
    """
    Build the forward-pass input tuple for one test point.
    ORION-BiX discovered path: pipeline.model.model_

    The raw OrionBix module signature may differ from the fallback.
    We try two common signatures and return whichever works.
    """
    # Signature 1: (X_context, y_context, x_test) - fallback style
    # Signature 2: (X_seq, y_seq) where X_seq = cat([X_ctx, x_test])
    return (X_ctx, y_ctx, x_test_single)


def model_predict(raw_model, X_ctx, y_ctx, x_test_single):
    """
    Run raw_model on one test point and return predicted class (int).
    Tries multiple call signatures to handle both real ORION-BiX
    and the structural fallback.
    """
    raw_model.eval()
    with torch.no_grad():
        # Try OrionBix native signature first
        try:
            X_seq = torch.cat([X_ctx, x_test_single.unsqueeze(0)], dim=0).unsqueeze(0)
            y_seq = y_ctx.unsqueeze(0)
            out = raw_model(X_seq, y_seq)
        except Exception:
            # Fallback signature
            out = raw_model(X_ctx, y_ctx, x_test_single)

        if out.dim() > 2:
            out = out.squeeze()
        if out.dim() == 2:
            out = out[0] if out.shape[0] == 1 else out[-1]
        return int(out.argmax().item())


# ── Corrupt context builder ───────────────────────────────────────────────────

def make_corrupted_context(y_ctx):
    """
    Flip all labels in the context window.
    This is the strongest corruption signal - guarantees wrong prediction
    on a model that relies on label-feature associations in context.
    """
    return 1 - y_ctx


# ── Submodule path discovery ──────────────────────────────────────────────────

def discover_attention_paths(raw_model):
    """
    Return list of (path_string, axis) tuples for all attention
    submodules in the model, discovered from named_modules().

    axis is 'row', 'col', or 'unknown' based on name heuristics.
    """
    paths = []
    for name, module in raw_model.named_modules():
        mtype = type(module).__name__
        if "MultiheadAttention" not in mtype and "Attention" not in mtype:
            continue
        # Skip out_proj - it's a Linear inside MHA, not the MHA itself
        if "out_proj" in name:
            continue
        if "row" in name.lower():
            axis = "row"
        elif "col" in name.lower():
            axis = "col"
        else:
            axis = "unknown"
        paths.append((name, axis))

    print(f"[2A] Discovered {len(paths)} attention submodules:")
    for p, ax in paths:
        print(f"     {p}  [{ax}]")
    return paths


# ── Single patch experiment ───────────────────────────────────────────────────

def patch_one_submodule(nn_model, raw_model, clean_input, corrupt_input,
                        submod_path, clean_pred, corrupt_pred):
    """
    Run the patching experiment for one (layer, axis) submodule.

    Returns restoration_score in [0, 1]:
      1.0 = patching fully restores clean prediction
      0.0 = patching has no effect
      intermediate = partial logit restoration (measured via KL)
    """
    # Navigate to submodule on NNsight model
    obj = nn_model
    for part in submod_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return 0.0, None

    # Step 1: save output[0] on clean run
    try:
        with nn_model.trace(*clean_input):
            saved_raw = obj.output[0].save()
        clean_act = unwrap_saved(saved_raw).detach().clone().to(DEVICE)
    except Exception as e:
        print(f"     [WARN] Save failed on {submod_path}: {e}")
        return 0.0, None

    # Step 2: patch clean activation into corrupt run, save logits
    try:
        with nn_model.trace(*corrupt_input):
            obj.output[0][:] = clean_act
            patched_out_raw = nn_model.output.save()
        patched_out = unwrap_saved(patched_out_raw).detach()
    except Exception as e:
        print(f"     [WARN] Patch failed on {submod_path}: {e}")
        return 0.0, None

    # Step 3: get corrupt baseline logits (no patch)
    try:
        with nn_model.trace(*corrupt_input):
            corrupt_out_raw = nn_model.output.save()
        corrupt_out = unwrap_saved(corrupt_out_raw).detach()
    except Exception as e:
        return 0.0, None

    # Flatten outputs to (n_classes,)
    def flatten_logits(t):
        t = t.cpu().float()
        while t.dim() > 1:
            t = t.squeeze(0)
        return t

    patched_logits = flatten_logits(patched_out)
    corrupt_logits = flatten_logits(corrupt_out)

    patched_pred = int(patched_logits.argmax().item())

    # Restoration score: did we recover the clean prediction?
    if patched_pred == clean_pred:
        # Full restoration - measure how much logit shifted toward clean
        score = 1.0
    elif patched_pred == corrupt_pred:
        # No effect
        score = 0.0
    else:
        # Partial - use logit difference as proxy
        n_classes = patched_logits.shape[0]
        score = 0.5

    return score, clean_act


# ── Full patching sweep ────────────────────────────────────────────────────────

def run_patching_sweep(raw_model, X_ctx, y_ctx, X_test, y_test,
                       n_samples=50):
    """
    Sweep all attention submodules across n_samples test points.
    Returns restoration_scores: dict {submod_path: mean_score}
    """
    if not NNSIGHT_AVAILABLE:
        print("[2B] NNsight unavailable - cannot run patching sweep.")
        print("[2B] Returning empty scores. Use HookShim path instead.")
        return {}

    print(f"\n[2B] Patching sweep: {n_samples} test samples")
    print(f"[2B] Clean context size: {len(X_ctx)}")

    raw_model.eval()
    nn_model = NNsight(raw_model)

    attn_paths = discover_attention_paths(raw_model)
    if not attn_paths:
        print("[2B] No attention paths found - check submodule discovery")
        return {}

    y_ctx_corrupt = make_corrupted_context(y_ctx)
    restoration_scores = defaultdict(list)

    # Select test points where clean prediction is correct
    correct_idx = []
    for i in range(min(len(X_test), 200)):
        xi = X_test[i]
        yi = y_test[i].item()
        pred = model_predict(raw_model, X_ctx, y_ctx, xi)
        if pred == yi:
            correct_idx.append(i)
        if len(correct_idx) >= n_samples:
            break

    print(f"[2B] Found {len(correct_idx)} correctly-predicted samples for patching")

    if len(correct_idx) == 0:
        print("[2B] No correct predictions found - check model and data")
        return {}

    # Build forward signatures
    def build_inputs(xi, use_corrupt=False):
        yc = y_ctx_corrupt if use_corrupt else y_ctx
        try:
            X_seq = torch.cat([X_ctx, xi.unsqueeze(0)], dim=0).unsqueeze(0)
            y_seq = yc.unsqueeze(0)
            return (X_seq, y_seq)
        except Exception:
            return (X_ctx, yc, xi)

    for sample_num, idx in enumerate(correct_idx):
        xi = X_test[idx]
        yi = y_test[idx].item()

        clean_pred = model_predict(raw_model, X_ctx, y_ctx, xi)
        corrupt_pred = model_predict(raw_model, X_ctx, y_ctx_corrupt, xi)

        # Skip if corruption doesn't change prediction (not a useful sample)
        if clean_pred == corrupt_pred:
            for path, _ in attn_paths:
                restoration_scores[path].append(0.0)
            continue

        clean_input   = build_inputs(xi, use_corrupt=False)
        corrupt_input = build_inputs(xi, use_corrupt=True)

        for path, axis in attn_paths:
            score, _ = patch_one_submodule(
                nn_model, raw_model,
                clean_input, corrupt_input,
                path, clean_pred, corrupt_pred
            )
            restoration_scores[path].append(score)

        if (sample_num + 1) % 10 == 0:
            print(f"[2B] Processed {sample_num+1}/{len(correct_idx)} samples...")

    # Aggregate
    mean_scores = {
        path: np.mean(scores) if scores else 0.0
        for path, scores in restoration_scores.items()
    }

    print(f"\n[2B] Patching sweep complete.")
    return mean_scores


# ── Circuit atlas ─────────────────────────────────────────────────────────────

def build_circuit_atlas(mean_scores, attn_paths):
    """
    Build ranked circuit atlas from restoration scores.
    Returns a DataFrame sorted by causal importance.
    """
    rows = []
    path_to_axis = dict(attn_paths)
    for path, score in mean_scores.items():
        # Parse layer number from path
        parts = path.split(".")
        layer_num = None
        for p in parts:
            if p.isdigit():
                layer_num = int(p)
                break
        axis = path_to_axis.get(path, "unknown")
        rows.append({
            "path":  path,
            "layer": layer_num,
            "axis":  axis,
            "restoration_score": round(score, 4),
        })

    df = pd.DataFrame(rows).sort_values("restoration_score", ascending=False)
    df = df.reset_index(drop=True)
    df.index.name = "rank"
    return df


# ── Attention pattern visualisation ──────────────────────────────────────────

def visualise_attention_patterns(raw_model, X_ctx, y_ctx, X_test,
                                  top_paths, feature_names, n_viz=3):
    """
    Save attention weight heatmaps for the top-k causally critical heads.
    Uses clean context - shows what the model attends to normally.
    """
    if not NNSIGHT_AVAILABLE:
        return

    raw_model.eval()
    nn_model = NNsight(raw_model)

    n_rows = min(len(top_paths), 5)
    fig, axes = plt.subplots(n_rows, n_viz, figsize=(4*n_viz, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    def build_input(xi):
        try:
            X_seq = torch.cat([X_ctx, xi.unsqueeze(0)], dim=0).unsqueeze(0)
            y_seq = y_ctx.unsqueeze(0)
            return (X_seq, y_seq)
        except Exception:
            return (X_ctx, y_ctx, xi)

    for row_i, (path, axis) in enumerate(top_paths[:n_rows]):
        # Navigate to submodule
        obj = nn_model
        valid = True
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
            if obj is None:
                valid = False
                break
        if not valid:
            continue

        for col_i in range(n_viz):
            xi = X_test[col_i]
            sample_input = build_input(xi)
            ax = axes[row_i, col_i]

            try:
                # Save attention weights (output[1])
                with nn_model.trace(*sample_input):
                    weights_raw = obj.output[1].save()
                weights = unwrap_saved(weights_raw).detach().cpu().float()

                # weights shape: (batch, heads, seq, seq) or (batch, seq, seq)
                if weights.dim() == 4:
                    weights = weights[0].mean(0)  # avg over heads
                elif weights.dim() == 3:
                    weights = weights[0]

                # Limit to last row (test token attending to context)
                attn_map = weights[-1:, :].numpy()
                im = ax.imshow(attn_map, aspect="auto", cmap="viridis",
                               vmin=0, vmax=attn_map.max())
                ax.set_title(f"{path.split('.')[-2]}.{axis}\nsample {col_i}",
                             fontsize=7)
                ax.set_yticks([])
                ax.set_xlabel("context pos", fontsize=6)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception as e:
                ax.text(0.5, 0.5, f"failed\n{str(e)[:40]}",
                        ha="center", va="center", fontsize=6,
                        transform=ax.transAxes)
                ax.set_title(f"{path} failed", fontsize=7)

    plt.suptitle("Attention patterns - top causally critical heads\n"
                 "(test token attending to context)", fontsize=9)
    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/exp2_attention_patterns.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[2C] Attention patterns saved to {out_path}")


# ── Bidirectional patching check ─────────────────────────────────────────────

def run_bidirectional_check(raw_model, X_ctx, y_ctx, X_test, y_test,
                             top_5_paths, n_samples=20):
    """
    For top-5 heads, run both:
      clean -> corrupt (restoration): does patching recover clean pred?
      corrupt -> clean (injection):   does patching corrupt a clean pred?

    A head that scores high on BOTH is a true critical circuit node.
    """
    if not NNSIGHT_AVAILABLE:
        return {}

    print("\n[2D] Bidirectional patching check on top-5 heads")
    raw_model.eval()
    nn_model = NNsight(raw_model)
    y_ctx_corrupt = make_corrupted_context(y_ctx)

    results = {}

    def build_inputs(xi, use_corrupt=False):
        yc = y_ctx_corrupt if use_corrupt else y_ctx
        try:
            X_seq = torch.cat([X_ctx, xi.unsqueeze(0)], dim=0).unsqueeze(0)
            return (X_seq, yc.unsqueeze(0))
        except Exception:
            return (X_ctx, yc, xi)

    for path, axis in top_5_paths:
        restore_scores = []
        inject_scores  = []

        obj = nn_model
        valid = True
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
            if obj is None:
                valid = False
                break
        if not valid:
            results[path] = {"restore": 0.0, "inject": 0.0}
            continue

        checked = 0
        for i in range(min(len(X_test), 200)):
            xi = X_test[i]
            clean_pred   = model_predict(raw_model, X_ctx, y_ctx, xi)
            corrupt_pred = model_predict(raw_model, X_ctx, y_ctx_corrupt, xi)
            if clean_pred == corrupt_pred:
                continue

            # --- Restoration (corrupt run, patch clean act) ---
            try:
                with nn_model.trace(*build_inputs(xi, use_corrupt=False)):
                    clean_act_raw = obj.output[0].save()
                clean_act = unwrap_saved(clean_act_raw).detach().clone().to(DEVICE)

                with nn_model.trace(*build_inputs(xi, use_corrupt=True)):
                    obj.output[0][:] = clean_act
                    restored_raw = nn_model.output.save()
                restored = unwrap_saved(restored_raw).detach().cpu()
                while restored.dim() > 1: restored = restored.squeeze(0)
                restore_scores.append(float(restored.argmax() == clean_pred))
            except Exception:
                restore_scores.append(0.0)

            # --- Injection (clean run, patch corrupt act) ---
            try:
                with nn_model.trace(*build_inputs(xi, use_corrupt=True)):
                    corrupt_act_raw = obj.output[0].save()
                corrupt_act = unwrap_saved(corrupt_act_raw).detach().clone().to(DEVICE)

                with nn_model.trace(*build_inputs(xi, use_corrupt=False)):
                    obj.output[0][:] = corrupt_act
                    injected_raw = nn_model.output.save()
                injected = unwrap_saved(injected_raw).detach().cpu()
                while injected.dim() > 1: injected = injected.squeeze(0)
                inject_scores.append(float(injected.argmax() == corrupt_pred))
            except Exception:
                inject_scores.append(0.0)

            checked += 1
            if checked >= n_samples:
                break

        r = np.mean(restore_scores) if restore_scores else 0.0
        inj = np.mean(inject_scores) if inject_scores else 0.0
        results[path] = {"restore": round(r, 3), "inject": round(inj, 3),
                         "axis": axis}
        print(f"  {path:45s} [{axis:3s}]  restore={r:.3f}  inject={inj:.3f}")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_experiment_2(n_patch_samples=50, n_bidir_samples=20):
    print("=" * 64)
    print("EXPERIMENT 2 - Activation Patching & Circuit Identification")
    print(f"Device: {DEVICE}  |  Seed: 42")
    print("=" * 64)

    # ── Data ──────────────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    # ── Model (fit first for lazy loading) ────────────────────────────────
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    # ── 2A: Discover all attention paths ──────────────────────────────────
    attn_paths = discover_attention_paths(raw)

    # ── 2B: Patching sweep ────────────────────────────────────────────────
    mean_scores = run_patching_sweep(raw, X_ctx, y_ctx, X_test, y_test,
                                     n_samples=n_patch_samples)

    if not mean_scores:
        print("[EXP2] No scores returned - NNsight may need HookShim fallback")
        return None

    # ── 2C: Circuit atlas ─────────────────────────────────────────────────
    atlas = build_circuit_atlas(mean_scores, attn_paths)
    print("\n[2C] CIRCUIT ATLAS (ranked by restoration score):")
    print("=" * 64)
    print(atlas.to_string())
    print("=" * 64)

    # Top 5 for downstream experiments
    top_5 = [(row["path"], row["axis"])
             for _, row in atlas.head(5).iterrows()]
    print(f"\n[2C] Top-5 critical heads:")
    for rank, (p, ax) in enumerate(top_5):
        score = atlas.iloc[rank]["restoration_score"]
        print(f"  #{rank+1}  {p}  [{ax}]  score={score}")

    # ── 2C: Attention visualisation ───────────────────────────────────────
    visualise_attention_patterns(raw, X_ctx, y_ctx, X_test,
                                  top_5, feat_names, n_viz=3)

    # ── 2D: Bidirectional check ───────────────────────────────────────────
    bidir = run_bidirectional_check(raw, X_ctx, y_ctx, X_test, y_test,
                                     top_5, n_samples=n_bidir_samples)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EXPERIMENT 2 - CIRCUIT ATLAS SUMMARY")
    print("=" * 64)
    print(atlas.head(5).to_string())
    print("\nKey finding:")
    top_axis_counts = atlas.head(5)["axis"].value_counts().to_dict()
    for ax, cnt in top_axis_counts.items():
        print(f"  {ax} attention heads: {cnt}/5 in top-5")

    row_top = atlas[atlas["axis"] == "row"]["restoration_score"].max() \
              if "row" in atlas["axis"].values else 0.0
    col_top = atlas[atlas["axis"] == "col"]["restoration_score"].max() \
              if "col" in atlas["axis"].values else 0.0
    print(f"\n  Best row-attn score: {row_top:.4f}")
    print(f"  Best col-attn score: {col_top:.4f}")

    if row_top > col_top:
        print("\n  --> Row attention (retrieval circuit) dominates.")
        print("      Experiment 3 attack targets: row_attn heads")
    elif col_top > row_top:
        print("\n  --> Column attention (feature interaction) dominates.")
        print("      Experiment 3 attack targets: col_attn heads")
    else:
        print("\n  --> Row and column equally critical.")
        print("      Experiment 3 will target both axes.")

    print("=" * 64)

    # Save atlas
    atlas_path = "/mnt/user-data/outputs/exp2_circuit_atlas.csv"
    atlas.to_csv(atlas_path)
    print(f"\n[2C] Circuit atlas saved to {atlas_path}")

    return atlas, top_5, bidir


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    atlas, top_5, bidir = run_experiment_2(
        n_patch_samples=50,
        n_bidir_samples=20
    )