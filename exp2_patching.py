# ============================================================
# EXPERIMENT 2 — Activation Patching & Circuit Identification
# Colab-compatible: assumes exp_0 cell was already executed.
# ============================================================
#
# Method: causal tracing via NNsight
#   1. Clean run (correct labels)       → save activations
#   2. Corrupt run (ALL labels flipped)  → wrong prediction
#   3. Patch clean activation into corrupt run per submodule
#   4. Measure: does patching restore the correct prediction?
#   5. Bidirectional: also inject corrupt acts into clean run
#
# Output: circuit atlas — ranked (stage, layer, axis) tuples
#         by restoration score, with attention pattern heatmaps
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
    print("[EXP2] ⚠️  NNsight not available — will use HookShim")

# Colab: DEVICE defined in exp_0 cell. Fallback if running standalone.
if 'DEVICE' not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def unwrap_saved(proxy):
    """NNsight version guard: <=0.3 has .value, >=0.4 proxy IS the tensor."""
    return proxy.value if hasattr(proxy, "value") else proxy


def _nav_to_submodule(nn_model, path):
    """Navigate NNsight attribute proxy to a submodule by dotted path."""
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _build_input(raw_model, X_ctx, y_ctx, x_test_i):
    """Build the input tuple OrionBix expects: (X_seq, y_seq)."""
    if type(raw_model).__name__ in ["OrionBix", "OrionMSP"]:
        X_seq = torch.cat([X_ctx, x_test_i.unsqueeze(0)], dim=0).unsqueeze(0)
        y_seq = y_ctx.unsqueeze(0)
        return (X_seq, y_seq)
    else:
        return (X_ctx, y_ctx, x_test_i)


def _flatten_logits(t):
    """Flatten model output to (n_classes,) tensor on CPU."""
    if isinstance(t, tuple):
        t = t[0]
    t = t.cpu().float()
    while t.dim() > 1:
        t = t.squeeze(0)
    return t


def _predict(raw_model, X_ctx, y_ctx, x_test_i):
    """Run raw_model on one test point → predicted class (int)."""
    raw_model.eval()
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    with torch.no_grad():
        out = raw_model(*inp)
    return int(_flatten_logits(out).argmax().item())


# ── Submodule discovery ──────────────────────────────────────────────────────

def discover_attention_paths(raw_model):
    """
    Dynamically find all attention submodules via named_modules().
    Returns list of (path_string, axis) where axis ∈ {row, col, icl}.

    Stage classification uses the top-level component name from OrionBix:
      col_embedder.*  → col   (feature interaction circuit)
      row_interactor.* → row  (retrieval circuit)
      icl_predictor.*  → icl  (label prediction circuit)
    """
    # Print raw submodule tree first so we can verify classification
    print("\n[2A] Submodule tree (attention modules only):")
    print("─" * 60)
    for name, module in raw_model.named_modules():
        mtype = type(module).__name__
        if any(k in mtype.lower() for k in ["attention", "attn"]):
            depth = name.count(".")
            print(f"  {'  ' * depth}{name}: {mtype}")
    print("─" * 60)

    paths = []
    for name, module in raw_model.named_modules():
        mtype = type(module).__name__
        if not any(k in mtype.lower() for k in ["attention", "attn"]):
            continue
        if "out_proj" in name:
            continue

        # Classify by top-level OrionBix component
        top_component = name.split(".")[0] if name else ""
        if top_component == "col_embedder" or "col" in name.lower():
            axis = "col"
        elif top_component == "row_interactor" or "row" in name.lower():
            axis = "row"
        elif top_component == "icl_predictor" or "icl" in name.lower():
            axis = "icl"
        else:
            axis = "unk"

        paths.append((name, axis))

    print(f"\n[2A] Classified {len(paths)} attention submodules:")
    for p, ax in paths:
        print(f"     [{ax:3s}] {p}")
    return paths


# ── Corruption ────────────────────────────────────────────────────────────────

def make_corrupted_context(y_ctx):
    """Flip ALL labels — maximum corruption signal for binary classification."""
    return 1 - y_ctx


# ── Core patching functions ──────────────────────────────────────────────────

def _save_activation(nn_model, submod, inp):
    """Save the output activation of a submodule during a trace.
    Tries output then output[0] (handles both tensor and tuple returns)."""
    try:
        with nn_model.trace(*inp):
            saved = submod.output.save()
        val = unwrap_saved(saved)
        if isinstance(val, tuple):
            val = val[0]
        return val.detach().clone()
    except Exception:
        pass
    try:
        with nn_model.trace(*inp):
            saved = submod.output[0].save()
        return unwrap_saved(saved).detach().clone()
    except Exception:
        return None


def _patch_and_get_pred(nn_model, submod, inp, clean_act):
    """Patch clean_act into submod during a trace, return predicted class.
    Patches output[0] directly — MHA always returns tuple (attn_out, weights)."""
    try:
        logits = _patch_indexed(nn_model, submod, inp, clean_act)
        if logits is not None:
            return int(_flatten_logits(logits).argmax().item())
    except Exception:
        pass
    return -1


def _patch_indexed(nn_model, submod, inp, act):
    with nn_model.trace(*inp):
        submod.output[0][:] = act
        out = nn_model.output.save()
    return unwrap_saved(out).detach()


# ── Patching sweep ───────────────────────────────────────────────────────────

def run_patching_sweep(raw_model, X_ctx, y_ctx, X_test, y_test,
                       attn_paths, n_samples=50):
    """
    For each test sample (correct under clean, wrong under corrupt):
      patch each submodule's clean activation into the corrupt run.
    Returns dict {path: mean_restoration_rate}.
    """
    if not NNSIGHT_AVAILABLE:
        print("[2B] ⚠️  NNsight unavailable — skipping patching sweep.")
        return {}

    device = next(raw_model.parameters()).device
    raw_model.eval()
    nn_model = NNsight(raw_model)
    y_corrupt = make_corrupted_context(y_ctx)

    print(f"\n[2B] Patching sweep")
    print(f"  Context size : {len(X_ctx)}")
    print(f"  Targets      : {len(attn_paths)} submodules")
    print(f"  Corruption   : full label flip")

    # Find usable samples: clean=correct AND corruption flips prediction
    # This takes O(400) forward passes max — ~30-60s on T4, not hung.
    print(f"  Finding usable samples (up to 200 candidates)...")
    usable = []
    scanned = 0
    for i in range(min(len(X_test), 200)):
        xi = X_test[i]
        true_y = y_test[i].item()
        clean_pred = _predict(raw_model, X_ctx, y_ctx, xi)
        scanned += 1
        if clean_pred != true_y:
            continue
        corrupt_pred = _predict(raw_model, X_ctx, y_corrupt, xi)
        scanned += 1
        if corrupt_pred == clean_pred:
            continue
        usable.append((i, clean_pred, corrupt_pred))
        if len(usable) >= n_samples:
            break
        if scanned % 50 == 0:
            print(f"    scanned {scanned} fwd passes, found {len(usable)} usable so far...")

    print(f"  Usable samples: {len(usable)} / {scanned} fwd passes (correct & flipped)")
    if not usable:
        print("[2B] ⚠️  No usable samples — try weaker corruption or more data.")
        return {}

    # Resolve NNsight submodule proxies
    submods = {}
    for path, axis in attn_paths:
        s = _nav_to_submodule(nn_model, path)
        if s is not None:
            submods[path] = s

    restoration_counts = defaultdict(int)

    for step, (idx, clean_pred, corrupt_pred) in enumerate(usable):
        xi = X_test[idx]
        clean_inp   = _build_input(raw_model, X_ctx, y_ctx, xi)
        corrupt_inp = _build_input(raw_model, X_ctx, y_corrupt, xi)

        for path in submods:
            submod = submods[path]

            # Save clean activation
            clean_act = _save_activation(nn_model, submod, clean_inp)
            if clean_act is None:
                continue

            # Patch into corrupt run
            patched_pred = _patch_and_get_pred(nn_model, submod, corrupt_inp, clean_act)
            if patched_pred == clean_pred:
                restoration_counts[path] += 1

        if (step + 1) % 10 == 0:
            print(f"  [{step+1:>3d}/{len(usable)} samples]")

    n = len(usable)
    mean_scores = {path: restoration_counts.get(path, 0) / n for path in submods}

    print(f"[2B] Sweep complete.")
    return mean_scores


# ── Bidirectional patching ───────────────────────────────────────────────────

def run_bidirectional_check(raw_model, X_ctx, y_ctx, X_test, y_test,
                            top_paths, n_samples=20):
    """
    For top-k heads, test BOTH directions:
      restore: corrupt run + clean act → does it recover?
      inject:  clean run + corrupt act → does it corrupt?
    Heads scoring high on both are true critical circuit nodes.
    """
    if not NNSIGHT_AVAILABLE:
        return {}

    print(f"\n[2D] Bidirectional patching — top {len(top_paths)} heads × {n_samples} samples")
    device = next(raw_model.parameters()).device
    raw_model.eval()
    nn_model = NNsight(raw_model)
    y_corrupt = make_corrupted_context(y_ctx)

    results = {}

    for path, axis in top_paths:
        submod = _nav_to_submodule(nn_model, path)
        if submod is None:
            results[path] = {"restore": 0.0, "inject": 0.0, "axis": axis}
            continue

        restore_hits, inject_hits, checked = 0, 0, 0

        for i in range(min(len(X_test), 200)):
            xi = X_test[i]
            clean_pred   = _predict(raw_model, X_ctx, y_ctx, xi)
            corrupt_pred = _predict(raw_model, X_ctx, y_corrupt, xi)
            if clean_pred == corrupt_pred:
                continue

            clean_inp   = _build_input(raw_model, X_ctx, y_ctx, xi)
            corrupt_inp = _build_input(raw_model, X_ctx, y_corrupt, xi)

            # Restore direction
            clean_act = _save_activation(nn_model, submod, clean_inp)
            if clean_act is not None:
                pred = _patch_and_get_pred(nn_model, submod, corrupt_inp, clean_act)
                if pred == clean_pred:
                    restore_hits += 1

            # Inject direction
            corrupt_act = _save_activation(nn_model, submod, corrupt_inp)
            if corrupt_act is not None:
                pred = _patch_and_get_pred(nn_model, submod, clean_inp, corrupt_act)
                if pred == corrupt_pred:
                    inject_hits += 1

            checked += 1
            if checked >= n_samples:
                break

        r = restore_hits / max(checked, 1)
        inj = inject_hits / max(checked, 1)
        results[path] = {"restore": round(r, 3), "inject": round(inj, 3), "axis": axis}
        print(f"  [{axis:3s}] {path:<50s}  restore={r:.3f}  inject={inj:.3f}")

    return results


# ── Circuit atlas ────────────────────────────────────────────────────────────

def build_circuit_atlas(mean_scores, attn_paths):
    """Build ranked DataFrame from restoration scores."""
    path_to_axis = dict(attn_paths)
    rows = []
    for path, score in mean_scores.items():
        parts = path.split(".")
        layer_num = next((int(p) for p in parts if p.isdigit()), None)

        # Stage classification matches discover_attention_paths
        top_component = parts[0] if parts else ""
        if top_component == "col_embedder" or "col" in path.lower().split(".")[0]:
            stage = "COL"
        elif top_component == "row_interactor" or "row" in path.lower().split(".")[0]:
            stage = "ROW"
        elif top_component == "icl_predictor" or "icl" in path.lower().split(".")[0]:
            stage = "ICL"
        else:
            stage = path_to_axis.get(path, "UNK").upper()

        rows.append({
            "path": path,
            "stage": stage,
            "layer": layer_num,
            "axis": path_to_axis.get(path, "unk"),
            "restoration": round(score, 4),
        })

    df = pd.DataFrame(rows).sort_values("restoration", ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_circuit_atlas(atlas_df):
    """Horizontal bar chart of restoration rates, color-coded by stage."""
    stage_colors = {"COL": "#4C72B0", "ROW": "#DD8452", "ICL": "#55A868"}

    fig, ax = plt.subplots(figsize=(10, max(6, len(atlas_df) * 0.35)))
    y_pos = range(len(atlas_df))
    colors = [stage_colors.get(s, "#999") for s in atlas_df["stage"]]
    labels = [f"[{row.stage}] {row.path.split('.')[-1]} (L{row.layer})"
              for _, row in atlas_df.iterrows()]

    ax.barh(y_pos, atlas_df["restoration"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Restoration Rate", fontsize=11)
    ax.set_title("Circuit Atlas — Causal Importance by Submodule", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=s) for s, c in stage_colors.items()]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig("exp2_circuit_atlas.png", dpi=150, bbox_inches="tight")
    print("[VIZ] Saved exp2_circuit_atlas.png")
    plt.close()


def plot_attention_patterns(raw_model, X_ctx, y_ctx, X_test,
                            top_paths, n_viz=3):
    """Activation norm visualization for top-k causally critical heads.

    OrionBix uses a custom MultiheadAttention that doesn't support
    need_weights, so we visualize ||output[0]|| per context position
    instead — shows which positions produce the strongest signal.
    Uses the same NNsight output[0] save path validated in patching.
    """
    if not NNSIGHT_AVAILABLE:
        return

    raw_model.eval()
    nn_model = NNsight(raw_model)

    n_rows = min(len(top_paths), 5)
    fig, axes = plt.subplots(n_rows, n_viz, figsize=(4 * n_viz, 3 * n_rows))
    if n_rows == 1: axes = axes.reshape(1, -1)
    if n_viz  == 1: axes = axes.reshape(-1, 1)

    for row_i, (path, axis) in enumerate(top_paths[:n_rows]):
        print(f"[VIZ] Row {row_i}: {path} [{axis}]")

        submod = _nav_to_submodule(nn_model, path)
        if submod is None:
            print(f"    ❌ submod not found")
            for c in range(n_viz):
                axes[row_i, c].text(0.5, 0.5, "submod not found",
                    ha="center", va="center", fontsize=7, transform=axes[row_i, c].transAxes)
            continue

        for col_i in range(n_viz):
            xi = X_test[col_i]
            inp = _build_input(raw_model, X_ctx, y_ctx, xi)
            ax = axes[row_i, col_i]

            try:
                act = _save_activation(nn_model, submod, inp)
                if act is None:
                    raise ValueError("save returned None")

                act = act.cpu().float()
                while act.dim() > 2:
                    act = act.squeeze(0)

                # L2 norm per position → (seq_len,)
                norms = act.norm(dim=-1).numpy()

                positions = np.arange(len(norms))
                colors = ['#55A868'] * (len(norms) - 1) + ['#DD8452']
                ax.bar(positions, norms, color=colors, width=1.0, edgecolor='none')
                ax.set_title(f"[{axis}] {path.split('.')[-1]}  s={col_i}", fontsize=7)
                ax.set_xlabel("pos (last=test)", fontsize=6)
                ax.set_ylabel("||act||₂", fontsize=6)
                ax.tick_params(labelsize=5)
                print(f"    [s={col_i}] ✅ norms shape={norms.shape}")
            except Exception as e:
                ax.text(0.5, 0.5, f"fail: {str(e)[:30]}",
                        ha="center", va="center", fontsize=6, transform=ax.transAxes)
                ax.set_title(f"[{axis}] {path.split('.')[-1]}", fontsize=7)
                print(f"    [s={col_i}] ❌ {e}")

    plt.suptitle("Activation Norms — Top Causally Critical Heads\n"
                 "(green=context, orange=test token)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp2_attention_patterns.png", dpi=120, bbox_inches="tight")
    print("[VIZ] Saved exp2_attention_patterns.png")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_experiment_2(n_patch_samples=50, n_bidir_samples=20):
    print("=" * 64)
    print("EXPERIMENT 2 — Activation Patching & Circuit Identification")
    print(f"Device: {DEVICE}  |  Seed: 42")
    print("=" * 64)

    # ── Data ──────────────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    # ── Model (lazy — must fit first) ─────────────────────────────────────
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    # ── 2A: Discover attention submodules ─────────────────────────────────
    attn_paths = discover_attention_paths(raw)

    # ── 2B: Patching sweep ────────────────────────────────────────────────
    mean_scores = run_patching_sweep(
        raw, X_ctx, y_ctx, X_test, y_test,
        attn_paths, n_samples=n_patch_samples,
    )
    if not mean_scores:
        print("[EXP2] ❌ No results — check NNsight or corruption.")
        return None, None, None

    # ── 2C: Circuit atlas ─────────────────────────────────────────────────
    atlas = build_circuit_atlas(mean_scores, attn_paths)

    print("\n" + "=" * 64)
    print("CIRCUIT ATLAS — Ranked by Restoration Rate")
    print("=" * 64)
    print(atlas.to_string())
    print("=" * 64)

    top_5 = [(row.path, row.axis) for _, row in atlas.head(5).iterrows()]
    print(f"\n[2C] ★ Top-5 causally critical circuits:")
    for rank, (p, ax) in enumerate(top_5, 1):
        score = atlas.iloc[rank - 1]["restoration"]
        print(f"  #{rank}  [{ax:3s}] {p}  →  {score*100:.1f}%")

    # ── 2C: Visualisations ────────────────────────────────────────────────
    plot_circuit_atlas(atlas)
    plot_attention_patterns(raw, X_ctx, y_ctx, X_test, top_5, n_viz=3)

    # ── 2D: Bidirectional check on top-5 ──────────────────────────────────
    bidir = run_bidirectional_check(
        raw, X_ctx, y_ctx, X_test, y_test,
        top_5, n_samples=n_bidir_samples,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EXPERIMENT 2 — SUMMARY")
    print("=" * 64)

    # Stage-level averages
    for stage in ["COL", "ROW", "ICL"]:
        stage_rows = atlas[atlas["stage"] == stage]
        if len(stage_rows):
            avg = stage_rows["restoration"].mean()
            mx  = stage_rows["restoration"].max()
            print(f"  {stage}  avg={avg*100:.1f}%  max={mx*100:.1f}%")

    # Axis dominance
    top5_axes = atlas.head(5)["axis"].value_counts().to_dict()
    print(f"\n  Top-5 axis distribution: {top5_axes}")

    row_best = atlas[atlas["stage"] == "ROW"]["restoration"].max() \
               if "ROW" in atlas["stage"].values else 0
    col_best = atlas[atlas["stage"] == "COL"]["restoration"].max() \
               if "COL" in atlas["stage"].values else 0
    icl_best = atlas[atlas["stage"] == "ICL"]["restoration"].max() \
               if "ICL" in atlas["stage"].values else 0

    dominant = max([("ROW", row_best), ("COL", col_best), ("ICL", icl_best)],
                   key=lambda x: x[1])
    print(f"\n  Dominant stage: {dominant[0]} (best restoration={dominant[1]*100:.1f}%)")
    print(f"  → Experiment 3 attack should target {dominant[0]} attention heads.")
    print("=" * 64)

    # Save
    atlas.to_csv("exp2_circuit_atlas.csv")
    print("\n[EXP2] Saved exp2_circuit_atlas.csv")

    return atlas, top_5, bidir


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    atlas, top_5, bidir = run_experiment_2(
        n_patch_samples=50,
        n_bidir_samples=20,
    )
