# ============================================================
# EXPERIMENT 6 — Gap-Filling: Transfer, Dataset, D2 FPR
# Colab-compatible: assumes exp_0 cells already executed.
#
# Three credibility gaps to close:
#   1. ORION-MSP transfer (is vulnerability architectural?)
#   2. HELOC dataset (does it generalize beyond Adult Income?)
#   3. D2 FPR at operating point (is D2 deployable?)
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
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

def _predict_proba(raw_model, X_ctx, y_ctx, x_test_i):
    raw_model.eval()
    with torch.no_grad():
        out = raw_model(*_build_input(raw_model, X_ctx, y_ctx, x_test_i))
    return torch.softmax(_flatten_logits(out), dim=0).numpy()

def unwrap_saved(proxy):
    return proxy.value if hasattr(proxy, "value") else proxy


# ── Attacks (from exp4/5) ─────────────────────────────────────────────────────

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


# ── HELOC loader ──────────────────────────────────────────────────────────────

def load_heloc(n_context=256, n_test=512, seed=42):
    """Load HELOC (Home Equity Line of Credit) from OpenML."""
    print("\n[DATA] Loading HELOC dataset...")
    data = fetch_openml(data_id=45023, as_frame=True, parser="auto")
    df = data.frame.dropna().reset_index(drop=True)

    target_col = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else df.columns[-1]
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode target to 0/1
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw.astype(str)), name="target")

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.astype(float)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_scaled))
    n_context = min(n_context, len(idx) // 3)
    n_test = min(n_test, len(idx) // 3)
    ctx_idx = idx[:n_context]
    test_idx = idx[n_context:n_context + n_test]

    X_ctx = torch.tensor(X_scaled[ctx_idx], dtype=DTYPE).to(DEVICE)
    y_ctx = torch.tensor(y.values[ctx_idx], dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(X_scaled[test_idx], dtype=DTYPE).to(DEVICE)
    y_test = torch.tensor(y.values[test_idx], dtype=torch.long).to(DEVICE)

    print(f"[DATA] HELOC: Context={X_ctx.shape}  Test={X_test.shape}")
    print(f"[DATA] Features ({len(feature_names)}): {feature_names[:5]}...")
    print(f"[DATA] Positive rate (test): {y_test.sum().item()}/{len(y_test)} "
          f"({y_test.float().mean()*100:.1f}%)")

    return X_ctx, y_ctx, X_test, y_test, feature_names, scaler


# ── D2 FPR analysis ──────────────────────────────────────────────────────────

ICL_REPR_PATH = "icl_predictor.tf_icl.blocks.0"

def get_last_token_repr(nn_model, raw_model, X_ctx, y_ctx, x_test_i,
                         path=ICL_REPR_PATH):
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    try:
        with nn_model.trace(*inp):
            saved = obj.output[0].save()
        act = unwrap_saved(saved).detach().cpu().float()
    except Exception:
        try:
            with nn_model.trace(*inp):
                saved = obj.output.save()
            act = unwrap_saved(saved)
            if isinstance(act, tuple): act = act[0]
            act = act.detach().cpu().float()
        except Exception:
            return None
    while act.dim() > 2:
        act = act.squeeze(0)
    return act[-1].numpy()


# ── Core evaluation ───────────────────────────────────────────────────────────

def eval_attacks(raw_model, X_ctx, y_ctx, X_test, y_test,
                  n_samples=100, k=3, sigma_values=None):
    """Run Attack D (sigma sweep) and Attack G, return ASR dict."""
    if sigma_values is None:
        sigma_values = [0.01, 0.05, 0.10]
    n_eval = min(n_samples, len(X_test))

    # Clean accuracy
    clean_correct = sum(
        _predict(raw_model, X_ctx, y_ctx, X_test[i]) == y_test[i].item()
        for i in range(n_eval)
    )
    clean_acc = clean_correct / n_eval

    results = {"clean_acc": clean_acc, "n_eval": n_eval}

    # Attack D (sigma sweep)
    for sigma in sigma_values:
        flips, tested = 0, 0
        for i in range(n_eval):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            tested += 1
            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma)
            if _predict(raw_model, X_p, y_p, xi) != yi:
                flips += 1
        results[f"D_s{sigma}"] = round(flips / max(tested, 1), 4)

    # Attack G (pool-only)
    g_flips, g_tested = 0, 0
    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue
        g_tested += 1
        rng = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng)
        if _predict(raw_model, X_g, y_g, xi) != yi:
            g_flips += 1
    results["G_pool"] = round(g_flips / max(g_tested, 1), 4)
    results["n_correct"] = g_tested

    return results


def eval_d2_fpr(nn_model, raw_model, X_ctx, y_ctx, X_test, y_test,
                 k=3, sigma=0.01, n_samples=80):
    """Compute D2 ROC curve and FPR at useful thresholds."""
    from sklearn.decomposition import PCA

    n_eval = min(n_samples, len(X_test))

    # Collect clean reprs
    clean_reprs, clean_indices = [], []
    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue
        r = get_last_token_repr(nn_model, raw_model, X_ctx, y_ctx, xi)
        if r is not None:
            clean_reprs.append(r)
            clean_indices.append(i)

    if len(clean_reprs) < 10:
        print("  [D2-FPR] Not enough clean reprs")
        return {}

    clean_reprs = np.stack(clean_reprs)
    n_comp = min(32, clean_reprs.shape[0] - 1, clean_reprs.shape[1])
    pca = PCA(n_components=n_comp).fit(clean_reprs)
    clean_proj = pca.transform(clean_reprs)
    proj_cov = np.cov(clean_proj.T) + np.eye(n_comp) * 1e-6
    proj_cov_inv = np.linalg.inv(proj_cov)
    proj_mean = clean_proj.mean(0)

    def mahal(r):
        rp = pca.transform(r.reshape(1, -1))[0]
        d = rp - proj_mean
        return float(d @ proj_cov_inv @ d)

    clean_scores = np.array([mahal(r) for r in clean_reprs])

    # Poison reprs
    poison_scores = []
    for i in clean_indices:
        xi, yi = X_test[i], y_test[i].item()
        rng = np.random.default_rng(42 + i)
        X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma)
        r = get_last_token_repr(nn_model, raw_model, X_p, y_p, xi)
        if r is not None:
            poison_scores.append(mahal(r))
    poison_scores = np.array(poison_scores)

    # ROC
    y_true = np.array([0]*len(clean_scores) + [1]*len(poison_scores))
    scores = np.concatenate([clean_scores, poison_scores])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    # Key operating points
    results = {"auroc": auroc, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    # Report FPR at TPR=80%, 85%, 90%
    for target_tpr in [0.80, 0.85, 0.90]:
        idx = np.searchsorted(tpr, target_tpr)
        if idx < len(fpr):
            results[f"fpr_at_tpr{int(target_tpr*100)}"] = float(fpr[idx])
            results[f"threshold_at_tpr{int(target_tpr*100)}"] = float(thresholds[idx])
        else:
            results[f"fpr_at_tpr{int(target_tpr*100)}"] = 1.0

    # Precision at each operating point
    for target_tpr in [0.80, 0.85, 0.90]:
        tp = target_tpr * len(poison_scores)
        fp_rate = results[f"fpr_at_tpr{int(target_tpr*100)}"]
        fp = fp_rate * len(clean_scores)
        precision = tp / max(tp + fp, 1e-8)
        results[f"precision_at_tpr{int(target_tpr*100)}"] = round(precision, 3)

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_experiment_6(n_samples=100, k=3):
    print("=" * 64)
    print("EXPERIMENT 6 — Gap-Filling: Transfer, Dataset, D2 FPR")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    sigma_values = [0.01, 0.05, 0.10]
    all_results = {}

    # ══════════════════════════════════════════════════════════════════════
    # GAP 1: ORION-MSP TRANSFER
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*64}")
    print("GAP 1: ORION-MSP Transfer (Adult Income)")
    print(f"{'━'*64}")

    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    for model_name in ["orion-bix", "orion-msp"]:
        print(f"\n[MODEL] Loading {model_name}...")
        wrapper = load_model(model_name)
        X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
        y_ctx_s = pd.Series(y_ctx.cpu().numpy())
        wrapper.fit(X_ctx_df, y_ctx_s)
        raw = extract_raw_module(wrapper)
        raw.eval()

        results = eval_attacks(raw, X_ctx, y_ctx, X_test, y_test,
                                n_samples=n_samples, k=k, sigma_values=sigma_values)
        all_results[f"{model_name}_adult"] = results

        print(f"\n  [{model_name} / Adult Income]")
        print(f"  Clean acc: {results['clean_acc']*100:.1f}%")
        for sigma in sigma_values:
            print(f"  Attack D (σ={sigma}): ASR = {results[f'D_s{sigma}']*100:.1f}%")
        print(f"  Attack G (pool):   ASR = {results['G_pool']*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # GAP 2: HELOC DATASET
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*64}")
    print("GAP 2: HELOC Dataset (OrionBix)")
    print(f"{'━'*64}")

    X_ctx_h, y_ctx_h, X_test_h, y_test_h, feat_h, scaler_h = load_heloc()

    wrapper = load_model("orion-bix")
    X_ctx_df_h = pd.DataFrame(X_ctx_h.cpu().numpy(), columns=feat_h)
    y_ctx_s_h = pd.Series(y_ctx_h.cpu().numpy())
    wrapper.fit(X_ctx_df_h, y_ctx_s_h)
    raw_h = extract_raw_module(wrapper)
    raw_h.eval()

    results_h = eval_attacks(raw_h, X_ctx_h, y_ctx_h, X_test_h, y_test_h,
                              n_samples=n_samples, k=k, sigma_values=sigma_values)
    all_results["orion-bix_heloc"] = results_h

    print(f"\n  [OrionBix / HELOC]")
    print(f"  Clean acc: {results_h['clean_acc']*100:.1f}%")
    for sigma in sigma_values:
        print(f"  Attack D (σ={sigma}): ASR = {results_h[f'D_s{sigma}']*100:.1f}%")
    print(f"  Attack G (pool):   ASR = {results_h['G_pool']*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # GAP 3: D2 FPR AT OPERATING POINT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*64}")
    print("GAP 3: D2 (Mahalanobis) FPR at Operating Point")
    print(f"{'━'*64}")

    d2_results = {}
    if NNSIGHT_AVAILABLE:
        # Use OrionBix on Adult Income (same as Exp 5)
        wrapper_bix = load_model("orion-bix")
        X_ctx_df_a = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
        y_ctx_s_a = pd.Series(y_ctx.cpu().numpy())
        wrapper_bix.fit(X_ctx_df_a, y_ctx_s_a)
        raw_bix = extract_raw_module(wrapper_bix)
        raw_bix.eval()
        nn_bix = NNsight(raw_bix)

        for sigma in [0.01, 0.05]:
            print(f"\n  [D2 FPR] σ={sigma}")
            d2 = eval_d2_fpr(nn_bix, raw_bix, X_ctx, y_ctx, X_test, y_test,
                              k=k, sigma=sigma, n_samples=min(n_samples, 80))
            d2_results[sigma] = d2
            if d2:
                print(f"  AUROC: {d2['auroc']:.3f}")
                for tpr_pct in [80, 85, 90]:
                    fpr_val = d2.get(f'fpr_at_tpr{tpr_pct}', float('nan'))
                    prec_val = d2.get(f'precision_at_tpr{tpr_pct}', float('nan'))
                    print(f"  At TPR={tpr_pct}%: FPR={fpr_val*100:.1f}%, "
                          f"Precision={prec_val*100:.1f}%")
    else:
        print("  NNsight unavailable — skipping D2")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("EXPERIMENT 6 — TRANSFER MATRIX")
    print(f"{'='*64}")
    print(f"\n  {'Config':<25s}  {'Clean':>6s}  {'D σ=.01':>8s}  {'D σ=.05':>8s}  {'G pool':>8s}")
    print(f"  {'─'*60}")
    for key, r in all_results.items():
        clean = r['clean_acc'] * 100
        d1 = r.get('D_s0.01', 0) * 100
        d5 = r.get('D_s0.05', 0) * 100
        g = r['G_pool'] * 100
        print(f"  {key:<25s}  {clean:>5.1f}%  {d1:>7.1f}%  {d5:>7.1f}%  {g:>7.1f}%")

    # Transfer check
    bix_adult = all_results.get("orion-bix_adult", {})
    msp_adult = all_results.get("orion-msp_adult", {})
    bix_heloc = all_results.get("orion-bix_heloc", {})

    print(f"\n  Transfer Checks:")
    if msp_adult:
        g_bix = bix_adult.get('G_pool', 0)
        g_msp = msp_adult.get('G_pool', 0)
        transfer = "✅ YES" if g_msp > 0.10 else "❌ NO"
        print(f"    OrionBix→OrionMSP (Attack G): {g_bix*100:.0f}% → {g_msp*100:.0f}%  {transfer}")
    if bix_heloc:
        g_adult = bix_adult.get('G_pool', 0)
        g_heloc = bix_heloc.get('G_pool', 0)
        transfer = "✅ YES" if g_heloc > 0.10 else "❌ NO"
        print(f"    Adult→HELOC (Attack G):        {g_adult*100:.0f}% → {g_heloc*100:.0f}%  {transfer}")

    if d2_results:
        print(f"\n{'='*64}")
        print("EXPERIMENT 6 — D2 DEPLOYABILITY (OrionBix / Adult)")
        print(f"{'='*64}")
        print(f"  {'sigma':>7s}  {'AUROC':>7s}  {'FPR@80':>7s}  {'FPR@85':>7s}  {'FPR@90':>7s}  {'Prec@85':>8s}")
        print(f"  {'─'*51}")
        for sigma, d in d2_results.items():
            if not d:
                continue
            print(f"  {sigma:>7.3f}  {d['auroc']:>7.3f}  "
                  f"{d.get('fpr_at_tpr80',0)*100:>6.1f}%  "
                  f"{d.get('fpr_at_tpr85',0)*100:>6.1f}%  "
                  f"{d.get('fpr_at_tpr90',0)*100:>6.1f}%  "
                  f"{d.get('precision_at_tpr85',0)*100:>7.1f}%")
        print(f"\n  Deployable = Precision@TPR85 > 80% (acceptable alert fatigue)")
    print(f"{'='*64}")

    # ── Visualization ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Transfer matrix heatmap
    ax = axes[0]
    configs = list(all_results.keys())
    attacks = ["D_s0.01", "D_s0.05", "D_s0.1", "G_pool"]
    attack_labels = ["D σ=.01", "D σ=.05", "D σ=.10", "G pool"]
    matrix = np.array([[all_results[c].get(a, 0)*100 for a in attacks] for c in configs])
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attack_labels, fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    for i in range(len(configs)):
        for j in range(len(attacks)):
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if matrix[i,j] > 50 else "black")
    ax.set_title("Transfer Matrix (ASR %)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Model comparison bars
    ax = axes[1]
    x_pos = np.arange(len(attacks))
    w = 0.25
    for idx, config in enumerate(configs):
        asrs = [all_results[config].get(a, 0)*100 for a in attacks]
        ax.bar(x_pos + idx*w - w, asrs, w, label=config.replace("_", " "),
               alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attack_labels, fontsize=8)
    ax.set_ylabel("ASR (%)")
    ax.set_title("Attack Comparison Across Configs", fontweight="bold")
    ax.legend(fontsize=7)

    # Panel 3: D2 ROC curve if available
    ax = axes[2]
    if d2_results:
        for sigma, d in d2_results.items():
            if d and 'fpr' in d:
                ax.plot(d['fpr'], d['tpr'], linewidth=2,
                        label=f"σ={sigma} (AUROC={d['auroc']:.3f})")
                # Mark operating points
                for tpr_pct in [80, 85, 90]:
                    fpr_val = d.get(f'fpr_at_tpr{tpr_pct}', None)
                    if fpr_val is not None:
                        ax.scatter(fpr_val, tpr_pct/100, s=40, zorder=5)
        ax.plot([0,1], [0,1], "k--", alpha=0.3, label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("D2 ROC Curve\n(FPR = clean queries flagged)", fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "D2 analysis\nnot available", ha="center",
                va="center", fontsize=12, transform=ax.transAxes)

    plt.suptitle("Experiment 6 — Transfer, Generalization & D2 Deployability",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp6_gaps.png", dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved exp6_gaps.png")
    plt.close()

    return all_results, d2_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, d2 = run_experiment_6(n_samples=100, k=3)
