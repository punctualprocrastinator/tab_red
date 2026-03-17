# ============================================================
# EXPERIMENT 5 — Detection & Defense Analysis (Enhanced)
# Colab-compatible: assumes exp_0 cells already executed.
#
# Detection surfaces:
#   D1) Input-space (min-dist to test point)
#   D2) Representation-space (Mahalanobis on residual stream)
#   D3) Prediction-space (confidence shift)
#
# Attacks (including adaptive):
#   D  = Synthetic Near-Dup (from Exp 4)
#   F  = Filter-Aware Interpolation (adapts to evade F1)
#   G  = Pool-Only Label Flip (undetectable by D1)
#
# Defenses:
#   F1) Near-duplicate filter
#   F2) Majority-vote ensemble
#   F3) Leave-one-out influence (catches G)
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

if 'DEVICE' not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared helpers ────────────────────────────────────────────────────────────

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


# ── Attack D reproduced (from exp4) ──────────────────────────────────────────

def attack_synthetic_near_dup(X_ctx, y_ctx, x_test, true_label,
                               k, rng, sigma=0.01):
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    wrong = 1 - true_label
    x_np = x_test.cpu().numpy()
    for pos in positions:
        noise = rng.normal(0, sigma, size=x_np.shape).astype(np.float32)
        X_p[pos] = torch.tensor(x_np + noise,
                                 dtype=X_ctx.dtype, device=X_ctx.device)
        y_p[pos] = wrong
    return X_p, y_p


# ── D1: Input-space detection ─────────────────────────────────────────────────

def detect_input_space(X_ctx_clean, X_ctx_poison, x_test, sigma_test):
    """
    Detector: flag a context as poisoned if any example is suspiciously
    close to x_test in L2 distance.

    Threshold swept from 0 to max observed distance.
    Returns AUROC over clean vs poisoned contexts (oracle evaluation:
    we know which contexts are poisoned).

    Real deployment: threshold set at training-time based on expected
    minimum pairwise distance in a clean context.
    """
    x_np = x_test.cpu().numpy()

    # Score for each context: minimum distance from any context example to x_test
    def min_dist_to_test(X_ctx):
        dists = np.linalg.norm(X_ctx.cpu().numpy() - x_np, axis=1)
        return dists.min()

    clean_score  = min_dist_to_test(X_ctx_clean)   # should be large
    poison_score = min_dist_to_test(X_ctx_poison)  # should be small

    return clean_score, poison_score


def sweep_detection_thresholds(raw_model, X_ctx, y_ctx, X_test, y_test,
                                sigma_values, k=3, n_samples=100):
    """
    For each sigma, compute the distribution of min-dist scores across
    n_samples test points (clean and poisoned contexts).
    Returns dict {sigma: {"clean": [...], "poison": [...]}}
    """
    results = {}

    for sigma in sigma_values:
        clean_scores, poison_scores = [], []

        for i in range(min(n_samples, len(X_test))):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue

            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                  k, rng, sigma)

            c, p = detect_input_space(X_ctx, X_p, xi, sigma)
            clean_scores.append(c)
            poison_scores.append(p)

        results[sigma] = {
            "clean":  np.array(clean_scores),
            "poison": np.array(poison_scores),
        }
        print(f"  sigma={sigma:.2f}  "
              f"clean_dist={np.mean(clean_scores):.3f}±{np.std(clean_scores):.3f}  "
              f"poison_dist={np.mean(poison_scores):.3f}±{np.std(poison_scores):.3f}")

    return results


def compute_detection_auroc(clean_scores, poison_scores):
    """
    AUROC for separating clean from poisoned contexts.
    Score = min_dist (lower = more suspicious = positive label).
    Negate so that lower distance = higher score = positive.
    """
    y_true  = np.array([0]*len(clean_scores) + [1]*len(poison_scores))
    # negate: small distance = poisoned = label 1
    scores  = np.array(list(-clean_scores) + list(-poison_scores))
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, scores)


# ── D2: Representation-space detection ───────────────────────────────────────

ICL_REPR_PATH = "icl_predictor.tf_icl.blocks.0"

def get_last_token_repr(nn_model, raw_model, X_ctx, y_ctx, x_test_i,
                         path=ICL_REPR_PATH):
    """
    Extract the residual stream state at the test-token position
    from the first ICL block (most causally important per Exp 2).
    Returns (d_model,) numpy array, or None if extraction fails.
    """
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    submod = None
    if nn_model is not None:
        obj = nn_model
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
            if obj is None:
                break
        submod = obj

    if submod is None:
        return None

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
            return None

    while act.dim() > 2:
        act = act.squeeze(0)
    # Last position = test token
    return act[-1].numpy()


def collect_repr_scores(nn_model, raw_model, X_ctx, y_ctx, X_test, y_test,
                         sigma_values, k=3, n_samples=80):
    """
    For each sigma: collect residual stream representations for clean
    and poisoned runs. Fit a centroid detector on clean reprs, score
    poisoned reprs by Mahalanobis distance from clean centroid.
    """
    results = {}

    # Collect clean representations first
    clean_reprs = []
    for i in range(min(n_samples, len(X_test))):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue
        r = get_last_token_repr(nn_model, raw_model, X_ctx, y_ctx, xi)
        if r is not None:
            clean_reprs.append(r)

    if len(clean_reprs) < 10:
        print("  [D2] Not enough clean representations — skipping repr detection")
        return {}

    clean_reprs = np.stack(clean_reprs)
    centroid = clean_reprs.mean(0)
    # PCA for stable Mahalanobis (residual stream is high-dim)
    from sklearn.decomposition import PCA
    n_components = min(32, clean_reprs.shape[0] - 1, clean_reprs.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(clean_reprs)
    clean_proj = pca.transform(clean_reprs)
    proj_cov = np.cov(clean_proj.T) + np.eye(n_components) * 1e-6
    proj_cov_inv = np.linalg.inv(proj_cov)

    def mahal_score(r):
        rp = pca.transform(r.reshape(1, -1))[0]
        diff = rp - clean_proj.mean(0)
        return float(diff @ proj_cov_inv @ diff)

    clean_mahal = np.array([mahal_score(r) for r in clean_reprs])
    print(f"  [D2] Clean repr Mahalanobis: mean={clean_mahal.mean():.2f} "
          f"std={clean_mahal.std():.2f}")

    for sigma in sigma_values:
        poison_mahal = []
        n_collected = 0
        for i in range(min(n_samples, len(X_test))):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                  k, rng, sigma)
            r = get_last_token_repr(nn_model, raw_model, X_p, y_p, xi)
            if r is not None:
                poison_mahal.append(mahal_score(r))
                n_collected += 1

        if not poison_mahal:
            continue

        poison_mahal = np.array(poison_mahal)
        y_true = np.array([0]*len(clean_mahal) + [1]*len(poison_mahal))
        scores = np.concatenate([clean_mahal, poison_mahal])
        try:
            auroc = roc_auc_score(y_true, scores)
        except Exception:
            auroc = 0.5
        results[sigma] = {
            "clean_mahal":  clean_mahal,
            "poison_mahal": poison_mahal,
            "auroc":        auroc,
        }
        print(f"  sigma={sigma:.2f}  "
              f"poison_mahal={poison_mahal.mean():.2f}±{poison_mahal.std():.2f}  "
              f"AUROC={auroc:.3f}")

    return results


# ── D3: Prediction-space detection ───────────────────────────────────────────

def collect_confidence_scores(raw_model, X_ctx, y_ctx, X_test, y_test,
                               sigma_values, k=3, n_samples=100):
    """
    For each sigma: collect confidence (max softmax prob) on poisoned runs.
    A high-confidence wrong prediction is a detectable signal.
    Returns {sigma: {"clean_conf": [...], "poison_conf": [...], "auroc": float}}
    """
    results = {}

    clean_conf = []
    for i in range(min(n_samples, len(X_test))):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue
        p = _predict_proba(raw_model, X_ctx, y_ctx, xi)
        clean_conf.append(p.max())

    for sigma in sigma_values:
        poison_conf = []
        for i in range(min(n_samples, len(X_test))):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                  k, rng, sigma)
            p = _predict_proba(raw_model, X_p, y_p, xi)
            poison_conf.append(p.max())

        # AUROC: can we separate clean from poisoned by confidence alone?
        y_true = np.array([0]*len(clean_conf) + [1]*len(poison_conf))
        scores = np.array(clean_conf + poison_conf)
        try:
            auroc = roc_auc_score(y_true, scores)
        except Exception:
            auroc = 0.5

        results[sigma] = {
            "clean_conf":  np.array(clean_conf),
            "poison_conf": np.array(poison_conf),
            "auroc":       auroc,
        }
        print(f"  sigma={sigma:.3f}  "
              f"clean_conf={np.mean(clean_conf):.3f}  "
              f"poison_conf={np.mean(poison_conf):.3f}  "
              f"AUROC={auroc:.3f}")

    return results


# ── F1: Near-duplicate filter defense ────────────────────────────────────────

def defense_neardup_filter(raw_model, X_ctx, y_ctx, X_test, y_test,
                            sigma_values, k=3, n_samples=100,
                            filter_radius=None):
    """
    Before inference: remove any context example within filter_radius
    of x_test in L2 distance.

    filter_radius=None: auto-set to 3 * sigma (covers 99.7% of noise ball).
    Returns {sigma: {"asr_undefended": float, "asr_defended": float}}
    """
    results = {}

    for sigma in sigma_values:
        radius = filter_radius if filter_radius is not None else 3 * sigma * np.sqrt(14)

        n_undefended, n_defended, n_tested = 0, 0, 0

        for i in range(min(n_samples, len(X_test))):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            n_tested += 1

            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                  k, rng, sigma)

            # Undefended ASR
            if _predict(raw_model, X_p, y_p, xi) != yi:
                n_undefended += 1

            # Defended: filter out near-dups of xi, then predict
            x_np = xi.cpu().numpy()
            dists = np.linalg.norm(X_p.cpu().numpy() - x_np, axis=1)
            keep_mask = dists > radius

            # If too many removed, fall back to clean context
            if keep_mask.sum() < 10:
                X_def, y_def = X_ctx, y_ctx
            else:
                X_def = X_p[keep_mask]
                y_def = y_p[keep_mask]

            if _predict(raw_model, X_def, y_def, xi) != yi:
                n_defended += 1

        n = max(n_tested, 1)
        asr_u = n_undefended / n
        asr_d = n_defended / n
        results[sigma] = {
            "asr_undefended": round(asr_u, 4),
            "asr_defended":   round(asr_d, 4),
            "radius_used":    round(radius, 4),
        }
        print(f"  sigma={sigma:.3f}  radius={radius:.3f}  "
              f"ASR: {asr_u*100:.1f}% -> {asr_d*100:.1f}% (defended)")

    return results


# ── F2: Majority-vote ensemble ────────────────────────────────────────────────

def defense_majority_vote(raw_model, X_ctx, y_ctx, X_test, y_test,
                           sigma_values, k=3, n_resamples=10,
                           subsample_frac=0.8, n_samples=100):
    """
    Resample a random subset of context n_resamples times and take
    majority vote across predictions.

    If k poisoned examples are in a context of 256, and we subsample
    80% each time, each resample has ~k*(0.8) = 0.8k expected poison
    examples vs 205 clean ones. Majority vote over 10 resamples
    should dilute isolated poison injections.

    Effective against low-k attacks where poison examples are sparse.
    """
    results = {}
    ctx_size = len(X_ctx)
    sub_size = int(ctx_size * subsample_frac)

    for sigma in sigma_values:
        n_undefended, n_defended, n_tested = 0, 0, 0

        for i in range(min(n_samples, len(X_test))):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            n_tested += 1

            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_synthetic_near_dup(X_ctx, y_ctx, xi, yi,
                                                  k, rng, sigma)

            # Undefended
            if _predict(raw_model, X_p, y_p, xi) != yi:
                n_undefended += 1

            # Majority vote over resampled contexts
            votes = []
            vote_rng = np.random.default_rng(999 + i)
            for _ in range(n_resamples):
                idx = vote_rng.choice(ctx_size, size=sub_size, replace=False)
                X_sub = X_p[idx]
                y_sub = y_p[idx]
                votes.append(_predict(raw_model, X_sub, y_sub, xi))

            majority = int(np.bincount(votes).argmax())
            if majority != yi:
                n_defended += 1

        n = max(n_tested, 1)
        results[sigma] = {
            "asr_undefended": round(n_undefended / n, 4),
            "asr_defended":   round(n_defended / n, 4),
        }
        print(f"  sigma={sigma:.3f}  "
              f"ASR: {n_undefended/n*100:.1f}% -> {n_defended/n*100:.1f}%  "
              f"(majority vote, m={n_resamples}, f={subsample_frac})")

    return results


# ── Attack F: Filter-Aware Interpolation ─────────────────────────────────────

def attack_filter_aware(X_ctx, y_ctx, x_test, true_label,
                        X_pool, y_pool, k, rng, filter_radius):
    """
    Adaptive attack: craft poison that stays JUST OUTSIDE the filter radius.
    Uses interpolation between x_test and nearest same-class pool example,
    with alpha tuned so L2(poison, x_test) ≈ filter_radius + epsilon.

    If F1 uses radius r, we set:
      alpha = max(0, 1 - (r + 0.01) / dist(x_test, neighbor))
    """
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    wrong = 1 - true_label
    x_np = x_test.cpu().numpy()
    X_pool_np = X_pool.cpu().numpy()
    y_pool_np = y_pool.cpu().numpy()

    same_mask = (y_pool_np == true_label)
    X_same = X_pool_np[same_mask]
    if len(X_same) == 0:
        return X_p, y_p

    dists = np.linalg.norm(X_same - x_np, axis=1)
    nn_idx = np.argsort(dists)[:k]

    for i, pos in enumerate(positions):
        if i < len(nn_idx):
            neighbor = X_same[nn_idx[i]]
            d = np.linalg.norm(neighbor - x_np)
            # alpha so that interpolated point is just outside radius
            if d > 1e-8:
                alpha = max(0.0, 1.0 - (filter_radius + 0.01) / d)
            else:
                alpha = 0.0
            interpolated = alpha * x_np + (1 - alpha) * neighbor
            X_p[pos] = torch.tensor(interpolated,
                                     dtype=X_ctx.dtype, device=X_ctx.device)
            y_p[pos] = wrong

    return X_p, y_p


# ── Attack G: Pool-Only Label Flip ────────────────────────────────────────────

def attack_pool_only(X_ctx, y_ctx, x_test, true_label, k, rng):
    """
    Undetectable by D1: use ONLY real context examples, no synthetic features.
    Find the k nearest same-class context examples to x_test and flip their
    labels. Features are 100% genuine — only labels are wrong.

    F1 cannot catch this because no example is suspiciously close to x_test.
    """
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    x_np = x_test.cpu().numpy()
    X_np = X_ctx.cpu().numpy()
    y_np = y_ctx.cpu().numpy()

    # Find same-class examples nearest to x_test
    same_mask = (y_np == true_label)
    same_indices = np.where(same_mask)[0]
    if len(same_indices) == 0:
        return X_p, y_p

    dists = np.linalg.norm(X_np[same_indices] - x_np, axis=1)
    nearest = same_indices[np.argsort(dists)[:k]]

    for pos in nearest:
        y_p[pos] = 1 - y_p[pos]  # flip label only

    return X_p, y_p


# ── Defense F3: Leave-One-Out Influence ───────────────────────────────────────

def defense_leave_one_out(raw_model, X_ctx, y_ctx, x_test, yi,
                           top_k=5):
    """
    For each context example, check if removing it changes the prediction.
    Flag the top_k highest-influence examples.

    Returns indices of flagged examples and whether defense changes prediction.
    """
    base_pred = _predict(raw_model, X_ctx, y_ctx, x_test)
    ctx_size = len(X_ctx)
    influence = np.zeros(ctx_size)

    base_proba = _predict_proba(raw_model, X_ctx, y_ctx, x_test)
    base_conf = base_proba[base_pred]

    for j in range(ctx_size):
        # Remove example j
        mask = torch.ones(ctx_size, dtype=torch.bool)
        mask[j] = False
        X_loo = X_ctx[mask]
        y_loo = y_ctx[mask]
        loo_proba = _predict_proba(raw_model, X_loo, y_loo, x_test)
        loo_conf = loo_proba[base_pred]
        influence[j] = base_conf - loo_conf  # positive = removing this HELPS correctness

    # Flag top_k most influential (highest positive influence = most suspicious)
    flagged = np.argsort(influence)[-top_k:][::-1]

    # Defense: remove flagged examples
    keep_mask = torch.ones(ctx_size, dtype=torch.bool)
    for j in flagged:
        keep_mask[j] = False
    X_def = X_ctx[keep_mask]
    y_def = y_ctx[keep_mask]
    defended_pred = _predict(raw_model, X_def, y_def, x_test)

    return flagged, influence, defended_pred


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_experiment_5(n_samples=100, k=3):
    print("=" * 64)
    print("EXPERIMENT 5 — Detection & Defense Analysis (Enhanced)")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    sigma_values = [0.01, 0.05, 0.10, 0.20, 0.50]

    # ── Data + model ──────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()
    nn_model = NNsight(raw) if NNSIGHT_AVAILABLE else None

    n_eval = min(n_samples, len(X_test))
    X_pool, y_pool = X_ctx.clone(), y_ctx.clone()

    # ── D1: Input-space detection ─────────────────────────────────────────
    print(f"\n[D1] Input-space (min-dist) detection")
    d1 = sweep_detection_thresholds(raw, X_ctx, y_ctx, X_test, y_test,
                                     sigma_values, k=k, n_samples=n_samples)
    d1_aurocs = {}
    for sigma, data in d1.items():
        auroc = compute_detection_auroc(data["clean"], data["poison"])
        d1_aurocs[sigma] = auroc
        print(f"  sigma={sigma:.2f}  D1 AUROC={auroc:.3f}")

    # ── D2: Representation-space detection ───────────────────────────────
    print(f"\n[D2] Representation-space (Mahalanobis) detection")
    if NNSIGHT_AVAILABLE:
        d2 = collect_repr_scores(nn_model, raw, X_ctx, y_ctx, X_test, y_test,
                                  sigma_values, k=k, n_samples=min(n_samples, 80))
        d2_aurocs = {s: d2[s]["auroc"] for s in d2}
    else:
        print("  [D2] NNsight unavailable — skipping")
        d2, d2_aurocs = {}, {}

    # ── D3: Confidence-based detection ───────────────────────────────────
    print(f"\n[D3] Prediction-space (confidence) detection")
    d3 = collect_confidence_scores(raw, X_ctx, y_ctx, X_test, y_test,
                                    sigma_values, k=k, n_samples=n_samples)
    d3_aurocs = {s: d3[s]["auroc"] for s in d3}

    # ── F1: Near-duplicate filter ─────────────────────────────────────────
    print(f"\n[F1] Defense: near-duplicate filter")
    f1 = defense_neardup_filter(raw, X_ctx, y_ctx, X_test, y_test,
                                 sigma_values, k=k, n_samples=n_samples)

    # ── F2: Majority-vote ensemble ────────────────────────────────────────
    print(f"\n[F2] Defense: majority-vote ensemble (m=10, f=0.8)")
    f2 = defense_majority_vote(raw, X_ctx, y_ctx, X_test, y_test,
                                sigma_values, k=k, n_samples=n_samples)

    # ═══════════════════════════════════════════════════════════════════════
    # NEW: Adaptive attacks + Leave-One-Out defense
    # ═══════════════════════════════════════════════════════════════════════

    # ── Attack F vs F1 (filter-aware) ─────────────────────────────────────
    print(f"\n[ATK-F] Attack F: filter-aware interpolation vs F1")
    atk_f_results = {}
    for sigma in sigma_values:
        radius = 3 * sigma * np.sqrt(14)  # same as F1
        n_f_undefended, n_f_defended, n_tested = 0, 0, 0
        for i in range(n_eval):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw, X_ctx, y_ctx, xi) != yi:
                continue
            n_tested += 1
            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_filter_aware(
                X_ctx, y_ctx, xi, yi, X_pool, y_pool, k, rng, radius
            )
            pred = _predict(raw, X_p, y_p, xi)
            if pred != yi:
                n_f_undefended += 1
            # Apply F1 defense on Attack F output
            x_np = xi.cpu().numpy()
            dists = np.linalg.norm(X_p.cpu().numpy() - x_np, axis=1)
            keep = dists > radius
            if keep.sum() < 10:
                X_def, y_def = X_ctx, y_ctx
            else:
                X_def, y_def = X_p[keep], y_p[keep]
            if _predict(raw, X_def, y_def, xi) != yi:
                n_f_defended += 1
        n = max(n_tested, 1)
        atk_f_results[sigma] = {
            "asr_undefended": round(n_f_undefended / n, 4),
            "asr_vs_f1":      round(n_f_defended / n, 4),
        }
        print(f"  sigma={sigma:.3f}  radius={radius:.3f}  "
              f"ASR(raw)={n_f_undefended/n*100:.1f}%  "
              f"ASR(vs F1)={n_f_defended/n*100:.1f}%")

    # ── Attack G: Pool-Only (undetectable by D1) ──────────────────────────
    print(f"\n[ATK-G] Attack G: pool-only label flip")
    print(f"         (features unchanged, only labels flipped)")
    atk_g_asr, atk_g_tested = 0, 0
    atk_g_min_dists = []
    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw, X_ctx, y_ctx, xi) != yi:
            continue
        atk_g_tested += 1
        rng = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng)
        if _predict(raw, X_g, y_g, xi) != yi:
            atk_g_asr += 1
        # Check D1 min-dist (should be same as clean)
        x_np = xi.cpu().numpy()
        md = np.linalg.norm(X_g.cpu().numpy() - x_np, axis=1).min()
        atk_g_min_dists.append(md)

    n = max(atk_g_tested, 1)
    atk_g_rate = atk_g_asr / n
    print(f"  ASR: {atk_g_rate*100:.1f}%  ({atk_g_asr}/{n})")
    print(f"  D1 min-dist: {np.mean(atk_g_min_dists):.3f}  "
          f"(same as clean: {np.mean(d1[0.01]['clean']):.3f})")
    print(f"  → D1 CANNOT detect Attack G (no near-dups injected)")

    # ── F3: Leave-One-Out defense vs Attack G ─────────────────────────────
    print(f"\n[F3] Defense: leave-one-out influence vs Attack G")
    n_f3_defended, n_f3_tested = 0, 0
    n_f3_samples = min(50, n_eval)  # LOO is expensive (256 fwd passes per sample)
    for i in range(n_f3_samples):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw, X_ctx, y_ctx, xi) != yi:
            continue
        n_f3_tested += 1
        rng = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng)
        if _predict(raw, X_g, y_g, xi) == yi:
            continue  # attack didn't flip in the first place
        flagged, influence, defended_pred = defense_leave_one_out(
            raw, X_g, y_g, xi, yi, top_k=k
        )
        if defended_pred == yi:
            n_f3_defended += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_f3_samples}] processed")

    print(f"  Attack G flips caught by F3: {n_f3_defended} "
          f"(of {max(atk_g_asr, 1)} total flips in first {n_f3_samples} samples)")

    # ── Summary tables ────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPERIMENT 5 — DETECTION AUROC SUMMARY")
    print(f"{'='*64}")
    print(f"  {'sigma':>7s}  {'D1(dist)':>9s}  {'D2(repr)':>9s}  {'D3(conf)':>9s}")
    print(f"  {'─'*44}")
    for sigma in sigma_values:
        d1a = d1_aurocs.get(sigma, float('nan'))
        d2a = d2_aurocs.get(sigma, float('nan'))
        d3a = d3_aurocs.get(sigma, float('nan'))
        print(f"  {sigma:>7.3f}  {d1a:>9.3f}  {d2a:>9.3f}  {d3a:>9.3f}")

    print(f"\n{'='*64}")
    print("EXPERIMENT 5 — DEFENSE SUMMARY")
    print(f"{'='*64}")
    print(f"  {'sigma':>7s}  {'D(raw)':>7s}  {'D+F1':>7s}  {'F(raw)':>7s}  {'F+F1':>7s}  {'G(raw)':>7s}")
    print(f"  {'─'*50}")
    for sigma in sigma_values:
        d_raw = f1.get(sigma, {}).get('asr_undefended', 0) * 100
        d_f1 = f1.get(sigma, {}).get('asr_defended', 0) * 100
        f_raw = atk_f_results.get(sigma, {}).get('asr_undefended', 0) * 100
        f_f1 = atk_f_results.get(sigma, {}).get('asr_vs_f1', 0) * 100
        print(f"  {sigma:>7.3f}  {d_raw:>6.1f}%  {d_f1:>6.1f}%  "
              f"{f_raw:>6.1f}%  {f_f1:>6.1f}%  {atk_g_rate*100:>6.1f}%")
    print(f"\n  Attack G (pool-only): {atk_g_rate*100:.1f}% ASR (D1 invisible)")
    print(f"{'='*64}")

    # ── Visualization (4 panels) ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sigs = sigma_values

    # Panel 1: Detection AUROC
    ax = axes[0, 0]
    ax.plot(sigs, [d1_aurocs.get(s, 0.5) for s in sigs],
            "o-", label="D1: min-dist", color="#4C72B0", linewidth=2)
    if d2_aurocs:
        ax.plot(sigs, [d2_aurocs.get(s, 0.5) for s in sigs],
                "s-", label="D2: repr Mahal", color="#DD8452", linewidth=2)
    ax.plot(sigs, [d3_aurocs.get(s, 0.5) for s in sigs],
            "^-", label="D3: confidence", color="#55A868", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Detection AUROC")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Detection AUROC vs Attack Sigma", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Defense comparison (D, F, G)
    ax = axes[0, 1]
    x_pos = np.arange(len(sigs))
    w = 0.18
    d_raw = [f1.get(s, {}).get('asr_undefended', 0)*100 for s in sigs]
    d_f1  = [f1.get(s, {}).get('asr_defended', 0)*100 for s in sigs]
    f_raw = [atk_f_results.get(s, {}).get('asr_undefended', 0)*100 for s in sigs]
    f_f1  = [atk_f_results.get(s, {}).get('asr_vs_f1', 0)*100 for s in sigs]
    ax.bar(x_pos - 1.5*w, d_raw, w, label="D: undefended",   color="#E84040", alpha=0.85)
    ax.bar(x_pos - 0.5*w, d_f1,  w, label="D: + F1 filter",  color="#4C72B0", alpha=0.85)
    ax.bar(x_pos + 0.5*w, f_raw, w, label="F: filter-aware", color="#DD8452", alpha=0.85)
    ax.bar(x_pos + 1.5*w, f_f1,  w, label="F: + F1 filter",  color="#55A868", alpha=0.85)
    ax.axhline(y=atk_g_rate*100, color="purple", linestyle="--", alpha=0.6,
               label=f"G: pool-only ({atk_g_rate*100:.0f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"σ={s}" for s in sigs], fontsize=8)
    ax.set_ylabel("ASR (%)")
    ax.set_title("Attack vs Defense Arms Race", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: Stealth-Efficacy with Attack G annotation
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(sigs, d_raw, "o-", color="#E84040",
                  linewidth=2.5, label="Attack D ASR")
    l2, = ax2.plot(sigs, [d1_aurocs.get(s, 0.5) for s in sigs],
                   "s--", color="#4C72B0", linewidth=2, label="D1 AUROC")
    ax.axhline(y=atk_g_rate*100, color="purple", linestyle=":",
               alpha=0.8, linewidth=2, label=f"Attack G ASR ({atk_g_rate*100:.0f}%)")
    ax2.axhline(y=0.5, color="purple", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("ASR (%)", color="#E84040")
    ax2.set_ylabel("D1 AUROC", color="#4C72B0")
    ax.set_title("Stealth-Efficacy Tradeoff\n"
                 "Attack G escapes D1 (purple line)", fontweight="bold")
    ax.legend(handles=[l1, l2], loc="center left", fontsize=8)

    # Panel 4: The Key Paper Figure — ASR vs Detection 2D scatter
    ax = axes[1, 1]
    # Collect all attack variants as (ASR, D1_AUROC, label)
    points = []
    for s in sigs:
        d_asr = f1.get(s, {}).get('asr_undefended', 0)
        d_det = d1_aurocs.get(s, 0.5)
        points.append((d_asr*100, d_det, f"D σ={s}", "#E84040"))
        f_asr = atk_f_results.get(s, {}).get('asr_undefended', 0)
        points.append((f_asr*100, d_det, f"F σ={s}", "#DD8452"))
    # Attack G (D1 ~ same as clean = 0.5 AUROC)
    points.append((atk_g_rate*100, 0.50, "G pool-only", "purple"))

    for asr, det, label, color in points:
        ax.scatter(asr, det, c=color, s=60, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(label, (asr, det), fontsize=5, ha='left',
                    xytext=(4, 3), textcoords='offset points')

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(26, color="gray", linestyle="--", alpha=0.3, label="clean error")
    ax.set_xlabel("Attack Success Rate (%) →\n(higher = better for attacker)", fontsize=9)
    ax.set_ylabel("Detection AUROC →\n(higher = easier to catch)", fontsize=9)
    ax.set_title("★ Key Figure: ASR vs Detectability", fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(0.35, 1.05)

    # Ideal attacker region
    ax.fill_between([50, 105], 0.35, 0.55, alpha=0.08, color='green')
    ax.text(77, 0.42, 'attacker ideal\n(high ASR, low detect)', fontsize=7,
            color='green', ha='center', style='italic')

    plt.suptitle("Experiment 5 — Detection, Defense & Adaptive Attacks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp5_detection.png", dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved exp5_detection.png")
    plt.close()

    return {
        "d1_aurocs": d1_aurocs,
        "d2_aurocs": d2_aurocs,
        "d3_aurocs": d3_aurocs,
        "f1": f1,
        "f2": f2,
        "atk_f": atk_f_results,
        "atk_g_asr": atk_g_rate,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_experiment_5(n_samples=100, k=3)