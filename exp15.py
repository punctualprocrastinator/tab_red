"""
Experiment 15 — Preprocessing Distance Diagnostic

Hypothesis: TabICL's resistance to SILHOUETTE (σ=0.01) is explained by its
preprocessor inflating small perturbations to larger distances in internal
feature space, while ORION-BiX's preprocessor preserves input-space distances.

Three measurements:
  1. Raw input space:       L2(x, x̃) at σ ∈ {0.01, 0.05, 0.10, 0.20, 0.50}
  2. ORION-BiX internal:    L2(preproc_bix(x), preproc_bix(x̃))
  3. TabICL internal:       L2(preproc_tabicl(x), preproc_tabicl(x̃))

If TabICL inflates σ=0.01 to ~0.3+ in internal space, that explains the
ASR collapse — the "near-duplicate" is no longer near in the model's
operative distance space.

Also checks the contrast-seeking head (H0, pos-frac=0.258) contribution
via a simple per-head attention simulation, and the block-1 OV norm
dominance hypothesis.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
OUT    = os.environ.get("OUT", "/kaggle/working/")
SIGMAS = [0.01, 0.05, 0.10, 0.20, 0.50]
N_SAMPLES = 100   # test points to average over


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_adult_raw():
    """Return raw (un-standardised) X so preprocessors see the original data."""
    print("[DATA] Loading Adult Income (raw, no standardisation)...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df   = data.frame
    tc   = "income" if "income" in df.columns else df.columns[-1]
    y    = LabelEncoder().fit_transform(df[tc])
    X_df = df.drop(columns=[tc])
    for c in X_df.select_dtypes(include=["category","object"]).columns:
        X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))
    X    = X_df.values.astype(np.float32)
    feat = list(X_df.columns)
    print(f"[DATA] Shape: {X.shape}  features: {feat[:4]}...")
    return X, y, feat


# ══════════════════════════════════════════════════════════════════════════════
# MODEL + PREPROCESSOR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(model_name, X_ctx, y_ctx, feat_names):
    """
    Returns (raw_nn_module, fitted_tabtune_wrapper).
    The wrapper retains the fitted preprocessor for transform calls.
    """
    from tabtune import TabularPipeline
    name_map = {
        "orion_bix": "OrionBix",
        "orion_msp": "OrionMSP",
        "tabicl":    "TabICL",
    }
    print(f"\n[MODEL] Loading {name_map[model_name]}...")
    wrapper = TabularPipeline(
        model_name=name_map[model_name],
        task_type="classification",
        tuning_strategy="inference",
        tuning_params={"device": "cuda" if DEVICE.type=="cuda" else "cpu"}
    )
    wrapper.fit(
        pd.DataFrame(X_ctx, columns=feat_names),
        pd.Series(y_ctx)
    )
    # Extract raw module
    raw = wrapper.model.model_
    raw.eval().to(DEVICE)
    print(f"[MODEL] {type(raw).__name__} ready on {next(raw.parameters()).device}")
    return raw, wrapper


def get_preprocessor(wrapper):
    """
    Navigate TabTune wrapper to find the fitted DataProcessor / preprocessor.
    Returns an object with a .transform() method, or None.
    """
    # Common paths TabTune uses
    candidates = [
        # path as list of attrs
        ["model", "data_processor"],
        ["model", "preprocessor"],
        ["data_processor"],
        ["preprocessor"],
        ["model", "tuning_manager", "data_processor"],
        ["model", "_data_processor"],
    ]
    for path in candidates:
        obj = wrapper
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "transform"):
            return obj

    # Fallback: walk all attributes looking for something with transform
    for attr in dir(wrapper):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(wrapper, attr)
            if hasattr(v, "transform") and hasattr(v, "fit"):
                return v
        except Exception:
            continue

    # Deep walk on wrapper.model
    if hasattr(wrapper, "model"):
        for attr in dir(wrapper.model):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(wrapper.model, attr)
                if hasattr(v, "transform") and hasattr(v, "fit"):
                    return v
            except Exception:
                continue

    return None


def preprocess_sample(preprocessor, x_raw, feat_names):
    """
    Apply preprocessor to a single sample.
    Returns numpy array in preprocessor's internal space, or None on failure.
    """
    if preprocessor is None:
        return None
    try:
        df = pd.DataFrame([x_raw], columns=feat_names)
        result = preprocessor.transform(df)
        if hasattr(result, "values"):
            result = result.values
        return result.astype(np.float32).flatten()
    except Exception as e:
        # Some preprocessors take numpy directly
        try:
            result = preprocessor.transform(x_raw.reshape(1, -1))
            if hasattr(result, "values"):
                result = result.values
            return result.astype(np.float32).flatten()
        except Exception as e2:
            print(f"  [WARN] preprocess_sample failed: {e2}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# EXP 15A — DISTANCE INFLATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def exp_15a_distance_inflation(X_raw, feat_names,
                                preproc_bix, preproc_tabicl,
                                n_samples=N_SAMPLES):
    """
    For each sigma, measure:
      - raw L2 distance between x and x̃ = x + N(0, sigma²I)
      - L2 distance in ORION-BiX preprocessed space
      - L2 distance in TabICL preprocessed space
      - inflation ratio = internal_dist / raw_dist

    Key question: does TabICL inflate σ=0.01 to the equivalent of σ=0.3
    in ORION-BiX space?
    """
    print("\n" + "="*64)
    print("EXP 15a — Preprocessing Distance Inflation")
    print("="*64)

    results = {s: {"raw":[], "bix":[], "tabicl":[]} for s in SIGMAS}
    n_valid = 0

    for i in range(min(n_samples * 3, len(X_raw) - 256)):
        if n_valid >= n_samples:
            break
        x = X_raw[256 + i]   # test region

        # Preprocess x once for both models
        px_bix    = preprocess_sample(preproc_bix,    x, feat_names)
        px_tabicl = preprocess_sample(preproc_tabicl, x, feat_names)

        if px_bix is None or px_tabicl is None:
            continue

        n_valid += 1
        rng = np.random.default_rng(42 + i)

        for sigma in SIGMAS:
            noise  = rng.normal(0, sigma, x.shape).astype(np.float32)
            x_dup  = x + noise

            px_dup_bix    = preprocess_sample(preproc_bix,    x_dup, feat_names)
            px_dup_tabicl = preprocess_sample(preproc_tabicl, x_dup, feat_names)

            raw_dist    = float(np.linalg.norm(noise))
            bix_dist    = float(np.linalg.norm(px_bix - px_dup_bix))    \
                          if px_dup_bix    is not None else float("nan")
            tabicl_dist = float(np.linalg.norm(px_tabicl - px_dup_tabicl)) \
                          if px_dup_tabicl is not None else float("nan")

            results[sigma]["raw"].append(raw_dist)
            results[sigma]["bix"].append(bix_dist)
            results[sigma]["tabicl"].append(tabicl_dist)

    print(f"\n  Samples: {n_valid}")
    print(f"\n  {'Sigma':>7s}  {'Raw L2':>10s}  {'BiX L2':>10s}  "
          f"{'TabICL L2':>12s}  {'BiX ratio':>10s}  {'ICL ratio':>10s}  "
          f"{'Inflation?':>12s}")
    print(f"  {'─'*80}")

    summary = {}
    for sigma in SIGMAS:
        raw    = np.nanmean(results[sigma]["raw"])
        bix    = np.nanmean(results[sigma]["bix"])
        tabicl = np.nanmean(results[sigma]["tabicl"])
        r_bix    = bix    / raw if raw > 0 else float("nan")
        r_tabicl = tabicl / raw if raw > 0 else float("nan")

        # Key diagnostic: does TabICL inflate σ=0.01 to internal > 0.15?
        # (0.15 = σ=0.05 equivalent in raw space, where BiX ASR drops to 87%)
        inflated = (not np.isnan(r_tabicl)) and (r_tabicl > r_bix * 1.5)
        flag = "INFLATED ✓" if inflated else "similar"

        print(f"  {sigma:>7.2f}  {raw:>10.4f}  {bix:>10.4f}  "
              f"{tabicl:>12.4f}  {r_bix:>10.3f}x  {r_tabicl:>10.3f}x  "
              f"{flag:>12s}")

        summary[sigma] = {
            "raw": raw, "bix": bix, "tabicl": tabicl,
            "ratio_bix": r_bix, "ratio_tabicl": r_tabicl,
            "inflated": inflated
        }

    # Verdict
    sigma_001 = summary.get(0.01, {})
    if sigma_001.get("inflated"):
        print(f"\n  ✅ HYPOTHESIS 1 CONFIRMED:")
        print(f"     TabICL inflates σ=0.01 by {sigma_001['ratio_tabicl']:.1f}x vs "
              f"ORION-BiX {sigma_001['ratio_bix']:.1f}x")
        print(f"     σ=0.01 raw → {sigma_001['tabicl']:.4f} internal (TabICL) "
              f"vs {sigma_001['bix']:.4f} (ORION-BiX)")
        # Find equivalent sigma in ORION-BiX space
        bix_vals = [(s, summary[s]["bix"]) for s in SIGMAS]
        equiv = min(bix_vals, key=lambda x: abs(x[1] - sigma_001["tabicl"]))
        print(f"     TabICL internal dist ≈ ORION-BiX σ={equiv[0]} (dist={equiv[1]:.4f})")
        print(f"     ORION-BiX ASR at equivalent sigma: see Table 4 in paper")
    else:
        print(f"\n  ❌ HYPOTHESIS 1 NOT CONFIRMED:")
        print(f"     Distance inflation ratio similar across models.")
        print(f"     TabICL resistance explained by H0 cancellation or block-1 OV dominance.")

    return summary, results


# ══════════════════════════════════════════════════════════════════════════════
# EXP 15B — CONTRAST HEAD CANCELLATION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def exp_15b_contrast_head_simulation(raw_bix, raw_tabicl,
                                      X_ctx_std, y_ctx, X_test_std,
                                      feat_names, n_samples=40):
    """
    Simulate per-head attention weights using a simplified QK dot product.

    For each head h at blocks.0, compute the attention score of the test token
    to the poison positions relative to clean positions:

        score_h(test, pos) = (W_Q_h @ x_test) · (W_K_h @ x_ctx_pos)

    Net attention on poison positions = mean over similarity-seeking heads
    minus contribution from contrast-seeking heads (H0 in TabICL).

    If the contrast head strongly down-weights poison positions, this
    explains the ASR collapse independently of preprocessing.
    """
    print("\n" + "="*64)
    print("EXP 15b — Contrast Head Cancellation Simulation")
    print("="*64)

    def nav(model, path):
        obj = model
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
            if obj is None: return None
        return obj

    def get_qk_weights(raw_model, block_path):
        mod = nav(raw_model, block_path)
        if mod is None: return None, None, None, None
        for name, m in mod.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                ipw = m.in_proj_weight.detach().float().cpu()
                d   = m.embed_dim
                n_h = m.num_heads
                d_h = d // n_h
                W_Q = ipw[:d].reshape(n_h, d_h, d)
                W_K = ipw[d:2*d].reshape(n_h, d_h, d)
                return W_Q, W_K, n_h, d_h
        return None, None, None, None

    def simulate_attn(W_Q, W_K, x_test, x_ctx, poison_positions, n_h, d_h, scale):
        """
        Compute softmax attention weights for each head.
        x_test: (d,), x_ctx: (N, d)
        Returns: (n_h, N) attention weights, (n_h,) mass on poison positions
        """
        scores_by_head = []
        poison_mass    = []
        for h in range(n_h):
            q = (W_Q[h] @ x_test)             # (d_h,)
            k = (W_K[h] @ x_ctx.T).T          # (N, d_h)
            s = (k @ q) / scale                # (N,)
            s_max = s - s.max()
            w = np.exp(s_max) / np.exp(s_max).sum()
            scores_by_head.append(w)
            poison_mass.append(float(w[poison_positions].sum()))
        return np.array(scores_by_head), np.array(poison_mass)

    results_bix    = {"clean": [], "poison": [], "by_head_clean": [], "by_head_poison": []}
    results_tabicl = {"clean": [], "poison": [], "by_head_clean": [], "by_head_poison": []}

    B0 = "icl_predictor.tf_icl.blocks.0"
    scale = np.sqrt(128)   # d_head = 128

    WQ_bix, WK_bix, nh_bix, dh_bix = get_qk_weights(raw_bix,    B0)
    WQ_icl, WK_icl, nh_icl, dh_icl = get_qk_weights(raw_tabicl, B0)

    if WQ_bix is None:
        print("  [SKIP] Could not extract QK weights for ORION-BiX")
        return {}
    if WQ_icl is None:
        print("  [SKIP] Could not extract QK weights for TabICL")
        return {}

    WQ_bix = WQ_bix.numpy(); WK_bix = WK_bix.numpy()
    WQ_icl = WQ_icl.numpy(); WK_icl = WK_icl.numpy()
    
    # Helper to capture 512-d embeddings via a forward hook
    def get_embeddings(raw_model, X_seq, y_seq):
        captured_embs = []
        def hook_fn(module, args, kwargs):
            # The tensor is usually (batch, seq, d_model)
            # Find the first 3D tensor in args or kwargs
            found_t = None
            for a in args:
                if isinstance(a, torch.Tensor) and a.dim() == 3:
                    found_t = a
                    break
            if found_t is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                        found_t = v
                        break
            if found_t is not None:
                captured_embs.append(found_t.detach().cpu().numpy())
            
        b0 = raw_model.icl_predictor.tf_icl.blocks[0]
        hook = b0.register_forward_pre_hook(hook_fn, with_kwargs=True)
        
        try:
            with torch.no_grad():
                raw_model(X_seq, y_seq)
        except Exception:
            pass # Ignore subsequent errors
                
        hook.remove()
        
        if captured_embs:
            emb = captured_embs[0]
            if emb.ndim == 3: emb = emb[0]
            return emb # (seq_len, 512)
        return None

    k = 3
    n_valid = 0
    for i in range(min(n_samples * 3, len(X_test_std))):
        if n_valid >= n_samples:
            break
        xi = X_test_std[i].cpu().numpy()

        # Pick k poison positions (random)
        rng = np.random.default_rng(42 + i)
        poison_pos = rng.choice(len(X_ctx_std), size=k, replace=False)

        # Build poisoned context (near-duplicate in standardised space)
        Xc = X_ctx_std.cpu().numpy()
        Xp = Xc.copy()
        for p in poison_pos:
            Xp[p] = xi + rng.normal(0, 0.01, xi.shape).astype(np.float32)

        # Add test token at end (standard OrionBix sequence format)
        ctx_clean  = np.vstack([Xc, xi.reshape(1,-1)])
        ctx_poison = np.vstack([Xp, xi.reshape(1,-1)])
        q_idx      = len(ctx_clean) - 1   # test token is last

        # Convert to tensors
        t_clean_x = torch.tensor(ctx_clean, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        t_poison_x = torch.tensor(ctx_poison, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        
        # Build dummy y sequence (not strictly needed for Q/K geometry but required by API)
        # Note: the test token label is conventionally masked or set to 0 during inference
        y_seq_p = y_ctx.clone()
        y_seq_p[poison_pos] = 1 - y_seq_p[poison_pos]  # Flip poison labels
        t_clean_y = torch.cat([y_ctx, torch.tensor([0], device=DEVICE)]).unsqueeze(0)
        t_poison_y = torch.cat([y_seq_p, torch.tensor([0], device=DEVICE)]).unsqueeze(0)

        # ORION-BiX
        emb_bix_clean = get_embeddings(raw_bix, t_clean_x, t_clean_y)
        emb_bix_poison = get_embeddings(raw_bix, t_poison_x, t_poison_y)
        
        _, pm_bix_clean  = simulate_attn(WQ_bix, WK_bix, emb_bix_clean[-1],
                                          emb_bix_clean,  poison_pos, nh_bix, dh_bix, scale)
        _, pm_bix_poison = simulate_attn(WQ_bix, WK_bix, emb_bix_poison[-1],
                                          emb_bix_poison, poison_pos, nh_bix, dh_bix, scale)

        # TabICL
        emb_icl_clean = get_embeddings(raw_tabicl, t_clean_x, t_clean_y)
        emb_icl_poison = get_embeddings(raw_tabicl, t_poison_x, t_poison_y)
        
        _, pm_icl_clean  = simulate_attn(WQ_icl, WK_icl, emb_icl_clean[-1],
                                          emb_icl_clean,  poison_pos, nh_icl, dh_icl, scale)
        _, pm_icl_poison = simulate_attn(WQ_icl, WK_icl, emb_icl_poison[-1],
                                          emb_icl_poison, poison_pos, nh_icl, dh_icl, scale)

        results_bix["clean"].append(pm_bix_clean)
        results_bix["poison"].append(pm_bix_poison)
        results_tabicl["clean"].append(pm_icl_clean)
        results_tabicl["poison"].append(pm_icl_poison)
        n_valid += 1

    print(f"\n  Samples: {n_valid}  |  k={k} poison positions  |  uniform baseline={k/len(X_ctx_std):.4f}")
    uniform = k / len(X_ctx_std)

    def summarise(arr, label, model):
        arr = np.array(arr)           # (n, n_heads)
        means = arr.mean(axis=0)      # (n_heads,)
        total = means.sum()
        print(f"\n  {model} — {label} context:")
        print(f"  {'Head':>5s}  {'Attn mass on poison':>22s}  {'vs uniform':>12s}  {'Sim-seeking':>12s}")
        print(f"  {'─'*56}")
        for h, m in enumerate(means):
            vs = m / uniform
            sim = "✓" if (h > 0 and model == "TabICL") or \
                          (model == "ORION-BiX") else "✗ (contrast)"
            print(f"  {h:>5d}  {m:>22.4f}  {vs:>12.2f}x  {sim:>12s}")
        print(f"  {'total':>5s}  {total:>22.4f}  {total/uniform:>12.2f}x")
        return means

    bm_clean  = summarise(results_bix["clean"],     "clean",   "ORION-BiX")
    bm_poison = summarise(results_bix["poison"],    "poisoned","ORION-BiX")
    im_clean  = summarise(results_tabicl["clean"],  "clean",   "TabICL")
    im_poison = summarise(results_tabicl["poison"], "poisoned","TabICL")

    # Lift = how much more attention poison gets under poisoning vs clean
    bix_lift = bm_poison.sum() / max(bm_clean.sum(), 1e-8)
    icl_lift = im_poison.sum() / max(im_clean.sum(), 1e-8)

    print(f"\n  Attention lift on poison positions (poisoned / clean):")
    print(f"    ORION-BiX: {bix_lift:.3f}x")
    print(f"    TabICL:    {icl_lift:.3f}x")

    # H0 contrast contribution in TabICL
    icl_h0_clean  = float(np.array(results_tabicl["clean"])[:, 0].mean())
    icl_h0_poison = float(np.array(results_tabicl["poison"])[:, 0].mean())
    print(f"\n  TabICL H0 (contrast-seeking, pos-frac=0.258):")
    print(f"    Clean context:   {icl_h0_clean:.4f}  ({icl_h0_clean/uniform:.2f}x uniform)")
    print(f"    Poisoned context:{icl_h0_poison:.4f}  ({icl_h0_poison/uniform:.2f}x uniform)")
    h0_cancels = icl_h0_poison < icl_h0_clean * 0.8
    print(f"    → {'H0 DOWN-WEIGHTS poison — contrast cancellation ACTIVE ✓' if h0_cancels else 'H0 does not cancel poison positions'}")

    if icl_lift < bix_lift * 0.5:
        print(f"\n  ✅ HYPOTHESIS 2 SUPPORTED:")
        print(f"     TabICL lifts attention on poison by only {icl_lift:.2f}x vs ORION-BiX {bix_lift:.2f}x")
    else:
        print(f"\n  ❌ HYPOTHESIS 2 NOT CONFIRMED:")
        print(f"     Attention lift similar across models ({icl_lift:.2f}x vs {bix_lift:.2f}x)")

    return {
        "bix_lift": bix_lift, "icl_lift": icl_lift,
        "h0_cancels": h0_cancels,
        "bix_poison_by_head": bm_poison.tolist(),
        "icl_poison_by_head": im_poison.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXP 15C — BLOCK-1 OV DOMINANCE
# ══════════════════════════════════════════════════════════════════════════════

def exp_15c_ov_dominance():
    """
    Structural comparison: OV norm ratio between block 0 and block 1.

    If TabICL block 1 OV norms dwarf block 0, the entry signal from
    blocks.0 is overwritten before it propagates.

    ORION-BiX:  block 1 max OV = 3.06,  block 0 max OV = 3.40  → ratio ≈ 0.90
    TabICL:     block 1 max OV = 40.39, block 0 max OV = 8.04  → ratio ≈ 5.02
    """
    print("\n" + "="*64)
    print("EXP 15c — Block-1 OV Dominance (from Exp 14 weights)")
    print("="*64)

    # Values from Exp 14 output — no recompute needed
    bix_b0_max    = 3.40
    bix_b1_max    = 3.06
    tabicl_b0_max = 8.04
    tabicl_b1_max = 40.39

    bix_ratio    = bix_b1_max    / bix_b0_max
    tabicl_ratio = tabicl_b1_max / tabicl_b0_max

    print(f"\n  {'Model':>12s}  {'Block 0 OV max':>16s}  {'Block 1 OV max':>16s}  {'B1/B0 ratio':>12s}")
    print(f"  {'─'*60}")
    print(f"  {'ORION-BiX':>12s}  {bix_b0_max:>16.2f}  {bix_b1_max:>16.2f}  {bix_ratio:>12.2f}x")
    print(f"  {'TabICL':>12s}  {tabicl_b0_max:>16.2f}  {tabicl_b1_max:>16.2f}  {tabicl_ratio:>12.2f}x")

    print(f"\n  Block-1 OV dominance ratio:")
    print(f"    ORION-BiX: block 1 writes {bix_ratio:.2f}x more than block 0  → signal from b0 propagates")
    print(f"    TabICL:    block 1 writes {tabicl_ratio:.2f}x more than block 0  → b0 signal may be overwritten")

    if tabicl_ratio > bix_ratio * 2:
        print(f"\n  ✅ HYPOTHESIS 3 SUPPORTED:")
        print(f"     TabICL block-1 OV dominance ({tabicl_ratio:.1f}x) is {tabicl_ratio/bix_ratio:.1f}x "
              f"larger than ORION-BiX ({bix_ratio:.1f}x).")
        print(f"     Block 0 entry signal in TabICL is overwritten by a ~{tabicl_b1_max:.0f}-norm "
              f"write at block 1, vs only ~{bix_b1_max:.0f} in ORION-BiX.")
    else:
        print(f"\n  ❌ HYPOTHESIS 3 NOT CONFIRMED: ratios too similar.")

    return {
        "bix_b0": bix_b0_max, "bix_b1": bix_b1_max, "bix_ratio": bix_ratio,
        "tabicl_b0": tabicl_b0_max, "tabicl_b1": tabicl_b1_max, "tabicl_ratio": tabicl_ratio,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize(dist_summary, attn_results, ov_results, out_path):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    sigmas = SIGMAS
    colors = {"raw":"#888888", "bix":"#E84040", "tabicl":"#4C72B0"}

    # ── 15a: Distance curves ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    if dist_summary:
        raw_d    = [dist_summary[s]["raw"]    for s in sigmas]
        bix_d    = [dist_summary[s]["bix"]    for s in sigmas]
        tabicl_d = [dist_summary[s]["tabicl"] for s in sigmas]
        ax.plot(sigmas, raw_d,    "o--", color=colors["raw"],    lw=2, label="Raw input space")
        ax.plot(sigmas, bix_d,    "s-",  color=colors["bix"],    lw=2, label="ORION-BiX internal")
        ax.plot(sigmas, tabicl_d, "^-",  color=colors["tabicl"], lw=2, label="TabICL internal")
        ax.set_xlabel("Sigma (input noise)")
        ax.set_ylabel("L2 distance (mean over samples)")
        ax.set_title("15a: Preprocessing Distance Inflation\n"
                     "If TabICL line >> ORION-BiX line → preprocessing explains resistance",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xscale("log")
        # Annotate sigma=0.01
        if 0.01 in dist_summary:
            for key, color, label in [("bix",colors["bix"],"BiX"),
                                       ("tabicl",colors["tabicl"],"ICL")]:
                v = dist_summary[0.01][key]
                ax.annotate(f"{label}={v:.3f}", (0.01, v),
                            textcoords="offset points", xytext=(8,4),
                            color=color, fontsize=8)
    else:
        ax.text(0.5,0.5,"Distance data\nnot available",ha="center",va="center",transform=ax.transAxes)

    # ── 15a: Inflation ratio bar ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if dist_summary:
        r_bix    = [dist_summary[s]["ratio_bix"]    for s in sigmas]
        r_tabicl = [dist_summary[s]["ratio_tabicl"] for s in sigmas]
        x_pos = np.arange(len(sigmas))
        ax.bar(x_pos - 0.2, r_bix,    0.4, color=colors["bix"],    alpha=0.85, label="ORION-BiX")
        ax.bar(x_pos + 0.2, r_tabicl, 0.4, color=colors["tabicl"], alpha=0.85, label="TabICL")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(s) for s in sigmas], fontsize=8)
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Internal / raw distance ratio")
        ax.set_title("15a: Inflation Ratio\n(higher = preprocessor amplifies noise)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

    # ── 15b: Per-head attention mass ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if attn_results and "bix_poison_by_head" in attn_results:
        bix_h = attn_results["bix_poison_by_head"]
        icl_h = attn_results["icl_poison_by_head"]
        heads = range(len(bix_h))
        uniform = 3 / 256
        ax.bar([h - 0.18 for h in heads], bix_h, 0.36,
               color=colors["bix"],    alpha=0.85, label="ORION-BiX")
        ax.bar([h + 0.18 for h in heads], icl_h, 0.36,
               color=colors["tabicl"], alpha=0.85, label="TabICL")
        ax.axhline(uniform, color="gray", linestyle="--", alpha=0.5, label="Uniform")
        ax.set_xlabel("Head index at blocks.0")
        ax.set_ylabel("Attention mass on poison positions")
        ax.set_title("15b: Per-Head Poison Attention (poisoned ctx)\n"
                     "TabICL H0 suppresses poison? → contrast cancellation",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xticks(list(heads))
    else:
        ax.text(0.5,0.5,"Attention data\nnot available",ha="center",va="center",transform=ax.transAxes)

    # ── 15b: Attention lift ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    if attn_results and "bix_lift" in attn_results:
        lifts  = [attn_results["bix_lift"], attn_results["icl_lift"]]
        labels = ["ORION-BiX", "TabICL"]
        clrs   = [colors["bix"], colors["tabicl"]]
        bars   = ax.bar(labels, lifts, color=clrs, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, lifts):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                    f"{v:.3f}x", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Attention lift (poisoned / clean)")
        ax.set_title("15b: Total Attention Lift on Poison Positions\n"
                     "(how much more attention poison gets under poisoning)",
                     fontweight="bold", fontsize=10)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="No lift")
        ax.legend(fontsize=8)

    # ── 15c: OV norm dominance ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if ov_results:
        cats    = ["Block 0 max OV", "Block 1 max OV", "B1/B0 ratio (×10)"]
        bix_v   = [ov_results["bix_b0"],    ov_results["bix_b1"],    ov_results["bix_ratio"]*10]
        tabicl_v= [ov_results["tabicl_b0"], ov_results["tabicl_b1"], ov_results["tabicl_ratio"]*10]
        x_pos   = np.arange(len(cats))
        ax.bar(x_pos - 0.2, bix_v,    0.4, color=colors["bix"],    alpha=0.85, label="ORION-BiX")
        ax.bar(x_pos + 0.2, tabicl_v, 0.4, color=colors["tabicl"], alpha=0.85, label="TabICL")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel("OV norm")
        ax.set_title("15c: Block-1 OV Dominance\n"
                     "TabICL B1 = 40.39 vs B0 = 8.04 (5x) — signal overwrite?",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Experiment 15: Why Does TabICL Resist SILHOUETTE?\n"
                 "Preprocessing Inflation  ·  Contrast Cancellation  ·  OV Dominance",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_15():
    print("="*64)
    print("EXPERIMENT 15 — Why Does TabICL Resist SILHOUETTE?")
    print(f"Device: {DEVICE}")
    print("="*64)

    # ── Load raw data (no standardisation — preprocessors need raw input) ─
    X_raw, y_raw, feat = load_adult_raw()
    n_ctx = 256

    X_ctx_raw  = X_raw[:n_ctx]
    y_ctx_raw  = y_raw[:n_ctx]
    X_test_raw = X_raw[n_ctx:]

    # Also standardised version for attention simulation
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_raw)
    X_ctx_std  = torch.tensor(X_std[:n_ctx],  dtype=DTYPE, device=DEVICE)
    X_test_std = torch.tensor(X_std[n_ctx:],  dtype=DTYPE, device=DEVICE)
    y_ctx_t    = torch.tensor(y_ctx_raw,       dtype=torch.long, device=DEVICE)

    # ── Load both models ──────────────────────────────────────────────────
    raw_bix,    wrapper_bix    = load_pipeline("orion_bix", X_ctx_raw, y_ctx_raw, feat)
    raw_tabicl, wrapper_tabicl = load_pipeline("tabicl",    X_ctx_raw, y_ctx_raw, feat)

    # ── Extract preprocessors ─────────────────────────────────────────────
    preproc_bix    = get_preprocessor(wrapper_bix)
    preproc_tabicl = get_preprocessor(wrapper_tabicl)

    print(f"\n[PREP] ORION-BiX preprocessor: "
          f"{type(preproc_bix).__name__ if preproc_bix else 'NOT FOUND'}")
    print(f"[PREP] TabICL preprocessor:    "
          f"{type(preproc_tabicl).__name__ if preproc_tabicl else 'NOT FOUND'}")

    # ── Exp 15a: Distance inflation ───────────────────────────────────────
    dist_summary, dist_raw = {}, {}
    if preproc_bix is not None and preproc_tabicl is not None:
        dist_summary, dist_raw = exp_15a_distance_inflation(
            X_test_raw, feat, preproc_bix, preproc_tabicl,
            n_samples=N_SAMPLES
        )
    else:
        print("\n[15a] SKIP — one or both preprocessors not accessible via public API.")
        print("       Run the diagnostic block below to find the correct attribute path:")
        print("       for attr in dir(wrapper_bix.model):")
        print("           v = getattr(wrapper_bix.model, attr, None)")
        print("           if hasattr(v, 'transform'): print(attr, type(v))")

    # ── Exp 15b: Contrast head cancellation ───────────────────────────────
    attn_results = exp_15b_contrast_head_simulation(
        raw_bix, raw_tabicl,
        X_ctx_std, y_ctx_t, X_test_std,
        feat, n_samples=40
    )

    # ── Exp 15c: OV dominance (from Exp 14 weights, no recompute needed) ──
    ov_results = exp_15c_ov_dominance()

    # ── Visualize ─────────────────────────────────────────────────────────
    out_png = f"{OUT}/exp15_tabicl_resistance.png"
    visualize(dist_summary, attn_results, ov_results, out_png)

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPERIMENT 15 — SUMMARY")
    print(f"{'='*64}")

    verdicts = []
    if dist_summary:
        s001 = dist_summary.get(0.01, {})
        if s001.get("inflated"):
            verdicts.append(
                f"H1 CONFIRMED: preprocessing inflation "
                f"(BiX ratio={s001['ratio_bix']:.2f}x, TabICL ratio={s001['ratio_tabicl']:.2f}x)"
            )
        else:
            verdicts.append("H1 NOT CONFIRMED: distance inflation similar across models")

    if attn_results:
        if attn_results.get("icl_lift", 1) < attn_results.get("bix_lift", 1) * 0.5:
            verdicts.append(
                f"H2 CONFIRMED: attention lift TabICL={attn_results['icl_lift']:.3f}x "
                f"vs BiX={attn_results['bix_lift']:.3f}x"
            )
        elif attn_results.get("h0_cancels"):
            verdicts.append("H2 PARTIAL: H0 contrast-seeking head suppresses poison positions")
        else:
            verdicts.append("H2 NOT CONFIRMED: attention patterns similar")

    if ov_results:
        r = ov_results["tabicl_ratio"] / ov_results["bix_ratio"]
        if r > 2:
            verdicts.append(
                f"H3 CONFIRMED: TabICL B1/B0 OV ratio = {ov_results['tabicl_ratio']:.1f}x "
                f"vs BiX {ov_results['bix_ratio']:.1f}x ({r:.1f}x larger)"
            )
        else:
            verdicts.append("H3 NOT CONFIRMED: OV dominance ratio similar")

    print("\n  Verdict per hypothesis:")
    for v in verdicts:
        print(f"    {v}")

    if not verdicts:
        print("  No hypotheses could be tested (preprocessors not accessible).")
        print("  Run the attribute dump diagnostic in 15a to find the preprocessor path.")

    confirmed = [v for v in verdicts if "CONFIRMED" in v and "NOT" not in v]
    print(f"\n  {'─'*50}")
    if confirmed:
        print(f"  PRIMARY EXPLANATION: {confirmed[0]}")
        print(f"  This explains TabICL's ASR collapse and should be added to §5.X")
    else:
        print("  No single hypothesis confirmed. TabICL resistance is multiply determined.")
        print("  Report as open question in paper (§5.X already drafted for this case).")

    return {
        "dist_summary":  dist_summary,
        "attn_results":  attn_results,
        "ov_results":    ov_results,
        "verdicts":      verdicts,
    }


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    results = run_experiment_15()