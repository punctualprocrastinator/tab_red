"""
Experiment 11 — Weight-Based Decomposition of the Poisoning Circuit

Three analyses that go deeper than activation patching:
  11a. OV Circuit Decomposition — What does each head WRITE?
  11b. QK Circuit Analysis — What does each head ATTEND TO?
  11c. Direct Logit Attribution — Per-head contribution to final prediction

Requires: exp_0 baseline cells already run (or standalone execution below).
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

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_adult_income():
    print("[0A] Loading Adult Income dataset...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame
    target_col = "income" if "income" in df.columns else df.columns[-1]
    y_raw = df[target_col]
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    X_df = df.drop(columns=[target_col])
    for c in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_df.values.astype(np.float32))
    feat_names = list(X_df.columns)
    n_ctx, n_test = 256, 512
    X_ctx  = torch.tensor(X_all[:n_ctx],             dtype=DTYPE, device=DEVICE)
    y_ctx  = torch.tensor(y_all[:n_ctx],             dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_all[n_ctx:n_ctx+n_test], dtype=DTYPE, device=DEVICE)
    y_test = torch.tensor(y_all[n_ctx:n_ctx+n_test], dtype=torch.long, device=DEVICE)
    print(f"[0A] Context : {X_ctx.shape}  |  Test: {X_test.shape}")
    pos = int(y_test.sum().item())
    print(f"[0A] Label balance — positive: {pos}/{len(y_test)} ({pos/len(y_test)*100:.1f}%)")
    return X_ctx, y_ctx, X_test, y_test, feat_names, scaler


def load_model(model_name="orion-bix"):
    print(f"\n[0B] Loading {model_name} via TabTune...")
    from tabtune import TabularPipeline
    name_map = {"orion-bix": "OrionBix", "orion-msp": "OrionMSP"}
    wrapper = TabularPipeline(
        model_name=name_map[model_name],
        task_type="classification",
        tuning_strategy="inference",
        tuning_params={"device": "cuda" if DEVICE.type == "cuda" else "cpu"}
    )
    print(f"[0B] ✅ TabularPipeline({name_map[model_name]}) ready")
    return wrapper


def extract_raw_module(wrapper) -> nn.Module:
    """
    Confirmed working path: pipeline.model.model_  (trailing underscore).
    Checks all known variants in priority order.
    """
    # ── Priority 1: confirmed working path ───────────────────────────────────
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "model_"):
        raw = wrapper.model.model_
        print(f"[0C] Raw module at pipeline.model.model_ -> "
              f"{type(raw).__name__} on {next(raw.parameters()).device}")
        raw.eval().to(DEVICE)
        return raw

    # ── Priority 2: pipeline.model.model (no underscore) ─────────────────────
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "model"):
        raw = wrapper.model.model
        print(f"[0C] Raw module at pipeline.model.model -> {type(raw).__name__}")
        raw.eval().to(DEVICE)
        return raw

    # ── Priority 3: walk the wrapper tree ────────────────────────────────────
    for attr in ["model_", "net", "module", "estimator_", "base_estimator_"]:
        obj = wrapper
        for step in ["model", attr]:
            obj = getattr(obj, step, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.Module):
            print(f"[0C] Raw module via .model.{attr} -> {type(obj).__name__}")
            obj.eval().to(DEVICE)
            return obj

    # ── Priority 4: direct attributes on wrapper ─────────────────────────────
    for attr in ["model_", "model", "net_", "net"]:
        obj = getattr(wrapper, attr, None)
        if obj is not None and isinstance(obj, nn.Module):
            print(f"[0C] Raw module at pipeline.{attr} -> {type(obj).__name__}")
            obj.eval().to(DEVICE)
            return obj

    # ── Diagnostic dump ───────────────────────────────────────────────────────
    print("[0C] ERROR: Could not find raw model. Wrapper attributes:")
    for a in dir(wrapper):
        if not a.startswith("__"):
            v = getattr(wrapper, a, None)
            print(f"  .{a}: {type(v).__name__}")
    raise RuntimeError(
        "Could not extract raw nn.Module from TabularPipeline.\n"
        "Known working path is pipeline.model.model_ — check TabTune version."
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (matching confirmed-working patterns from exp_8_9_10)
# ══════════════════════════════════════════════════════════════════════════════

def _build_input(raw_model, X_ctx, y_ctx, x_test_i):
    """
    Confirmed working for OrionBix/OrionMSP:
      stack [X_ctx | x_test] → (1, ctx_len+1, n_features)
      labels: y_ctx unsqueezed → (1, ctx_len)
    """
    if type(raw_model).__name__ in ["OrionBix", "OrionMSP"]:
        X_seq = torch.cat([X_ctx, x_test_i.unsqueeze(0)], dim=0).unsqueeze(0)
        return (X_seq, y_ctx.unsqueeze(0))
    # Generic fallback
    return (X_ctx, y_ctx, x_test_i)


def _flatten_logits(out):
    t = out[0] if isinstance(out, tuple) else out
    if isinstance(t, tuple): t = t[0]
    t = t.cpu().float()
    while t.dim() > 1:
        t = t.squeeze(0)
    return t


def _predict(raw_model, X_ctx, y_ctx, x_test_i):
    raw_model.eval()
    with torch.no_grad():
        out = raw_model(*_build_input(raw_model, X_ctx, y_ctx, x_test_i))
    return int(_flatten_logits(out).argmax().item())


def unwrap(proxy):
    return proxy.value if hasattr(proxy, "value") else proxy


def nav(obj, path):
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ── Attacks ──────────────────────────────────────────────────────────────────

def attack_near_dup(X_ctx, y_ctx, x_test, true_label, k, rng, sigma=0.01):
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    positions = rng.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)
    x_np = x_test.cpu().numpy()
    for pos in positions:
        noise = rng.normal(0, sigma, size=x_np.shape).astype(np.float32)
        X_p[pos] = torch.tensor(x_np + noise, dtype=X_ctx.dtype, device=X_ctx.device)
        y_p[pos] = 1 - true_label
    return X_p, y_p


def attack_pool_only(X_ctx, y_ctx, x_test, true_label, k, rng):
    X_p, y_p = X_ctx.clone(), y_ctx.clone()
    x_np = x_test.cpu().numpy()
    X_np, y_np = X_ctx.cpu().numpy(), y_ctx.cpu().numpy()
    same = np.where(y_np == true_label)[0]
    if len(same) == 0:
        return X_p, y_p
    nearest = same[np.argsort(np.linalg.norm(X_np[same] - x_np, axis=1))[:k]]
    for pos in nearest:
        y_p[pos] = 1 - y_p[pos]
    return X_p, y_p


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_weights(raw_model):
    """
    Auto-discover all attention weight matrices and the decoder.
    Returns a dict with all the weight info needed for 11a/11b/11c.
    """
    # ── Find ICL block paths ──────────────────────────────────────────────────
    block_paths = []
    seen = set()
    for name, _ in raw_model.named_modules():
        if "icl_predictor.tf_icl.blocks." in name:
            parts = name.split(".")
            # want exactly 'icl_predictor.tf_icl.blocks.N'
            try:
                idx = parts.index("blocks")
                block_name = ".".join(parts[:idx+2])
                if parts[idx+1].isdigit() and block_name not in seen:
                    seen.add(block_name)
                    block_paths.append(block_name)
            except ValueError:
                continue
    block_paths = sorted(block_paths, key=lambda x: int(x.split(".")[-1]))

    if not block_paths:
        print("[WEIGHT] No ICL blocks found — scanning all named_modules...")
        for name, _ in raw_model.named_modules():
            print(f"  {name}")
        return None

    print(f"[WEIGHT] {len(block_paths)} ICL blocks: {block_paths[0]} ... {block_paths[-1]}")

    # ── Inspect blocks.0 structure ───────────────────────────────────────────
    b0_path = block_paths[0]
    b0 = nav(raw_model, b0_path)
    print(f"\n[WEIGHT] Structure of {b0_path}:")
    for name, mod in b0.named_modules():
        if name:
            params = [(pn, list(p.shape))
                      for pn, p in mod.named_parameters(recurse=False)]
            print(f"  {name:45s}  {type(mod).__name__}")
            for pn, ps in params:
                print(f"    .{pn:42s}  {ps}")

    # ── Discover decoder ──────────────────────────────────────────────────────
    decoder_module = None
    decoder_path   = None
    for name, mod in raw_model.named_modules():
        if name == "icl_predictor.decoder":
            decoder_module = mod
            decoder_path   = name
            break
    if decoder_module is None:
        for name, mod in raw_model.named_modules():
            if "decoder" in name and hasattr(mod, "forward"):
                decoder_module = mod
                decoder_path   = name
                break

    print(f"\n[WEIGHT] Decoder: {decoder_path}  ({type(decoder_module).__name__ if decoder_module else 'NOT FOUND'})")

    # ── Collect weight matrices per block ─────────────────────────────────────
    d_model, n_heads, weight_type = None, None, "unknown"
    block_weights = {}

    for bp in block_paths:
        mod = nav(raw_model, bp)
        if mod is None:
            continue
        w = {}

        for pname, param in mod.named_parameters():
            pl = pname.lower()
            p  = param.detach()

            if "in_proj_weight" in pl:
                # Fused QKV: shape (3*d, d)
                d = p.shape[1]
                w["W_Q"] = p[:d].clone()
                w["W_K"] = p[d:2*d].clone()
                w["W_V"] = p[2*d:].clone()
                d_model, weight_type = d, "fused_qkv"

            elif "in_proj_bias" in pl:
                d = p.shape[0] // 3
                w["b_Q"] = p[:d].clone()
                w["b_K"] = p[d:2*d].clone()
                w["b_V"] = p[2*d:].clone()

            elif any(k in pl for k in ("q_proj", "wq", "w_q", "query")) and "weight" in pl:
                w["W_Q"] = p.clone()
                d_model, weight_type = p.shape[1], "separate_qkv"
            elif any(k in pl for k in ("k_proj", "wk", "w_k", "key")) and "weight" in pl:
                w["W_K"] = p.clone()
            elif any(k in pl for k in ("v_proj", "wv", "w_v", "value")) and "weight" in pl:
                w["W_V"] = p.clone()

            elif any(k in pl for k in ("out_proj", "o_proj", "wo", "w_o")) and "weight" in pl:
                w["W_O"] = p.clone()
            elif any(k in pl for k in ("out_proj", "o_proj")) and "bias" in pl:
                w["b_O"] = p.clone()

        # Fallback: find MHA module directly
        if "W_Q" not in w:
            for mname, mmod in mod.named_modules():
                if isinstance(mmod, nn.MultiheadAttention):
                    n_heads = mmod.num_heads
                    d_model = mmod.embed_dim
                    ipw = mmod.in_proj_weight
                    if ipw is not None:
                        d = d_model
                        w["W_Q"] = ipw[:d].detach().clone()
                        w["W_K"] = ipw[d:2*d].detach().clone()
                        w["W_V"] = ipw[2*d:].detach().clone()
                        weight_type = "mha_module"
                    opw = mmod.out_proj.weight
                    if opw is not None:
                        w["W_O"] = opw.detach().clone()
                    break

        block_weights[bp] = w

    # ── Infer n_heads ─────────────────────────────────────────────────────────
    for name, mod in raw_model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            n_heads = mod.num_heads
            if d_model is None:
                d_model = mod.embed_dim
            break
    if n_heads is None:
        for name, mod in raw_model.named_modules():
            for attr in ("num_heads", "n_heads", "nhead", "num_attention_heads"):
                v = getattr(mod, attr, None)
                if v is not None and isinstance(v, int):
                    n_heads = v
                    break

    d_head = (d_model // n_heads) if (d_model and n_heads) else None

    # ── Summary ───────────────────────────────────────────────────────────────
    w0 = block_weights.get(block_paths[0], {})
    found = [k for k in ("W_Q", "W_K", "W_V", "W_O") if k in w0]
    print(f"\n[WEIGHT] Discovered:")
    print(f"  d_model={d_model}  n_heads={n_heads}  d_head={d_head}")
    print(f"  type={weight_type}  matrices={found}")
    for k in found:
        print(f"    {k}: {list(w0[k].shape)}")

    return {
        "blocks":         block_paths,
        "d_model":        d_model,
        "n_heads":        n_heads,
        "d_head":         d_head,
        "weight_type":    weight_type,
        "block_weights":  block_weights,
        "decoder_module": decoder_module,
        "decoder_path":   decoder_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11a: OV CIRCUIT DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def exp_11a_ov_circuit(weight_info):
    """
    W_OV_h = W_O_h @ W_V_h
    SVD gives: dominant writing directions and their strength.
    Project top direction through decoder to get logit impact.
    """
    print("\n" + "=" * 64)
    print("EXP 11a — OV Circuit Decomposition")
    print("=" * 64)

    d_model        = weight_info["d_model"]
    n_heads        = weight_info["n_heads"]
    d_head         = weight_info["d_head"]
    decoder_module = weight_info["decoder_module"]

    if not (d_model and n_heads and d_head):
        print("  [SKIP] Missing d_model / n_heads / d_head.")
        return {}

    results = {}

    for bp in weight_info["blocks"]:
        w = weight_info["block_weights"].get(bp, {})
        if "W_V" not in w or "W_O" not in w:
            continue

        W_V = w["W_V"].float().cpu()   # (d_model, d_model)  — (3d_model→d_model) sliced
        W_O = w["W_O"].float().cpu()   # (d_model, d_model)

        try:
            # Per-head slicing
            # W_V rows are d_model output, cols are d_model input
            # Each head uses d_head rows of W_V and d_head cols of W_O
            W_V_h = W_V.reshape(n_heads, d_head, d_model)   # (H, d_head, d_model)
            # W_O: (d_model, d_model), each head contributes d_head columns
            W_O_h = W_O.reshape(d_model, n_heads, d_head).permute(1, 0, 2)  # (H, d_model, d_head)

            ov_norms, top_svs, logit_impacts = [], [], []

            for h in range(n_heads):
                # W_OV_h: (d_model, d_model)
                W_OV_h = W_O_h[h] @ W_V_h[h]    # (d_model, d_head) @ (d_head, d_model)

                ov_norms.append(float(W_OV_h.norm()))

                U, S, Vh = torch.linalg.svd(W_OV_h, full_matrices=False)
                top_svs.append(float(S[0]))

                # Project top writing direction through decoder
                if decoder_module is not None:
                    top_dir = U[:, 0].unsqueeze(0)
                    try:
                        with torch.no_grad():
                            dev = next(decoder_module.parameters()).device
                            logits = decoder_module(top_dir.to(dev))
                            logit_impacts.append(logits.squeeze().cpu().numpy())
                    except Exception:
                        logit_impacts.append(None)
                else:
                    logit_impacts.append(None)

            results[bp] = {
                "ov_norms":           ov_norms,
                "top_singular_vals":  top_svs,
                "logit_impacts":      logit_impacts,
            }

            bn = bp.split(".")[-1]
            print(f"  Block {bn}:  OV norms {[f'{x:.2f}' for x in ov_norms]}"
                  f"  top-σ {[f'{x:.2f}' for x in top_svs]}")

        except Exception as e:
            print(f"  [WARN] block {bp.split('.')[-1]} failed: {e}")

    if results:
        all_heads = [(norm, f"block {bp.split('.')[-1]} head {h}")
                     for bp, r in results.items()
                     for h, norm in enumerate(r["ov_norms"])]
        all_heads.sort(reverse=True)
        print(f"\n  Top-5 OV-norm heads (dominant writers):")
        for norm, desc in all_heads[:5]:
            print(f"    {desc}: {norm:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 11b: QK CIRCUIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def exp_11b_qk_circuit(weight_info):
    """
    W_QK_h = W_Q_h @ W_K_h^T
    Eigendecomposition reveals similarity-seeking vs contrast-seeking heads.
    Positive eigenvalues = attend to similar features; negative = attend to different.
    """
    print("\n" + "=" * 64)
    print("EXP 11b — QK Circuit Analysis")
    print("=" * 64)

    d_model = weight_info["d_model"]
    n_heads = weight_info["n_heads"]
    d_head  = weight_info["d_head"]

    if not (d_model and n_heads and d_head):
        print("  [SKIP] Missing d_model / n_heads / d_head.")
        return {}

    results = {}

    for bp in weight_info["blocks"]:
        w = weight_info["block_weights"].get(bp, {})
        if "W_Q" not in w or "W_K" not in w:
            continue

        W_Q = w["W_Q"].float().cpu()   # (d_model, d_model)
        W_K = w["W_K"].float().cpu()

        try:
            W_Q_h = W_Q.reshape(n_heads, d_head, d_model)  # (H, d_head, d_model)
            W_K_h = W_K.reshape(n_heads, d_head, d_model)

            qk_norms, eigvals_all, pos_fracs = [], [], []

            for h in range(n_heads):
                # W_QK_h = W_Q_h @ W_K_h^T → (d_head, d_head)
                W_QK_h = W_Q_h[h] @ W_K_h[h].T
                qk_norms.append(float(W_QK_h.norm()))

                # Symmetrize for eigvalsh (real eigenvalues)
                W_QK_sym = 0.5 * (W_QK_h + W_QK_h.T)
                eigvals = torch.linalg.eigvalsh(W_QK_sym).cpu().numpy()
                eigvals_all.append(eigvals)
                pos_fracs.append(float((eigvals > 0).mean()))

            results[bp] = {
                "qk_norms":  qk_norms,
                "eigvals":   eigvals_all,
                "pos_fracs": pos_fracs,
            }

            bn = bp.split(".")[-1]
            head_summaries = [f"H{h}: {pf:.2f}" for h, pf in enumerate(pos_fracs)]
            print(f"  Block {bn}  pos-frac {head_summaries}")

        except Exception as e:
            print(f"  [WARN] block {bp.split('.')[-1]} QK failed: {e}")

    # Key diagnostic: blocks.0 is the attack entry point.
    # If blocks.0 heads have high positive eigenvalue fraction, they are
    # "similarity-seeking" — exactly the retrieval mechanism the attack exploits.
    b0 = weight_info["blocks"][0]
    if b0 in results:
        pf = results[b0]["pos_fracs"]
        print(f"\n  blocks.0 positive eigenvalue fractions: {[f'{x:.3f}' for x in pf]}")
        dominant_sim = [h for h, f in enumerate(pf) if f > 0.6]
        print(f"  Similarity-seeking heads (>0.6 pos): {dominant_sim}")
        print(f"  → These are the heads the attack exploits via near-duplicate retrieval")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 11c: DIRECT LOGIT ATTRIBUTION (DLA)
# ══════════════════════════════════════════════════════════════════════════════

def exp_11c_direct_logit_attribution(nn_model, raw_model, weight_info,
                                      X_ctx, y_ctx, X_test, y_test,
                                      n_samples=30, k=3):
    """
    For each block, project its residual stream output through the decoder.
    DLA(block_ℓ) = decoder(block_ℓ_output)[true] - decoder(block_ℓ_output)[false]

    Positive = this block supports the correct prediction.
    Flip contribution = clean_DLA - poisoned_DLA = how much this block
    is responsible for the prediction change under poisoning.

    Convergence check with Exp 8b:
    If DLA flip is large at blocks 0-1 → consistent with 61% behavioral restoration.
    If DLA flip is distributed across 0-6 → consistent with 100% cumulative restoration.
    """
    print("\n" + "=" * 64)
    print("EXP 11c — Direct Logit Attribution")
    print("=" * 64)

    if nn_model is None:
        print("  [SKIP] NNsight required.")
        return {}

    decoder_module = weight_info["decoder_module"]
    if decoder_module is None:
        print("  [SKIP] Decoder module not found.")
        return {}

    block_paths = weight_info["blocks"]
    n_layers    = len(block_paths)
    n_eval      = min(n_samples * 3, len(X_test))

    clean_dla    = [[] for _ in range(n_layers)]
    poison_d_dla = [[] for _ in range(n_layers)]
    poison_g_dla = [[] for _ in range(n_layers)]
    n_collected  = 0

    def get_block_outputs(X_c, y_c, xi):
        """Single forward pass saving test-token output of every block."""
        inp    = _build_input(raw_model, X_c, y_c, xi)
        saved  = {}
        try:
            with nn_model.trace(*inp):
                for j, bp in enumerate(block_paths):
                    sub = nav(nn_model, bp)
                    if sub is not None:
                        try:
                            saved[j] = sub.output[0].save()
                        except Exception:
                            saved[j] = sub.output.save()
        except Exception as e:
            if "ExitTracing" in type(e).__name__:
                raise
            return None

        result = {}
        for j, proxy in saved.items():
            act = unwrap(proxy)
            if isinstance(act, tuple): act = act[0]
            act = act.detach().cpu().float()
            while act.dim() > 2:
                act = act.squeeze(0)
            result[j] = act[-1].numpy()   # test token position
        return result

    def dla_score(vec, true_label):
        """Project block output through decoder → logit(true) - logit(false)."""
        try:
            with torch.no_grad():
                t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                t = t.to(next(decoder_module.parameters()).device)
                logits = decoder_module(t).squeeze().cpu()
            c0, c1 = int(logits[0]), int(logits[1])
            tc = min(true_label, len(logits) - 1)
            fc = 1 - true_label if true_label <= 1 else 0
            return float(logits[tc]) - float(logits[fc])
        except Exception:
            return float("nan")

    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue

        rng = np.random.default_rng(42 + i)
        X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)
        if _predict(raw_model, X_d, y_d, xi) == yi:
            continue   # attack didn't land

        rng_g = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)

        acts_c = get_block_outputs(X_ctx, y_ctx, xi)
        acts_d = get_block_outputs(X_d, y_d, xi)
        acts_g = get_block_outputs(X_g, y_g, xi)

        if acts_c is None or acts_d is None or acts_g is None:
            continue
        if len(acts_c) < n_layers:
            continue

        for j in range(n_layers):
            if j not in acts_c:
                continue
            sc = dla_score(acts_c[j], yi)
            sd = dla_score(acts_d[j], yi)
            sg = dla_score(acts_g[j], yi)
            if not np.isnan(sc): clean_dla[j].append(sc)
            if not np.isnan(sd): poison_d_dla[j].append(sd)
            if not np.isnan(sg): poison_g_dla[j].append(sg)

        n_collected += 1
        if n_collected % 10 == 0:
            print(f"  [{n_collected}] samples collected")

    print(f"  Total: {n_collected} samples")
    if n_collected == 0:
        print("  [SKIP] No valid samples collected.")
        return {}

    clean_m    = [float(np.mean(v)) if v else float("nan") for v in clean_dla]
    poison_d_m = [float(np.mean(v)) if v else float("nan") for v in poison_d_dla]
    poison_g_m = [float(np.mean(v)) if v else float("nan") for v in poison_g_dla]
    flip_d     = [c - d if not (np.isnan(c) or np.isnan(d)) else float("nan")
                  for c, d in zip(clean_m, poison_d_m)]
    flip_g     = [c - g if not (np.isnan(c) or np.isnan(g)) else float("nan")
                  for c, g in zip(clean_m, poison_g_m)]

    print(f"\n  {'Block':>6s}  {'Clean DLA':>10s}  {'AtkD DLA':>10s}  "
          f"{'AtkG DLA':>10s}  {'Flip(D)':>10s}  {'Flip(G)':>10s}")
    print(f"  {'─'*60}")
    for j in range(n_layers):
        fmt = lambda v: f"{v:10.3f}" if not np.isnan(v) else "       n/a"
        print(f"  {j:>6d} {fmt(clean_m[j])} {fmt(poison_d_m[j])} "
              f"{fmt(poison_g_m[j])} {fmt(flip_d[j])} {fmt(flip_g[j])}")

    # Top flip blocks
    valid_flip_d = [(j, abs(v)) for j, v in enumerate(flip_d) if not np.isnan(v)]
    valid_flip_d.sort(key=lambda x: x[1], reverse=True)
    top_d = valid_flip_d[:3]
    print(f"\n  Top-3 flip blocks (Attack D): {[(j, f'{flip_d[j]:.3f}') for j, _ in top_d]}")

    # Convergence check against Exp 8b
    cumul_flip_d = np.nancumsum([abs(v) for v in flip_d])
    total_flip_d = cumul_flip_d[-1] if len(cumul_flip_d) > 0 else 0
    if total_flip_d > 0:
        print(f"\n  Cumulative DLA flip fraction (Attack D):")
        for j in range(min(8, n_layers)):
            frac = cumul_flip_d[j] / total_flip_d
            bar  = "█" * int(frac * 20)
            print(f"    Blocks 0-{j}: {frac*100:5.1f}%  {bar}")
        print(f"  → Compare with Exp 8b: blocks 0-1 = 61%, blocks 0-6 = 100%")

    return {
        "clean_dla":    clean_m,
        "poison_d_dla": poison_d_m,
        "poison_g_dla": poison_g_m,
        "flip_d":       flip_d,
        "flip_g":       flip_g,
        "n_collected":  n_collected,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_exp11(ov_results, qk_results, dla_results, weight_info, out_path):
    n_blocks = len(weight_info["blocks"])
    n_heads  = weight_info["n_heads"] or 4

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors = {"clean": "#4C72B0", "atkd": "#E84040", "atkg": "#55A868"}

    # ── OV norms heatmap ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if ov_results:
        mat = np.array([ov_results[bp]["ov_norms"]
                        if bp in ov_results else [0]*n_heads
                        for bp in weight_info["blocks"]])
        im = ax.imshow(mat.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("ICL Block"); ax.set_ylabel("Head")
        ax.set_xticks(range(n_blocks)); ax.set_xticklabels(range(n_blocks), fontsize=8)
        ax.set_yticks(range(n_heads))
        ax.set_title("11a: OV Norms\n(W_V · W_O per head)", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "OV data\nnot available", ha="center", va="center", transform=ax.transAxes)

    # ── Top singular values heatmap ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if ov_results:
        mat_sv = np.array([ov_results[bp]["top_singular_vals"]
                           if bp in ov_results else [0]*n_heads
                           for bp in weight_info["blocks"]])
        im = ax.imshow(mat_sv.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("ICL Block"); ax.set_ylabel("Head")
        ax.set_xticks(range(n_blocks)); ax.set_xticklabels(range(n_blocks), fontsize=8)
        ax.set_yticks(range(n_heads))
        ax.set_title("11a: Top σ of W_OV\n(dominant writing strength)", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "OV data\nnot available", ha="center", va="center", transform=ax.transAxes)

    # ── QK positive eigenvalue fraction ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if qk_results:
        labels_qk, pfracs = [], []
        for bp in list(qk_results.keys())[:4]:
            bn = bp.split(".")[-1]
            for h, pf in enumerate(qk_results[bp]["pos_fracs"]):
                labels_qk.append(f"B{bn}H{h}")
                pfracs.append(pf)
        bar_colors = ["#E84040" if pf > 0.6 else "#4C72B0" for pf in pfracs]
        ax.barh(range(len(labels_qk)), pfracs, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(labels_qk)))
        ax.set_yticklabels(labels_qk, fontsize=7)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
        ax.axvline(0.6, color="red",  linestyle=":",  alpha=0.4, label="60% (sim-seeking)")
        ax.set_xlabel("Fraction positive eigenvalues")
        ax.set_title("11b: QK Eigenvalue Polarity\n(red>60% = similarity-seeking heads)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "QK data\nnot available", ha="center", va="center", transform=ax.transAxes)

    # ── DLA per block (grouped bars) ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    if dla_results:
        layers = np.arange(len(dla_results["clean_dla"]))
        w = 0.25
        ax.bar(layers - w, dla_results["clean_dla"],    width=w, color=colors["clean"], alpha=0.85, label="Clean")
        ax.bar(layers,     dla_results["poison_d_dla"], width=w, color=colors["atkd"],  alpha=0.85, label="Attack D")
        ax.bar(layers + w, dla_results["poison_g_dla"], width=w, color=colors["atkg"],  alpha=0.85, label="Attack G")
        ax.axhline(0, color="gray", alpha=0.3)
        ax.set_xlabel("ICL Block"); ax.set_ylabel("DLA: logit(true) − logit(false)")
        ax.set_title("11c: Direct Logit Attribution per Block\n"
                     "(positive = supports correct prediction)", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.set_xticks(layers)
    else:
        ax.text(0.5, 0.5, "DLA data not available", ha="center", va="center", transform=ax.transAxes)

    # ── Flip contribution + cumulative ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if dla_results:
        fd = np.array([v if not np.isnan(v) else 0 for v in dla_results["flip_d"]])
        fg = np.array([v if not np.isnan(v) else 0 for v in dla_results["flip_g"]])
        layers = np.arange(len(fd))
        ax.bar(layers - 0.18, fd, width=0.36, color=colors["atkd"], alpha=0.8, label="Flip D")
        ax.bar(layers + 0.18, fg, width=0.36, color=colors["atkg"], alpha=0.8, label="Flip G")
        ax.axhline(0, color="gray", alpha=0.3)

        # Overlay cumulative line
        ax2 = ax.twinx()
        cumul = np.nancumsum(np.abs(fd))
        total = cumul[-1] if cumul[-1] > 0 else 1
        ax2.plot(layers, cumul / total * 100, "k--", linewidth=1.5,
                 marker=".", markersize=4, label="Cumulative %")
        ax2.axhline(61, color="gray", linestyle=":", alpha=0.5)   # Exp 8b 61% marker
        ax2.axhline(100, color="gray", linestyle=":", alpha=0.3)
        ax2.set_ylabel("Cumulative flip % (Attack D)", fontsize=8)
        ax2.set_ylim(0, 110)

        ax.set_xlabel("ICL Block")
        ax.set_ylabel("Flip contribution")
        ax.set_title("11c: Per-Block Flip + Cumulative\n(dashed=61% & 100% Exp 8b reference)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="lower right")
        ax.set_xticks(layers)
    else:
        ax.text(0.5, 0.5, "Flip data\nnot available", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Experiment 11: Weight-Based Decomposition of the Poisoning Circuit\n"
                 "OV Circuits  ·  QK Circuits  ·  Direct Logit Attribution",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_11(n_samples=30, k=3):
    print("=" * 64)
    print("EXPERIMENT 11 — Weight-Based Decomposition")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper  = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    nn_model = NNsight(raw) if NNSIGHT_AVAILABLE else None

    weight_info = discover_weights(raw)
    if weight_info is None:
        print("[ERROR] Weight discovery failed.")
        return {}

    w0   = weight_info["block_weights"].get(weight_info["blocks"][0], {})
    has_qkvo = all(k in w0 for k in ("W_Q", "W_K", "W_V", "W_O"))

    ov_results  = exp_11a_ov_circuit(weight_info)    if has_qkvo else {}
    qk_results  = exp_11b_qk_circuit(weight_info)    if has_qkvo else {}
    dla_results = exp_11c_direct_logit_attribution(
        nn_model, raw, weight_info,
        X_ctx, y_ctx, X_test, y_test,
        n_samples=n_samples, k=k
    ) if nn_model is not None else {}

    out_png = f"{OUT}/exp11_weight_decomposition.png"
    visualize_exp11(ov_results, qk_results, dla_results, weight_info, out_png)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPERIMENT 11 — SUMMARY")
    print(f"{'='*64}")
    print(f"  d_model={weight_info['d_model']}  "
          f"n_heads={weight_info['n_heads']}  d_head={weight_info['d_head']}")
    print(f"  Weights found: {has_qkvo}  (type={weight_info['weight_type']})")

    if ov_results:
        all_norms = sorted(
            [(bp.split(".")[-1], h, n)
             for bp, r in ov_results.items()
             for h, n in enumerate(r["ov_norms"])],
            key=lambda x: x[2], reverse=True
        )
        print(f"\n  Top OV writers:")
        for bn, h, n in all_norms[:5]:
            print(f"    Block {bn} Head {h}: norm={n:.3f}")

    if qk_results:
        b0 = weight_info["blocks"][0]
        if b0 in qk_results:
            pf = qk_results[b0]["pos_fracs"]
            sim_heads = [h for h, f in enumerate(pf) if f > 0.6]
            print(f"\n  blocks.0 similarity-seeking heads: {sim_heads}")
            print(f"    (These are the heads the near-dup attack exploits)")

    if dla_results:
        fd = dla_results["flip_d"]
        valid = [(j, abs(v)) for j, v in enumerate(fd) if not np.isnan(v)]
        valid.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Top flip blocks (Attack D DLA): {[(j, f'{fd[j]:.3f}') for j, _ in valid[:3]]}")

        # Key convergence check
        cumul = np.nancumsum([abs(v) for v in fd])
        total = cumul[-1] if cumul[-1] > 0 else 1
        b01_frac = cumul[1] / total if len(cumul) > 1 else 0
        b06_frac = cumul[6] / total if len(cumul) > 6 else 0
        print(f"\n  DLA cumulative flip (convergence with Exp 8b):")
        print(f"    Blocks 0-1: {b01_frac*100:.1f}%  (Exp 8b behavioral: 61%)")
        print(f"    Blocks 0-6: {b06_frac*100:.1f}%  (Exp 8b behavioral: 100%)")
        agreement = abs(b01_frac - 0.61) < 0.15
        print(f"    → {'CONVERGENT ✓' if agreement else 'DIVERGENT — structural vs behavioral differ'}")

    print(f"\n{'='*64}")

    return {
        "weight_info": weight_info,
        "ov_results":  ov_results,
        "qk_results":  qk_results,
        "dla_results": dla_results,
    }


if __name__ == "__main__":
    results = run_experiment_11(n_samples=30, k=3)