# ============================================================
# EXPERIMENTS 8, 9, 10 — Mechanistic Interpretability Suite
# Colab-compatible: assumes exp_0 cells already executed.
#
# Exp 8: Patch-through causal chain
#   Does patching blocks.0 also restore the layer-8 commitment?
#   Establishes blocks.0 → layer 8 as a single causal pathway.
#
# Exp 9: Linear probing across layers
#   Does probe accuracy track the logit-lens trajectory?
#   Validates layer-8 commitment as a residual-stream phenomenon,
#   not a decoder artifact.
#
# Exp 10: Attention weight extraction at blocks.0
#   Where does the test token attend under clean vs poisoned?
#   Visualises the retrieval hijack directly.
#
# All output saved to /mnt/user-data/outputs/
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as SkScaler

try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False
    print("[WARN] NNsight not available — Exps 8 and 9 will be skipped.")

if "DEVICE" not in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "DTYPE" not in globals():
    DTYPE = torch.float32

OUT = "/mnt/user-data/outputs"


# ── Shared helpers ─────────────────────────────────────────────────────────────

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

def unwrap(proxy):
    return proxy.value if hasattr(proxy, "value") else proxy

def nav(nn_model, path):
    obj = nn_model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ── Attacks ────────────────────────────────────────────────────────────────────

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


# ── Block discovery ────────────────────────────────────────────────────────────

def discover_blocks(raw_model):
    """Return sorted list of ICL block paths and the decoder module."""
    block_paths, decoder_module, decoder_path = [], None, None
    for name, mod in raw_model.named_modules():
        if "icl_predictor.tf_icl.blocks." in name:
            parts = name.split(".")
            if parts[-1].isdigit():
                block_paths.append(name)
    block_paths = sorted(block_paths, key=lambda x: int(x.split(".")[-1]))

    for name, mod in raw_model.named_modules():
        if name.endswith(".decoder") or name == "decoder":
            if hasattr(mod, "forward"):
                decoder_path, decoder_module = name, mod
                break
    if decoder_module is None:
        for name, mod in raw_model.named_modules():
            if "predictor" in name and "decoder" in name:
                if isinstance(mod, nn.Sequential) or hasattr(mod, "__len__"):
                    decoder_path, decoder_module = name, mod
                    break

    print(f"  [STRUCT] {len(block_paths)} ICL blocks: "
          f"{block_paths[0]} ... {block_paths[-1]}")
    if decoder_module:
        print(f"  [STRUCT] Decoder: {decoder_path}")
    return block_paths, decoder_path, decoder_module


# ── Residual-stream extraction (shared by Exp 8 and 9) ────────────────────────

def get_block_activations(nn_model, raw_model, X_ctx, y_ctx, x_test_i,
                           block_paths):
    """
    Single forward pass, save test-token residual stream at every block.
    Returns dict {path: (d_model,) numpy array} or {} on failure.
    """
    inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
    saved = {}
    try:
        with nn_model.trace(*inp):
            for path in block_paths:
                sub = nav(nn_model, path)
                if sub is not None:
                    try:
                        saved[path] = sub.output[0].save()
                    except Exception:
                        saved[path] = sub.output.save()
    except Exception as e:
        print(f"  [TRACE] Failed: {e}")
        return {}

    result = {}
    for path, proxy in saved.items():
        act = unwrap(proxy)
        if isinstance(act, tuple): act = act[0]
        act = act.detach().cpu().float()
        while act.dim() > 2: act = act.squeeze(0)
        result[path] = act[-1].numpy()   # test token
    return result


def decode_repr(decoder_module, repr_vec):
    """Project a residual stream vector through the decoder → probabilities."""
    decoder_module.eval()
    with torch.no_grad():
        t = torch.tensor(repr_vec, dtype=torch.float32)
        t = t.to(next(decoder_module.parameters()).device)
        logits = decoder_module(t)
    logits = logits.detach().cpu().float()
    if logits.dim() > 1: logits = logits.squeeze(0)
    return torch.softmax(logits, dim=0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8 — Patch-Through Causal Chain
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_8(nn_model, raw_model, X_ctx, y_ctx, X_test, y_test,
                      block_paths, decoder_module,
                      n_samples=50, k=3):
    """
    Protocol:
      1. Collect clean blocks.0 activation for each sample.
      2. Create poisoned context (Attack D, σ=0.01).
      3. Run poisoned forward pass but patch clean blocks.0 activation back in.
      4. Check: does the patched layer-8 logit output revert to clean?

    Three conditions per sample:
      - Clean:    clean context, clean blocks.0            → ground truth
      - Poisoned: poisoned context, poisoned blocks.0      → attack succeeds
      - Patched:  poisoned context, CLEAN blocks.0 patched → does it fix layer 8?

    Metric: P(true_label) at layer 8 and at final output under all three.
    """
    print("\n" + "=" * 64)
    print("EXPERIMENT 8 — Patch-Through Causal Chain")
    print("=" * 64)

    if not NNSIGHT_AVAILABLE:
        print("  [SKIP] NNsight required.")
        return {}

    BLOCKS_0 = block_paths[0]
    BLOCKS_8 = block_paths[8] if len(block_paths) > 8 else block_paths[-1]
    print(f"  Entry point: {BLOCKS_0}")
    print(f"  Commitment:  {BLOCKS_8}")

    results = []
    n_eval = min(n_samples, len(X_test))

    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue

        rng = np.random.default_rng(42 + i)
        X_p, y_p = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)

        # Verify attack flips the prediction
        if _predict(raw_model, X_p, y_p, xi) == yi:
            continue   # attack didn't land; skip

        # ── Step 1: get clean blocks.0 activation ────────────────────────────
        clean_acts = get_block_activations(nn_model, raw_model, X_ctx, y_ctx, xi,
                                            [BLOCKS_0, BLOCKS_8])
        if not clean_acts:
            continue
        clean_b0  = clean_acts[BLOCKS_0]   # (d_model,)
        clean_b8_probs = decode_repr(decoder_module, clean_acts.get(BLOCKS_8, clean_b0))

        # ── Step 2: get poisoned blocks.0 and blocks.8 ───────────────────────
        poison_acts = get_block_activations(nn_model, raw_model, X_p, y_p, xi,
                                             [BLOCKS_0, BLOCKS_8])
        if not poison_acts:
            continue
        poison_b8_probs = decode_repr(decoder_module,
                                       poison_acts.get(BLOCKS_8, poison_acts[BLOCKS_0]))

        # ── Step 3: patch clean blocks.0 into poisoned forward pass ──────────
        clean_b0_tensor = torch.tensor(clean_b0, dtype=DTYPE,
                                        device=next(raw_model.parameters()).device)
        inp_p = _build_input(raw_model, X_p, y_p, xi)

        patched_b8_act = None
        patched_final_pred = None
        patched_b8_saved = None
        try:
            sub0 = nav(nn_model, BLOCKS_0)
            sub8 = nav(nn_model, BLOCKS_8)
            with nn_model.trace(*inp_p):
                # Patch blocks.0 output: replace test token with clean activation
                if sub0 is not None:
                    try:
                        act_out = sub0.output[0]
                        # Try 3D (B, S, D) first, fall back to 2D (S, D)
                        try:
                            act_out[:, -1, :] = clean_b0_tensor
                        except Exception:
                            act_out[-1, :] = clean_b0_tensor
                    except Exception:
                        act_out = sub0.output
                        try:
                            act_out[:, -1, :] = clean_b0_tensor
                        except Exception:
                            act_out[-1, :] = clean_b0_tensor
                # Save blocks.8 after patching
                if sub8 is not None:
                    try:
                        patched_b8_saved = sub8.output[0].save()
                    except Exception:
                        patched_b8_saved = sub8.output.save()
        except Exception as e:
            # Fallback: second attempt with different indexing
            try:
                with nn_model.trace(*inp_p):
                    sub0 = nav(nn_model, BLOCKS_0)
                    if sub0 is not None:
                        try:
                            sub0.output[0][:, -1, :] = clean_b0_tensor
                        except Exception:
                            try:
                                sub0.output[0][-1, :] = clean_b0_tensor
                            except Exception:
                                pass
                    sub8 = nav(nn_model, BLOCKS_8)
                    if sub8 is not None:
                        try:
                            patched_b8_saved = sub8.output[0].save()
                        except Exception:
                            patched_b8_saved = sub8.output.save()
            except Exception as e2:
                print(f"  [EXP8] Patch failed for sample {i}: {e2}")
                continue

        # Also get patched final prediction directly
        try:
            with nn_model.trace(*inp_p):
                sub0 = nav(nn_model, BLOCKS_0)
                if sub0 is not None:
                    try:
                        sub0.output[0][:, -1, :] = clean_b0_tensor
                    except Exception:
                        try:
                            sub0.output[-1, :] = clean_b0_tensor
                        except Exception:
                            pass
                final_out = nn_model.output.save()
            final = unwrap(final_out)
            if isinstance(final, tuple): final = final[0]
            final = final.detach().cpu().float()
            while final.dim() > 1: final = final.squeeze(0)
            patched_final_pred = int(final.argmax().item())
        except Exception:
            patched_final_pred = None

        # Decode patched blocks.8
        patched_b8_probs = None
        if patched_b8_saved is not None:
            try:
                pb8 = unwrap(patched_b8_saved)
                if isinstance(pb8, tuple): pb8 = pb8[0]
                pb8 = pb8.detach().cpu().float()
                while pb8.dim() > 2: pb8 = pb8.squeeze(0)
                patched_b8_probs = decode_repr(decoder_module, pb8[-1].numpy())
            except Exception:
                patched_b8_probs = None

        results.append({
            "true_label":          yi,
            "clean_b8_p_true":     float(clean_b8_probs[yi]),
            "poison_b8_p_true":    float(poison_b8_probs[yi]),
            "patched_b8_p_true":   float(patched_b8_probs[yi]) if patched_b8_probs is not None else float("nan"),
            "patched_final_pred":  patched_final_pred,
            "patched_restores":    (patched_final_pred == yi) if patched_final_pred is not None else False,
        })

    if not results:
        print("  [EXP8] No valid samples collected.")
        return {}

    n = len(results)
    clean_b8    = np.mean([r["clean_b8_p_true"]  for r in results])
    poison_b8   = np.mean([r["poison_b8_p_true"] for r in results])
    patched_b8  = np.nanmean([r["patched_b8_p_true"] for r in results])
    restore_rate = np.mean([r["patched_restores"] for r in results])

    print(f"\n  Samples collected: {n}")
    print(f"  Mean P(true_label) at layer 8:")
    print(f"    Clean context:         {clean_b8:.3f}")
    print(f"    Poisoned context:      {poison_b8:.3f}")
    print(f"    Patched (clean b0):    {patched_b8:.3f}")
    print(f"  Patch restoration rate (final pred): {restore_rate*100:.1f}%")
    print(f"  → {'blocks.0 → layer 8 CAUSAL CHAIN CONFIRMED' if restore_rate > 0.5 else 'Partial / no chain — something else between b0 and b8'}")

    # ── Exp 8b: Cumulative patching ───────────────────────────────────────
    print(f"\n  [8b] Cumulative patching: how many blocks needed to restore?")
    max_patch = min(8, len(block_paths))  # patch up to the commitment layer
    cumul_restore = []   # (n_patched, restore_rate, p_true_at_layer8)

    for n_patch in range(1, max_patch + 1):
        patch_blocks = block_paths[:n_patch]
        n_restored, n_tested = 0, 0
        p_true_vals = []

        for i in range(n_eval):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue
            rng = np.random.default_rng(42 + i)
            X_p, y_p = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)
            if _predict(raw_model, X_p, y_p, xi) == yi:
                continue

            # Get clean activations at all blocks to patch
            clean_acts = get_block_activations(nn_model, raw_model, X_ctx, y_ctx, xi,
                                                patch_blocks + [BLOCKS_8])
            if not clean_acts or BLOCKS_8 not in clean_acts:
                continue

            # Patch all blocks simultaneously
            inp_p = _build_input(raw_model, X_p, y_p, xi)
            dev = next(raw_model.parameters()).device
            try:
                with nn_model.trace(*inp_p):
                    for bp in patch_blocks:
                        if bp in clean_acts:
                            clean_vec = torch.tensor(clean_acts[bp], dtype=DTYPE, device=dev)
                            sub = nav(nn_model, bp)
                            if sub is not None:
                                try:
                                    sub.output[0][:, -1, :] = clean_vec
                                except Exception:
                                    try:
                                        sub.output[0][-1, :] = clean_vec
                                    except Exception:
                                        pass
                    # Save layer-8 output
                    sub8 = nav(nn_model, BLOCKS_8)
                    try:
                        saved_b8 = sub8.output[0].save()
                    except Exception:
                        saved_b8 = sub8.output.save()

                pb8 = unwrap(saved_b8)
                if isinstance(pb8, tuple): pb8 = pb8[0]
                pb8 = pb8.detach().cpu().float()
                while pb8.dim() > 2: pb8 = pb8.squeeze(0)
                probs = decode_repr(decoder_module, pb8[-1].numpy())
                p_true_vals.append(float(probs[yi]))

                # Check final prediction with same patching
                with nn_model.trace(*inp_p):
                    for bp in patch_blocks:
                        if bp in clean_acts:
                            clean_vec = torch.tensor(clean_acts[bp], dtype=DTYPE, device=dev)
                            sub = nav(nn_model, bp)
                            if sub is not None:
                                try:
                                    sub.output[0][:, -1, :] = clean_vec
                                except Exception:
                                    try:
                                        sub.output[0][-1, :] = clean_vec
                                    except Exception:
                                        pass
                    final_out = nn_model.output.save()
                final = unwrap(final_out)
                if isinstance(final, tuple): final = final[0]
                final = final.detach().cpu().float()
                while final.dim() > 1: final = final.squeeze(0)
                if int(final.argmax().item()) == yi:
                    n_restored += 1
                n_tested += 1
            except Exception:
                continue

        rr = n_restored / max(n_tested, 1)
        pt = float(np.mean(p_true_vals)) if p_true_vals else float('nan')
        cumul_restore.append((n_patch, rr, pt))
        print(f"    Patch blocks 0-{n_patch-1}: restore={rr*100:.1f}%  P(true)@L8={pt:.3f}  (n={n_tested})")

    # Find minimum blocks needed for >50% restoration
    min_blocks_50 = -1
    for np_, rr_, _ in cumul_restore:
        if rr_ > 0.5:
            min_blocks_50 = np_
            break

    print(f"\n  Minimum blocks to restore >50%: {'layers 0-' + str(min_blocks_50-1) if min_blocks_50 > 0 else 'NOT ACHIEVED (circuit is fully diffuse)'}")

    print(f"\n{'='*64}")
    print("EXPERIMENT 8 — SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Condition':<30s}  {'P(true) @ layer 8':>18s}")
    print(f"  {'─'*52}")
    print(f"  {'Clean context':<30s}  {clean_b8:>18.3f}")
    print(f"  {'Poisoned context':<30s}  {poison_b8:>18.3f}")
    print(f"  {'Patched (clean blocks.0)':<30s}  {patched_b8:>18.3f}")
    print(f"  {'─'*52}")
    print(f"  Single-block restoration: {restore_rate*100:.1f}%")
    if cumul_restore:
        print(f"\n  Cumulative patching:")
        print(f"  {'Blocks patched':<20s}  {'Restore %':>10s}  {'P(true)@L8':>12s}")
        print(f"  {'─'*46}")
        for np_, rr_, pt_ in cumul_restore:
            print(f"  {'0-'+str(np_-1):<20s}  {rr_*100:>9.1f}%  {pt_:>12.3f}")
    print(f"{'='*64}")

    return {
        "results":       results,
        "clean_b8":      clean_b8,
        "poison_b8":     poison_b8,
        "patched_b8":    patched_b8,
        "restore_rate":  restore_rate,
        "cumul_restore": cumul_restore,
        "min_blocks_50": min_blocks_50,
        "n":             n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 9 — Linear Probing Across Layers
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_9(nn_model, raw_model, X_ctx, y_ctx, X_test, y_test,
                      block_paths, n_samples=80, k=3):
    """
    At each of the 12 ICL blocks, extract the test-token residual stream and
    train a logistic regression probe to predict true_label.

    Three conditions:
      - Clean:    clean context
      - Attack D: σ=0.01 near-duplicate
      - Attack G: pool-only label flip

    If probe accuracy tracks the logit-lens curve (low through layers 4–7,
    sharp jump at layer 8), the commitment is a genuine residual-stream
    phenomenon, not a decoder artifact.
    """
    print("\n" + "=" * 64)
    print("EXPERIMENT 9 — Linear Probing Across Layers")
    print("=" * 64)

    if not NNSIGHT_AVAILABLE:
        print("  [SKIP] NNsight required.")
        return {}

    n_layers = len(block_paths)
    n_eval   = min(n_samples, len(X_test))

    # Collect representations for all conditions
    # clean_reprs[layer] = list of (repr, true_label)
    clean_reprs   = {i: [] for i in range(n_layers)}
    poison_d_reprs = {i: [] for i in range(n_layers)}
    poison_g_reprs = {i: [] for i in range(n_layers)}

    n_collected = 0
    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue

        rng = np.random.default_rng(42 + i)
        X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)
        rng_g = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)

        c_acts = get_block_activations(nn_model, raw_model, X_ctx, y_ctx, xi, block_paths)
        d_acts = get_block_activations(nn_model, raw_model, X_d,   y_d,   xi, block_paths)
        g_acts = get_block_activations(nn_model, raw_model, X_g,   y_g,   xi, block_paths)

        if not (c_acts and d_acts and g_acts):
            continue
        if len(c_acts) < n_layers:
            continue

        for j, path in enumerate(block_paths):
            if path in c_acts:
                clean_reprs[j].append((c_acts[path], yi))
            if path in d_acts:
                poison_d_reprs[j].append((d_acts[path], yi))
            if path in g_acts:
                poison_g_reprs[j].append((g_acts[path], yi))

        n_collected += 1
        if n_collected % 10 == 0:
            print(f"  [{n_collected}/{n_eval}] collected")

    print(f"  Total samples: {n_collected}")

    if n_collected < 10:
        print("  [EXP9] Too few samples for probing.")
        return {}

    # Train probe at each layer (leave-10%-out cross-val approximation)
    def probe_accuracy(layer_reprs):
        if len(layer_reprs) < 10:
            return float("nan")
        X_ = np.stack([r[0] for r in layer_reprs])
        y_ = np.array([r[1] for r in layer_reprs])
        if len(np.unique(y_)) < 2:
            return float("nan")
        sc = SkScaler()
        X_ = sc.fit_transform(X_)
        # Use repeated random splits for robustness with skewed classes
        accs = []
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(len(X_))
            Xs, ys = X_[perm], y_[perm]
            split = max(2, int(0.2 * len(Xs)))
            X_te, X_tr = Xs[:split], Xs[split:]
            y_te, y_tr = ys[:split], ys[split:]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            try:
                clf.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, clf.predict(X_te)))
            except Exception:
                continue
        return float(np.mean(accs)) if accs else float("nan")

    print("\n  Computing probe accuracy per layer...")
    clean_acc   = []
    poison_d_acc = []
    poison_g_acc = []

    for j in range(n_layers):
        ca  = probe_accuracy(clean_reprs[j])
        da  = probe_accuracy(poison_d_reprs[j])
        ga  = probe_accuracy(poison_g_reprs[j])
        clean_acc.append(ca)
        poison_d_acc.append(da)
        poison_g_acc.append(ga)
        print(f"  Layer {j:>2d}  clean={ca:.3f}  AtkD={da:.3f}  AtkG={ga:.3f}")

    # Find commitment layer by max clean accuracy jump
    acc_arr = np.array(clean_acc)
    valid = ~np.isnan(acc_arr)
    if valid.sum() > 1:
        diffs = np.diff(acc_arr)
        diffs_valid = np.where(~np.isnan(diffs), diffs, -999)
        probe_commit_layer = int(np.argmax(diffs_valid)) + 1
    else:
        probe_commit_layer = -1

    print(f"\n  Probe commitment layer (max clean accuracy jump): {probe_commit_layer}")
    print(f"  Logit-lens commitment layer (from Exp 7): 8")
    if probe_commit_layer == 8:
        print(f"  → AGREEMENT: Layer-8 commitment is a residual-stream phenomenon.")
    else:
        print(f"  → DIVERGENCE at layer {probe_commit_layer}: decoder contributes non-trivially.")

    # ── Exp 9b: Cross-condition probe transfer ────────────────────────────
    print(f"\n  [9b] Cross-condition probe transfer")
    print(f"       Train on clean → test on poisoned (and vice versa)")
    print(f"       If clean→AtkD fails: attack creates new representation subspace")
    print(f"       If clean→AtkG works: pool-only attack preserves representation structure")

    transfer_clean_to_d = []
    transfer_clean_to_g = []
    transfer_d_to_clean = []

    for j in range(n_layers):
        c_data = clean_reprs[j]
        d_data = poison_d_reprs[j]
        g_data = poison_g_reprs[j]

        if len(c_data) < 10 or len(d_data) < 10 or len(g_data) < 10:
            transfer_clean_to_d.append(float('nan'))
            transfer_clean_to_g.append(float('nan'))
            transfer_d_to_clean.append(float('nan'))
            continue

        X_c = np.stack([r[0] for r in c_data])
        y_c = np.array([r[1] for r in c_data])
        X_d = np.stack([r[0] for r in d_data])
        y_d = np.array([r[1] for r in d_data])
        X_g = np.stack([r[0] for r in g_data])
        y_g = np.array([r[1] for r in g_data])

        if len(np.unique(y_c)) < 2:
            transfer_clean_to_d.append(float('nan'))
            transfer_clean_to_g.append(float('nan'))
            transfer_d_to_clean.append(float('nan'))
            continue

        sc = SkScaler()
        X_c_s = sc.fit_transform(X_c)
        X_d_s = sc.transform(X_d)
        X_g_s = sc.transform(X_g)

        try:
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            clf.fit(X_c_s, y_c)
            # Clean-trained probe tested on poisoned data
            # Note: y_d and y_g are TRUE labels, same as y_c
            # The probe predicts the true label from the representation
            transfer_clean_to_d.append(accuracy_score(y_d, clf.predict(X_d_s)))
            transfer_clean_to_g.append(accuracy_score(y_g, clf.predict(X_g_s)))
        except Exception:
            transfer_clean_to_d.append(float('nan'))
            transfer_clean_to_g.append(float('nan'))

        try:
            if len(np.unique(y_d)) >= 2:
                clf_d = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf_d.fit(X_d_s, y_d)
                transfer_d_to_clean.append(accuracy_score(y_c, clf_d.predict(X_c_s)))
            else:
                transfer_d_to_clean.append(float('nan'))
        except Exception:
            transfer_d_to_clean.append(float('nan'))

    print(f"\n  {'Layer':>5s}  {'Clean→AtkD':>12s}  {'Clean→AtkG':>12s}  {'AtkD→Clean':>12s}")
    print(f"  {'─'*46}")
    for j in range(n_layers):
        cd = f"{transfer_clean_to_d[j]:.3f}" if not np.isnan(transfer_clean_to_d[j]) else "  n/a"
        cg = f"{transfer_clean_to_g[j]:.3f}" if not np.isnan(transfer_clean_to_g[j]) else "  n/a"
        dc = f"{transfer_d_to_clean[j]:.3f}" if not np.isnan(transfer_d_to_clean[j]) else "  n/a"
        print(f"  {j:>5d}  {cd:>12s}  {cg:>12s}  {dc:>12s}")

    # Interpretation
    if not all(np.isnan(transfer_clean_to_d)):
        avg_cd = np.nanmean(transfer_clean_to_d[-4:])  # late layers
        avg_cg = np.nanmean(transfer_clean_to_g[-4:])
        print(f"\n  Late-layer (8-11) transfer accuracy:")
        print(f"    Clean→AtkD: {avg_cd:.3f}  {'(preserved)' if avg_cd > 0.7 else '(DISRUPTED — new subspace)'}")
        print(f"    Clean→AtkG: {avg_cg:.3f}  {'(preserved)' if avg_cg > 0.7 else '(DISRUPTED — new subspace)'}")

    print(f"\n{'='*64}")
    print("EXPERIMENT 9 — SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Layer':>5s}  {'Clean':>8s}  {'AtkD':>8s}  {'AtkG':>8s}  {'Cl→D':>8s}  {'Cl→G':>8s}")
    print(f"  {'─'*52}")
    for j in range(n_layers):
        c = f"{clean_acc[j]:.3f}" if not np.isnan(clean_acc[j]) else "  n/a"
        d = f"{poison_d_acc[j]:.3f}" if not np.isnan(poison_d_acc[j]) else "  n/a"
        g = f"{poison_g_acc[j]:.3f}" if not np.isnan(poison_g_acc[j]) else "  n/a"
        cd = f"{transfer_clean_to_d[j]:.3f}" if not np.isnan(transfer_clean_to_d[j]) else "  n/a"
        cg = f"{transfer_clean_to_g[j]:.3f}" if not np.isnan(transfer_clean_to_g[j]) else "  n/a"
        print(f"  {j:>5d}  {c:>8s}  {d:>8s}  {g:>8s}  {cd:>8s}  {cg:>8s}")
    print(f"{'='*64}")

    return {
        "clean_acc":          clean_acc,
        "poison_d_acc":       poison_d_acc,
        "poison_g_acc":       poison_g_acc,
        "transfer_clean_to_d": transfer_clean_to_d,
        "transfer_clean_to_g": transfer_clean_to_g,
        "transfer_d_to_clean": transfer_d_to_clean,
        "probe_commit_layer": probe_commit_layer,
        "n_collected":        n_collected,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 10 — Attention Weight Extraction at blocks.0
# ══════════════════════════════════════════════════════════════════════════════

def _extract_attn_weights_monkeypatch(raw_model, X_ctx, y_ctx, x_test_i,
                                       block_path="icl_predictor.tf_icl.blocks.0"):
    """
    Try to extract attention weights from the MHA at the given block.
    Uses hooks instead of monkey-patching kwargs, to support custom MHAs
    that don't accept need_weights.
    Returns attention weights tensor or None.
    """
    import inspect

    # Navigate to the MHA submodule
    target_mha = None
    for name, mod in raw_model.named_modules():
        if name == block_path + ".attn" or name == block_path + ".self_attn":
            target_mha = mod
            break
    # Fallback: any attention-like module inside the block
    if target_mha is None:
        for name, mod in raw_model.named_modules():
            if block_path in name and "attn" in name.split(".")[-1].lower():
                target_mha = mod

    if target_mha is None:
        return None

    captured = {}

    # ── Strategy 1: Hook on Softmax layers inside the MHA ─────────────────
    # Custom MHAs compute attn = softmax(QK^T / sqrt(d)) internally.
    # We intercept the softmax output which IS the attention weights.
    def softmax_hook(module, input, output):
        if output.dim() >= 2:
            s1, s2 = output.shape[-1], output.shape[-2] if output.dim() >= 3 else output.shape[-1]
            # Attention weight matrices are (near) square with seq_len > 10
            if s1 == s2 and s1 > 10:
                captured["weights"] = output.detach()

    hooks = []
    for sub_name, sub_mod in target_mha.named_modules():
        if isinstance(sub_mod, nn.Softmax):
            hooks.append(sub_mod.register_forward_hook(softmax_hook))

    # Also hook the MHA output in case it returns (out, weights)
    def mha_output_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            w = output[1]
            if w is not None and isinstance(w, torch.Tensor) and w.dim() >= 2:
                captured["weights"] = w.detach()

    hooks.append(target_mha.register_forward_hook(mha_output_hook))

    try:
        raw_model.eval()
        with torch.no_grad():
            inp = _build_input(raw_model, X_ctx, y_ctx, x_test_i)
            raw_model(*inp)
    finally:
        for h in hooks:
            h.remove()

    return captured.get("weights", None)


def run_experiment_10(raw_model, X_ctx, y_ctx, X_test, y_test,
                       block_paths, nn_model=None, n_samples=30, k=3, n_show_ctx=20):
    """
    For each sample, extract attention weights at blocks.0 under:
      - Clean context
      - Attack D (σ=0.01)
      - Attack G (pool-only)

    Key quantity: test-token attention (last row of weight matrix).
    Expected under Attack D: mass concentrated on k near-duplicate positions.
    Expected under Attack G: mass on k nearest real neighbors (slightly spread).
    Expected under Clean:    approximately uniform over context.

    Visualise:
      - Average attention heatmap (context position vs head) per condition
      - Sorted attention to test-token (bar chart)
      - Attention entropy per condition
    """
    print("\n" + "=" * 64)
    print("EXPERIMENT 10 — Attention Weights at blocks.0")
    print("=" * 64)

    BLOCK0 = block_paths[0]
    n_eval = min(n_samples, len(X_test))

    # Structures: list of (test_token_attn, poison_positions, yi)
    clean_attns   = []
    poison_d_attns = []
    poison_g_attns = []
    poison_d_positions = []   # which ctx positions were poisoned
    poison_g_positions = []

    n_collected = 0
    for i in range(n_eval):
        xi, yi = X_test[i], y_test[i].item()
        if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
            continue

        rng = np.random.default_rng(42 + i)
        X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)

        # Track which positions were injected for Attack D
        rng2 = np.random.default_rng(42 + i)
        d_positions = rng2.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)

        rng_g = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)
        # Track which positions were flipped for Attack G
        x_np = xi.cpu().numpy()
        X_np, y_np_orig = X_ctx.cpu().numpy(), y_ctx.cpu().numpy()
        same = np.where(y_np_orig == yi)[0]
        if len(same) > 0:
            g_positions = same[np.argsort(np.linalg.norm(X_np[same] - x_np, axis=1))[:k]]
        else:
            g_positions = []

        # Extract weights
        w_clean = _extract_attn_weights_monkeypatch(raw_model, X_ctx, y_ctx, xi, BLOCK0)
        w_d     = _extract_attn_weights_monkeypatch(raw_model, X_d,   y_d,   xi, BLOCK0)
        w_g     = _extract_attn_weights_monkeypatch(raw_model, X_g,   y_g,   xi, BLOCK0)

        if w_clean is None or w_d is None or w_g is None:
            if not hasattr(run_experiment_10, '_warned_none'):
                print("  [EXP10] Monkey-patch returned None — will use activation-norm fallback.")
                run_experiment_10._warned_none = True
            continue

        def process_weights(w):
            """Extract test-token row → (n_heads, seq_len) then average to (seq_len,)"""
            w = w.detach().cpu().float()
            # shapes: (B, H, S, S) or (H, S, S) or (B, S, S) or (S, S)
            while w.dim() > 3: w = w.squeeze(0)
            if w.dim() == 3:    # (H, S, S)
                test_row = w[:, -1, :-1]   # (H, ctx_len) — exclude test token itself
                return test_row.mean(0).numpy()  # average heads → (ctx_len,)
            elif w.dim() == 2:  # (S, S)
                return w[-1, :-1].numpy()
            return None

        ac = process_weights(w_clean)
        ad = process_weights(w_d)
        ag = process_weights(w_g)

        if ac is None or ad is None or ag is None:
            continue

        # Truncate to first n_show_ctx positions for visualisation
        ac = ac[:n_show_ctx]
        ad = ad[:n_show_ctx]
        ag = ag[:n_show_ctx]

        clean_attns.append(ac)
        poison_d_attns.append(ad)
        poison_g_attns.append(ag)
        poison_d_positions.append([p for p in d_positions if p < n_show_ctx])
        poison_g_positions.append([p for p in g_positions if p < n_show_ctx])
        n_collected += 1

    # ── Activation-norm fallback if monkey-patch failed ──────────────────────
    if n_collected == 0 and NNSIGHT_AVAILABLE and nn_model is not None:
        print("  [EXP10] Using activation-norm proxy (need_weights unavailable).")

        def get_attn_norm_proxy(X_c, y_c, xi, block_path):
            """Per-position activation norm as attention proxy."""
            inp = _build_input(raw_model, X_c, y_c, xi)
            sub = nav(nn_model, block_path)
            if sub is None:
                return None
            try:
                # Single trace — NO try/except inside the with block
                # (inner try/except catches NNsight's ExitTracingException)
                with nn_model.trace(*inp):
                    saved = sub.output.save()
                act = unwrap(saved)
                if isinstance(act, tuple):
                    act = act[0]
                act = act.detach().cpu().float()
                while act.dim() > 2:
                    act = act.squeeze(0)
                norms = act.norm(dim=-1).numpy()
                ctx_norms = norms[:-1]
                ctx_norms = ctx_norms / (ctx_norms.sum() + 1e-8)
                return ctx_norms[:n_show_ctx]
            except Exception as e:
                if "ExitTracing" in type(e).__name__:
                    raise  # re-raise NNsight control flow
                return None

        for i in range(n_eval):
            xi, yi = X_test[i], y_test[i].item()
            if _predict(raw_model, X_ctx, y_ctx, xi) != yi:
                continue

            rng = np.random.default_rng(42 + i)
            X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng, sigma=0.01)
            rng2 = np.random.default_rng(42 + i)
            d_pos = rng2.choice(len(X_ctx), size=min(k, len(X_ctx)), replace=False)

            rng_g = np.random.default_rng(42 + i)
            X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)
            x_np = xi.cpu().numpy()
            X_np, y_np_orig = X_ctx.cpu().numpy(), y_ctx.cpu().numpy()
            same = np.where(y_np_orig == yi)[0]
            g_pos = same[np.argsort(np.linalg.norm(X_np[same] - x_np, axis=1))[:k]] \
                    if len(same) > 0 else []

            ac = get_attn_norm_proxy(X_ctx, y_ctx, xi, BLOCK0)
            ad = get_attn_norm_proxy(X_d,   y_d,   xi, BLOCK0)
            ag = get_attn_norm_proxy(X_g,   y_g,   xi, BLOCK0)

            if ac is None or ad is None or ag is None:
                continue

            clean_attns.append(ac)
            poison_d_attns.append(ad)
            poison_g_attns.append(ag)
            poison_d_positions.append([p for p in d_pos if p < n_show_ctx])
            poison_g_positions.append([p for p in g_pos if p < n_show_ctx])
            n_collected += 1

        mode = "activation-norm proxy"
    elif n_collected > 0:
        mode = "true attention weights"
    else:
        mode = "unknown"

    if n_collected == 0:
        print("  [EXP10] Could not collect any attention data.")
        return {}

    print(f"  Collected {n_collected} samples ({mode})")

    # ── Compute averages and entropy ─────────────────────────────────────────
    avg_clean   = np.mean(clean_attns, axis=0)
    avg_d       = np.mean(poison_d_attns, axis=0)
    avg_g       = np.mean(poison_g_attns, axis=0)

    def entropy(a):
        a = np.clip(a, 1e-8, 1)
        a = a / a.sum()
        return float(-np.sum(a * np.log(a)))

    ent_clean = np.mean([entropy(a) for a in clean_attns])
    ent_d     = np.mean([entropy(a) for a in poison_d_attns])
    ent_g     = np.mean([entropy(a) for a in poison_g_attns])

    # Concentration at poison positions
    def mean_attn_at_positions(attn_list, pos_list):
        vals = []
        for attn, positions in zip(attn_list, pos_list):
            if len(positions) > 0:
                vals.append(attn[list(positions)].sum())
        return float(np.mean(vals)) if vals else float("nan")

    conc_d = mean_attn_at_positions(poison_d_attns, poison_d_positions)
    conc_g = mean_attn_at_positions(poison_g_attns, poison_g_positions)

    # Equivalent uniform: 1/n_show_ctx * k
    uniform_k = k / n_show_ctx

    print(f"\n  Attention entropy at blocks.0:")
    print(f"    Clean:    {ent_clean:.3f}  (max uniform = {np.log(n_show_ctx):.2f})")
    print(f"    Attack D: {ent_d:.3f}")
    print(f"    Attack G: {ent_g:.3f}")
    print(f"\n  Mean attention mass on poison positions (k={k}, uniform={uniform_k:.3f}):")
    print(f"    Attack D: {conc_d:.3f}  ({conc_d/uniform_k:.1f}x over uniform)")
    print(f"    Attack G: {conc_g:.3f}  ({conc_g/uniform_k:.1f}x over uniform)")
    print(f"  Mode: {mode}")

    print(f"\n{'='*64}")
    print("EXPERIMENT 10 — SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Condition':<15s}  {'Entropy':>10s}  {'Attn @ poison pos':>18s}  {'Uniform baseline':>17s}")
    print(f"  {'─'*65}")
    print(f"  {'Clean':<15s}  {ent_clean:>10.3f}  {'n/a':>18s}  {np.log(n_show_ctx):>17.3f}")
    print(f"  {'Attack D':<15s}  {ent_d:>10.3f}  {conc_d:>18.3f}  {uniform_k:>17.3f}")
    print(f"  {'Attack G':<15s}  {ent_g:>10.3f}  {conc_g:>18.3f}  {uniform_k:>17.3f}")
    print(f"{'='*64}")

    return {
        "avg_clean":    avg_clean,
        "avg_d":        avg_d,
        "avg_g":        avg_g,
        "ent_clean":    ent_clean,
        "ent_d":        ent_d,
        "ent_g":        ent_g,
        "conc_d":       conc_d,
        "conc_g":       conc_g,
        "uniform_k":    uniform_k,
        "n_show_ctx":   n_show_ctx,
        "n_collected":  n_collected,
        "mode":         mode,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — all three experiments in one figure
# ══════════════════════════════════════════════════════════════════════════════

def visualize_all(exp8, exp9, exp10, out_path):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.45, wspace=0.38)

    # ── Row 0: Exp 8 ─────────────────────────────────────────────────────────
    # Panel 0,0: P(true_label) at layer 8 — bar chart
    ax = fig.add_subplot(gs[0, 0])
    if exp8:
        conds = ["Clean", "Poisoned\n(Attack D)", "Patched\n(clean b0)"]
        vals  = [exp8["clean_b8"], exp8["poison_b8"], exp8["patched_b8"]]
        colors = ["#4C72B0", "#E84040", "#55A868"]
        bars = ax.bar(conds, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_ylabel("P(true_label) at layer 8")
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_title("Exp 8: Layer-8 P(true) After\nblocks.0 Patch",
                     fontweight="bold", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Exp 8\nnot available", ha="center",
                va="center", transform=ax.transAxes)

    # Panel 0,1: Cumulative patching curve
    ax = fig.add_subplot(gs[0, 1])
    if exp8 and "cumul_restore" in exp8 and exp8["cumul_restore"]:
        n_patched = [c[0] for c in exp8["cumul_restore"]]
        rr_vals   = [c[1]*100 for c in exp8["cumul_restore"]]
        pt_vals   = [c[2] for c in exp8["cumul_restore"]]
        ax.plot(n_patched, rr_vals, "o-", color="#55A868", linewidth=2.5,
                markersize=6, label="Restore rate (%)")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.4, label="50% threshold")
        ax.set_xlabel("# ICL blocks patched (0 to N-1)")
        ax.set_ylabel("Restoration rate (%)")
        ax.set_ylim(-5, 105)
        ax.set_title("Exp 8b: Cumulative Patching\nHow many blocks to undo attack?",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        # Annotate the minimum blocks for >50%
        mb = exp8.get("min_blocks_50", -1)
        if mb > 0:
            ax.axvline(mb, color="#E84040", linestyle=":", alpha=0.7)
            ax.annotate(f"≥50% @ {mb} blocks", xy=(mb, 50), fontsize=8,
                        xytext=(mb+0.5, 65), arrowprops=dict(arrowstyle="->", color="#E84040"))
    elif exp8:
        rr = exp8["restore_rate"]
        ax.barh(["Restore\nrate"], [rr*100], color="#55A868", alpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of poisoned samples")
        ax.set_title("Exp 8: Restoration (single block)", fontweight="bold", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Exp 8\nnot available", ha="center",
                va="center", transform=ax.transAxes)

    # Panel 0,2: text summary of Exp 8
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    if exp8:
        mb = exp8.get("min_blocks_50", -1)
        mb_str = f"blocks 0-{mb-1}" if mb > 0 else "NOT REACHED"
        txt = (
            f"Causal chain test\n"
            f"Samples: {exp8['n']}\n\n"
            f"P(true) @ layer 8\n"
            f"  Clean:   {exp8['clean_b8']:.3f}\n"
            f"  Poisoned:{exp8['poison_b8']:.3f}\n"
            f"  Patched: {exp8['patched_b8']:.3f}\n\n"
            f"Single-block: {exp8['restore_rate']*100:.1f}%\n"
            f">50% needs: {mb_str}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F4E8"))

    # ── Row 1: Exp 9 ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    if exp9:
        layers = np.arange(len(exp9["clean_acc"]))
        ax.plot(layers, exp9["clean_acc"],   "o-", color="#4C72B0",
                linewidth=2.5, markersize=5, label="Clean (same-cond)")
        ax.plot(layers, exp9["poison_d_acc"], "s--", color="#E84040",
                linewidth=2, markersize=5, label="AtkD (same-cond)")
        ax.plot(layers, exp9["poison_g_acc"], "^:", color="#55A868",
                linewidth=2, markersize=5, label="AtkG (same-cond)")
        # Add cross-condition transfer lines
        if "transfer_clean_to_d" in exp9:
            ax.plot(layers, exp9["transfer_clean_to_d"], "x-", color="#C0392B",
                    linewidth=1.5, markersize=5, alpha=0.7, label="Clean→AtkD (transfer)")
            ax.plot(layers, exp9["transfer_clean_to_g"], "+-", color="#27AE60",
                    linewidth=1.5, markersize=5, alpha=0.7, label="Clean→AtkG (transfer)")
        commit = exp9["probe_commit_layer"]
        if commit >= 0:
            ax.axvline(commit, color="gray", linestyle=":", alpha=0.7,
                       label=f"Probe commit (layer {commit})")
        ax.axvline(8, color="red", linestyle="--", alpha=0.4,
                   label="Logit-lens commit (layer 8)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("ICL Layer")
        ax.set_ylabel("Probe Accuracy")
        ax.set_ylim(0.3, 1.05)
        ax.set_title("Exp 9: Linear Probe Accuracy per Layer\n"
                     "(validates logit-lens trajectory as residual-stream phenomenon)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Exp 9 not available", ha="center",
                va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    if exp9:
        ca = exp9["clean_acc"]
        agree = exp9["probe_commit_layer"] == 8
        l8_str = f"  {ca[8]:.3f}" if len(ca) > 8 else "  n/a"
        txt = (
            f"Linear probe summary\n"
            f"Samples: {exp9['n_collected']}\n\n"
            f"Probe commit layer: {exp9['probe_commit_layer']}\n"
            f"Logit-lens layer:   8\n"
            f"Agreement: {'YES' if agree else 'NO'}\n\n"
            f"Layer-8 clean acc:\n"
            f"{l8_str}"
        )
        fc = "#E8F4E8" if agree else "#FFF3E0"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=fc))

    # ── Row 2: Exp 10 ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    if exp10:
        n_pos = exp10["n_show_ctx"]
        positions = np.arange(n_pos)
        ax.plot(positions, exp10["avg_clean"], color="#4C72B0",
                linewidth=1.5, alpha=0.8, label="Clean")
        ax.plot(positions, exp10["avg_d"],     color="#E84040",
                linewidth=1.5, alpha=0.8, label="Attack D")
        ax.plot(positions, exp10["avg_g"],     color="#55A868",
                linewidth=1.5, alpha=0.8, label="Attack G")
        ax.axhline(1/n_pos, color="gray", linestyle="--", alpha=0.5,
                   label="Uniform")
        ax.set_xlabel("Context position")
        ax.set_ylabel("Attention weight (test token)")
        ax.set_title(f"Exp 10: Test-Token Attention\nat blocks.0 ({exp10['mode'][:24]})",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Exp 10\nnot available", ha="center",
                va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 1])
    if exp10:
        conds = ["Clean", "Attack D", "Attack G"]
        ents  = [exp10["ent_clean"], exp10["ent_d"], exp10["ent_g"]]
        colors = ["#4C72B0", "#E84040", "#55A868"]
        bars = ax.bar(conds, ents, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(np.log(exp10["n_show_ctx"]), color="gray",
                   linestyle="--", alpha=0.5, label="Max entropy (uniform)")
        for bar, v in zip(bars, ents):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Attention Entropy (nats)")
        ax.set_title("Exp 10: Attention Entropy\n(lower = more concentrated)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Exp 10\nnot available", ha="center",
                va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    if exp10:
        ud = exp10["uniform_k"]
        txt = (
            f"Attention @ poison pos\n\n"
            f"Uniform baseline: {ud:.3f}\n\n"
            f"Attack D: {exp10['conc_d']:.3f}\n"
            f"  ({exp10['conc_d']/ud:.1f}x over uniform)\n\n"
            f"Attack G: {exp10['conc_g']:.3f}\n"
            f"  ({exp10['conc_g']/ud:.1f}x over uniform)\n\n"
            f"Mode:\n{exp10['mode']}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF2FF"))

    fig.suptitle(
        "Experiments 8–10: Mechanistic Interpretability Suite\n"
        "Causal Chain · Linear Probing · Attention Weights at blocks.0",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[VIZ] Saved {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiments_8_9_10(n_samples=50, k=3):
    print("=" * 64)
    print("EXPERIMENTS 8–10 — Mechanistic Interpretability Suite")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    # ── Data + model ──────────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    raw.eval()

    nn_model = NNsight(raw) if NNSIGHT_AVAILABLE else None

    # ── Discover structure ────────────────────────────────────────────────────
    block_paths, dec_path, dec_module = discover_blocks(raw)

    if not block_paths:
        print("[ERROR] No ICL blocks found.")
        return {}

    # ── Run experiments ───────────────────────────────────────────────────────
    exp8 = run_experiment_8(
        nn_model, raw, X_ctx, y_ctx, X_test, y_test,
        block_paths, dec_module, n_samples=n_samples, k=k
    ) if (nn_model is not None and dec_module is not None) else {}

    exp9 = run_experiment_9(
        nn_model, raw, X_ctx, y_ctx, X_test, y_test,
        block_paths, n_samples=min(n_samples, 80), k=k
    ) if nn_model is not None else {}

    exp10 = run_experiment_10(
        raw, X_ctx, y_ctx, X_test, y_test,
        block_paths, nn_model=nn_model, n_samples=n_samples, k=k
    )

    # ── Visualise ─────────────────────────────────────────────────────────────
    out_png = f"{OUT}/exp8_9_10_mech_interp.png"
    visualize_all(exp8, exp9, exp10, out_png)

    # ── Final cross-experiment summary ────────────────────────────────────────
    print(f"\n{'='*64}")
    print("CROSS-EXPERIMENT MECHANISTIC SUMMARY")
    print(f"{'='*64}")

    if exp8:
        rr = exp8["restore_rate"]
        print(f"\n  Causal chain (Exp 8):")
        print(f"    blocks.0 → layer 8 restoration rate: {rr*100:.1f}%")
        print(f"    Verdict: {'CONFIRMED' if rr > 0.5 else 'PARTIAL — other pathway involved'}")

    if exp9:
        agree = exp9["probe_commit_layer"] == 8
        print(f"\n  Linear probe (Exp 9):")
        print(f"    Probe commitment layer: {exp9['probe_commit_layer']}")
        print(f"    Logit-lens layer: 8")
        print(f"    Agreement: {'YES — layer-8 is residual-stream phenomenon' if agree else 'NO — decoder contributes significantly'}")

    if exp10:
        print(f"\n  Attention weights (Exp 10):")
        print(f"    Clean entropy:    {exp10['ent_clean']:.3f}")
        print(f"    Attack D entropy: {exp10['ent_d']:.3f}  "
              f"({exp10['conc_d']/exp10['uniform_k']:.1f}x concentration at poison positions)")
        print(f"    Attack G entropy: {exp10['ent_g']:.3f}  "
              f"({exp10['conc_g']/exp10['uniform_k']:.1f}x concentration at poison positions)")

    print(f"\n{'='*64}")

    return {"exp8": exp8, "exp9": exp9, "exp10": exp10}


if __name__ == "__main__":
    results = run_experiments_8_9_10(n_samples=50, k=3)
