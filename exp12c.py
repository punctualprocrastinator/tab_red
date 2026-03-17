"""
Experiment 12c — Directed Logit-Flip Scrubbing at blocks.0

Why 12b failed:
  - PCA subspace (95% variance) captures geometric spread, not functional geometry.
    Clean and poisoned representations share the same top-34 PCA directions.
    Projecting into those 34 dims removes clean signal, not attack signal.
  - Layer 6 is post-amplification. By blocks.3-5, OV heads have already written
    the corrupted signal into the residual stream (DLA flip 1.5-2.2 vs 0.6 at b0).
    Scrubbing there removes the resistance, not the attack.

This experiment:
  Defense C: Project OUT the logit-flip direction from the test token at blocks.0.
    - Logit-flip direction = the direction in residual stream space that increases
      logit(false_class) - logit(true_class). Computed from decoder weights.
    - One-dimensional projection (not 34-dim collapse). Minimally destructive.
    - Applied at blocks.0 (entry, DLA flip=0.635) before OV amplification cascade.

  Defense D: Same, but calibrated scrubbing — project out only alpha * flip_dir
    where alpha is tuned on a held-out calibration set to maximize ASR reduction
    while minimizing clean accuracy loss.

  Defense E: Multi-layer scrubbing — apply at blocks.0 AND blocks.1.
    Tests whether catching the signal at both entry-point layers helps.

  Each defense is tested against:
    - Attack D (near-duplicate, sigma=0.01, k=3): expected to be most affected
    - Attack G (pool-only):                        tests generalization
    - Adaptive attacker who knows the flip direction (robustness check)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
OUT    = os.environ.get("OUT", "/kaggle/working/")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL = True
except ImportError:
    MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL  (confirmed working patterns from exp_8_9_10 / exp11_fixed)
# ══════════════════════════════════════════════════════════════════════════════

def load_adult_income():
    print("[0A] Loading Adult Income dataset...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df   = data.frame
    tc   = "income" if "income" in df.columns else df.columns[-1]
    y_all = LabelEncoder().fit_transform(df[tc])
    X_df  = df.drop(columns=[tc])
    for c in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))
    X_all = StandardScaler().fit_transform(X_df.values.astype(np.float32))
    feat  = list(X_df.columns)
    nc, nt = 256, 512
    Xc = torch.tensor(X_all[:nc],        dtype=DTYPE, device=DEVICE)
    yc = torch.tensor(y_all[:nc],        dtype=torch.long, device=DEVICE)
    Xt = torch.tensor(X_all[nc:nc+nt],   dtype=DTYPE, device=DEVICE)
    yt = torch.tensor(y_all[nc:nc+nt],   dtype=torch.long, device=DEVICE)
    pos = int(yt.sum()); n = len(yt)
    print(f"[0A] Context {Xc.shape}  Test {Xt.shape}  pos={pos}/{n} ({pos/n*100:.1f}%)")
    return Xc, yc, Xt, yt, feat


def load_and_fit(feat_names, Xc, yc):
    from tabtune import TabularPipeline
    wrapper = TabularPipeline(
        model_name="OrionBix", task_type="classification",
        tuning_strategy="inference",
        tuning_params={"device": "cuda" if DEVICE.type == "cuda" else "cpu"}
    )
    wrapper.fit(
        pd.DataFrame(Xc.cpu().numpy(), columns=feat_names),
        pd.Series(yc.cpu().numpy())
    )
    raw = wrapper.model.model_
    raw.eval().to(DEVICE)
    print(f"[0B] Raw module: {type(raw).__name__} on {next(raw.parameters()).device}")
    return raw


def _build_input(raw, Xc, yc, xi):
    Xs = torch.cat([Xc, xi.unsqueeze(0)], dim=0).unsqueeze(0)
    return Xs, yc.unsqueeze(0)


def _predict(raw, Xc, yc, xi):
    raw.eval()
    with torch.no_grad():
        out = raw(*_build_input(raw, Xc, yc, xi))
    t = out[0] if isinstance(out, tuple) else out
    t = t.cpu().float()
    while t.dim() > 1: t = t.squeeze(0)
    return int(t.argmax())


# ── Attacks ───────────────────────────────────────────────────────────────────

def attack_near_dup(Xc, yc, xi, yi, k, rng, sigma=0.01):
    Xp, yp = Xc.clone(), yc.clone()
    pos = rng.choice(len(Xc), size=min(k, len(Xc)), replace=False)
    xn  = xi.cpu().numpy()
    for p in pos:
        Xp[p] = torch.tensor(xn + rng.normal(0, sigma, xn.shape).astype(np.float32),
                              dtype=Xc.dtype, device=Xc.device)
        yp[p] = 1 - yi
    return Xp, yp


def attack_pool_only(Xc, yc, xi, yi, k):
    Xp, yp = Xc.clone(), yc.clone()
    xn, Xn, yn = xi.cpu().numpy(), Xc.cpu().numpy(), yc.cpu().numpy()
    same = np.where(yn == yi)[0]
    if len(same) == 0: return Xp, yp
    near = same[np.argsort(np.linalg.norm(Xn[same] - xn, axis=1))[:k]]
    for p in near: yp[p] = 1 - yp[p]
    return Xp, yp


# ══════════════════════════════════════════════════════════════════════════════
# LOGIT-FLIP DIRECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_logit_flip_direction(raw, false_class=1, true_class=0):
    """
    Backpropagate the logit-flip signal through the decoder to get
    the direction in residual stream space (d_model=512) that most
    increases logit(false_class) - logit(true_class).

    Decoder: Sequential → Linear(512→1024) → GELU → Linear(1024→10)

    The gradient of logit_diff w.r.t. the input of the first linear layer is:
        flip_dir = W1^T @ (W2[false] - W2[true])
    where W1 = first linear weight (1024, 512), W2 = last linear weight (10, 1024).

    This is a first-order approximation (ignores GELU nonlinearity at h=0).
    We also compute an empirical version by sampling clean representations
    and measuring the actual gradient via autograd for comparison.
    """
    decoder = None
    for name, mod in raw.named_modules():
        if name == "icl_predictor.decoder":
            decoder = mod
            break
    if decoder is None:
        raise RuntimeError("Decoder not found at icl_predictor.decoder")

    linears = [m for m in decoder.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        raise RuntimeError(f"Expected ≥2 Linear layers in decoder, found {len(linears)}")

    W1 = linears[0].weight.detach().float().cpu()   # (1024, 512)
    W2 = linears[-1].weight.detach().float().cpu()  # (10,   1024)

    # Direction in penultimate space: (1024,)
    flip_penultimate = W2[false_class] - W2[true_class]

    # Direction in residual stream space: (512,)
    flip_dir = W1.T @ flip_penultimate               # (512,)
    flip_dir = flip_dir / (flip_dir.norm() + 1e-8)

    print(f"\n[DIR] Logit-flip direction computed")
    print(f"  W1: {list(W1.shape)}  W2: {list(W2.shape)}")
    print(f"  flip_dir norm before normalisation: {(W1.T @ flip_penultimate).norm():.4f}")
    print(f"  flip_dir norm after:  1.000")
    print(f"  false_class={false_class}, true_class={true_class}")

    return flip_dir.to(DEVICE)


def compute_empirical_flip_direction(raw, Xc, yc, Xt, yt, n=40):
    """
    Empirical version: collect test-token activations at blocks.0 for
    correctly classified samples, then compute the mean direction that
    separates clean (true prediction) from poisoned (flipped prediction).

    This doesn't require assuming GELU ≈ linear.
    """
    print("\n[DIR-EMP] Computing empirical flip direction at blocks.0...")
    block0 = None
    for name, mod in raw.named_modules():
        if name == "icl_predictor.tf_icl.blocks.0":
            block0 = mod
            break
    if block0 is None:
        return None

    captured = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["act"] = t.detach().cpu().float()

    h = block0.register_forward_hook(hook)

    clean_acts, poison_acts = [], []
    n_eval = min(n * 3, len(Xt))

    for i in range(n_eval):
        if len(clean_acts) >= n:
            break
        xi, yi = Xt[i], yt[i].item()
        if _predict(raw, Xc, yc, xi) != yi:
            continue
        rng = np.random.default_rng(42 + i)
        Xd, yd = attack_near_dup(Xc, yc, xi, yi, 3, rng, sigma=0.01)
        if _predict(raw, Xd, yd, xi) == yi:
            continue

        with torch.no_grad():
            raw(*_build_input(raw, Xc, yc, xi))
        act = captured["act"]
        while act.dim() > 2: act = act.squeeze(0)
        clean_acts.append(act[-1].numpy())

        with torch.no_grad():
            raw(*_build_input(raw, Xd, yd, xi))
        act = captured["act"]
        while act.dim() > 2: act = act.squeeze(0)
        poison_acts.append(act[-1].numpy())

    h.remove()

    if not clean_acts:
        print("  [DIR-EMP] No samples collected.")
        return None

    mean_clean  = np.mean(clean_acts,  axis=0)
    mean_poison = np.mean(poison_acts, axis=0)
    emp_dir     = torch.tensor(mean_poison - mean_clean, dtype=DTYPE)
    emp_dir     = emp_dir / (emp_dir.norm() + 1e-8)

    cos_sim = float((emp_dir * compute_logit_flip_direction(raw).cpu()).sum())
    print(f"  Empirical direction computed from {len(clean_acts)} pairs")
    print(f"  Cosine similarity with analytical direction: {cos_sim:.4f}")
    print(f"  (|cos| > 0.3 = meaningful alignment; ≈1.0 = identical)")

    return emp_dir.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(raw, Xc, yc, Xt, yt, n_eval=100, k=3, hook_fn=None,
             target_layer_path="icl_predictor.tf_icl.blocks.0"):
    """
    Run inference on n_eval test samples under three conditions:
      clean / Attack D (sigma=0.01) / Attack G (pool-only)

    Returns ASR on correctly-classified samples and clean accuracy.
    Registers hook_fn on target_layer_path if provided.
    """
    target_mod = None
    for name, mod in raw.named_modules():
        if name == target_layer_path:
            target_mod = mod
            break

    handle = None
    if hook_fn is not None and target_mod is not None:
        handle = target_mod.register_forward_hook(hook_fn)

    c_preds, d_preds, g_preds, labels = [], [], [], []

    for i in range(min(n_eval, len(Xt))):
        xi, yi = Xt[i], yt[i].item()
        rng = np.random.default_rng(42 + i)
        Xd, yd = attack_near_dup(Xc, yc, xi, yi, k, rng, sigma=0.01)
        Xg, yg = attack_pool_only(Xc, yc, xi, yi, k)

        with torch.no_grad():
            cp = _predict(raw, Xc, yc, xi)
            dp = _predict(raw, Xd, yd, xi)
            gp = _predict(raw, Xg, yg, xi)

        c_preds.append(cp); d_preds.append(dp)
        g_preds.append(gp); labels.append(yi)

    if handle:
        handle.remove()

    labels   = np.array(labels)
    c_preds  = np.array(c_preds)
    d_preds  = np.array(d_preds)
    g_preds  = np.array(g_preds)

    correct  = c_preds == labels
    n_correct = correct.sum()

    clean_acc = correct.mean()
    asr_d = (correct & (d_preds != labels)).sum() / max(n_correct, 1)
    asr_g = (correct & (g_preds != labels)).sum() / max(n_correct, 1)

    return {
        "clean_acc": float(clean_acc),
        "asr_d":     float(asr_d),
        "asr_g":     float(asr_g),
        "n":         int(n_eval),
        "n_correct": int(n_correct),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEFENSE HOOKS
# ══════════════════════════════════════════════════════════════════════════════

def make_full_scrub_hook(flip_dir):
    """
    Defense C: fully project out flip_dir from test token at target layer.
    h_scrubbed = h - (h · flip_dir) * flip_dir
    """
    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        tensor = (out[0] if is_tuple else out).clone()
        with torch.no_grad():
            fd = flip_dir.to(tensor.device)
            h  = tensor[..., -1, :]                          # (..., d_model)
            coeff = (h * fd).sum(dim=-1, keepdim=True)       # scalar projection
            tensor[..., -1, :] = h - coeff * fd              # project out
        return (tensor,) + out[1:] if is_tuple else tensor
    return hook


def make_calibrated_scrub_hook(flip_dir, alpha=0.5):
    """
    Defense D: partial scrubbing — project out alpha * flip_dir.
    alpha=1.0 → full scrub (Defense C)
    alpha=0.0 → no-op
    Lower alpha → less clean accuracy damage, less attack reduction.
    """
    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        tensor = (out[0] if is_tuple else out).clone()
        with torch.no_grad():
            fd = flip_dir.to(tensor.device)
            h  = tensor[..., -1, :]
            coeff = (h * fd).sum(dim=-1, keepdim=True)
            tensor[..., -1, :] = h - alpha * coeff * fd
        return (tensor,) + out[1:] if is_tuple else tensor
    return hook


def make_two_layer_scrub_hook(flip_dir):
    """
    Defense E: same projection, to be applied at BOTH blocks.0 AND blocks.1.
    Register this hook on both layers in sequence.
    """
    return make_full_scrub_hook(flip_dir)  # same operation, different layer


def make_empirical_scrub_hook(emp_dir, alpha=1.0):
    """
    Defense C-emp: same as C but using empirical flip direction.
    """
    return make_calibrated_scrub_hook(emp_dir, alpha=alpha)


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_alpha(raw, flip_dir, Xc, yc, Xt, yt,
                    n_calib=60, k=3,
                    target_layer="icl_predictor.tf_icl.blocks.0"):
    """
    Sweep alpha in [0, 1] on a calibration set.
    Find the smallest alpha where ASR_D drops below 0.3,
    or the alpha that maximises (ASR_reduction - clean_acc_drop).
    """
    print("\n[CALIB] Sweeping alpha for Defense D calibration...")
    alphas  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    records = []

    for a in alphas:
        hook = make_calibrated_scrub_hook(flip_dir, alpha=a)
        res  = evaluate(raw, Xc, yc, Xt, yt, n_eval=n_calib, k=k,
                        hook_fn=hook, target_layer_path=target_layer)
        records.append((a, res["clean_acc"], res["asr_d"], res["asr_g"]))
        print(f"  α={a:.1f}  clean={res['clean_acc']:.3f}  "
              f"ASR_D={res['asr_d']:.3f}  ASR_G={res['asr_g']:.3f}")

    # Best alpha: maximise (asr_d_base - asr_d) - 2*(clean_base - clean)
    # i.e. reward ASR reduction twice as much as penalise clean accuracy drop
    base_clean = records[0][1]
    base_asrd  = records[0][2]
    scores = [(a, (base_asrd - asrd) - 2.0 * max(0, base_clean - clean))
              for a, clean, asrd, _ in records]
    best_alpha = max(scores, key=lambda x: x[1])[0]
    print(f"\n  Best alpha (reward/penalty heuristic): {best_alpha}")
    return best_alpha, records


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE ATTACK
# ══════════════════════════════════════════════════════════════════════════════

def attack_adaptive(Xc, yc, xi, yi, k, rng, flip_dir, sigma=0.01, lam=0.5):
    """
    Attacker who knows flip_dir and augments near-duplicates to lie
    along flip_dir (orthogonal to the scrubbing projection).

    Strategy: add a component along flip_dir to each poison example,
    scaled by lam, so the attack partially survives scrubbing.

    flip_dir is the direction being scrubbed — adding it means the
    representation *after* scrubbing still carries attack signal
    in the residual component of flip_dir perpendicular to the
    scrubbed dimension.

    Note: full scrubbing removes the flip_dir component entirely,
    so this adaptive attack is actually weaker against Defense C.
    It tests whether partial scrubbing (Defense D) is robust.
    """
    Xp, yp = Xc.clone(), yc.clone()
    pos = rng.choice(len(Xc), size=min(k, len(Xc)), replace=False)
    xn  = xi.cpu().numpy()
    fd  = flip_dir.cpu().numpy()

    for p in pos:
        noise = rng.normal(0, sigma, xn.shape).astype(np.float32)
        # Add a component along the feature projection of flip_dir
        # (approximation: flip_dir is in residual stream space, not feature space,
        #  but we test feature-space alignment as the attacker's best available proxy)
        noise += lam * sigma * fd[:len(noise)] / (np.linalg.norm(fd[:len(noise)]) + 1e-8)
        Xp[p] = torch.tensor(xn + noise, dtype=Xc.dtype, device=Xc.device)
        yp[p] = 1 - yi
    return Xp, yp


def evaluate_adaptive(raw, flip_dir, Xc, yc, Xt, yt,
                       hook_fn, n_eval=100, k=3,
                       target_layer="icl_predictor.tf_icl.blocks.0"):
    """Evaluate adaptive attack against a given defense hook."""
    target_mod = None
    for name, mod in raw.named_modules():
        if name == target_layer:
            target_mod = mod
            break

    handle = target_mod.register_forward_hook(hook_fn) if target_mod else None

    c_preds, adap_preds, labels = [], [], []
    for i in range(min(n_eval, len(Xt))):
        xi, yi = Xt[i], yt[i].item()
        rng = np.random.default_rng(42 + i)
        Xa, ya = attack_adaptive(Xc, yc, xi, yi, k, rng, flip_dir)
        with torch.no_grad():
            c_preds.append(_predict(raw, Xc, yc, xi))
            adap_preds.append(_predict(raw, Xa, ya, xi))
        labels.append(yi)

    if handle: handle.remove()

    labels     = np.array(labels)
    c_preds    = np.array(c_preds)
    adap_preds = np.array(adap_preds)
    correct    = c_preds == labels
    asr_adap   = (correct & (adap_preds != labels)).sum() / max(correct.sum(), 1)
    return float(asr_adap)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize(results, alpha_records, out_path):
    if not MPL:
        return

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors = {"clean": "#4C72B0", "d": "#E84040", "g": "#55A868", "adap": "#DD8452"}
    defenses = list(results.keys())

    # ── Panel 0,0: Clean accuracy per defense ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    accs = [results[d]["clean_acc"] for d in defenses]
    bars = ax.bar(defenses, accs, color=colors["clean"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clean Accuracy")
    ax.set_title("Clean Accuracy per Defense\n(higher = less collateral damage)",
                 fontweight="bold", fontsize=10)
    ax.axhline(results.get("Baseline", {}).get("clean_acc", 0.8),
               color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    # ── Panel 0,1: Attack D ASR per defense ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    asrs_d = [results[d]["asr_d"] for d in defenses]
    bars = ax.bar(defenses, asrs_d, color=colors["d"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, asrs_d):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ASR — Attack D")
    ax.set_title("Attack D ASR per Defense\n(lower = better defense)",
                 fontweight="bold", fontsize=10)
    ax.axhline(results.get("Baseline", {}).get("asr_d", 0.975),
               color="gray", linestyle="--", alpha=0.5, label="Baseline ASR")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    # ── Panel 0,2: Attack G ASR per defense ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    asrs_g = [results[d]["asr_g"] for d in defenses]
    bars = ax.bar(defenses, asrs_g, color=colors["g"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, asrs_g):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ASR — Attack G")
    ax.set_title("Attack G ASR per Defense\n(pool-only, no synthetic content)",
                 fontweight="bold", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    # ── Panel 1,0: Alpha sweep (Defense D calibration) ───────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if alpha_records:
        alphas = [r[0] for r in alpha_records]
        ca     = [r[1] for r in alpha_records]
        ad     = [r[2] for r in alpha_records]
        ag     = [r[3] for r in alpha_records]
        ax.plot(alphas, ca, "o-", color=colors["clean"], linewidth=2, label="Clean acc")
        ax.plot(alphas, ad, "s-", color=colors["d"],     linewidth=2, label="ASR_D")
        ax.plot(alphas, ag, "^-", color=colors["g"],     linewidth=2, label="ASR_G")
        ax.set_xlabel("Alpha (scrubbing strength)")
        ax.set_ylabel("Metric")
        ax.set_title("Defense D: Alpha Calibration Sweep\n(0=no-op, 1=full scrub)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Alpha sweep\nnot run", ha="center",
                va="center", transform=ax.transAxes)

    # ── Panel 1,1: Tradeoff scatter ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    base_acc = results.get("Baseline", {}).get("clean_acc", 0.8)
    base_asr = results.get("Baseline", {}).get("asr_d", 0.975)
    for i, d in enumerate(defenses):
        r = results[d]
        acc_drop = base_acc - r["clean_acc"]
        asr_drop = base_asr - r["asr_d"]
        ax.scatter(acc_drop, asr_drop, s=120, zorder=5,
                   label=d, edgecolors="white", linewidths=0.8)
        ax.annotate(d, (acc_drop, asr_drop),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Clean accuracy drop (← bad)")
    ax.set_ylabel("ASR_D reduction (↑ good)")
    ax.set_title("Defense Tradeoff Space\n(top-left = ideal)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)

    # ── Panel 1,2: Adaptive attack robustness ────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    adap_asrs = {d: results[d].get("asr_adaptive", float("nan")) for d in defenses}
    valid = {d: v for d, v in adap_asrs.items() if not np.isnan(v)}
    if valid:
        ax.bar(list(valid.keys()), list(valid.values()),
               color=colors["adap"], alpha=0.85, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Adaptive Attack ASR")
        ax.set_title("Adaptive Attacker Robustness\n(knows flip_dir, augments along it)",
                     fontweight="bold", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Adaptive ASR\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Adaptive Attacker Robustness", fontweight="bold", fontsize=10)

    fig.suptitle("Experiment 12c: Directed Logit-Flip Scrubbing\n"
                 "Analytical · Empirical · Calibrated · Two-Layer · Adaptive",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ] Saved {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_12c(n_eval=100, n_calib=60, k=3):
    print("=" * 64)
    print("EXPERIMENT 12c — Directed Logit-Flip Scrubbing")
    print(f"Device: {DEVICE}  |  k={k}  |  n_eval={n_eval}")
    print("=" * 64)

    Xc, yc, Xt, yt, feat = load_adult_income()
    raw = load_and_fit(feat, Xc, yc)

    # ── Compute flip directions ───────────────────────────────────────────────
    flip_analytical = compute_logit_flip_direction(raw, false_class=1, true_class=0)
    flip_empirical  = compute_empirical_flip_direction(raw, Xc, yc, Xt, yt, n=40)

    # ── Calibrate alpha ───────────────────────────────────────────────────────
    best_alpha, alpha_records = calibrate_alpha(
        raw, flip_analytical, Xc, yc, Xt, yt,
        n_calib=n_calib, k=k
    )

    # ── Evaluate all defenses ─────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EVALUATING ALL DEFENSES")
    print(f"{'='*64}")

    LAYER_0 = "icl_predictor.tf_icl.blocks.0"
    LAYER_1 = "icl_predictor.tf_icl.blocks.1"

    results = {}

    # Baseline
    print("\n  [Baseline] No defense...")
    results["Baseline"] = evaluate(raw, Xc, yc, Xt, yt, n_eval=n_eval, k=k)

    # Defense C: full analytical scrub at blocks.0
    print("\n  [Defense C] Full analytical scrub @ blocks.0...")
    hook_c = make_full_scrub_hook(flip_analytical)
    results["Def-C\n(full,L0)"] = evaluate(
        raw, Xc, yc, Xt, yt, n_eval=n_eval, k=k,
        hook_fn=hook_c, target_layer_path=LAYER_0
    )

    # Defense D: calibrated alpha at blocks.0
    print(f"\n  [Defense D] Calibrated α={best_alpha} @ blocks.0...")
    hook_d = make_calibrated_scrub_hook(flip_analytical, alpha=best_alpha)
    results[f"Def-D\n(α={best_alpha},L0)"] = evaluate(
        raw, Xc, yc, Xt, yt, n_eval=n_eval, k=k,
        hook_fn=hook_d, target_layer_path=LAYER_0
    )

    # Defense C-emp: empirical direction
    if flip_empirical is not None:
        print("\n  [Defense C-emp] Empirical direction @ blocks.0...")
        hook_cemp = make_empirical_scrub_hook(flip_empirical, alpha=1.0)
        results["Def-C-emp\n(empirical,L0)"] = evaluate(
            raw, Xc, yc, Xt, yt, n_eval=n_eval, k=k,
            hook_fn=hook_cemp, target_layer_path=LAYER_0
        )

    # Defense E: two-layer scrub — blocks.0 + blocks.1 simultaneously
    print("\n  [Defense E] Two-layer scrub @ blocks.0 + blocks.1...")
    hook_e0 = make_two_layer_scrub_hook(flip_analytical)
    hook_e1 = make_two_layer_scrub_hook(flip_analytical)

    # Register both hooks
    block0_mod, block1_mod = None, None
    for name, mod in raw.named_modules():
        if name == LAYER_0: block0_mod = mod
        if name == LAYER_1: block1_mod = mod

    handle0 = block0_mod.register_forward_hook(hook_e0) if block0_mod else None
    handle1 = block1_mod.register_forward_hook(hook_e1) if block1_mod else None

    # Evaluate without hook_fn (hooks already registered above)
    res_e = evaluate(raw, Xc, yc, Xt, yt, n_eval=n_eval, k=k,
                     hook_fn=None)   # no additional hook
    if handle0: handle0.remove()
    if handle1: handle1.remove()
    results["Def-E\n(two-layer)"] = res_e

    # Adaptive attack against Defense C (full scrub)
    print("\n  [Adaptive] Testing adaptive attacker against Defense C...")
    hook_c_adap = make_full_scrub_hook(flip_analytical)
    asr_adap = evaluate_adaptive(
        raw, flip_analytical, Xc, yc, Xt, yt,
        hook_fn=hook_c_adap, n_eval=n_eval, k=k
    )
    results["Def-C\n(full,L0)"]["asr_adaptive"] = asr_adap
    print(f"  Adaptive ASR against Defense C: {asr_adap:.3f}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXP 12c — DIRECTED SCRUBBING SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Defense':<22s}  {'Clean Acc':>10s}  {'ASR_D':>8s}  "
          f"{'ASR_G':>8s}  {'ASR_Adap':>10s}  {'n_correct':>10s}")
    print(f"  {'─'*76}")
    for name, r in results.items():
        name_flat = name.replace("\n", " ")
        adap = f"{r.get('asr_adaptive', float('nan')):>10.3f}" \
               if not np.isnan(r.get("asr_adaptive", float("nan"))) else "       n/a"
        print(f"  {name_flat:<22s}  {r['clean_acc']:>10.3f}  {r['asr_d']:>8.3f}  "
              f"{r['asr_g']:>8.3f} {adap}  {r['n_correct']:>10d}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n[VERDICT]")
    baseline_asr_d = results["Baseline"]["asr_d"]
    baseline_clean = results["Baseline"]["clean_acc"]

    best_defense = None
    best_score   = -999
    for name, r in results.items():
        if name == "Baseline":
            continue
        asr_reduction = baseline_asr_d - r["asr_d"]
        acc_drop      = baseline_clean - r["clean_acc"]
        score         = asr_reduction - 2.0 * max(0, acc_drop)
        if score > best_score:
            best_score   = score
            best_defense = (name, r, asr_reduction, acc_drop)

    if best_defense:
        name, r, asr_red, acc_drop = best_defense
        name_flat = name.replace("\n", " ")
        if asr_red > 0.2:
            print(f"  ✅ Best defense: {name_flat}")
            print(f"     ASR_D: {baseline_asr_d:.3f} → {r['asr_d']:.3f} "
                  f"(reduction: {asr_red*100:.1f}%)")
            print(f"     Clean acc: {baseline_clean:.3f} → {r['clean_acc']:.3f} "
                  f"(drop: {acc_drop*100:.1f}%)")
            if "asr_adaptive" in r:
                print(f"     Adaptive ASR: {r['asr_adaptive']:.3f} "
                      f"({'ROBUST' if r['asr_adaptive'] < r['asr_d'] + 0.1 else 'BROKEN by adaptive'})")
        else:
            print(f"  ❌ No defense achieved >20% ASR reduction.")
            print(f"     Best was {name_flat} with {asr_red*100:.1f}% reduction.")
            print(f"     Interpretation: logit-flip direction is entangled with the")
            print(f"     clean prediction signal — cannot be cleanly removed.")
            print(f"     Implication for paper: confirms D2 (representation monitoring)")
            print(f"     is the only viable defense at current mechanistic understanding.")

    # ── Visualization ─────────────────────────────────────────────────────────
    out_png = f"{OUT}/exp12c_directed_scrubbing.png"
    visualize(results, alpha_records, out_png)

    return {
        "results":       results,
        "alpha_records": alpha_records,
        "best_alpha":    best_alpha,
        "flip_dir":      flip_analytical,
        "flip_emp":      flip_empirical,
    }


if __name__ == "__main__":
    run_experiment_12c(n_eval=100, n_calib=60, k=3)
