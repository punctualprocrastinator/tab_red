# ============================================================
# EXPERIMENT 1 - Baseline Attacks
# Colab-compatible: assumes exp_0 cell was already executed,
# so load_adult_income, load_model, extract_raw_module, DEVICE,
# DTYPE are all available in the global namespace.
# ============================================================
#
# NOTE: OrionBix's forward pass internally uses torch.no_grad()
# or detaches tensors, so standard backprop-based PGD fails.
# All attacks use FINITE-DIFFERENCE gradient estimation instead.
# With only 14 features this is perfectly efficient.
# ============================================================

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ── DLBacktrace availability ──────────────────────────────────────────────────
try:
    from dl_backtrace.pytorch_backtrace import DLBacktrace
    HAS_DLBACKTRACE = True
    print("[EXP1] ✅ DLBacktrace available")
except ImportError:
    HAS_DLBACKTRACE = False
    print("[EXP1] ⚠️  DLBacktrace not available — gradient fallback will be used")


# ── OrionBix Wrapper ──────────────────────────────────────────────────────────

class OrionBixAttackWrapper(torch.nn.Module):
    """Differentiable wrapper: (B, F) test batch → (B, C) logits."""

    def __init__(self, raw_model, X_ctx, y_ctx):
        super().__init__()
        self.raw_model = raw_model
        self.X_ctx = X_ctx
        self.y_ctx = y_ctx
        self.is_orion = type(raw_model).__name__ in ["OrionBix", "OrionMSP"]

    @torch.no_grad()
    def forward(self, x_test):
        """x_test: (B, F) → logits: (B, C)"""
        B = x_test.shape[0]
        all_logits = []
        for i in range(B):
            xi = x_test[i:i+1]
            if self.is_orion:
                X_seq = torch.cat([self.X_ctx, xi], dim=0).unsqueeze(0)
                y_seq = self.y_ctx.unsqueeze(0)
                out = self.raw_model(X_seq, y_seq)
            else:
                out = self.raw_model(self.X_ctx, self.y_ctx, xi.squeeze(0))
            if out.dim() == 1:
                out = out.unsqueeze(0)
            while out.dim() > 2:
                out = out.squeeze(1)
            all_logits.append(out)
        return torch.cat(all_logits, dim=0)

    def get_loss(self, x_test, y_true):
        """Score-based: returns scalar cross-entropy loss."""
        logits = self.forward(x_test)
        return torch.nn.functional.cross_entropy(logits, y_true).item()

    def get_probs(self, x_test):
        """Returns (B, C) softmax probabilities."""
        logits = self.forward(x_test)
        return torch.softmax(logits, dim=-1)


# ── Finite-Difference Gradient Estimation ─────────────────────────────────────

def estimate_gradient_fd(wrapper, X, y, h=0.01):
    """
    Estimate ∂loss/∂X using central finite differences.
    X: (N, F), y: (N,) → grad: (N, F)
    For F=14, this requires 28 forward passes per sample — fast on T4.
    """
    N, F = X.shape
    grad = torch.zeros_like(X)

    for f in range(F):
        X_plus = X.clone();  X_plus[:, f] += h
        X_minus = X.clone(); X_minus[:, f] -= h

        loss_plus = wrapper.get_loss(X_plus, y)
        loss_minus = wrapper.get_loss(X_minus, y)

        grad[:, f] = (loss_plus - loss_minus) / (2 * h)

    return grad


# ── ATTACK A: PGD (Finite-Difference) ────────────────────────────────────────

def run_pgd_attack(raw_model, X_ctx, y_ctx, X_test, y_test,
                   epsilon=0.1, alpha=0.02, num_iter=10):
    """L∞-PGD using finite-difference gradient estimation."""
    print(f"\n[ATTACK A] PGD  (ε={epsilon}, α={alpha}, iters={num_iter})")
    print(f"  Using finite-difference gradients (h=0.01)")

    wrapper = OrionBixAttackWrapper(raw_model, X_ctx, y_ctx).eval()

    N = min(100, len(X_test))
    X_batch = X_test[:N].clone().detach()
    y_batch = y_test[:N].clone().detach()

    # Random init inside ε-ball
    delta = torch.zeros_like(X_batch).uniform_(-epsilon, epsilon)

    for t in range(num_iter):
        adv = torch.clamp(X_batch + delta, -5.0, 5.0)
        grad = estimate_gradient_fd(wrapper, adv, y_batch, h=0.01)

        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = delta.clamp(-epsilon, epsilon)
            adv = torch.clamp(X_batch + delta, -5.0, 5.0)
            delta = adv - X_batch

        if (t + 1) % 5 == 0 or t == 0:
            with torch.no_grad():
                preds = wrapper(adv).argmax(dim=-1)
            iter_asr = 1.0 - accuracy_score(y_batch.cpu().numpy(), preds.cpu().numpy())
            print(f"  iter {t+1:>2d}/{num_iter} — ASR: {iter_asr*100:.1f}%")

    adv_X = (X_batch + delta).detach()

    with torch.no_grad():
        adv_preds = wrapper(adv_X).argmax(dim=-1)

    acc = accuracy_score(y_batch.cpu().numpy(), adv_preds.cpu().numpy())
    asr = 1.0 - acc
    l_inf = (adv_X - X_batch).abs().max(dim=1)[0].mean().item()
    l_2   = (adv_X - X_batch).norm(p=2, dim=1).mean().item()

    print(f"  ────────────────────────────────────────")
    print(f"  FINAL  ASR: {asr*100:.1f}%  |  L∞: {l_inf:.4f}  |  L₂: {l_2:.4f}")
    return {"asr": asr, "l_inf": l_inf, "l2": l_2}


# ── ATTACK B: Feature-Guided (DLBacktrace / FD-Gradient) ─────────────────────

def run_feature_guided_attack(raw_model, X_ctx, y_ctx, X_test, y_test,
                              epsilon=0.1, k_features=3):
    """Perturb only the k most important features."""
    print(f"\n[ATTACK B] Feature-Guided  (ε={epsilon}, k={k_features})")

    wrapper = OrionBixAttackWrapper(raw_model, X_ctx, y_ctx).eval()

    N = min(100, len(X_test))
    X_batch = X_test[:N].clone().detach()
    y_batch = y_test[:N].clone().detach()

    dlb_used = False
    if HAS_DLBACKTRACE:
        try:
            print("  Attempting DLBacktrace relevance …")
            dlb = DLBacktrace(
                wrapper.raw_model,
                input_for_graph=(
                    torch.cat([X_batch[0:1], X_batch[0:1]], dim=0).unsqueeze(0),
                    y_batch[0:1].unsqueeze(0),
                ),
                device=str(X_batch.device).split(":")[0],  # 'cuda:0' → 'cuda'
                verbose=False,
            )
            # Use single-sample trace
            X_seq_sample = torch.cat([X_ctx, X_batch[0:1]], dim=0).unsqueeze(0)
            y_seq_sample = y_ctx.unsqueeze(0)
            _ = dlb.predict(X_seq_sample, y_seq_sample)
            relevance = dlb.evaluation(
                mode="default", multiplier=100.0, task="binary-classification"
            )
            if isinstance(relevance, dict):
                feat_imp = list(relevance.values())[0]
            elif isinstance(relevance, list):
                feat_imp = relevance[0]
            else:
                feat_imp = relevance

            feat_imp = torch.tensor(feat_imp).to(X_batch.device) \
                       if not isinstance(feat_imp, torch.Tensor) else feat_imp.to(X_batch.device)
            if feat_imp.dim() == 1:
                feat_imp = feat_imp.unsqueeze(0).expand(N, -1)

            dlb_used = True
            print("  ✅ DLBacktrace relevance obtained")
        except Exception as e:
            print(f"  ⚠️  DLBacktrace failed ({e}), using FD gradients")

    if not dlb_used:
        # Use finite-difference gradient magnitude as feature importance
        feat_imp = estimate_gradient_fd(wrapper, X_batch, y_batch, h=0.01).abs()
        print("  Using FD-gradient attribution")

    # Compute gradient direction for perturbation via FD
    grad = estimate_gradient_fd(wrapper, X_batch, y_batch, h=0.01)

    # Select top-k features per sample and perturb
    top_k_idx = feat_imp.abs().topk(k_features, dim=1).indices

    adv_X = X_batch.clone()
    for i in range(N):
        adv_X[i, top_k_idx[i]] += epsilon * grad[i, top_k_idx[i]].sign()
    adv_X = torch.clamp(adv_X, -5.0, 5.0)

    with torch.no_grad():
        adv_preds = wrapper(adv_X).argmax(dim=-1)

    acc = accuracy_score(y_batch.cpu().numpy(), adv_preds.cpu().numpy())
    asr = 1.0 - acc
    l_inf = (adv_X - X_batch).abs().max(dim=1)[0].mean().item()

    label = "DLBacktrace" if dlb_used else "FD-Gradient"
    print(f"  ASR: {asr*100:.1f}%  |  L∞: {l_inf:.4f}  ({label})")
    return {"asr": asr, "l_inf": l_inf, "method": label}


# ── ATTACK C: Random Context Poisoning ───────────────────────────────────────

def run_context_poisoning(raw_model, X_ctx, y_ctx, X_test, y_test,
                          k_values=(1, 3, 5, 10), noise_scale=0.5):
    """Flip labels (and optionally perturb features) of k context examples."""
    print(f"\n[ATTACK C] Random Context Poisoning  (k={list(k_values)})")

    results = {}
    N = min(100, len(X_test))
    rng = np.random.default_rng(42)

    for k in k_values:
        poisoned_X = X_ctx.clone()
        poisoned_y = y_ctx.clone()

        idx = rng.choice(len(y_ctx), size=min(k, len(y_ctx)), replace=False)
        poisoned_y[idx] = 1 - poisoned_y[idx]
        noise = torch.randn_like(poisoned_X[idx]) * noise_scale
        poisoned_X[idx] += noise
        poisoned_X = torch.clamp(poisoned_X, -5.0, 5.0)

        wrapper = OrionBixAttackWrapper(raw_model, poisoned_X, poisoned_y).eval()

        with torch.no_grad():
            preds = wrapper(X_test[:N]).argmax(dim=-1)

        acc = accuracy_score(y_test[:N].cpu().numpy(), preds.cpu().numpy())
        asr = 1.0 - acc
        results[k] = asr
        print(f"  k={k:>2d}  →  ASR: {asr*100:.1f}%")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_experiment_1():
    print("=" * 64)
    print("EXPERIMENT 1 — Baseline Attacks")
    print(f"Device: {DEVICE}  |  Seed: 42")
    print("=" * 64)

    # ── Data ──────────────────────────────────────────────────────────────
    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()

    # ── Model (lazy-loaded — must fit first) ──────────────────────────────
    wrapper = load_model("orion-bix")
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)

    # ── Clean baseline ────────────────────────────────────────────────────
    N = min(100, len(X_test))
    clean_w = OrionBixAttackWrapper(raw, X_ctx, y_ctx).eval()
    with torch.no_grad():
        clean_preds = clean_w(X_test[:N]).argmax(dim=-1)
    clean_acc = accuracy_score(y_test[:N].cpu().numpy(), clean_preds.cpu().numpy())
    print(f"\n[CLEAN] Accuracy on first {N}: {clean_acc*100:.1f}%")

    # ── Attacks ───────────────────────────────────────────────────────────
    eps = 0.1
    res_A = run_pgd_attack(raw, X_ctx, y_ctx, X_test, y_test, epsilon=eps)
    res_B = run_feature_guided_attack(raw, X_ctx, y_ctx, X_test, y_test,
                                      epsilon=eps, k_features=3)
    res_C = run_context_poisoning(raw, X_ctx, y_ctx, X_test, y_test)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EXPERIMENT 1 — SUMMARY")
    print("=" * 64)
    print(f"  Clean accuracy        : {clean_acc*100:.1f}%")
    print(f"  PGD           ASR     : {res_A['asr']*100:.1f}%  (ε={eps} L∞)")
    print(f"  Feature-Guided ASR    : {res_B['asr']*100:.1f}%  (ε={eps}, k=3, {res_B['method']})")
    for k, asr in res_C.items():
        print(f"  Context Poison k={k:<2d}  ASR: {asr*100:.1f}%")
    print("=" * 64)

    return res_A, res_B, res_C


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    res_A, res_B, res_C = run_experiment_1()
