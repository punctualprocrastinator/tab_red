"""
15-Day NeurIPS Roadmap - Experiment 12
Mechanistically-Informed Defenses

Defense B: Orthogonal Subspace Scrubbing
Insight from Exp 9b: Attack D creates an orthogonal representation subspace by layer 6.
Method: 
  1. Profile clean representations at Layer 6 using PCA.
  2. At inference, project Layer 6 outputs into this clean PCA subspace to scrub
     the orthogonal attack vector before it reaches the Phase Transition (Layer 8).
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

sys.path.append(os.path.abspath("."))
from exp_8_9_10 import load_adult_income, _build_input, _flatten_logits, attack_near_dup, attack_pool_only

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    return wrapper

def extract_raw_module(wrapper) -> nn.Module:
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "model_"):
        raw = wrapper.model.model_
        raw.eval().to(DEVICE)
        return raw
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "model"):
        raw = wrapper.model.model
        raw.eval().to(DEVICE)
        return raw
    for attr in ["model_", "net", "module", "estimator_", "base_estimator_"]:
        obj = wrapper
        for step in ["model", attr]:
            obj = getattr(obj, step, None)
            if obj is None: break
        if obj is not None and isinstance(obj, nn.Module):
            obj.eval().to(DEVICE)
            return obj
    for attr in ["model_", "model", "net_", "net"]:
        obj = getattr(wrapper, attr, None)
        if obj is not None and isinstance(obj, nn.Module):
            obj.eval().to(DEVICE)
            return obj
    raise RuntimeError("Could not unwrap TabularPipeline")

def get_block_module(raw_model, block_idx=6):
    for name, mod in raw_model.named_modules():
        if f"icl_predictor.tf_icl.blocks.{block_idx}" == name:
            return mod
    raise RuntimeError(f"Could not find blocks.{block_idx}")

# ══════════════════════════════════════════════════════════════════════════════
# DEFENSE B: SUBSPACE SCRUBBING
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_12b(n_samples=50, k=3, scrub_layer=6, pca_components=0.95):
    print("=" * 64)
    print(f"EXPERIMENT 12b — Orthogonal Subspace Scrubbing at Layer {scrub_layer}")
    print(f"Device: {DEVICE}  |  k={k}")
    print("=" * 64)

    X_ctx, y_ctx, X_test, y_test, feat_names, scaler = load_adult_income()
    wrapper  = load_model("orion-bix")
    
    X_ctx_df = pd.DataFrame(X_ctx.cpu().numpy(), columns=feat_names)
    y_ctx_s  = pd.Series(y_ctx.cpu().numpy())
    wrapper.fit(X_ctx_df, y_ctx_s)
    raw = extract_raw_module(wrapper)
    target_layer = get_block_module(raw, scrub_layer)
    
    print(f"\n[PHASE 1] Extracting Clean PCA Subspace from Layer {scrub_layer}")
    clean_activations = []
    
    # Hook to capture activations
    global captured_act
    captured_act = None
    def capture_hook(module, args, output):
        global captured_act
        captured_act = output.detach() if not isinstance(output, tuple) else output[0].detach()
        return output
        
    h = target_layer.register_forward_hook(capture_hook)
    
    for i in range(min(150, len(X_test))):
        xi, yi = X_test[i], y_test[i].item()
        inp = _build_input(raw, X_ctx, y_ctx, xi)
        with torch.no_grad(): raw(*inp)
        
        # Test token is always the last token in the sequence dimensions
        act = captured_act.cpu().float()
        while act.dim() > 2: act = act.squeeze(0)
        clean_activations.append(act[-1].numpy())
        
    h.remove()
    
    X_train_pca = np.array(clean_activations)
    pca = PCA(n_components=pca_components) # Retain 95% of clean variance
    pca.fit(X_train_pca)
    
    print(f"  Clean activations shape: {X_train_pca.shape}")
    print(f"  PCA retained {pca.n_components_} components out of {X_train_pca.shape[1]} to explain {pca_components*100}% variance.")
    
    # Move PCA basis to GPU for fast projection
    components = torch.tensor(pca.components_, dtype=DTYPE, device=DEVICE) # (K, d_model)
    mean       = torch.tensor(pca.mean_,       dtype=DTYPE, device=DEVICE) # (d_model,)
    
    print(f"\n[PHASE 2] Evaluating Scrubbing Defense...")
    
    # The Scrubbing Hook
    def scrub_hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0] if is_tuple else output
        
        # Look at the test token index
        # Shape usually: (seq_len, batch, d_model) or (batch, seq_len, d_model)
        # We will project the entire tensor along the d_model dimension just to be safe.
        d_model = tensor.shape[-1]
        
        with torch.no_grad():
            x_centered = tensor - mean
            # Project onto PCA components: (..., d_model) @ (d_model, K) = (..., K)
            projection = torch.matmul(x_centered, components.T)
            # Reconstruct: (..., K) @ (K, d_model) = (..., d_model)
            x_scrubbed = torch.matmul(projection, components) + mean
            
            # Since some models use residual connections inside the block, fully overwriting
            # the tensor scrubs everything orthogonal to the clean subspace.
            tensor.copy_(x_scrubbed)
            
        return (tensor,) + output[1:] if is_tuple else tensor

    # We evaluate normal (no defense) and defensive (with scrub hook)
    
    clean_preds_base, atkd_preds_base, atkg_preds_base = [], [], []
    clean_preds_def,  atkd_preds_def,  atkg_preds_def  = [], [], []
    
    targets = []
    for i in range(min(n_samples, len(X_test))):
        xi, yi = X_test[i], y_test[i].item()
        rng_d = np.random.default_rng(42 + i)
        X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng_d, sigma=0.01)
        rng_g = np.random.default_rng(42 + i)
        X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)
        targets.append((xi, yi, X_d, y_d, X_g, y_g))

    def eval_targets(use_defense):
        hook_handle = None
        if use_defense:
            hook_handle = target_layer.register_forward_hook(scrub_hook)
            
        c_preds, d_preds, g_preds = [], [], []
        for i, (xi, yi, X_d, y_d, X_g, y_g) in enumerate(targets):
            with torch.no_grad():
                c_preds.append(int(_flatten_logits(raw(*_build_input(raw, X_ctx, y_ctx, xi))).argmax().item()))
                d_preds.append(int(_flatten_logits(raw(*_build_input(raw, X_d, y_d, xi))).argmax().item()))
                g_preds.append(int(_flatten_logits(raw(*_build_input(raw, X_g, y_g, xi))).argmax().item()))
                
        if hook_handle: hook_handle.remove()
        valid_labels = [t[1] for t in targets]
        
        acc = accuracy_score(valid_labels, c_preds)
        d_asr = sum(1 for c, d, t in zip(c_preds, d_preds, valid_labels) if c == t and d != t) / max(1, sum(1 for c, t in zip(c_preds, valid_labels) if c == t))
        g_asr = sum(1 for c, g, t in zip(c_preds, g_preds, valid_labels) if c == t and g != t) / max(1, sum(1 for c, t in zip(c_preds, valid_labels) if c == t))
        
        return acc, d_asr, g_asr

    print("  Evaluating Baseline (No Defense)...")
    acc_base, d_base, g_base = eval_targets(False)
    
    print("  Evaluating PCA Scrubbing (Defense B)...")
    acc_def, d_def, g_def = eval_targets(True)
    
    print("\n" + "=" * 64)
    print("EXP 12b — ORTHOGONAL SUBSPACE SCRUBBING SUMMARY")
    print("=" * 64)
    print(f"{'Configuration':<15} | {'Clean Acc':<10} | {'AtkD ASR':<10} | {'AtkG ASR':<10}")
    print("-" * 55)
    print(f"{'Base (No Def)':<15} | {acc_base:.3f}      | {d_base:.3f}      | {g_base:.3f}")
    print(f"{'PCA Scrub L6':<15} | {acc_def:.3f}      | {d_def:.3f}      | {g_def:.3f}")

    print("\n[CONCLUSION]")
    acc_drop = acc_base - acc_def
    d_blocked = d_base - d_def
    
    if d_blocked > 0.2:
        print(f"✅ Hypothesis Confirmed: Subspace Scrubbing blocks Attack D by {d_blocked*100:.1f}%!")
        print(f"   Trade-off: Clean accuracy changes by {acc_drop*100:.1f}%.")
    else:
        print(f"❌ Hypothesis Failed: Scrubbing failed to block Attack D. Subspace might be intrinsically entangled.")

if __name__ == "__main__":
    run_experiment_12b(n_samples=50, k=3, scrub_layer=6, pca_components=0.95)
