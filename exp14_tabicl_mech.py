"""
Experiment 14: Mechanistic Interpretation of Robustness (TabICL vs OrionBix)
Why is the baseline TabICL model robust to context poisoning (ASR ~25%) while 
foundational models (OrionBix/OrionMSP) are completely vulnerable (ASR ~100%)?

Hypothesis: TabICL lacks the strong "similarity-seeking" attention heads in its 
early layers that the foundational models developed during extensive pre-training.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# REUSE EXP 11 TOOLS
# ══════════════════════════════════════════════════════════════════════════════
sys.path.append(os.path.abspath("."))
from exp11_weight_decomp import discover_weights, exp_11a_ov_circuit, exp_11b_qk_circuit
from exp13_generalization import load_dataset, load_tabtune_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_model_weights(model_name, X_ctx, y_ctx):
    print("\n" + "=" * 64)
    print(f"Analyzing Model: {model_name.upper()}")
    print("=" * 64)
    
    try:
        raw_model = load_tabtune_model(model_name, X_ctx.cpu().numpy(), y_ctx.cpu().numpy())
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return
        
    print(f"\n[1] Discovering Weights for {model_name}...")
    weight_info = discover_weights(raw_model)
    
    if not weight_info or not weight_info.get("blocks"):
        print("Failed to discover attention blocks.")
        return
        
    # Run OV and QK Circuit Analyses
    exp_11a_ov_circuit(weight_info)
    exp_11b_qk_circuit(weight_info)
    
    # Custom Print for QK Eigenvalue Polarity at Block 0
    if weight_info["blocks"]:
        b0 = weight_info["blocks"][0]
        # Quick re-compute for summary
        w = weight_info["block_weights"].get(b0, {})
        if "W_Q" in w and "W_K" in w:
            n_heads = weight_info["n_heads"]
            d_head  = weight_info["d_head"]
            d_model = weight_info["d_model"]
            W_Q_h = w["W_Q"].float().cpu().reshape(n_heads, d_head, d_model)
            W_K_h = w["W_K"].float().cpu().reshape(n_heads, d_head, d_model)
            
            pfracs = []
            for h in range(n_heads):
                W_QK_h = W_Q_h[h] @ W_K_h[h].T
                W_QK_sym = 0.5 * (W_QK_h + W_QK_h.T)
                eigvals = torch.linalg.eigvalsh(W_QK_sym).cpu().numpy()
                pfracs.append(float((eigvals > 0).mean()))
                
            print(f"\n[SUMMARY] {model_name.upper()} Block 0 Similarity-Seeking Heads:")
            print(f"  Positive Eigenvalue Fractions: {[f'{x:.3f}' for x in pfracs]}")
            sim_heads = [h for h, f in enumerate(pfracs) if f > 0.6]
            if sim_heads:
                print(f"  🚨 VULNERABLE: Found similarity-seeking heads >0.6: {sim_heads}")
            else:
                print(f"  🛡️ ROBUST: No strong similarity-seeking heads >0.6 found.")

if __name__ == "__main__":
    print("[0] Loading dataset to initialize models...")
    X_ctx, y_ctx, _, _, _ = load_dataset("adult", n_ctx=256, n_test=10)
    
    # Contrast the highly vulnerable model vs the robust baseline
    analyze_model_weights("orion_bix", X_ctx, y_ctx)
    analyze_model_weights("tabicl", X_ctx, y_ctx)
