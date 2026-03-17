"""
15-Day NeurIPS Roadmap - Experiment 13
Multi-Model and Multi-Dataset Generalization

Scaling up the evaluation to prove statistical significance across domains and architectures.
Models: OrionBix, OrionMSP, TabICL, Mitra, TabPFN
Datasets: Adult, HELOC, Bank Marketing
Sample size: N=500 per configuration
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
import time

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

sys.path.append(os.path.abspath("."))
# Keeping only what we strictly need from exp_8_9_10 to avoid NNsight issues if possible
from exp_8_9_10 import _build_input, _flatten_logits, attack_near_dup, attack_pool_only

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name, n_ctx=256, n_test=500):
    print(f"\n[DATA] Loading {name}...")
    if name == "adult":
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
        target_col = "income" if "income" in df.columns else df.columns[-1]
    elif name == "heloc":
        # HELOC dataset on OpenML
        data = fetch_openml(data_id=45041, as_frame=True, parser="auto")
        df = data.frame
        target_col = "RiskPerformance" if "RiskPerformance" in df.columns else df.columns[-1]
    elif name == "bank":
        # Bank Marketing on OpenML
        data = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
        df = data.frame
        target_col = "Class" if "Class" in df.columns else df.columns[-1]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    y_raw = df[target_col]
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    
    X_df = df.drop(columns=[target_col])
    for c in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))
        
    # Drop rows with NaN if any exist to ensure clean evaluation
    mask = ~X_df.isna().any(axis=1)
    X_df = X_df[mask]
    y_all = y_all[mask]
        
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_df.values.astype(np.float32))
    feat_names = list(X_df.columns)
    
    # Shuffle for evaluation
    rng = np.random.default_rng(1337)
    indices = rng.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]
    
    X_ctx  = torch.tensor(X_all[:n_ctx], dtype=DTYPE, device=DEVICE)
    y_ctx  = torch.tensor(y_all[:n_ctx], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_all[n_ctx:n_ctx+n_test], dtype=DTYPE, device=DEVICE)
    y_test = torch.tensor(y_all[n_ctx:n_ctx+n_test], dtype=torch.long, device=DEVICE)
    
    print(f"  Context : {X_ctx.shape}  |  Test: {X_test.shape}")
    pos = int(y_test.sum().item())
    print(f"  Label balance (test) — pos: {pos}/{len(y_test)} ({pos/len(y_test)*100:.1f}%)")
    return X_ctx, y_ctx, X_test, y_test, feat_names

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_tabtune_model(model_name, X_ctx_df, y_ctx_s):
    print(f"  Loading {model_name} via TabTune...")
    from tabtune import TabularPipeline
    name_map = {
        "orion_bix": "OrionBix", 
        "orion_msp": "OrionMSP", 
        "tabicl": "TabICL", 
        "mitra": "Mitra",
        "tabpfn": "TabPFN"
    }
    wrapper = TabularPipeline(
        model_name=name_map[model_name],
        task_type="classification",
        tuning_strategy="inference",
        tuning_params={"device": "cuda" if DEVICE.type == "cuda" else "cpu"}
    )
    # Important: Fit on the context set to initialize preprocessors/classifiers
    wrapper.fit(pd.DataFrame(X_ctx_df), pd.Series(y_ctx_s))
    
    # Fast unwrapping using our robust method
    for attr in ["model_", "model"]:
        if hasattr(wrapper, "model") and hasattr(wrapper.model, attr):
            raw = getattr(wrapper.model, attr)
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
            
    print(f"  [WARN] Fallback: Returning wrapper itself for {model_name}")
    return wrapper

# Provide unified predict function since not all tabtune raw models use the same _build_input
def universal_predict(raw_model, model_name, X_ctx, y_ctx, xi):
    raw_model.eval()
    with torch.no_grad():
        if model_name in ["orion_bix", "orion_msp", "tabicl"]:
            X_seq = torch.cat([X_ctx, xi.unsqueeze(0)], dim=0).unsqueeze(0)
            y_seq = y_ctx.unsqueeze(0)
            out = raw_model(X_seq, y_seq)
        elif model_name == "mitra":
            # Mitra might use separate context and query args depending on version
            X_seq = torch.cat([X_ctx, xi.unsqueeze(0)], dim=0).unsqueeze(0)
            y_seq = y_ctx.unsqueeze(0)
            out = raw_model(X_seq, y_seq)
        elif model_name == "tabpfn":
            # TabPFN native often takes train_x, train_y, test_x
            # We attempt standard generic sequence passing if unwrapped:
            out = raw_model(X_ctx.unsqueeze(0), y_ctx.unsqueeze(0), xi.unsqueeze(0).unsqueeze(0))
        else:
            # Fallback wrapper predict
            pass
            
        return int(_flatten_logits(out).argmax().item())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_13(n_test=500, k=3):
    print("=" * 64)
    print("EXPERIMENT 13 — Multi-Model & Multi-Dataset Generalization")
    print(f"Device: {DEVICE}  |  k={k}  |  N_samples/config: {n_test}")
    print("=" * 64)

    datasets = ["adult", "heloc", "bank"]
    models = ["orion_bix", "orion_msp", "tabicl"] # Starting with the 3 confirmed TabICL-family ones
    
    results = []

    for d_name in datasets:
        X_ctx, y_ctx, X_test, y_test, feat_names = load_dataset(d_name, n_ctx=256, n_test=n_test)
        
        for m_name in models:
            print(f"\n[RUN] Dataset: {d_name.upper()}  |  Model: {m_name.upper()}")
            try:
                raw_model = load_tabtune_model(m_name, X_ctx.cpu().numpy(), y_ctx.cpu().numpy())
                
                c_preds, d_preds, g_preds = [], [], []
                valid_ys = []
                
                t0 = time.time()
                for i in range(len(X_test)):
                    xi, yi = X_test[i], y_test[i].item()
                    
                    # Compute Baseline Clean Prediction
                    try:
                        c_p = universal_predict(raw_model, m_name, X_ctx, y_ctx, xi)
                    except Exception as e:
                        if i == 0: print(f"  Prediction failed. Skipping {m_name}. Error: {e}")
                        break
                        
                    # Target selection criteria: the attack is only meaningful 
                    # if the model predicts the CLEAN class correctly.
                    if c_p != yi: 
                        continue
                        
                    valid_ys.append(yi)
                    c_preds.append(c_p)
                    
                    # Attack D: Near-Duplicate Injection
                    rng_d = np.random.default_rng(42 + i)
                    X_d, y_d = attack_near_dup(X_ctx, y_ctx, xi, yi, k, rng_d, sigma=0.01)
                    d_preds.append(universal_predict(raw_model, m_name, X_d, y_d, xi))
                    
                    # Attack G: Pool-Only Label Flip
                    rng_g = np.random.default_rng(42 + i)
                    X_g, y_g = attack_pool_only(X_ctx, y_ctx, xi, yi, k, rng_g)
                    g_preds.append(universal_predict(raw_model, m_name, X_g, y_g, xi))
                    
                    if len(c_preds) % 50 == 0:
                        print(f"  Processed {len(c_preds)} valid target points...")

                t1 = time.time()
                
                if not valid_ys:
                    print(f"  No valid predictions. Skipping.")
                    continue
                    
                n_targets = len(valid_ys)
                acc = n_targets / len(X_test) # Roughly clean accuracy 
                
                # ASR calculated on targets the model naturally gets right
                # meaning an attack is successful if it changes the prediction to the wrong class
                d_asr = sum(1 for c, d, t in zip(c_preds, d_preds, valid_ys) if c == t and d != t) / n_targets
                g_asr = sum(1 for c, g, t in zip(c_preds, g_preds, valid_ys) if c == t and g != t) / n_targets
                
                print(f"  Clean Acc: {acc:.3f} | AtkD ASR: {d_asr:.3f} | AtkG ASR: {g_asr:.3f}  ({t1-t0:.1f}s)")
                
                results.append({
                    "Dataset": d_name,
                    "Model": m_name,
                    "CleanAcc": acc,
                    "AtkD_ASR": d_asr,
                    "AtkG_ASR": g_asr,
                    "Samples": n_targets
                })
                
            except Exception as e:
                print(f"  ❌ Failed to evaluate {m_name} on {d_name}: {e}")

    # ── Print Final Matrix ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("FINAL GENERALIZATION MATRIX (N=500 queries per config)")
    print("=" * 64)
    df_res = pd.DataFrame(results)
    
    if len(df_res) > 0:
        print("\n[ATTACK D: Near-Duplicate Matrix (ASR)]")
        matrix_d = df_res.pivot(index="Model", columns="Dataset", values="AtkD_ASR")
        print(matrix_d.round(3))
        
        print("\n[ATTACK G: Pool-Only Matrix (ASR)]")
        matrix_g = df_res.pivot(index="Model", columns="Dataset", values="AtkG_ASR")
        print(matrix_g.round(3))
        
        df_res.to_csv("exp13_generalization.csv", index=False)
        print("\nResults saved to exp13_generalization.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_experiment_13()
