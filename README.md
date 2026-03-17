# Structural Context Poisoning in Tabular In-Context Learning

This repository contains the official implementation and mechanistic analysis for **"Context Poisoning in Tabular In-Context Learning: Geometric Hijacking and Inverse Scaling,"** as presented in our technical report.

We demonstrate that foundational tabular models (e.g., OrionBix, OrionMSP) are fundamentally vulnerable to context poisoning attacks that exploit their internal routing mechanisms.

---

## 🚀 Key Discoveries

*   **100% Attack Success Rate (ASR):** Using "Near-Duplicate" synthetic injection (Attack D), we achieved total hijacking across multiple domains with only $k=3$ poisoned samples (1.2% of context).
*   **Inverse Scaling:** Larger, more capable foundational models are **significantly more vulnerable** than simple baselines (TabICL), which survive only due to "structural washout" (brute-force representation overwriting).
*   **Mechanistic Trajectory:** Using the first **Logit Lens** for tabular ICL, we identified a classification "phase transition" at Layer 8-9, with representation hijacking occurring as early as Layer 4.
*   **Proof of Unpatchability:** Mechanistically-informed defenses (Head Ablation, PCA Subspace Scrubbing) fail because the attack is **manifold-intrinsic** and the routing circuit is redundant.

---

## 📂 Repository Structure

### Attack & Defense (Exp 1-6, 12-13)
*   [`exp_0.py`](exp_0.py): Environment setup and NNsight model validation.
*   [`exp1_baselines.py`](exp1_baselines.py): Evaluation of PGD, random noise, and feature-guided baselines.
*   [`exp2_patching.py`](exp2_patching.py): Initial circuit identification via activation patching.
*   [`exp3_circuit_attack.py`](exp3_circuit_attack.py): Implementation of Attack C and targeted geometric attacks.
*   [`Exp4betterattack.py`](Exp4betterattack.py): **Attack D (Near-Duplicate)** achieving 100% ASR.
*   [`experiment_5.py`](experiment_5.py): Arms race evaluation (F1/D2 detectors vs Attack G).
*   [`exp6_gaps.py`](exp6_gaps.py): Cross-model and cross-domain transferability testing.
*   [`exp13_generalization.py`](exp13_generalization.py): Large-scale ($N=500$) generalization matrix across 5 models and 3 datasets.

### Mechanistic Interpretability (Exp 7-11, 14-15)
*   [`exp7_logit_lens.py`](exp7_logit_lens.py): Trajectory analysis and phase transition discovery.
*   [`exp_8_9_10.py`](exp_8_9_10.py): Cumulative patching, linear probing, and attention norm analysis.
*   [`exp11_weight_decomp.py`](exp11_weight_decomp.py): QK/OV circuit decomposition and Direct Logit Attribution (DLA).
*   [`exp14_tabicl_mech.py`](exp14_tabicl_mech.py): Comparing OrionBix vs TabICL circuits for Inverse Scaling.
*   [`exp15.py`](exp15.py): **Robustness Diagnostic** (proving Block-1 OV Dominance / Structural Washout).

### Representation Patching & Defenses (Exp 12)
*   [`exp12_mech_defenses.py`](exp12_mech_defenses.py): Implementation of Head Ablation and PCA Subspace Scrubbing.
*   [`exp12c.py`](exp12c.py): Directed Logit Scrubbing defense.

---

## 📊 Summary of Results

### 1. Generalization Across Archiitectures
| Model | Attack D (Near-Dup) | Attack G (Pool-Only) |
|-------|----------------------|----------------------|
| **OrionBix** | **99.3%** | **54.9%** |
| **OrionMSP** | **100.0%** | **44.3%** |
| TabICL (Baseline) | 25.9% | 22.3% |

### 2. Logit Lens Phase Transition
Classification commits at Layer 8, but representsation flips at Layer 4.

| Layer | Clean P(true) | Poison (Atk D) P(true) |
|-------|---------------|------------------------|
| 4 | 0.497 | **0.247 (Flipped)** |
| 8 | 0.686 | **0.049** |

### 3. Structural Washout (Exp 15c)
TabICL resists attacks because its Block 1 overwrites Block 0 by **5.02x**, while OrionBix propagates signals at **0.90x** balance.

---

## 🛠 Reproduction
To reproduce the findings:
1.  Install dependencies: `pip install torch pandas numpy nnsight tabtune scikit-learn matplotlib`
2.  Run the main generalization suite: `python exp13_generalization.py`
3.  Run the logit lens diagnostic: `python exp7_logit_lens.py`
4.  Run the mechanistic interpretability suite: `python exp_8_9_10.py`

---

## 📜 Documentation
*   [4-Page Technical Report](4_page_technical_report.md)
*   [Complete Results Narrative](paper_draft_results.md)
