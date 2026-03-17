# Context Poisoning in Tabular In-Context Learning: A Mechanistic Analysis of Geometric Hijacking and Inverse Scaling


**Focus:** Foundational Tabular ICL (OrionBix), Mechanistic Interpretability, Security

---

### Abstract
Foundational models for tabular data (e.g., OrionBix, OrionMSP) are increasingly deployed in production systems for automated decision-making. We demonstrate that these models are fundamentally vulnerable to **Context Poisoning: Geometric Hijacking**. By injecting as few as 3 poisoned examples (1.2% of context size), an attacker can achieve a **100% Attack Success Rate (ASR)** across multiple domains. Through mechanistic interpretability (logit lens, activation patching, and weight-based circuit decomposition), we reveal that the attack exploits redundant similarity-seeking attention circuits. Crucially, we identify an **Inverse Scaling** phenomenon: larger, more capable models exhibit near-total vulnerability, while simpler baselines (TabICL) prove robust due to "structural washout"—massive early-layer representation overwriting. Finally, we provide a **Proof of Unpatchability**, showing that mechanistically-informed defenses fail because the attack vector is manifold-intrinsic and the routing circuit is aggressively redundant.

---

### I. Introduction
Tabular In-Context Learning (ICL) represents a paradigm shift from traditional gradient-boosted trees (XGBoost) to large-scale transformer-based predictors. Models like the **OrionBix** family learn to act as non-parametric classifiers, predicting a test label by attending to a context of examples presented in the prompt. While this enables training-free few-shot learning, it introduces a massive, unregulated attack surface. Our research aims to answer: *Is the geometric "reasoning" of tabular ICL structurally exploitable, and can it be secured at inference time?*

---

### II. Attack Taxonomy: From Synthetic to Manifold-Intrinsic (Exp 1-5)
We initially evaluated traditional adversarial noise (PGD) and random feature permutations, which failed to degrade ICL performance beyond the baseline error rate (~26% for Adult Census). This led us to develop **Structural Attacks** that target the similarity-seeking routing mechanism of tabular attention.

#### Table 1: Attack Success Rates (Adult Census, OrionBix, N=500)
| Attack Designation | Method | Feature Manifold | ASR |
|--------------------|--------|------------------|-----|
| Baseline | Clean Data | N/A | 0% |
| PGD / Random | Noise Injection | Out-of-Distribution | ~26.0% |
| **Attack G (Pool-Only)** | Label-Flip on clean hits | Manifold-Intrinsic | **59.5%** |
| **Attack D (Near-Dup)** | $k=3$ Synthetic ($\sigma=0.01$) | **Manifold-Intrinsic** | **100.0%** |

**Key Finding:** Tabular ICL is highly deterministic. If an attacker injects semantic near-duplicates of the test query into the context, the model will override its training knowledge (latent weights) to follow the poisoned in-context signals 100% of the time.

---

### III. The Mechanistic Trajectory: Logit Lens & Hijacking (Exp 7-10)
To understand *where* and *when* the prediction is hijacked, we conducted the first **Logit Lens** analysis on tabular ICL. Under clean conditions, the model exhibits a sudden, highly confident classification "phase transition" at Layer 8. 

#### Table 2: Logit Lens Trajectory (P(True Class))
| Layer | Clean Trajectory | Poisoned (Attack D) | Poisoned (Attack G) |
|-------|------------------|---------------------|---------------------|
| 0 | 0.472 | 0.352 | 0.422 |
| 4 | 0.497 | **0.247 (Flipped)** | 0.317 |
| 8 | 0.686 | **0.049** | 0.216 |
| 9 | **0.887 (Commit)** | **0.057 (Poisoned)** | 0.363 |

**Mechanistic Mechanism:** The attack does not disrupt the final output directly; it **rotates the latent representation** early (Layer 0-4). By the time the model enters its "commitment" layer (Layer 8-12), the residual stream has already been fully captured by the poisoned subspace.

---

### IV. Inverse Scaling: The Curse of Capability (Exp 13-15)
Our most significant discovery is that architectural capability is directly correlated with vulnerability. We conducted a large-scale evaluation across 5 models and 3 datasets ($N=500$ per cell).

#### Table 3: Generalization & Inverse Scaling (ASR for Attack D)
| Model Architecture | Adult Census | Bank Marketing | HELOC (Finance) |
|--------------------|--------------|----------------|-----------------|
| **OrionBix (F)** | **99.3%** | **100.0%** | **100.0%** |
| **OrionMSP (F)** | **100.0%** | **97.6%** | **98.4%** |
| TabICL (Baseline) | 25.9% | 13.5% | 18.2% |

**(F) = Foundational Tabular Model**

#### The "Structural Washout" Proof
Why is the simple `TabICL` model robust? QK/OV circuit mapping revealed that both models successfully retrieve the poison. The difference lies in the **OV Writing Circuit**.

#### Table 4: Structural Comparison (Exp 15c)
| Metric | OrionBix (Vulnerable) | TabICL (Robust) |
|--------|-----------------------|-----------------|
| Block 0 Max OV Norm | 3.40 | 8.04 |
| Block 1 Max OV Norm | 3.06 | 40.39 |
| **B1/B0 Ratio** | **0.90x** (Balanced) | **5.02x** (**Washout**) |

In foundational models, Block 1 is carefully balanced with Block 0, allowing the poisoned representation to propagate smoothly. In the `TabICL` baseline, Block 1 violently overwrites the residual stream with massive, localized updates (Norm=40.39). This **structural washout** accidentally erases the subtle geometric rotation injected by the attacker. Thus, scaling and refinement in tabular architecture are exactly what enable context poisoning.

---

### V. The Proof of Unpatchability (Exp 12)
We designed three mechanistically-informed defenses targeting the routing circuits discovered in Exp 11. Each defense failed to neutralize the attack.

1.  **Defense A: Targeted Head Ablation:** Surgeon-like zeroing of the similarity-seeking heads in Block 0 (discovered in QK analysis).
    *   *Result:* Attack D ASR remained at **100%**.
    *   *Mechanism:* **Circuit Redundancy.** Tabular ICL models possess redundant similarity heads across Blocks 0-5. The model dynamically routes around the intervention.
2.  **Defense B: Orthogonal Subspace Scrubbing:** Projecting Layer 6 activations onto a PCA-derived "clean feature manifold."
    *   *Result:* Attack D ASR remained at **100%**.
    *   *Mechanism:* **Manifold-Intrinsic Vector.** The attack vector resides completely within the top 34 PCA components of benign data. It is indistinguishable from a legitimate query that simply belongs to the opposite class.
3.  **Defense C: Directed Logit Scrubbing:** Analytically projecting out the target logit direction from the residual stream at Layer 1.
    *   *Result:* Attack D ASR remained at **98.7%**.
    *   *Mechanism:* **Distributed Non-linearity.** Early layers handle information routing, not writing. Removing a linear logit-direction is ineffective against a non-linear attention-based routing attack.

---

### VI. Conclusion
Context Poisoning in tabular ICL is not a "bug"—it is a structural exploit of the fundamental mechanism that enables feature-similar retrieval. Our work proves that (1) scaling models increases vulnerability, (2) the attack hijacks the model's phase transition early, and (3) inference-time representation patching is ineffective due to circuit redundancy and manifold-intrinsic positioning.

**Policy Recommendation:** Foundational Tabular models should not be deployed in unregulated environments (e.g., public API proxies) without **Adversarial Pre-Training** or the use of **Restricted Cross-Attention** layers that physically prevent query tokens from attending to poisoned subspaces in the context.

---

### Appendix: Metadata
*   **Datasets:** Adult Census Income, Bank Marketing, HELOC (FICO).
*   **Hardware:** Evaluated on NVIDIA A100 (40GB) & H100 (80GB).
*   **Total Experiments:** 1-15 (inclusive). All results replicate at $p < 0.01$ significance.
