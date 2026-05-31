# TCGNN-Surveillance: Topological Causal Graph Neural Networks for Detecting Structural Voids in Epidemiological Surveillance

> **Mboya, G. O.** (2026). *Topological Causal Graph Neural Networks for Detecting Structural Voids in Epidemiological Surveillance.*

> ORCID: [0009-0005-9102-4028](https://orcid.org/0009-0005-9102-4028) · Contact: gmotieno@jooust.ac.ke

---

## Abstract

Traditional epidemiological surveillance systems operate under a structural fallacy: the conflation of zero reported cases with zero disease prevalence. This epistemological blind spot obscures systemic reporting gaps and misguides public health resource allocation. Standard non-spatial correlative models fail to detect these voids because they treat health facilities as statistically isolated entities.

This repository implements the **Topological Causal Graph Neural Network (TCGNN)**, a transdisciplinary architecture integrating computational topology, biostatistical regularization, and Judea Pearl's causal calculus. The framework models the surveillance network as a spatial undirected graph, leveraging a message-passing algorithm to detect local topological invariants. Through a rigorous hyperparameter grid search, the architecture optimized spatial aggregation at an **ϵ = 5.0 km** radius and a dynamic void indicator threshold of **τ = 0.30**, minimizing validation Binary Cross Entropy (BCE) loss to **0.0373**. In a controlled ablation study, the TCGNN outperformed a standard Multi-Layer Perceptron (MLP) baseline by **96.63%**, demonstrating the necessity of spatial geometry in void detection.

> **Disclaimer:** This work is a personal initiative conducted independently and is not affiliated with, funded by, or representative of the official position of any Institution.

---

## Table of Contents

1. [Theoretical Framework](#1-theoretical-framework)
2. [Architecture](#2-architecture)
3. [Empirical Results](#3-empirical-results)
4. [Installation](#4-installation)
5. [Execution Pipeline](#5-execution-pipeline)
6. [Processing Real-World Data](#6-processing-real-world-data)
7. [Project Structure](#7-project-structure)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [References](#9-references)
10. [Declarations](#10-declarations)

---

## 1. Theoretical Framework

### 1.1 The Epistemological Problem

The integrity of global public health forecasting relies on the assumption that Health Management Information Systems (HMIS) capture a representative sample of underlying disease dynamics. In resource-constrained environments, surveillance data is frequently compromised not by random error, but by **structural reporting voids** — localized clusters of facilities that cease reporting or report anomalously, which standard analytics conflate with genuine zero-prevalence.

This framework introduces the concept of the **Geometry of Silence**: the spatial topology of missing surveillance data, which is not random noise but a structured, mappable phenomenon.

### 1.2 Topological Graph Construction

The surveillance network is formalized as an undirected graph **G = (V, E)**, where each vertex v ∈ V represents an individual health facility. Edges are constructed via a spatial ϵ-ball approach:

$$E = \{(u, v) \in V \times V \mid d_{Euc}(u, v) \leq \epsilon\}$$

The undirected structure models the **bidirectional nature of infectious disease catchment crossover** rather than administrative reporting hierarchies. Euclidean distance is used as the computational baseline; geodesic road-network distance remains a horizon for future deployment. The optimal radius ϵ is determined empirically by minimizing validation BCE loss on a **spatially stratified 20% held-out set**, preventing topological data leakage.

### 1.3 Void Supervision Signal

A structural void is formally defined via an indicator function:

$$y_v = \mathbb{I}(c_{\text{reported}} < \tau \cdot c_{\text{expected}})$$

Where `τ` is a tunable hyperparameter — not a rigid clinical assumption — calibrated via grid search. The empirically optimal τ = 0.30 aligns with established HMIS audit frameworks, in which reporting completeness below one-third of demographically expected volumes is flagged as a systemic surveillance failure (WHO, 2020).

### 1.4 Causal Refutation via Do-Calculus

To distinguish genuine surveillance voids from algorithmic artefacts, the architecture employs Pearl's backdoor criterion. The Structural Causal Model (SCM) is defined by the following DAG:

```
Z → T
Z → Y
T → Y
```

Where **T** = Resource Allocation (intervention), **Y** = Void Severity (outcome), and **Z** = {Population Density, Expected Caseloads} (confounder set). Because Z is chronologically and causally prior to T, conditioning on Z blocks the backdoor path T ← Z → Y without inducing collider bias. The backdoor adjustment for continuous confounders is:

$$P(Y \mid do(T=t)) = \int P(Y \mid T=t, Z=z)\, P(Z=z)\, dz$$

The Average Treatment Effect (ATE) is then isolated as:

$$ATE = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]$$

This transitions the architecture from predictive forecasting to **counterfactual causal simulation**.

---

## 2. Architecture

The framework decouples data ingestion, neural computation, and causal refutation into four independent modules.

### `data_adapter.py` — Biostatistical Regularization and Categorical Sanitation

Ingests raw field data (e.g., KHIS or KEMRI schemas). Executes a dual-sanitation pipeline:

- **Continuous features:** Z-score standardization forces the input matrix into $\mathcal{N}(0,1)$, stabilizing the gradient descent vector $\nabla L$ and preventing weight matrix saturation from skewed epidemiological integers:

$$z_{i,f} = \frac{x_{i,f} - \mu_f}{\sigma_f}$$
- **Categorical features:** Orthogonal vector embedding via one-hot encoding converts facility classification tiers into binary matrices prior to neural ingestion.
- Filters topologically invalid nodes (facilities missing GPS coordinates).

### `backend/graph_engine.py` — Topological Engine

Converts the normalized feature matrix into a spatial graph G = (V, E) using the ϵ-ball edge construction. Executes the hyperparameter sensitivity sweep across candidate radii (5 km, 15 km, 30 km, 50 km), selecting the ϵ that minimizes validation BCE loss on the spatially stratified hold-out set.

### `backend/tcgnn_model.py` & `training_loop.py` — Neural Computation Core

A custom **PyTorch Geometric** GNN implementing the message-passing algorithm:

$$h_v^{(k)} = \text{ReLU}\!\left( W_1 h_v^{(k-1)} + W_2 \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$

Key design constraints:
- **K = 2 layers** to prevent over-smoothing, preserving localized heterogeneity of void signatures.
- Output clamped to preserve gradient flow at probability boundaries:

$$\hat{y}_v = \text{Clamp}\!\left(\text{Sigmoid}(W_{out}\, h_v^{(K)}),\, \delta,\, 1 - \delta\right)$$

  where $\delta = 10^{-5}$, ensuring predictions never reach exactly 0 or 1.
- Optimized by minimizing **Binary Cross Entropy (BCE) loss** over the labeled node set.

### `causal_sandbox.py` — Causal Refutation Sandbox

Powered by **DoWhy**, this module executes Pearl's do-calculus. It instantiates the SCM, injects the confounder set Z = {population, expected_cases} via the backdoor criterion, and computes the ATE of simulated diagnostic resource allocation. A validated **ATE > 0.25** confirms the architecture's internal causal consistency.

---

## 3. Empirical Results

### 3.1 Hyperparameter Optimization

| Parameter | Candidates Tested | Optimal Value | Selection Criterion |
|---|---|---|---|
| Spatial radius (ϵ) | 5, 15, 30, 50 km | **5.0 km** | Min. validation BCE loss |
| Void threshold (τ) | 0.20, 0.30, 0.40 | **0.30** | Optimal signal-to-noise ratio |
| **Validation BCE Loss** | — | **0.0373** | — |

The optimal ϵ = 5.0 km — significantly tighter than the baseline assumption of 15.0 km — reveals that surveillance interdependency operates at a highly localized geographic scale, smaller than conventional county or district boundaries.

### 3.2 Ablation Study

| Model | Architecture | Final Validation BCE Loss |
|---|---|---|
| Baseline MLP | Tabular, no spatial topology | 0.0149 |
| **TCGNN (ours)** | Spatial graph + message passing | **0.0005** |
| **Improvement** | | **96.63%** |

The TCGNN's exceptionally low loss is partially attributable to the noise-free nature of the synthetic validation environment. This is acknowledged as a limitation; performance on real-world KHIS data, which carries the chaotic noise of field collection, is the next validation frontier.

---

## 4. Installation

> This architecture requires a **Linux or cloud-based environment** (e.g., GitHub Codespaces) configured for PyTorch Geometric dependencies.

```bash
pip install torch pandas numpy scikit-learn networkx matplotlib dowhy torch-geometric
```

---

## 5. Execution Pipeline

### Step 1 — Initialize Synthetic Baseline

Generates a mathematically sound synthetic dataset and builds the required directory structure (`data/raw/`, `data/processed/`, `models/`):

```bash
python main_pipeline.py
```

### Step 2 — Train the Neural Network

Executes the 200-epoch training loop with Z-score normalization and bounded gradient output:

```bash
python training_loop.py
```

Trained weights are serialized to `models/tcgnn_trained_weights.pth`. Expected convergence: validation BCE loss ≈ **0.0004–0.0005** on synthetic data.

### Step 3 — Causal Refutation

Stress-tests the trained network and computes the ATE of simulated resource interventions:

```bash
python causal_sandbox.py
```

Expected output: **ATE > 0.25**, confirming internal causal consistency.

---

## 6. Processing Real-World Data

To transition from synthetic validation to field deployment (e.g., KHIS or KEMRI):

1. Place raw tabular data into `data/raw/`.
2. Open `data_adapter.py` and modify `self.column_mapping` to match your dataset's column schema.
3. Run the adapter:

```bash
python data_adapter.py
```

The adapter will standardize the schema, execute numeric typing, apply one-hot encoding to categorical tiers, filter invalid GPS nodes, and output a validated matrix to `data/processed/`.

---

## 7. Project Structure

```
.
├── data/
│   ├── raw/                        # Raw input data (KHIS, KEMRI, or synthetic)
│   └── processed/                  # Adapter-validated matrices
├── models/
│   └── tcgnn_trained_weights.pth   # Serialized trained weights
├── backend/
│   ├── graph_engine.py             # Topological graph construction & ϵ sweep
│   └── tcgnn_model.py              # GNN architecture (K=2, ReLU, clamp)
├── data_adapter.py                 # Dual-sanitation ingestion pipeline
├── training_loop.py                # BCE-optimized training with Z-score normalization
├── causal_sandbox.py               # DoWhy backdoor-criterion causal refutation
└── main_pipeline.py                # Synthetic data generation & initialization
```

---

## 8. Limitations and Future Work

- **Synthetic validation only.** All reported metrics are derived from simulated data lacking real-world noise. External validity requires ingestion of real KHIS/KEMRI field data.
- **Euclidean distance baseline.** Geodesic road-network or travel-time distance would provide superior epidemiological fidelity for edge construction.
- **Simulated counterfactuals.** The causal ATE is computed in a simulated environment; it establishes internal structural consistency but does not guarantee field causal validity without real-world randomized or quasi-experimental data.

The immediate next phase is deployment on raw KHIS data to translate internal causal logic into real-world public health policy impact.

---

## 9. References

1. World Health Organization. (2020). *Data quality review: A toolkit for facility data quality assessment. Module 1: Framework and metrics.* Geneva: WHO. https://cdn.who.int/media/docs/default-source/world-health-data-platform/rhis-modules/dqa-module-1-framework-and-metrics.pdf

2. Lawson, A. B. (2013). *Statistical Methods in Spatial Epidemiology.* John Wiley & Sons. https://doi.org/10.1002/9780470035771

3. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction.* American Mathematical Society. https://doi.org/10.1090/mbk/069

4. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511803161

5. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/1609.02907

6. Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph convolutional networks for semi-supervised learning. *Proceedings of the AAAI Conference on Artificial Intelligence.* https://doi.org/10.1609/aaai.v32i1.11604

---

## 10. Declarations

**Funding:** This research received no external funding. It was conducted as a personal independent initiative.

**Conflicts of Interest:** The author declares no competing interests.

**Data Availability:** The core architecture and simulation code are publicly available at https://github.com/Grolds-Code/tcgnn-surveillance. Real-world field data (KHIS) was not utilized in this foundational validation phase.

**Author Contributions:** G.O.M. is the sole author. He conceived the architecture, wrote the code, conducted the experiments, and authored the manuscript.

---

*For questions, collaborations, or deployment support: gmotieno@jooust.ac.ke*