# TCGNN-Surveillance: The Geometry of Silence

> A Topological Causal Graph Neural Network for identifying, mapping, and quantifying structural voids in public health surveillance systems.

---

## Overview

TCGNN-Surveillance operates at the intersection of **computational topology**, **biostatistics**, and **causal inference**. Rather than relying on standard correlation analysis, this framework maps spatial relationships between epidemiological nodes (health facilities) and mathematically simulates the causal impact of targeted resource allocation.

Every neural prediction is subject to causal refutation — ensuring that identified structural voids represent actual physical gaps in public health surveillance, not algorithmic artefacts.

---

## Core Architecture

The framework decouples data ingestion from neural computation for enterprise-grade resilience and mathematical stability.

### `data_adapter.py` — Data Ingestion Interface
A strict sanitation pipeline built to ingest raw field data (e.g., KHIS or KEMRI databases). It standardises variable schemas, handles type casting, and filters out topologically invalid nodes (facilities missing GPS coordinates) before computation begins.

### `backend/graph_engine.py` — Topological Engine
Converts normalised tabular data into spatial graphs. Edges are defined by a Euclidean distance threshold (`radius`) between geographic coordinates, explicitly mapping the underlying surveillance network topology.

### `backend/tcgnn_model.py` & `training_loop.py` — Neural Compute Core
A custom **PyTorch Geometric (GNN)** architecture. To prevent gradient saturation and flatlining caused by large raw epidemiological integers, the engine enforces strict Z-score standardisation and a mathematical output clamp during backpropagation.

### `causal_sandbox.py` — Causal Refutation Sandbox
Powered by **DoWhy**, this module executes Pearl's do-calculus. It isolates the **Average Treatment Effect (ATE)** to simulate the precise impact of structural interventions, ensuring network predictions are rooted in causal reality rather than observational bias.

---

## Installation

> This architecture requires a **Linux or cloud-based environment** (e.g., GitHub Codespaces) configured for heavy mathematical computation.

Install all required statistical and deep learning dependencies:

```bash
pip install torch pandas numpy scikit-learn networkx matplotlib dowhy
```

---

## Execution Pipeline

### 1. Data Initialisation

If real-world reporting data is not yet available, generate a mathematically sound synthetic baseline. This also builds the required directory structure (`data/raw/`, `data/processed/`, `models/`):

```bash
python main_pipeline.py
```

### 2. Neural Network Training

Execute the training loop. The engine applies standard deviation scaling and bounded gradients to achieve optimal convergence.

```bash
python training_loop.py
```

> A standard 200-epoch cycle on synthetic data minimises loss to approximately **0.0004**. Trained weights are serialised to `models/tcgnn_trained_weights.pth`.

### 3. Causal Impact Simulation

Run the causal inference module to stress-test the trained network and calculate the ATE of simulated resource interventions:

```bash
python causal_sandbox.py
```

> The model should output a validated **ATE > 0.25**, confirming causal inference capability.

---

## Processing Real-World Data (KHIS / KEMRI)

To transition the engine from synthetic validation to real-world deployment:

1. Place your raw tabular dataset into `data/raw/`.
2. Open `data_adapter.py` and modify the `self.column_mapping` dictionary to match the proprietary column headers of your dataset.
3. Run the adapter:

```bash
python data_adapter.py
```

The adapter will format the schema, execute numeric typing, and output a validated matrix to `data/processed/` for the GNN to consume.

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw input data (KHIS, KEMRI, or synthetic)
│   └── processed/            # Validated, adapter-processed matrices
├── models/
│   └── tcgnn_trained_weights.pth
├── backend/
│   ├── graph_engine.py       # Topological graph construction
│   └── tcgnn_model.py        # GNN architecture
├── data_adapter.py           # Data ingestion & sanitation pipeline
├── training_loop.py          # Model training with Z-score normalisation
├── causal_sandbox.py         # DoWhy causal refutation module
└── main_pipeline.py          # Synthetic data generation & initialisation
```

---

## Methodological Integrity

TCGNN-Surveillance is designed with strict adherence to **virtue epistemology in AI**. Machine learning is not treated as a black box — every neural prediction is subject to causal refutation via DoWhy. This ensures that identified surveillance voids correspond to genuine physical gaps in public health infrastructure, rather than statistical noise or model hallucination.

---

## Authorship & Affiliations

**Architect:** Grold Otieno Mboya

| Affiliation | Role |
|---|---|
| NIXQUE LTD | CTO |
| Fatima Institute | Research Fellow |


Developed to advance transdisciplinary inquiry in **computational epidemiology** and **algorithmic robustness**.

---

*For questions, collaborations, or deployment support, please contact: groldotieno97@gmail.com.*