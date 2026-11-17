# CED-III Integrative Task 2: Sentiment Analysis on GoEmotions

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)](#)

> **Course:** Computación y Estructuras Discretas III (CED-III) — 2026-2<br>
> **Institution:** Universidad Icesi<br>
> **Unit:** Unit 3 — Turing Machines and Applications<br>
> **Team:** `acabados`<br>
> **Repository:** [Juanes3333/CED-III-IT2-acabados](https://github.com/Juanes3333/CED-III-IT2-acabados)

---

## 1. Project Overview

This project implements a complete **sentiment analysis** pipeline on Reddit comments using the **GoEmotions** dataset. We benchmark and compare four distinct model families under a rigorous experimental protocol:

1. **Baseline Model:** `DummyClassifier` (majority class and uniform baselines).
2. **Dense Neural Network:** Multi-Layer Perceptron (MLP) trained on TF-IDF feature representations.
3. **Vanilla Recurrent Neural Network (RNN):** Sequential model processing word embedding sequences with feedback loops.
4. **Long Short-Term Memory (LSTM):** Gated recurrent architecture designed to mitigate vanishing gradients and capture longer dependencies.
5. **Transformer Extension:** Pretrained sequence classifier (`distilbert-base-uncased`) leveraging multi-head self-attention.

Beyond empirical performance, this work connects neural architectures with theoretical concepts from **Turing Machines (TM)**, exploring memory mechanisms, sequential state transitions, bounded computation, and generalization boundaries.

---

## 2. Team Members

| Member | Role / Primary Focus | GitHub User |
|---|---|---|
| **Juan Esteban Restrepo (R)** | Data pipeline, Dense NN, Transformer, Turing Machine analysis | [`@Juanes3333`](https://github.com/Juanes3333) |
| **Daniel Varela (V)** | Environment setup, training utilities, EDA, Vanilla RNN, final comparison | [`@Danielvarelavictoria`](https://github.com/Danielvarelavictoria) |
| **Samuel Granda (G)** | Evaluation & visualization, Baseline model, LSTM, report scaffolding & presentation | [`@SamuelGranda`](https://github.com/SamuelGranda) |

---

## 3. Repository Structure

The repository conforms strictly to the directory structure defined in the course specification:

```
CED-III-IT2-acabados/
├── README.md                           # Project overview, team, and run instructions
├── requirements.txt                    # Pinned Python package dependencies (pip)
├── environment.yml                     # Conda environment specification (Python 3.11)
├── data/
│   ├── raw/                            # Raw GoEmotions TSV files and label mappings
│   ├── processed/                      # Balanced (1,000 samples) and split datasets
│   └── README.md                       # Dataset documentation and schema description
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis & preprocessing justification
│   ├── 02_baseline_model.ipynb         # DummyClassifier baseline evaluation
│   ├── 03_dense_rnn_lstm.ipynb         # Dense NN, Vanilla RNN, and LSTM training & tuning
│   └── 04_transformer_extension.ipynb  # DistilBERT fine-tuning and comparative evaluation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Data loading, mapping, cleaning, and tokenization
│   ├── models.py                       # Architecture builders (Dense, RNN, LSTM, Transformer)
│   ├── train.py                        # Seed management, grid search, and training runners
│   ├── evaluate.py                     # Performance metrics (accuracy, kappa, F1) & bootstrap CIs
│   └── visualize.py                    # Publication-quality plotting utilities
├── outputs/
│   ├── figures/                        # Generated figures and learning curves (150 DPI)
│   ├── metrics/                        # Metrics CSV files and grid search results
│   └── saved_models/                   # Exported model weights and pipeline artifacts
├── docs/
│   ├── report.pdf                      # Final IEEE 2-4 page academic report
│   ├── presentation.pdf                # 10-minute technical presentation slides
│   └── references.bib                  # BibTeX bibliography
├── prompts/
│   ├── prompt_logs.txt                 # Consolidated GenAI prompt history
│   └── AIGen_Interactions.md           # Critical AI audit and refinement log
└── logs/
    ├── training_logs.txt               # Aggregated execution and training logs
    └── experiment_notes.md             # Experimental decisions and notes
```

---

## 4. Environment Setup

The project requires **Python 3.11** to ensure compatibility across all deep learning frameworks (TensorFlow 2.15, Keras 2.15, Transformers).

### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/Juanes3333/CED-III-IT2-acabados.git
cd CED-III-IT2-acabados

# Create and activate the conda environment
conda env create -f environment.yml
conda activate goemotions-it2

# Download required NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Option B: Using Pip in a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 5. Execution Workflow

All experiments are reproducible using a fixed random seed (`SEED = 1234`). Follow the steps below in sequential order:

1. **Data Ingestion & Verification:**
   Download the GoEmotions dataset into `data/raw/` as outlined in [`data/README.md`](data/README.md).

2. **Exploratory Data Analysis (EDA):**
   Execute `notebooks/01_eda.ipynb` to verify class distribution, inspect sentence lengths, analyze token frequencies, and compute `max_len` / vocabulary parameters saved to `outputs/metrics/eda_summary.json`.

3. **Baseline Model:**
   Execute `notebooks/02_baseline_model.ipynb` to establish the lower-bound classification metrics using `DummyClassifier`.

4. **Dense, RNN & LSTM Benchmarks:**
   Execute `notebooks/03_dense_rnn_lstm.ipynb` to train, tune hyperparameters via cross-validation / validation splits, and evaluate recurrent architectures.

5. **Transformer Fine-Tuning:**
   Execute `notebooks/04_transformer_extension.ipynb` to evaluate `distilbert-base-uncased` and perform comparative error analysis.

<!-- TODO(IT2-32): Add command-line execution instructions for end-to-end headless runs via nbconvert -->

---

## 6. Experimental Standards & Reproducibility

- **Fixed Seed:** All stochastic operations (data splits, model weight initializations, grid searches, bootstrap resampling) enforce `SEED = 1234`.
- **No Data Leakage:** Tokenizers and feature extractors (`TfidfVectorizer`, `TextVectorization`) are fitted strictly on the 800-sample training partition. The 200-sample test set is evaluated once per model for final benchmarking.
- **Evaluation Metrics:** Every model is evaluated on both training and test partitions using Accuracy, Precision, Recall, F1-score (positive and macro), and **Cohen's kappa**, accompanied by confusion matrices and bootstrap 95% confidence intervals.

<!-- TODO(IT2-32): Update results summary table with finalized test metrics -->

---

## 7. AI Collaboration Disclosure

In accordance with course guidelines (**GenAI Level 3: Collaboration with GenAI**):
- Generative AI tools (Antigravity / LLMs) are used collaboratively to assist in code scaffolding, documentation, and conceptual validation.
- All AI-generated suggestions are critically audited, refined, and verified against course materials, ensuring theoretical correctness and preventing slide pitfalls (e.g., correcting scorer signatures, preserving negation tokens in stopword removal, and stratifying train/test splits).
- All prompt interactions and critical evaluations are systematically logged in the `prompts/` directory.
