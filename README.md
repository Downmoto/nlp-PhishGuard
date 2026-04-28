# PhishGuard

A fine-tuned BERT-based NLP pipeline for binary phishing email detection. PhishGuard fine-tunes `bert-base-uncased` to classify email text as **Legitimate (0)** or **Phishing (1)** with 99.48% accuracy on a held-out test set.

**Authors:** Arad Fadaei & Mahboobeh Yasini — AIG230 Final Project

Demo Link: https://youtu.be/J-1gts2d3Aw
---

## Final Report
The final report can be found in [docs/report/report.pdf](https://github.com/Downmoto/nlp-phishguard/blob/main/docs/report/report.pdf)

## Table of Contents

- [Overview](#overview)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts](#scripts)
  - [download_data.py](#download_datapy)
  - [train.py](#trainpy)
  - [evaluate.py](#evaluatepy)
  - [predict.py](#predictpy)
  - [run_eda.py](#run_edapy)
  - [run_app.py](#run_apppy)
  - [push_to_hub.py](#push_to_hubpy)
- [Running Tests](#running-tests)
- [Datasets](#datasets)

---

## Overview

PhishGuard implements a complete machine learning pipeline:

1. **Data ingestion** — Downloads and combines 7 email datasets (Enron, CEAS_08, Ling, Nazario, etc.)
2. **Preprocessing** — Strips HTML, replaces URLs with `[URL]`, removes PII (emails/phones)
3. **Training** — Fine-tunes `bert-base-uncased` via HuggingFace `Trainer` with early stopping and FP16 support
4. **Evaluation** — Classification report, confusion matrix, and ROC-AUC curve
5. **Inference** — CLI prediction with per-token attention-based explainability
6. **Web UI** — Gradio interface for interactive testing

---

## Performance

Evaluated on a stratified 15% test split (n = 12,374):

| Metric | Value |
|---|---|
| Accuracy | **99.48%** |
| Macro F1 | **0.9948** |
| ROC-AUC | **0.9998** |
| False Positives | 16 |
| False Negatives | 49 |

---

## Project Structure

```
nlp-PhishGuard/
├── config.yaml                  # Central hyperparameter and path configuration
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/                     # Original CSV datasets
│   └── processed/               # Train/val/test Parquet splits
├── models/
│   ├── best/                    # Best checkpoint (used for inference)
│   └── checkpoint-*/            # Training checkpoints
├── reports/figures/             # Generated plots (confusion matrix, ROC, EDA)
├── docs/                        # Proposal, progress notes, evaluation report
└── src/
    ├── phishguard/              # Main package
    │   ├── config.py
    │   ├── data/                # Downloading, loading, preprocessing
    │   ├── eda/                 # Exploratory data analysis
    │   ├── model/               # BERT classifier and dataset wrappers
    │   ├── training/            # HuggingFace Trainer wrapper
    │   ├── evaluation/          # Metrics, plots, report generation
    │   ├── inference/           # Predictor with attention explainability
    │   └── web/                 # Gradio app
    ├── scripts/                 # CLI entry points (see Scripts section)
    └── tests/                   # pytest test suite
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Downmoto/nlp-PhishGuard.git
cd nlp-PhishGuard

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> **GPU note:** A CUDA-capable GPU is strongly recommended for training. The pipeline uses FP16 mixed-precision by default. CPU inference is supported.

---

## Configuration

All hyperparameters and file paths are controlled by [`config.yaml`](config.yaml):

```yaml
model:
  model_name: "bert-base-uncased"
  num_labels: 2
  max_seq_length: 256

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
  warmup_steps: 500
  weight_decay: 0.01
  fp16: true
  metric_for_best_model: "f1"
  save_total_limit: 2

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

output:
  output_dir: "models"
  best_checkpoint_dir: "models/best"
  figures_dir: "reports/figures"
  reports_dir: "docs"
```

---

## Scripts

All scripts are located in [`src/scripts/`](src/scripts/) and are run from the project root with the virtual environment activated.

---

### `download_data.py`

Downloads the primary phishing email dataset from Kaggle into `data/raw/`.

**Usage:**

```bash
python src/scripts/download_data.py
```

**What it does:**
- Fetches the `naserabdullahalam/phishing-email-dataset` dataset via `kagglehub` (no Kaggle API key required)
- Copies all downloaded CSV files into `data/raw/`

> Run this once before training if you don't already have raw data in `data/raw/`.

---

### `train.py`

End-to-end training pipeline: preprocesses raw data, splits it, fine-tunes BERT, and saves the best checkpoint.

**Usage:**

```bash
python src/scripts/train.py
```

**What it does:**
1. Loads all CSVs from `data/raw/`, auto-detects text/label columns, normalises labels (`spam→1`, `ham→0`)
2. Applies text preprocessing (HTML stripping, URL/PII replacement)
3. Performs a stratified 70/15/15 train/val/test split and saves Parquet files to `data/processed/`
4. Tokenises with `bert-base-uncased` (max length 256)
5. Trains with HuggingFace `Trainer` using early stopping (patience=2), best model tracked by macro F1
6. Automatically resumes from the latest `checkpoint-*` folder if one exists
7. Saves the best model and tokenizer to `models/best/`

**Estimated training time:** ~2–3 hours on a single GPU (NVIDIA RTX series).

---

### `evaluate.py`

Runs the saved best checkpoint against the held-out test split and produces a full evaluation report.

**Usage:**

```bash
python src/scripts/evaluate.py
```

**What it does:**
- Loads the model from `models/best/` and the test split from `data/processed/`
- Computes precision, recall, F1, and accuracy per class
- Generates and saves:
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/roc_curve.png`
- Writes `docs/evaluation_report.md` with all metrics and figure references

> Requires a trained model in `models/best/` and processed data in `data/processed/`. Run `train.py` first.

---

### `predict.py`

CLI inference tool for classifying a single email as phishing or legitimate.

**Usage:**

```bash
# Classify text passed directly on the command line
python src/scripts/predict.py --text "Congratulations! You have won a $1000 gift card. Click here to claim."

# Classify the contents of a text file
python src/scripts/predict.py --file path/to/email.txt
```

**Arguments:**

| Argument | Description |
|---|---|
| `--text TEXT` | Raw email text string to classify |
| `--file PATH` | Path to a `.txt` file containing the email body |

**Output:**

```
Verdict    : PHISHING
Confidence : 99.87%
Top Tokens : click, gift, claim, free, congratulations, ...
```

Prints the predicted label, model confidence, and the top-10 attention-weighted tokens for explainability.

> Requires `models/best/` to be populated. Works on CPU without a GPU.

---

### `run_eda.py`

Generates exploratory data analysis plots from the training split.

**Usage:**

```bash
python src/scripts/run_eda.py
```

**What it does:**
- Loads `data/processed/train.parquet`
- Generates and saves the following figures to `reports/figures/`:
  - Class distribution bar chart
  - Email length histograms (by class)
  - Top-20 most frequent words per class
  - Word clouds (legitimate vs phishing)
  - URL presence rate by class

> Requires processed data in `data/processed/`. Run `train.py` (which also preprocesses) or `download_data.py` + the data loading step first.

---

### `run_app.py`

Launches the Gradio web interface for interactive phishing detection.

**Usage:**

```bash
python src/scripts/run_app.py
```

**What it does:**
- Starts a Gradio Blocks web app on `http://0.0.0.0:7860`
- Accepts email text via a text box
- Returns:
  - **Verdict** — Legitimate or Phishing
  - **Confidence** — Model probability score (%)
  - **Top Attention Tokens** — Top-20 tokens the model focused on

The predictor is loaded lazily on the first request to reduce startup time.

> Requires `models/best/` to be populated. Access the app at `http://localhost:7860` in your browser.

---

### `push_to_hub.py`

Publishes the best checkpoint and tokenizer to the Hugging Face Hub with an auto-generated model card.

**Usage:**

```bash
python src/scripts/push_to_hub.py --repo-id your-username/your-model-name
```

**Arguments:**

| Argument | Description |
|---|---|
| `--repo-id REPO_ID` | Target Hugging Face Hub repository in `owner/model-name` format (required) |

**What it does:**
- Loads the model and tokenizer from `models/best/`
- Pushes weights, tokenizer files, and a generated model card to the specified Hub repository
- Creates the repository if it does not already exist

> Requires a valid Hugging Face access token. Log in first with `huggingface-cli login`.

---

## Running Tests

```bash
pytest
```

The test suite covers dataset loading, preprocessing, and predictor logic. Configuration is in [`pytest.ini`](pytest.ini); tests live in [`src/tests/`](src/tests/).

---

## Datasets

The following datasets are combined for training:

| Dataset | Description |
|---|---|
| `CEAS_08.csv` | CEAS 2008 spam/phishing competition corpus |
| `Enron.csv` | Enron email corpus (ham + spam) |
| `Ling.csv` | Ling-Spam dataset |
| `Nazario.csv` | Nazario phishing corpus |
| `Nigerian_Fraud.csv` | Nigerian advance-fee fraud emails |
| `phishing_email.csv` | General phishing email collection |
| `SpamAssasin.csv` | SpamAssassin public mail corpus |

Primary download source: [`naserabdullahalam/phishing-email-dataset`](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) on Kaggle.
