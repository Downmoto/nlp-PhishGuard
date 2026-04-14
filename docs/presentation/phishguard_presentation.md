---
marp: true
theme: default
paginate: true
size: 16:9
title: PhishGuard: Fine-Tuned BERT for Phishing Email Detection
description: Final project presentation for AIG230
style: |
  section {
    font-family: 'Aptos', 'Segoe UI', sans-serif;
    background: linear-gradient(180deg, #f7f8f4 0%, #edf1e8 100%);
    color: #1f2a1f;
    padding: 48px;
  }
  h1, h2, h3 {
    color: #14342b;
    font-weight: 700;
  }
  h1 {
    font-size: 2.1rem;
    margin-bottom: 0.3rem;
  }
  h2 {
    font-size: 1.7rem;
    margin-bottom: 0.4rem;
  }
  strong {
    color: #7a2e1f;
  }
  code {
    background: #e7ece4;
    color: #163329;
    padding: 0.1em 0.3em;
    border-radius: 0.25em;
  }
  table {
    font-size: 0.82rem;
  }
  img {
    background: white;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .small {
    font-size: 0.85rem;
  }
  .center {
    text-align: center;
  }
---

# PhishGuard
## Fine-Tuned BERT for Phishing Email Detection

Arad Fadaei; Mahboobeh Yasini  
AIG230 Final Project

---

## Problem

- Phishing remains a major email-based social engineering threat.
- Modern phishing emails often look professional and mimic normal workplace communication.
- Rule-based filters and shallow keyword matching struggle when attackers change wording or tone.
- The task is fundamentally **semantic**: the model must learn intent, urgency, and impersonation patterns.

---

## Project Objectives

- Build an end-to-end phishing email detection pipeline.
- Fine-tune a transformer model for binary classification of email text.
- Preserve some interpretability instead of returning only a hard label.
- Provide both a CLI workflow and a lightweight web interface.

Success meant strong predictive performance and a clean engineering pipeline from raw data to deployment.

---

## Technical Solution Overview

1. Load labeled email data
2. Preprocess noisy raw email text
3. Split into train, validation, and test sets
4. Tokenize with BERT tokenizer
5. Fine-tune `bert-base-uncased`
6. Evaluate the best checkpoint
7. Serve inference through CLI and Gradio

Core stack: **PyTorch**, **Hugging Face Transformers**, **pandas**, **scikit-learn**, **Gradio**

---

## Dataset and Splits

- Repository includes seven raw corpora under `data/raw/`
- Training pipeline uses `phishing_email.csv` as the configured primary file
- Processed dataset size: **82,485** emails
- Stratified split ratio: **70 / 15 / 15**

| Split | Rows | Legitimate | Phishing |
| --- | ---: | ---: | ---: |
| Train | 57,739 | 27,716 | 30,023 |
| Validation | 12,372 | 5,939 | 6,433 |
| Test | 12,374 | 5,940 | 6,434 |

---

## Preprocessing Pipeline

- Strip HTML using BeautifulSoup with regex fallback
- Replace URLs with `[URL]`
- Mask email addresses as `[EMAIL]`
- Mask phone numbers as `[PHONE]`
- Normalize repeated whitespace

Why this mattered:

- reduces noise
- preserves the semantic structure of the message
- avoids memorizing raw identifiers that do not generalize

---

## Model and Training Strategy

- Base model: `bert-base-uncased`
- Task: binary sequence classification
- Max sequence length: **256** tokens
- Training framework: Hugging Face `Trainer`
- Dataset class pre-tokenizes text once at initialization
- Best checkpoint selected by **macro F1**

Why this setup worked:
- transfer learning reduced the amount of task-specific training needed
- pre-tokenization removed repeated CPU work across epochs
- macro F1 gave a better model-selection signal than accuracy alone

---
## Evaluation Results

| Metric | Value |
| --- | ---: |
| Accuracy | **99.48%** |
| Macro F1 | **0.9948** |
| ROC-AUC | **0.9998** |
| False Positives | 16 |
| False Negatives | 49 |

These results were measured on the held-out **12,374** test split.

---

## Confusion Matrix

![bg left](../../reports/figures/confusion_matrix.png)

Small false-positive count is especially valuable because excessive blocking of legitimate email reduces trust in a security tool.

---

## Explainability and Web Interface

- `PhishGuardPredictor` returns label, label ID, confidence, and token scores
- Token scores come from final-layer attention focused on `[CLS]`
- Gradio app allows users to paste email text and inspect the result interactively

Web app outputs:

- predicted verdict
- confidence score
- top attention-weighted tokens

---

## Demo
---

## Lessons Learned

- Data quality mattered as much as model choice.
- URL and PII normalization improved signal density.
- Pre-tokenization reduced repeated CPU work during training.
- FP16, pinned memory, and multiple data-loader workers improved throughput.
- Macro F1 was a better checkpoint metric than raw accuracy.
- Lightweight explainability made the system easier to inspect and trust.

---

## Conclusion

- PhishGuard delivered a complete NLP pipeline
- The project combined preprocessing, stratified splitting, transformer fine-tuning, evaluation, inference, and a web app.
- Final performance was strong: **99.48% accuracy** and **0.9948 macro F1**.
- The result is both technically effective and practical to demonstrate.
