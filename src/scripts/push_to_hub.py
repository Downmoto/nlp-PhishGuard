"""Push the best PhishGuard checkpoint to the Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_CARD = """\
---
language: en
license: mit
tags:
  - bert
  - text-classification
  - phishing
  - email
  - cybersecurity
datasets:
  - custom
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
---

# PhishGuard — Phishing Email Detector

Fine-tuned [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) for
binary phishing/legitimate email classification.

## Model description

`BertForSequenceClassification` with two output labels:
- `0` → Legitimate
- `1` → Phishing

## Training data

Aggregated from seven public email datasets: CEAS-08, Enron, Ling, Nazario,
Nigerian Fraud, Phishing Email, and SpamAssassin.

## Evaluation results

| Metric | Value |
|--------|-------|
| Accuracy | 0.9948 |
| Macro F1 | 0.9948 |
| ROC-AUC  | 0.9998 |

### Confusion matrix (test set, n = 12 374)

|  | Predicted Legitimate | Predicted Phishing |
|--|--|----|
| **Actual Legitimate** | 5 924 | 16 |
| **Actual Phishing** | 49 | 6 385 |

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="<your-username>/phishguard")
result = classifier("Congratulations! You've won a $1,000 gift card. Click here to claim.")
print(result)  # [{'label': 'Phishing', 'score': 0.999...}]
```

## Limitations

Trained on English-language email text only. Performance on non-English or
heavily obfuscated content may degrade.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push PhishGuard model to HF Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo in the form <username>/<model-name>, e.g. johndoe/phishguard",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="models/best",
        help="Path to the checkpoint directory (default: models/best)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face write token. If omitted, uses the cached login token "
             "(run `huggingface-cli login` first).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    print(f"Loading model and tokenizer from {ckpt} ...")
    model = BertForSequenceClassification.from_pretrained(str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    # Attach human-readable label names so the pipeline maps integers → strings.
    model.config.id2label = {0: "Legitimate", 1: "Phishing"}
    model.config.label2id = {"Legitimate": 0, "Phishing": 1}

    # Ensure the repo exists before uploading.
    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Pushing model to https://huggingface.co/{args.repo_id} ...")
    model.push_to_hub(args.repo_id, token=args.token)
    tokenizer.push_to_hub(args.repo_id, token=args.token)

    # Upload model card.
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
        commit_message="Add model card",
    )

    print(f"\nDone! Model available at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
