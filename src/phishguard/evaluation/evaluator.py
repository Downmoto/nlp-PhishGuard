"""Model evaluation: metrics, confusion matrix, ROC-AUC, attention extraction."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    trainer: Any,
    test_dataset: Any,
    figures_dir: str | os.PathLike,
) -> dict[str, Any]:
    """Run full evaluation suite on *test_dataset*.

    Parameters
    ----------
    trainer:
        A fitted :class:`~transformers.Trainer`.
    test_dataset:
        :class:`~phishguard.model.dataset.PhishingDataset` test split.
    figures_dir:
        Directory where plots are saved.

    Returns
    -------
    dict
        Keys: ``report`` (str), ``auc`` (float), ``cm`` (ndarray).
    """
    out_figs = Path(figures_dir)
    out_figs.mkdir(parents=True, exist_ok=True)

    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    labels = preds_output.label_ids

    probs = _softmax(logits)
    preds = np.argmax(logits, axis=-1)

    report = classification_report(labels, preds, target_names=["Legitimate", "Phishing"])
    cm = confusion_matrix(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])

    print("[evaluator] Classification Report:\n", report)
    print(f"[evaluator] ROC-AUC: {auc:.4f}")

    _plot_confusion_matrix(cm, out_figs)
    _plot_roc_curve(labels, probs[:, 1], auc, out_figs)

    return {"report": report, "auc": auc, "cm": cm}


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _plot_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("[evaluator] Saved confusion_matrix.png")


def _plot_roc_curve(labels: np.ndarray, probs: np.ndarray, auc: float, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    print("[evaluator] Saved roc_curve.png")


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_scores(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int = 512,
) -> tuple[list[str], list[float]]:
    """Return per-token attention scores from the [CLS] token's last layer.

    Parameters
    ----------
    model:
        The raw ``BertForSequenceClassification`` model (with ``output_attentions=True``).
    tokenizer:
        Corresponding tokenizer.
    text:
        Raw input email text.
    max_length:
        Truncation length.

    Returns
    -------
    tuple
        ``(tokens, scores)`` where ``scores`` are mean attention weights from
        the [CLS] token across heads in the last attention layer.
    """
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, heads, seq, seq) per layer — take last layer
    last_attention = outputs.attentions[-1]  # (1, heads, seq, seq)
    # Average over heads, take [CLS] row (index 0)
    cls_attention = last_attention[0].mean(dim=0)[0]  # (seq,)
    scores = cls_attention.cpu().numpy().tolist()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, scores


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_evaluation_report(
    results: dict[str, Any],
    output_path: str | os.PathLike,
) -> None:
    """Write a Markdown evaluation report to *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = results["cm"]
    tn, fp, fn, tp = cm.ravel()

    content = f"""\
# PhishGuard Evaluation Report

## Classification Report

```
{results['report']}
```

## ROC-AUC

| Metric | Value |
|--------|-------|
| ROC-AUC | {results['auc']:.4f} |

## Confusion Matrix

|  | Predicted Legitimate | Predicted Phishing |
|--|--|--|
| **Actual Legitimate** | {tn} | {fp} |
| **Actual Phishing** | {fn} | {tp} |

## Figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
"""
    path.write_text(content, encoding="utf-8")
    print(f"[evaluator] Report written → {path}")
