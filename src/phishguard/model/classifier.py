"""BERT-based sequence classifier wrapper."""
from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoConfig, BertForSequenceClassification


class PhishGuardClassifier:
    """Thin wrapper around :class:`~transformers.BertForSequenceClassification`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default ``"bert-base-uncased"``).
    num_labels:
        Number of output classes (2 for binary phishing detection).
    checkpoint_dir:
        Optional path to a saved checkpoint.  When provided the model weights
        are loaded from disk instead of being downloaded.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        checkpoint_dir: str | os.PathLike | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels

        if checkpoint_dir is not None:
            ckpt = Path(checkpoint_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                str(ckpt), num_labels=num_labels
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                # Keep training fast; attention maps can be requested explicitly at inference time.
                output_attentions=False,
            )
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, config=config
            )

    def save(self, output_dir: str | os.PathLike) -> None:
        """Persist model weights and config to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        print(f"[classifier] Model saved → {out}")
