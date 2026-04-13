"""Inference engine: load checkpoint and predict on raw text."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification


class PhishGuardPredictor:
    """Load a trained PhishGuard checkpoint and expose a ``predict`` method.

    Parameters
    ----------
    checkpoint_dir:
        Path to the directory saved by :func:`~phishguard.training.trainer.train`.
    max_length:
        Truncation length matching training.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        ckpt = Path(checkpoint_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        self._model = BertForSequenceClassification.from_pretrained(
            str(ckpt), output_attentions=True
        )
        self._max_length = max_length

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, Any]:
        """Classify a single email text.

        Parameters
        ----------
        text:
            Raw (unprocessed) email body.

        Returns
        -------
        dict
            ``label`` (str), ``label_id`` (int), ``confidence`` (float),
            ``token_scores`` (list[dict] with keys ``token`` and ``score``).
        """
        from phishguard.data.preprocessor import preprocess_text

        text = preprocess_text(text)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._max_length,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

        logits = outputs.logits.cpu().numpy()[0]
        probs = self._softmax(logits)
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        label = "phishing" if label_id == 1 else "legitimate"

        # Token-level attention scores from [CLS] in last layer
        last_attention = outputs.attentions[-1]
        cls_attention = last_attention[0].mean(dim=0)[0].cpu().numpy()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        token_scores = [
            {"token": tok, "score": float(score)}
            for tok, score in zip(tokens, cls_attention)
        ]

        return {
            "label": label,
            "label_id": label_id,
            "confidence": confidence,
            "token_scores": token_scores,
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()
