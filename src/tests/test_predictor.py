"""Unit tests for phishguard.inference.predictor (CPU, tiny model stub)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.phishguard.inference.predictor import PhishGuardPredictor
import torch


# ---------------------------------------------------------------------------
# Helpers to build mock objects without loading a real checkpoint
# ---------------------------------------------------------------------------

def _make_mock_model(logits: list[float]) -> MagicMock:
    """Return a mock BertForSequenceClassification."""
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model

    logit_tensor = torch.tensor([logits])
    # Build fake attention: 1 layer, 1 head, seq_len=5
    seq_len = 5
    fake_att = torch.softmax(torch.ones(1, 1, seq_len, seq_len), dim=-1)
    output = MagicMock()
    output.logits = logit_tensor
    output.attentions = [fake_att]
    mock_model.return_value = output
    return mock_model


def _make_predictor(logits: list[float]) -> PhishGuardPredictor:
    """Instantiate PhishGuardPredictor without loading real files."""
    from phishguard.inference.predictor import PhishGuardPredictor
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    mock_model = _make_mock_model(logits)

    predictor = object.__new__(PhishGuardPredictor)
    predictor._tokenizer = tokenizer
    predictor._model = mock_model
    predictor._max_length = 64
    predictor._device = "cpu"
    return predictor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhishGuardPredictor:
    def test_phishing_label(self):
        predictor = _make_predictor([-5.0, 5.0])  # strong phishing signal
        result = predictor.predict("Click here to win a prize http://evil.com")
        assert result["label"] == "phishing"
        assert result["label_id"] == 1
        assert result["confidence"] > 0.9

    def test_legitimate_label(self):
        predictor = _make_predictor([5.0, -5.0])  # strong legit signal
        result = predictor.predict("Hi team, see you at the meeting tomorrow.")
        assert result["label"] == "legitimate"
        assert result["label_id"] == 0
        assert result["confidence"] > 0.9

    def test_result_keys(self):
        predictor = _make_predictor([0.0, 0.0])
        result = predictor.predict("Some email text.")
        assert {"label", "label_id", "confidence", "token_scores"} <= result.keys()

    def test_token_scores_structure(self):
        predictor = _make_predictor([0.0, 0.0])
        result = predictor.predict("Test email.")
        assert isinstance(result["token_scores"], list)
        first = result["token_scores"][0]
        assert "token" in first and "score" in first

    def test_confidence_in_range(self):
        predictor = _make_predictor([1.0, 2.0])
        result = predictor.predict("Any text.")
        assert 0.0 <= result["confidence"] <= 1.0
