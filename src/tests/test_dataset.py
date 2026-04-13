"""Unit tests for phishguard.model.dataset."""
import pandas as pd
import pytest
import torch

from phishguard.model.dataset import PhishingDataset


@pytest.fixture()
def tiny_df():
    return pd.DataFrame(
        {
            "text": [
                "Click now to claim your prize!",
                "Hi team, please review the attached document.",
                "Urgent: verify your account at [URL].",
            ],
            "label": [1, 0, 1],
        }
    )


@pytest.fixture()
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


class TestPhishingDataset:
    def test_length(self, tiny_df, tokenizer):
        ds = PhishingDataset(tiny_df, tokenizer, max_length=64)
        assert len(ds) == 3

    def test_item_keys(self, tiny_df, tokenizer):
        ds = PhishingDataset(tiny_df, tokenizer, max_length=64)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_tensor_types(self, tiny_df, tokenizer):
        ds = PhishingDataset(tiny_df, tokenizer, max_length=64)
        item = ds[0]
        assert isinstance(item["input_ids"], list)
        assert isinstance(item["attention_mask"], list)
        assert all(isinstance(token, int) for token in item["input_ids"])
        assert all(isinstance(token, int) for token in item["attention_mask"])
        assert item["labels"].dtype == torch.long

    def test_shape(self, tiny_df, tokenizer):
        max_len = 64
        ds = PhishingDataset(tiny_df, tokenizer, max_length=max_len)
        item = ds[0]
        assert len(item["input_ids"]) <= max_len
        assert len(item["attention_mask"]) == len(item["input_ids"])

    def test_label_values(self, tiny_df, tokenizer):
        ds = PhishingDataset(tiny_df, tokenizer, max_length=64)
        assert ds[0]["labels"].item() == 1
        assert ds[1]["labels"].item() == 0
