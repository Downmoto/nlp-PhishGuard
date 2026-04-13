"""PyTorch Dataset wrapping tokenised phishing email data."""
from __future__ import annotations

from typing import cast

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class PhishingDataset(Dataset):
    """Map-style PyTorch Dataset for phishing email classification.

    Parameters
    ----------
    df:
        DataFrame with at least ``text`` (str) and ``label`` (int 0/1) columns.
    tokenizer:
        A HuggingFace tokenizer compatible with the chosen model.
    max_length:
        Maximum token length used for truncation / padding.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self._labels = df["label"].tolist()
        # Pre-tokenize once to avoid repeated CPU tokenization every epoch.
        encodings = cast(
            dict[str, list[list[int]]],
            tokenizer(
                df["text"].tolist(),
                truncation=True,
                max_length=max_length,
                padding=False,
            ),
        )
        # Keep a canonical dict to remain robust across worker pickling boundaries.
        self._encodings = encodings
        self._input_ids = encodings["input_ids"]
        self._attention_masks = encodings["attention_mask"]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, list[int] | torch.Tensor]:
        input_ids = getattr(self, "_input_ids", None)
        attention_masks = getattr(self, "_attention_masks", None)
        if input_ids is None or attention_masks is None:
            encodings = getattr(self, "_encodings")
            input_ids = encodings["input_ids"]
            attention_masks = encodings["attention_mask"]
        return {
            "input_ids": input_ids[idx],
            "attention_mask": attention_masks[idx],
            "labels": torch.tensor(self._labels[idx], dtype=torch.long),
        }


def build_datasets(
    processed_dir: str | os.PathLike,
    model_name: str,
    max_length: int = 512,
) -> tuple[PhishingDataset, PhishingDataset, PhishingDataset]:
    """Load train/val/test Parquet splits and return :class:`PhishingDataset` objects.

    Returns
    -------
    tuple
        ``(train_dataset, val_dataset, test_dataset)``
    """
    from pathlib import Path

    base = Path(processed_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datasets = {}
    for split in ("train", "val", "test"):
        df = pd.read_parquet(base / f"{split}.parquet")
        datasets[split] = PhishingDataset(df, tokenizer, max_length)

    return datasets["train"], datasets["val"], datasets["test"]
