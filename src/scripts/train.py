#!/usr/bin/env python
"""End-to-end training pipeline.

Steps:
  1. Load + preprocess data
  2. Build datasets
  3. Instantiate model
  4. Train
  5. Save best checkpoint
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from phishguard.config import load_config
from phishguard.data.loader import load_primary, split_and_save, verify_balance
from phishguard.data.preprocessor import preprocess_dataframe
from phishguard.model.classifier import PhishGuardClassifier
from phishguard.model.dataset import build_datasets
from phishguard.training.trainer import train


def main() -> None:
    device = "CUDA" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU detected!)"
    print(f"[train] Device: {device}")
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]

    # ------------------------------------------------------------------ #
    # 1. Load raw data
    # ------------------------------------------------------------------ #
    print("[train] Loading raw data …")
    df = load_primary(raw_dir, data_cfg["primary_file"])
    verify_balance(df)

    # ------------------------------------------------------------------ #
    # 2. Preprocess
    # ------------------------------------------------------------------ #
    print("[train] Preprocessing text …")
    df = preprocess_dataframe(df)

    # ------------------------------------------------------------------ #
    # 3. Split & save
    # ------------------------------------------------------------------ #
    print("[train] Splitting data …")
    split_and_save(
        df,
        processed_dir,
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
        random_seed=data_cfg["random_seed"],
    )

    # ------------------------------------------------------------------ #
    # 4. Build datasets
    # ------------------------------------------------------------------ #
    print("[train] Building datasets …")
    train_ds, val_ds, _ = build_datasets(
        processed_dir,
        model_name=model_cfg["model_name"],
        max_length=model_cfg["max_seq_length"],
    )

    # ------------------------------------------------------------------ #
    # 5. Instantiate model
    # ------------------------------------------------------------------ #
    print("[train] Loading model …")
    classifier = PhishGuardClassifier(
        model_name=model_cfg["model_name"],
        num_labels=model_cfg["num_labels"],
    )

    # ------------------------------------------------------------------ #
    # 6. Train
    # ------------------------------------------------------------------ #
    train(config, train_ds, val_ds, classifier, output_cfg["output_dir"])


if __name__ == "__main__":
    main()
