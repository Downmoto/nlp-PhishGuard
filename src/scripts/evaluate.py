#!/usr/bin/env python
"""Evaluate a trained PhishGuard checkpoint on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from phishguard.config import load_config
from phishguard.evaluation.evaluator import evaluate_model, write_evaluation_report
from phishguard.model.classifier import PhishGuardClassifier
from phishguard.model.dataset import PhishingDataset, build_datasets
from phishguard.training.trainer import compute_metrics


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    ckpt_dir = output_cfg["best_checkpoint_dir"]
    processed_dir = data_cfg["processed_dir"]
    figures_dir = output_cfg["figures_dir"]
    report_path = Path(output_cfg["reports_dir"]) / "evaluation_report.md"

    print(f"[evaluate] Loading checkpoint from {ckpt_dir} …")
    classifier = PhishGuardClassifier(
        model_name=model_cfg["model_name"],
        num_labels=model_cfg["num_labels"],
        checkpoint_dir=ckpt_dir,
    )

    _, _, test_ds = build_datasets(
        processed_dir,
        model_name=ckpt_dir,  # tokenizer is stored alongside checkpoint
        max_length=model_cfg["max_seq_length"],
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Minimal TrainingArguments just to drive Trainer.predict()
    eval_args = TrainingArguments(
        output_dir=output_cfg["output_dir"],
        per_device_eval_batch_size=config["training"].get("batch_size", 16),
        eval_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=classifier.model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results = evaluate_model(trainer, test_ds, figures_dir)
    write_evaluation_report(results, report_path)


if __name__ == "__main__":
    main()
