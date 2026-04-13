"""HuggingFace Trainer-based training loop."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute accuracy and macro-F1 from Trainer evaluation predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }


def build_training_args(config: dict, output_dir: str | os.PathLike) -> TrainingArguments:
    """Construct :class:`~transformers.TrainingArguments` from the YAML config dict."""
    tc = config.get("training", {})
    
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tc.get("num_epochs", 3),
        per_device_train_batch_size=tc.get("batch_size", 16),
        per_device_eval_batch_size=tc.get("batch_size", 16),
        learning_rate=tc.get("learning_rate", 2e-5),
        weight_decay=tc.get("weight_decay", 0.01),
        warmup_steps=tc.get("warmup_steps", 0),
        save_total_limit=tc.get("save_total_limit", 3),
        fp16=tc.get("fp16", False),
        
        # Change default to "epoch" to minimize overhead
        eval_strategy=tc.get("evaluation_strategy", "epoch"),
        eval_steps=tc.get("eval_steps", 500),
        save_strategy=tc.get("save_strategy", "epoch"),
        save_steps=tc.get("save_steps", 500),
        
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "f1"),
        
        # --- Performance Optimizations ---
        # Multi-worker data loading prevents CPU bottlenecks
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        # Ensures faster data transfer from RAM to GPU
        dataloader_pin_memory=True,
        
        logging_steps=50,
        report_to="none",
    )


def train(
    config: dict,
    train_dataset: Any,
    val_dataset: Any,
    model: Any,
    output_dir: str | os.PathLike,
) -> Trainer:
    """Run training and return the fitted :class:`~transformers.Trainer`.

    Parameters
    ----------
    config:
        Loaded YAML config dict.
    train_dataset / val_dataset:
        :class:`~phishguard.model.dataset.PhishingDataset` instances.
    model:
        :class:`~phishguard.model.classifier.PhishGuardClassifier` instance.
    output_dir:
        Directory for checkpoints.
    """
    training_args = build_training_args(config, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Auto-resume from the latest checkpoint if one exists
    resume_ckpt = _latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"[trainer] Resuming from checkpoint: {resume_ckpt}")
    else:
        print("[trainer] Starting training from scratch …")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    print("[trainer] Training complete.")

    # Save best checkpoint explicitly
    best_dir = Path(config["output"]["best_checkpoint_dir"])
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))

    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(best_dir))
    print(f"[trainer] Best model saved → {best_dir}")

    return trainer


def _latest_checkpoint(output_dir: str | os.PathLike) -> str | None:
    """Return the path of the most recent ``checkpoint-*`` folder, or *None*."""
    base = Path(output_dir)
    if not base.exists():
        return None
    checkpoints = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None
