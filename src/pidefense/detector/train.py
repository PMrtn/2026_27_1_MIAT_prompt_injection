"""Fine-tune multilingual DistilBERT for binary prompt-injection detection.

The best epoch is chosen by validation PR-AUC. The decision threshold is then calibrated on the validation benign
samples for FPR <= 1% and stored next to the model (``threshold.json``). Test data is never touched here.

Usage:
    python -m pidefense.detector.train [--epochs 3] [--lr 3e-5]
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from pidefense.detector.common import (
    ARTIFACT_DIR,
    MAX_TOKENS,
    MODEL_NAME,
    RESULTS_DIR,
    TARGET_FPR,
    binary_metrics,
    load_split,
    select_threshold,
    write_json,
)
from pidefense.detector.inference import InjectionDetector
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _to_dataset(split: str, tokenizer: AutoTokenizer) -> Dataset:
    samples = load_split(split)
    dataset = Dataset.from_dict({"text": [s.text for s in samples], "label": [s.label for s in samples]})
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_TOKENS),
        batched=True,
        remove_columns=["text"],
    )


def _compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = np.asarray(prediction.predictions)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)[:, 1] / np.exp(shifted).sum(axis=1)
    metrics = binary_metrics(prediction.label_ids, probs, 0.5)
    return {key: metrics[key] for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "fpr") if key in metrics}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = _to_dataset("train", tokenizer)
    val_ds = _to_dataset("val", tokenizer)
    logger.info("Train %d / val %d samples; CUDA available: %s", len(train_ds), len(val_ds), torch.cuda.is_available())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "benign", 1: "injection"}, label2id={"benign": 0, "injection": 1}
    )
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    run_dir = ARTIFACT_DIR / "run"
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60
    best_dir = ARTIFACT_DIR / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    history = [
        {key: value for key, value in entry.items() if isinstance(value, (int, float))}
        for entry in trainer.state.log_history
    ]
    # Threshold calibration on validation benign samples (sliding-window scores, as used at inference).
    del trainer, model
    torch.cuda.empty_cache()
    detector = InjectionDetector(best_dir)
    val_samples = load_split("val")
    val_scores = detector.score([s.text for s in val_samples])
    threshold = select_threshold([s.label for s in val_samples], val_scores, TARGET_FPR)
    write_json(best_dir / "threshold.json", {"threshold": threshold, "target_fpr": TARGET_FPR, "calibrated_on": "val"})
    val_metrics = binary_metrics([s.label for s in val_samples], val_scores, threshold)
    write_json(
        RESULTS_DIR / "training.json",
        {
            "model": MODEL_NAME,
            "hyperparameters": vars(args) | {"max_tokens": MAX_TOKENS},
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_minutes": round(train_minutes, 2),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "threshold": threshold,
            "val_metrics_at_threshold": val_metrics,
            "log_history": history,
        },
    )
    logger.info("Training done in %.1f min; threshold %.4f; val %s", train_minutes, threshold, val_metrics)


if __name__ == "__main__":
    main()
