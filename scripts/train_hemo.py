from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, set_seed

from src.data import CSVDatasetWithLabels, Collator, load_tokenizer
from src.modeling import ESMForMultiLabel


def compute_pos_weight_binary(train_ds: CSVDatasetWithLabels) -> torch.Tensor:
    """Compute pos_weight = neg/pos for each label column."""
    pos = None
    neg = None
    for i in range(len(train_ds)):
        y = train_ds[i]["labels"].to(torch.float64)
        if pos is None:
            pos = torch.zeros_like(y)
            neg = torch.zeros_like(y)
        pos += (y >= 0.5).to(torch.float64)
        neg += (y < 0.5).to(torch.float64)

    assert pos is not None and neg is not None
    pos = torch.clamp(pos, min=1.0)
    w = (neg / pos).to(torch.float32)
    return w


class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs.pop("seq_id", None)
        outputs = model(**inputs, labels=None)
        logits = outputs["logits"]

        labels = labels.float()
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    # single-label default
    y = labels[:, 0]
    p = probs[:, 0]

    metrics = {}
    try:
        metrics["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        metrics["pr_auc"] = float("nan")

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["mcc@0.5"] = float(matthews_corrcoef(y, (p >= 0.5).astype(int)))
    except Exception:
        metrics["mcc@0.5"] = float("nan")

    return metrics


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--val_csv", default="val.csv")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--id_col", default="Hash")
    ap.add_argument("--label_col", default="Hemolysis", help="Binary label column name, e.g. Hemolysis (0/1).")

    # Model
    ap.add_argument("--model_dir", required=True, help="Base ESM2 model folder (local HF).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--freeze_backbone", type=int, default=0)
    ap.add_argument("--grad_ckpt", type=int, default=0)

    # Train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=1)

    # Imbalance
    ap.add_argument("--pos_weight_mode", type=str, default="auto", choices=["auto", "none"])
    ap.add_argument("--pos_weight_max", type=float, default=50.0)

    args = ap.parse_args()

    set_seed(args.seed)

    train_path = os.path.join(args.data_dir, args.train_csv)
    val_path = os.path.join(args.data_dir, args.val_csv)

    tokenizer = load_tokenizer(args.model_dir)
    collator = Collator(tokenizer, max_length=args.max_len)

    train_ds = CSVDatasetWithLabels(
        train_path,
        label_cols=[args.label_col],
        seq_col=args.seq_col,
        id_col=args.id_col,
    )
    val_ds = CSVDatasetWithLabels(
        val_path,
        label_cols=[args.label_col],
        seq_col=args.seq_col,
        id_col=args.id_col,
    )

    pos_weight = None
    if args.pos_weight_mode == "auto":
        pos_weight = compute_pos_weight_binary(train_ds)
        if args.pos_weight_max is not None:
            pos_weight = torch.clamp(pos_weight, max=float(args.pos_weight_max))
        print("pos_weight:", pos_weight.tolist())

    model = ESMForMultiLabel(args.model_dir, num_labels=1)
    if args.freeze_backbone:
        model.freeze_backbone()

    if args.grad_ckpt and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        fp16=bool(args.fp16),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
