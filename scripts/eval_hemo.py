from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CSVDatasetWithLabels, Collator, load_tokenizer
from src.modeling import load_model_for_inference


def metrics_from_probs(y: np.ndarray, p: np.ndarray, thr_from: str = "self", fixed_thr: float = 0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score

    out = {}
    out["pr_auc"] = float(average_precision_score(y, p))
    out["roc_auc"] = float(roc_auc_score(y, p))

    if thr_from == "fixed":
        thr = float(fixed_thr)
    else:
        best_mcc = -1.0
        best_thr = 0.5
        for thr0 in np.linspace(0.05, 0.95, 19):
            pred = (p >= thr0).astype(int)
            mcc = matthews_corrcoef(y, pred)
            if mcc > best_mcc:
                best_mcc = float(mcc)
                best_thr = float(thr0)
        thr = best_thr
        out["best_thr"] = float(best_thr)
        out["best_mcc"] = float(best_mcc)

    out["mcc@0.5"] = float(matthews_corrcoef(y, (p >= 0.5).astype(int)))
    out["f1@thr"] = float(f1_score(y, (p >= thr).astype(int)))
    out["thr"] = float(thr)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--base_model_dir", default=None, help="Required if ckpt_dir only has weights (no config.json).")
    ap.add_argument("--merge_lora", type=int, default=1)

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--id_col", default="Hash")
    ap.add_argument("--label_col", default="Hemolysis")

    ap.add_argument("--out", required=True)
    ap.add_argument("--pred_out", default=None)

    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--thr_from", type=str, default="self", choices=["self", "fixed"])
    ap.add_argument("--fixed_thr", type=float, default=0.5)

    args = ap.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv)
    ds = CSVDatasetWithLabels(
        csv_path,
        label_cols=[args.label_col],
        seq_col=args.seq_col,
        id_col=args.id_col,
    )

    # Prefer tokenizer saved with the checkpoint; fall back to base_model_dir.
    try:
        tokenizer = load_tokenizer(args.ckpt_dir)
    except Exception:
        if args.base_model_dir is None:
            raise
        tokenizer = load_tokenizer(args.base_model_dir)
    collator = Collator(tokenizer, max_length=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device(args.device)
    model = load_model_for_inference(
        args.ckpt_dir,
        base_model_dir=args.base_model_dir,
        merge_lora=bool(args.merge_lora),
        num_labels=1,
    ).to(device)
    model.eval()

    all_logits = []
    all_labels = []
    all_ids = []

    for batch in tqdm(dl, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"].cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)
        all_ids.extend(batch["seq_id"])

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = 1 / (1 + np.exp(-logits))

    y = labels[:, 0]
    p = probs[:, 0]

    report = metrics_from_probs(y, p, thr_from=args.thr_from, fixed_thr=args.fixed_thr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.pred_out:
        df = pd.DataFrame({"seq_id": all_ids, f"p_{args.label_col}": p, f"y_{args.label_col}": y})
        os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
        df.to_csv(args.pred_out, index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
