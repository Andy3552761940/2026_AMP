#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

sys.path.append(os.path.abspath("src"))

from hemo_pred.data import load_dataset
from hemo_pred.infer import predict_proba, load_threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = load_dataset(args.test_csv, args.seq_col, args.label_col)
    y = df[args.label_col].values.astype(int)
    p = predict_proba(df, args.model_dir, seq_col=args.seq_col, device=args.device)
    thr = load_threshold(args.model_dir) if args.thr is None else args.thr
    pred = (p >= thr).astype(int)

    metrics = {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    print(metrics)


if __name__ == "__main__":
    main()
