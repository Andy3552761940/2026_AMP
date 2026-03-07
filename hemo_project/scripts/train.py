#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath("src"))

from hemo_pred.data import load_dataset
from hemo_pred.train_pipeline import train_with_cv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--esm_model_name", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--esm_batch_size", type=int, default=16)
    ap.add_argument("--esm_max_len", type=int, default=512)
    args = ap.parse_args()

    df = load_dataset(args.train_csv, args.seq_col, args.label_col)
    metrics = train_with_cv(
        df,
        seq_col=args.seq_col,
        label_col=args.label_col,
        out_dir=args.out_dir,
        folds=args.folds,
        seed=args.seed,
        device=args.device,
        esm_model_name=args.esm_model_name,
        esm_batch_size=args.esm_batch_size,
        esm_max_len=args.esm_max_len,
    )
    print(metrics)


if __name__ == "__main__":
    main()
