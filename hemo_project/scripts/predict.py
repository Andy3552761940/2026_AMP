#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath("src"))

from hemo_pred.data import load_dataset
from hemo_pred.infer import predict_proba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    df = load_dataset(args.input_csv, args.seq_col, label_col="__dummy__")
    p = predict_proba(df, args.model_dir, seq_col=args.seq_col, device=args.device)
    df_out = df.copy()
    df_out["p_hemolysis"] = p
    df_out["pred_label"] = (p >= args.thr).astype(int)
    df_out.to_csv(args.output_csv, index=False)
    print(f"saved to {args.output_csv}")


if __name__ == "__main__":
    main()
