from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data import load_tokenizer
from src.modeling import load_model_for_inference


class PredictCSVChunkDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_col: str = "Sequence",
        id_col: Optional[str] = "Hash",
        include_labels: bool = False,
        label_col: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.seq_col = seq_col
        self.id_col = id_col if (id_col and id_col in self.df.columns) else None
        self.label_col = label_col if (label_col and label_col in self.df.columns) else None

        if self.seq_col not in self.df.columns:
            raise ValueError(f"Missing sequence column: {self.seq_col}")

        self.has_labels = bool(include_labels) and (self.label_col is not None)

        # Keep original row order inside this chunk for later restore
        self.df["_orig_order"] = np.arange(len(self.df), dtype=np.int64)
        # Precompute sequence lengths for sorting/bucketing
        self.df["_seq_len"] = self.df[self.seq_col].astype(str).str.len().astype(np.int32)

    def sort_by_length(self):
        # Stable sort preserves order among equal lengths
        self.df = self.df.sort_values("_seq_len", kind="mergesort").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col]).strip().upper()
        if seq == "nan":
            seq = ""

        if self.id_col is not None:
            seq_id = str(row[self.id_col])
        else:
            seq_id = str(int(row["_orig_order"]))

        item = {
            "sequence": seq,
            "seq_id": seq_id,
            "orig_order": int(row["_orig_order"]),
            "seq_len": int(row["_seq_len"]),
        }

        if self.has_labels:
            item["label"] = float(row[self.label_col])

        return item


class PredictCollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        sequences = [f["sequence"] for f in features]
        enc = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["seq_id"] = [f["seq_id"] for f in features]
        enc["orig_order"] = torch.tensor([f["orig_order"] for f in features], dtype=torch.long)
        enc["seq_len"] = torch.tensor([f["seq_len"] for f in features], dtype=torch.int32)
        if "label" in features[0]:
            enc["label"] = torch.tensor([f["label"] for f in features], dtype=torch.float32)
        return enc


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_input_csv(args) -> str:
    if args.in_csv:
        return args.in_csv
    if args.data_dir and args.csv:
        return os.path.join(args.data_dir, args.csv)
    raise ValueError("Provide either --in_csv OR (--data_dir and --csv).")


def detect_sep(in_csv: str, seq_col: str) -> str:
    """Detect delimiter (comma vs tab) for a supposed CSV file.

    We try comma first; if `seq_col` is not present, we retry as TSV.
    """
    try:
        head = pd.read_csv(in_csv, nrows=2)
        if seq_col in head.columns:
            return ","
    except Exception:
        pass
    head = pd.read_csv(in_csv, sep="\t", nrows=2)
    if seq_col in head.columns:
        return "\t"
    return ","


def merge_chunk_csvs(chunks_dir: str, out_csv: str) -> int:
    files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.csv")))
    if not files:
        raise FileNotFoundError(f"No chunk csv files found in {chunks_dir}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "wb") as fout:
        for i, fp in enumerate(files):
            with open(fp, "rb") as fin:
                if i > 0:
                    fin.readline()  # skip header
                shutil.copyfileobj(fin, fout)
    return len(files)


@torch.no_grad()
def run_one_chunk(
    df_chunk: pd.DataFrame,
    chunk_idx: int,
    model,
    tokenizer,
    args,
    device: torch.device,
    chunk_csv_path: str,
) -> Dict[str, float]:
    label_col = args.label_col or args.label_name

    ds = PredictCSVChunkDataset(
        df_chunk,
        seq_col=args.seq_col,
        id_col=args.id_col,
        include_labels=bool(args.include_labels),
        label_col=label_col,
    )
    if bool(args.sort_by_len):
        ds.sort_by_length()

    collator = PredictCollator(tokenizer, max_length=args.max_len)

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
    dl = DataLoader(ds, **dl_kwargs)

    all_probs = []
    all_ids = []
    all_orig = []
    all_seq_len = []
    all_labels = [] if ds.has_labels else None

    t0 = time.perf_counter()
    for batch in tqdm(dl, desc=f"chunk {chunk_idx:04d}", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with amp_ctx(device, args.amp):
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.sigmoid(out["logits"]).detach().float().cpu().numpy()  # [B,1]
        all_probs.append(probs)
        all_ids.extend(batch["seq_id"])
        all_orig.append(batch["orig_order"].cpu().numpy())
        all_seq_len.append(batch["seq_len"].cpu().numpy())

        if all_labels is not None and "label" in batch:
            all_labels.append(batch["label"].cpu().numpy())

    infer_sec = time.perf_counter() - t0

    probs = np.concatenate(all_probs, axis=0).reshape(-1)
    orig = np.concatenate(all_orig, axis=0)
    seq_len = np.concatenate(all_seq_len, axis=0)

    out_df = pd.DataFrame(
        {
            "seq_id": all_ids,
            "_orig_order": orig,
            "seq_len": seq_len,
            f"p_{args.label_name}": probs,
        }
    )

    if args.thr is not None:
        out_df[f"pred_{args.label_name}"] = (out_df[f"p_{args.label_name}"] >= float(args.thr)).astype(int)

    if bool(args.include_labels) and all_labels is not None and len(all_labels) > 0:
        y = np.concatenate(all_labels, axis=0).reshape(-1)
        out_df[f"y_{args.label_name}"] = y

    if bool(args.include_seq):
        seq_map = {int(r["_orig_order"]): str(r[args.seq_col]) for _, r in ds.df.iterrows()}
        out_df["Sequence"] = [seq_map[int(i)] for i in out_df["_orig_order"].tolist()]

    # Restore original order within this chunk
    out_df = out_df.sort_values("_orig_order", kind="mergesort").reset_index(drop=True)
    out_df.to_csv(chunk_csv_path, index=False)

    n = len(out_df)
    return {
        "rows": int(n),
        "infer_sec": float(infer_sec),
        "rows_per_sec_infer": float(n / infer_sec) if infer_sec > 0 else 0.0,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # Input
    ap.add_argument("--in_csv", default=None, help="Full path to csv input")
    ap.add_argument("--data_dir", default=None, help="Alternative: data directory")
    ap.add_argument("--csv", default=None, help="Alternative: csv file name used with --data_dir")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--id_col", default="Hash")

    # Task
    ap.add_argument("--label_name", default="Hemolysis", help="Name used for output columns")
    ap.add_argument(
        "--label_col",
        default=None,
        help="Optional: ground-truth label column in input csv (only used when --include_labels 1). Defaults to --label_name.",
    )

    # Model
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--base_model_dir", default=None)
    ap.add_argument("--merge_lora", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_len", type=int, default=256)

    # Perf
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    ap.add_argument("--sort_by_len", type=int, default=1)

    # Chunking / Resume
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--pred_out", required=True)
    ap.add_argument("--chunks_dir", default=None)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--keep_chunks", type=int, default=1)

    # Output controls
    ap.add_argument("--include_labels", type=int, default=0)
    ap.add_argument("--include_seq", type=int, default=0)
    ap.add_argument("--thr", type=float, default=None, help="Optional threshold for pred_{label_name}")
    ap.add_argument("--summary_out", default=None)

    args = ap.parse_args()

    in_csv = resolve_input_csv(args)
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(in_csv)

    os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
    chunks_dir = args.chunks_dir or (args.pred_out + ".chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    t_all0 = time.perf_counter()

    # Prefer tokenizer saved with the checkpoint; fall back to base_model_dir.
    try:
        tokenizer = load_tokenizer(args.ckpt_dir)
    except Exception:
        if args.base_model_dir is None:
            raise
        tokenizer = load_tokenizer(args.base_model_dir)
    model = load_model_for_inference(
        args.ckpt_dir,
        base_model_dir=args.base_model_dir,
        merge_lora=bool(args.merge_lora),
        num_labels=1,
    ).to(device)
    model.eval()

    total_rows = 0
    total_infer_sec = 0.0
    chunk_infos = []

    sep = detect_sep(in_csv, seq_col=args.seq_col)
    reader = pd.read_csv(in_csv, chunksize=args.chunk_size, sep=sep)
    for chunk_idx, df_chunk in enumerate(reader):
        chunk_csv_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:06d}.csv")

        if bool(args.resume) and os.path.isfile(chunk_csv_path) and os.path.getsize(chunk_csv_path) > 0:
            rows = len(df_chunk)
            total_rows += rows
            chunk_infos.append({"chunk_idx": chunk_idx, "rows": rows, "skipped": True})
            print(f"[resume] skip chunk {chunk_idx:04d} ({rows} rows)")
            continue

        info = run_one_chunk(
            df_chunk=df_chunk,
            chunk_idx=chunk_idx,
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            chunk_csv_path=chunk_csv_path,
        )
        total_rows += info["rows"]
        total_infer_sec += info["infer_sec"]
        chunk_infos.append({"chunk_idx": chunk_idx, "skipped": False, **info})
        print(
            f"[chunk {chunk_idx:04d}] rows={info['rows']} "
            f"infer={info['infer_sec']:.2f}s rps={info['rows_per_sec_infer']:.1f}"
        )

    num_chunk_files = merge_chunk_csvs(chunks_dir, args.pred_out)
    total_wall_sec = time.perf_counter() - t_all0

    summary = {
        "input_csv": in_csv,
        "pred_out": args.pred_out,
        "chunks_dir": chunks_dir,
        "num_chunks_merged": num_chunk_files,
        "total_rows": int(total_rows),
        "total_infer_sec": float(total_infer_sec),
        "total_wall_sec": float(total_wall_sec),
        "rows_per_sec_infer": float(total_rows / total_infer_sec) if total_infer_sec > 0 else None,
        "rows_per_sec_wall": float(total_rows / total_wall_sec) if total_wall_sec > 0 else None,
        "config": {
            "label_name": args.label_name,
            "label_col": args.label_col or args.label_name,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "amp": args.amp,
            "sort_by_len": bool(args.sort_by_len),
            "chunk_size": int(args.chunk_size),
            "max_len": int(args.max_len),
            "device": str(device),
        },
        "chunks": chunk_infos,
    }

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not bool(args.keep_chunks):
        shutil.rmtree(chunks_dir, ignore_errors=True)
        print(f"[cleanup] removed {chunks_dir}")


if __name__ == "__main__":
    main()
