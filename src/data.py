from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL_COLS = ["Antimicrobial", "Antibacterial", "Antifungal", "Antiviral", "Antiparasitic"]


def read_csv_flexible(path: str, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a CSV/TSV file with a simple fallback.

    Many bio datasets are actually TSV but saved with a .csv suffix.
    We first try the default comma-separated read; if required columns are
    missing, we retry with tab separation.

    Parameters
    - path: input file path
    - required_cols: if provided, used to decide whether to fallback to TSV
    """
    df = pd.read_csv(path)
    if required_cols is None:
        return df

    req = set(required_cols)
    if req.issubset(set(df.columns)):
        return df

    # Retry as TSV
    df_tsv = pd.read_csv(path, sep="\t")
    if req.issubset(set(df_tsv.columns)):
        return df_tsv

    # Return the original df so the caller can raise a helpful error.
    return df


def parse_binary_label(x) -> float:
    """Parse a binary label from CSV.

    Accepts 0/1, True/False, or common strings (case-insensitive).
    """
    if pd.isna(x):
        raise ValueError("Label value is NaN")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "pos", "positive", "hemolytic", "toxic"}:
            return 1.0
        if s in {"0", "false", "f", "no", "n", "neg", "negative", "non-hemolytic", "nonhemolytic", "non_toxic", "nontoxic"}:
            return 0.0
        # fall back: try numeric string
        try:
            return float(s)
        except Exception as e:
            raise ValueError(f"Cannot parse binary label from: {x!r}") from e
    # fall back
    return float(x)

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor  # shape [B, 5] float
    seq_ids: Optional[List[str]] = None

class EscapeCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = read_csv_flexible(csv_path, required_cols=["Sequence", *LABEL_COLS])
        if "Sequence" not in self.df.columns:
            raise ValueError(f"Missing 'Sequence' column in {csv_path}")
        missing = [c for c in LABEL_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns {missing} in {csv_path}")
        # keep stable order
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = str(row["Sequence"]).strip().upper()
        labels = torch.tensor([float(row[c]) for c in LABEL_COLS], dtype=torch.float32)
        seq_id = str(row["Hash"]) if "Hash" in row else str(idx)
        return {"sequence": seq, "labels": labels, "seq_id": seq_id}


class CSVDatasetWithLabels(Dataset):
    """Generic CSV dataset for peptide property classification.

    Expected columns
    - sequence column: default 'Sequence'
    - label columns: provided via `label_cols`
    - optional id column: default 'Hash' (falls back to row index)

    This is useful for tasks other than AMP, e.g. Hemolysis/Cytotoxicity.
    """

    def __init__(
        self,
        csv_path: str,
        label_cols: List[str],
        seq_col: str = "Sequence",
        id_col: Optional[str] = "Hash",
        label_parser=parse_binary_label,
    ):
        # Only sequence + labels are required. `id_col` is optional.
        required = [seq_col, *list(label_cols)]
        self.df = read_csv_flexible(csv_path, required_cols=required)
        self.seq_col = seq_col
        self.id_col = id_col if (id_col and id_col in self.df.columns) else None
        self.label_cols = list(label_cols)
        self.label_parser = label_parser

        if self.seq_col not in self.df.columns:
            raise ValueError(f"Missing '{self.seq_col}' column in {csv_path}")
        missing = [c for c in self.label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns {missing} in {csv_path}")
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col]).strip().upper()
        labels = torch.tensor([self.label_parser(row[c]) for c in self.label_cols], dtype=torch.float32)
        if self.id_col is not None:
            seq_id = str(row[self.id_col])
        else:
            seq_id = str(idx)
        return {"sequence": seq, "labels": labels, "seq_id": seq_id}

class Collator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        sequences = [f["sequence"] for f in features]
        labels = torch.stack([f["labels"] for f in features], dim=0)  # [B, 5]
        enc = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Keep seq_ids for exporting logits
        enc["labels"] = labels
        enc["seq_id"] = [f["seq_id"] for f in features]
        return enc

def load_tokenizer(model_dir: str):
    # local model folder support
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
