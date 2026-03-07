from __future__ import annotations

import pandas as pd


def load_dataset(path: str, seq_col: str = "sequence", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column: {seq_col}")
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(int)
    return df
