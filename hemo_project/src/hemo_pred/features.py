from __future__ import annotations

import numpy as np
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWYH")
POS = set("KRH")
NEG = set("DE")

AA_MASS = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}


def clean_sequence(seq: str) -> str:
    seq = (seq or "").strip().upper()
    return "".join(ch for ch in seq if ch in AA_SET)


def aac_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return np.zeros(len(AA), dtype=float)
    counts = np.array([seq.count(a) for a in AA], dtype=float)
    return counts / len(seq)


def physchem_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    n = len(seq)
    if n == 0:
        return np.zeros(5, dtype=float)

    charge = (sum(ch in POS for ch in seq) - sum(ch in NEG for ch in seq)) / n
    hydrophobic_ratio = sum(ch in HYDROPHOBIC for ch in seq) / n
    aromatic_ratio = sum(ch in AROMATIC for ch in seq) / n
    mw = sum(AA_MASS[ch] for ch in seq) / n

    return np.array([n, charge, hydrophobic_ratio, aromatic_ratio, mw], dtype=float)


def build_handcrafted_matrix(df: pd.DataFrame, seq_col: str = "sequence") -> np.ndarray:
    mats = []
    for s in df[seq_col].astype(str):
        mats.append(np.concatenate([aac_features(s), physchem_features(s)]))
    return np.vstack(mats)
