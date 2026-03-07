from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .features import build_handcrafted_matrix
from .embedding import ESMEmbedder


def _build_meta_features(prob_1: np.ndarray, prob_2: np.ndarray) -> np.ndarray:
    avg = (prob_1 + prob_2) / 2.0
    return np.column_stack([
        prob_1,
        prob_2,
        avg,
        prob_1 * prob_2,
        np.abs(prob_1 - prob_2),
        np.maximum(prob_1, prob_2),
        np.minimum(prob_1, prob_2),
    ])


def _load_model_config(model_dir: Path) -> dict:
    cfg_file = model_dir / "model_config.json"
    if not cfg_file.exists():
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_threshold(model_dir: str, default_thr: float = 0.5) -> float:
    model_dir = Path(model_dir)
    thr_file = model_dir / "decision_threshold.json"
    if not thr_file.exists():
        return default_thr
    with open(thr_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("stacking", default_thr))


def predict_proba(df: pd.DataFrame, model_dir: str, seq_col: str = "sequence", device: str = "cpu") -> np.ndarray:
    model_dir = Path(model_dir)
    lgb = joblib.load(model_dir / "branch_handcrafted_lgbm.joblib")
    esm_lr = joblib.load(model_dir / "branch_esm_lr.joblib")
    meta = joblib.load(model_dir / "stacking_model.joblib")

    X_h = build_handcrafted_matrix(df, seq_col=seq_col)
    cfg = _load_model_config(model_dir)
    embedder = ESMEmbedder(model_name=cfg.get("esm_model_name", "facebook/esm2_t6_8M_UR50D"), device=device)
    X_e = embedder.encode(
        df[seq_col].astype(str).tolist(),
        batch_size=int(cfg.get("esm_batch_size", 16)),
        max_len=int(cfg.get("esm_max_len", 512)),
    )

    p1 = lgb.predict_proba(X_h)[:, 1]
    p2 = esm_lr.predict_proba(X_e)[:, 1]
    pm = meta.predict_proba(_build_meta_features(p1, p2))[:, 1]
    return pm
