from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .features import build_handcrafted_matrix
from .embedding import ESMEmbedder


def predict_proba(df: pd.DataFrame, model_dir: str, seq_col: str = "sequence", device: str = "cpu") -> np.ndarray:
    model_dir = Path(model_dir)
    lgb = joblib.load(model_dir / "branch_handcrafted_lgbm.joblib")
    esm_lr = joblib.load(model_dir / "branch_esm_lr.joblib")
    meta = joblib.load(model_dir / "stacking_model.joblib")

    X_h = build_handcrafted_matrix(df, seq_col=seq_col)
    embedder = ESMEmbedder(device=device)
    X_e = embedder.encode(df[seq_col].astype(str).tolist())

    p1 = lgb.predict_proba(X_h)[:, 1]
    p2 = esm_lr.predict_proba(X_e)[:, 1]
    pm = meta.predict_proba(np.vstack([p1, p2]).T)[:, 1]
    return pm
