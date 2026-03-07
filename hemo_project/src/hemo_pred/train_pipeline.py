from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import build_handcrafted_matrix
from .embedding import ESMEmbedder


def _metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


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


def find_best_threshold(y_true: np.ndarray, prob: np.ndarray, step: float = 0.005) -> tuple[float, float]:
    thresholds = np.arange(step, 1.0, step)
    acc_scores = [accuracy_score(y_true, (prob >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(acc_scores))
    return float(thresholds[best_idx]), float(acc_scores[best_idx])


def train_with_cv(
    df,
    seq_col: str,
    label_col: str,
    out_dir: str,
    folds: int = 5,
    seed: int = 42,
    device: str = "cpu",
    esm_model_name: str = "facebook/esm2_t12_35M_UR50D",
    esm_batch_size: int = 16,
    esm_max_len: int = 512,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_h = build_handcrafted_matrix(df, seq_col=seq_col)
    y = df[label_col].values.astype(int)
    seqs = df[seq_col].astype(str).tolist()

    embedder = ESMEmbedder(model_name=esm_model_name, device=device)
    X_e = embedder.encode(seqs, batch_size=esm_batch_size, max_len=esm_max_len)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(len(df), dtype=float)
    oof_esm = np.zeros(len(df), dtype=float)

    for tr, va in skf.split(X_h, y):
        lgb = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
        )
        lgb.fit(X_h[tr], y[tr])
        oof_lgb[va] = lgb.predict_proba(X_h[va])[:, 1]

        esm_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        esm_lr.fit(X_e[tr], y[tr])
        oof_esm[va] = esm_lr.predict_proba(X_e[va])[:, 1]

    X_meta = _build_meta_features(oof_lgb, oof_esm)
    meta = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12),
        cv=5,
        class_weight="balanced",
        max_iter=2000,
        scoring="accuracy",
        n_jobs=-1,
        random_state=seed,
    )
    meta.fit(X_meta, y)
    oof_stack = meta.predict_proba(X_meta)[:, 1]
    best_thr, best_acc = find_best_threshold(y, oof_stack)

    cv_metrics = {
        "handcrafted_lgbm": _metrics(y, oof_lgb),
        "esm_lr": _metrics(y, oof_esm),
        "stacking": _metrics(y, oof_stack),
        "stacking_tuned_threshold": {
            **_metrics(y, oof_stack, thr=best_thr),
            "threshold": best_thr,
            "best_cv_accuracy": best_acc,
        },
    }

    final_lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
    )
    final_lgb.fit(X_h, y)

    final_esm_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    final_esm_lr.fit(X_e, y)

    joblib.dump(final_lgb, out / "branch_handcrafted_lgbm.joblib")
    joblib.dump(final_esm_lr, out / "branch_esm_lr.joblib")

    final_p1 = final_lgb.predict_proba(X_h)[:, 1]
    final_p2 = final_esm_lr.predict_proba(X_e)[:, 1]
    final_meta_x = _build_meta_features(final_p1, final_p2)
    meta.fit(final_meta_x, y)
    joblib.dump(meta, out / "stacking_model.joblib")

    with open(out / "decision_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"stacking": best_thr}, f, indent=2, ensure_ascii=False)


    with open(out / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "esm_model_name": esm_model_name,
            "esm_batch_size": esm_batch_size,
            "esm_max_len": esm_max_len,
        }, f, indent=2, ensure_ascii=False)

    with open(out / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cv_metrics, f, indent=2, ensure_ascii=False)

    return cv_metrics
