"""Model pipeline construction, training, evaluation, and persistence.

Task: predict `target` — will this user listen to this song again within
a month of first hearing it. Binary classification, scored by **AUC**
(area under the ROC curve) on Kaggle's real leaderboard — unlike
academic_success's macro-F1 (multi-class, needs class balance handled)
or the CO2/Spotify projects' regression/precision@k metrics. `target` is
close to balanced (50.4%/49.6% — verified directly), so unlike
academic_success, no resampling (ADASYN etc.) is needed here.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .features import build_feature_pipeline


def default_estimator() -> BaseEstimator:
    """LightGBM: handles this project's mix of sparse one-hot categorical
    columns and skewed count features well, and is fast enough for 7.4M
    rows — the same reasoning HistGradientBoosting/LightGBM get chosen as
    defaults in the other three projects.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, random_state=config.RANDOM_SEED, verbosity=-1)


def logistic_estimator() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def random_forest_estimator() -> BaseEstimator:
    """`max_depth=12` — the same load-bearing cap the Spotify project's
    SHAP investigation established: unbounded trees on a multi-million
    row dataset make `TreeExplainer` impractically slow, and there's no
    reason to expect that lesson doesn't transfer here.
    """
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1
    )


def xgboost_estimator() -> BaseEstimator:
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, random_state=config.RANDOM_SEED, eval_metric="auc")


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Logistic Regression": logistic_estimator,
    "Random Forest": random_forest_estimator,
    "LightGBM": default_estimator,
    "XGBoost": xgboost_estimator,
}


def build_pipeline(estimator: BaseEstimator | None = None) -> Pipeline:
    pipeline = build_feature_pipeline()
    steps = list(pipeline.steps)
    steps.append(("model", estimator if estimator is not None else default_estimator()))
    return Pipeline(steps=steps)


def holdout_eval(
    panel: pd.DataFrame,
    estimator: BaseEstimator | None = None,
    test_size: float = 0.2,
) -> dict:
    """A single stratified train/holdout split, scored by AUC. Not
    k-fold cross-validation: at 7.4M rows, one fit of a gradient-boosted
    model already takes real time, and unlike the CO2/Spotify projects'
    lag-feature or nearest-neighbor leakage concerns, there's no
    chronological or identity-based leakage risk here that would make a
    single split misleading — a plain stratified split is the honest,
    tractable choice.
    """
    X, y = panel.drop(columns=[config.TARGET_COL]), panel[config.TARGET_COL]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_SEED
    )

    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_holdout)[:, 1]
    return {
        "auc": float(roc_auc_score(y_holdout, proba)),
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
    }


def train_pipeline(panel: pd.DataFrame, estimator: BaseEstimator | None = None) -> Pipeline:
    """Fit a fresh pipeline on the full given panel."""
    X, y = panel.drop(columns=[config.TARGET_COL]), panel[config.TARGET_COL]
    pipeline = build_pipeline(estimator)
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)
