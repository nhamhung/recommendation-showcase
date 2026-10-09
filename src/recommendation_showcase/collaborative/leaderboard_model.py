"""Training for the leaderboard model (LightGBM on `leaderboard_features`).

Validation is chronological, never random: the model is fit on the first
80% of `train.csv` and scored on the last 20%, because the real test set
comes *after* all of train in time. A random split would let the model see
the future of every user and score far higher than the leaderboard does.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import leaderboard_features as lf

VALIDATION_FRACTION = 0.2

PARAMS = dict(
    objective="binary",
    metric="auc",
    # Heavy regularisation, adapted from the competition's 1st-place solution
    # (github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution,
    # lgb_record.csv): small trees (99 leaves, depth 10), large leaves
    # (>= 1306 rows) and a very large L2 penalty that shrinks every leaf
    # value, so the model has to accumulate evidence over many rounds instead
    # of memorising individual users and songs. Validation AUC 0.7606 -> 0.7650
    # versus this project's earlier settings (255 leaves, no L2) at learning
    # rate 0.3; 0.7663 at 0.1 (best at 1,585 rounds), which with ~25% more
    # rounds for the full fit gave the best submission (private AUC 0.74320).
    learning_rate=0.1,
    num_leaves=99,
    max_depth=10,
    min_data_in_leaf=1306,
    feature_fraction=0.69,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l1=6.37,
    lambda_l2=65200,
    # The ID categoricals (msno, song_id, artist) have tens to hundreds of
    # thousands of levels; categorical smoothing stops LightGBM memorising
    # rare IDs.
    cat_smooth=50,
    cat_l2=50,
    min_data_per_group=500,
    max_cat_threshold=64,
    verbose=-1,
    seed=config.RANDOM_SEED,
)


@dataclass
class Split:
    fit: pd.DataFrame
    valid: pd.DataFrame | None
    test: pd.DataFrame
    features: list[str]


def prepare(table: pd.DataFrame, validation: bool) -> Split:
    """Attach target encodings for the chosen mode and split the table.

    validation=True: labels available only for the first 80% of train;
    the last 20% is scored. validation=False: every training label is
    available and the model is fit on all of train for the submission.
    """
    n_train = int((table["is_test"] == 0).sum())
    cutoff = int(n_train * (1 - VALIDATION_FRACTION)) if validation else n_train
    label_available = (table["t"] < cutoff).to_numpy()
    table = pd.concat([table, lf.add_target_encodings(table, label_available)], axis=1)
    features = lf.feature_columns(table)
    train = table.iloc[:n_train]
    return Split(
        fit=train.iloc[:cutoff],
        valid=train.iloc[cutoff:] if validation else None,
        test=table.iloc[n_train:],
        features=features,
    )


def train(split: Split, num_rounds: int, params: dict | None = None, log_every: int = 0) -> lgb.Booster:
    params = {**PARAMS, **(params or {})}
    categorical = [c for c in lf.CATEGORICAL_FEATURES if c in split.features]
    fit_set = lgb.Dataset(split.fit[split.features], split.fit[config.TARGET_COL], categorical_feature=categorical)
    callbacks, valid_sets = [], []
    if split.valid is not None:
        valid_sets = [lgb.Dataset(split.valid[split.features], split.valid[config.TARGET_COL],
                                  reference=fit_set, categorical_feature=categorical)]
        callbacks.append(lgb.early_stopping(50, verbose=False))
        if log_every:
            callbacks.append(lgb.log_evaluation(log_every))
    return lgb.train(params, fit_set, num_rounds, valid_sets=valid_sets, callbacks=callbacks)


def predict_test(booster: lgb.Booster, split: Split) -> pd.DataFrame:
    """A submission frame (id, target), sorted by id."""
    predictions = booster.predict(split.test[split.features])
    return (
        pd.DataFrame({config.ID_COL: split.test["test_id"].astype(np.int64).to_numpy(),
                      config.TARGET_COL: predictions})
        .sort_values(config.ID_COL)
        .reset_index(drop=True)
    )
