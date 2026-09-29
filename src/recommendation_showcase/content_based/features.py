"""Feature engineering and the shared preprocessing pipeline.

This is the single source of truth for turning Spotify's raw audio-feature
columns into the feature space the recommender's nearest-neighbor search
runs over. The notebook, `scripts/train.py`, `scripts/recommend.py`, and
the Streamlit app all call `build_feature_pipeline()` (wrapped inside the
fitted pipeline saved to `models/model.joblib`) so none of them can
silently diverge from what the recommender was actually built on.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Median valence/energy thresholds used to bucket every track into one of
# four quadrants of Russell's circumplex model of affect — a well-known
# framework for mapping (energy, valence) pairs onto mood labels. The
# actual cutoffs are learned from training data (see FeatureEngineer.fit),
# not hardcoded at 0.5, since Spotify's own valence/energy distributions
# aren't centered there.
_QUADRANT_LABELS = {
    (True, True): "energetic_happy",
    (True, False): "energetic_dark",
    (False, True): "calm_happy",
    (False, False): "calm_dark",
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives:

    - `key_sin`/`key_cos`: Spotify's `key` column is one of 12 pitch
      classes (0=C, 1=C#/Db, ..., 11=B) arranged in a circle, not a
      linear scale — pitch class 11 is one semitone from pitch class 0,
      not "far away" from it the way a plain integer or a flat one-hot
      encoding would imply. Encoding it as a point on a circle (like the
      CO2 Emissions project's week-of-year encoding) preserves that
      adjacency.
    - `duration_min`: `duration_ms` converted to minutes — purely a
      readability transform, doesn't change what the model learns, but
      makes the notebook's EDA and the app's displayed values sane.
    - `mood_quadrant`: buckets each track into one of four mood
      categories from Russell's circumplex model of affect, using
      training-data medians of `valence` (musical positivity) and
      `energy` as the quadrant boundaries. Purely a demo-friendly,
      interpretable feature — a genre classifier can learn the same
      distinction from raw valence/energy directly, but a human comparing
      two songs finds "energetic_happy vs. calm_dark" much more legible
      than two floats.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        self.valence_median_ = X["valence"].median()
        self.energy_median_ = X["energy"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        angle = 2 * np.pi * df[config.KEY_COL] / 12
        df["key_sin"] = np.sin(angle)
        df["key_cos"] = np.cos(angle)

        df["duration_min"] = df["duration_ms"] / 60_000

        high_energy = df["energy"] >= self.energy_median_
        high_valence = df["valence"] >= self.valence_median_
        df["mood_quadrant"] = [
            _QUADRANT_LABELS[(e, v)] for e, v in zip(high_energy, high_valence)
        ]

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + config.ENGINEERED_NUMERIC_COLS + config.ENGINEERED_CATEGORICAL_COLS)


def build_preprocessor() -> ColumnTransformer:
    """Numeric columns (raw audio features + engineered): median-impute,
    then standard-scale — nearest-neighbor distance is meaningless if
    `loudness` (roughly -60 to 0) and `danceability` (0 to 1) aren't on
    comparable scales, so this matters even more here than in a
    tree-based project, where scale-invariance is automatic.
    Categorical columns: most-frequent-impute, then one-hot encode.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_COLS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )


def build_feature_pipeline() -> Pipeline:
    """FeatureEngineer + preprocessor, without a final estimator. Useful
    on its own for inspecting transformed features (e.g. in the
    notebook's EDA section) and as the shared input both the recommender
    and the genre classifier run on.
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
        ]
    )
