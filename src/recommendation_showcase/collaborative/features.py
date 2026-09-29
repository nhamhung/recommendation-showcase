"""Feature engineering and the shared preprocessing pipeline.

This is the single source of truth for turning the merged raw panel
(`data.load_full_train_panel()`/`load_full_test_panel()`) into model-ready
features. The notebook, `scripts/train.py`, `scripts/make_submission.py`,
and the Streamlit app all call `build_feature_pipeline()` (wrapped inside
the fitted pipeline saved to `models/model.joblib`) so none of them can
silently diverge from how the model was actually trained.

Two transformers here learn a lookup from the training data only, then
apply it unchanged to any future row (train, holdout, or the real
competition test set) — the same discipline as the CO2 Emissions
project's `LocationMeanEncoder` and the Spotify project's
`SalesLagEncoder`-equivalent reasoning: `InteractionCountEncoder`
(frequency counts) and `RareCategoryGrouper` (which categories count as
"common").
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class InteractionCountEncoder(BaseEstimator, TransformerMixin):
    """Learns three frequency counts from the training rows only:
    `song_popularity` (how many training interactions involve this song),
    `user_activity_count` (how many involve this user), and
    `artist_song_count` (how many training interactions involve a song by
    this artist). A song/user/artist never seen in training gets 0 at
    transform time — itself a meaningful signal ("brand new to the
    catalog/never-active user"), not a missing value to impute away.

    This replaces one-hot encoding `song_id`/`msno`/`artist_name`
    directly, which would be both infeasible (millions/hundreds of
    thousands of distinct values — verified directly for `artist_name`:
    222,363) and pointless (a one-hot column for one specific song only
    ever fires for that song's own rows, telling a model nothing a raw
    identity lookup wouldn't).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "InteractionCountEncoder":
        self.song_counts_ = X[config.SONG_COL].value_counts()
        self.user_counts_ = X[config.USER_COL].value_counts()
        self.artist_counts_ = X["artist_name"].value_counts()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["song_popularity"] = df[config.SONG_COL].map(self.song_counts_).fillna(0)
        df["user_activity_count"] = df[config.USER_COL].map(self.user_counts_).fillna(0)
        df["artist_song_count"] = df["artist_name"].map(self.artist_counts_).fillna(0)
        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ["song_popularity", "user_activity_count", "artist_song_count"])


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapses low-frequency categories (below `min_frequency`) into a
    shared sentinel, learned from training data only. Same technique as
    academic_success's transformer of the same name, applied here to
    `primary_genre` (182 distinct values — verified directly) instead of
    occupation/nationality codes.
    """

    def __init__(self, columns: list[str] | None = None, min_frequency: float = 0.01):
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoryGrouper":
        columns = self.columns if self.columns is not None else config.RARE_GROUPED_COLS
        self.kept_categories_: dict[str, set] = {}
        for col in columns:
            freqs = X[col].value_counts(normalize=True)
            self.kept_categories_[col] = set(freqs[freqs >= self.min_frequency].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, kept in self.kept_categories_.items():
            df[col] = df[col].where(df[col].isin(kept), "rare")
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) if input_features is not None else [])


def _parse_isrc_year(isrc: pd.Series) -> pd.Series:
    """ISRC format: 2-letter country + 3-char registrant + 2-digit year +
    5-digit designation (e.g. `TWUM71200043` -> country `TW`, year `12`).
    The 2-digit year needs a century pivot: this dataset's tracks span up
    to ~2017, so a 2-digit year > 17 is assumed 19xx, otherwise 20xx —
    the standard convention for this specific competition's ISRC field.
    """
    year_2digit = pd.to_numeric(isrc.str.slice(5, 7), errors="coerce")
    return year_2digit.apply(lambda y: np.nan if pd.isna(y) else (1900 + y if y > 17 else 2000 + y))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic derivations from the merged panel's raw columns —
    no `y`, no state learned from training rows (unlike
    `InteractionCountEncoder`/`RareCategoryGrouper` above):

    - `member_tenure_days`: days between registration and (planned)
      expiration — a proxy for how invested a member is in the service.
    - `age_clean`: `bd` (self-reported age) is garbage outside roughly
      5-100 for ~58% of members (verified directly: 0s, negatives, and a
      max of 1051) — anything outside that range becomes NaN, imputed
      downstream by the shared median-imputer rather than trusted as a
      real age.
    - `genre_count`/`composer_count`: `genre_ids`/`composer` are
      pipe-separated multi-value strings for some songs (verified: ~7.9%
      of non-null `genre_ids`) — counted rather than just checked for
      presence.
    - `primary_genre`: the first listed genre code, fed to
      `RareCategoryGrouper` before one-hot encoding.
    - `has_composer`/`has_lyricist`: `composer`/`lyricist` are missing
      for ~47%/~85% of songs (verified directly) — far too sparse to use
      the actual names as features, but *whether* a song credits one at
      all is cheap, always-available signal.
    - `isrc_country`/`isrc_year`: parsed from `song_extra_info.isrc` —
      see `_parse_isrc_year`.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        reg = pd.to_datetime(df["registration_init_time"], format="%Y%m%d", errors="coerce")
        exp = pd.to_datetime(df["expiration_date"], format="%Y%m%d", errors="coerce")
        df["member_tenure_days"] = (exp - reg).dt.days

        low, high = config.VALID_AGE_RANGE
        df["age_clean"] = df["bd"].where(df["bd"].between(low, high))

        genre_ids = df["genre_ids"].fillna("")
        df["genre_count"] = genre_ids.apply(lambda s: len(s.split("|")) if s else 0)
        df["primary_genre"] = genre_ids.apply(lambda s: s.split("|")[0] if s else "unknown")

        composer = df["composer"].fillna("")
        df["composer_count"] = composer.apply(lambda s: len(s.split("|")) if s else 0)
        df["has_composer"] = (df["composer"].notna()).astype(int)
        df["has_lyricist"] = (df["lyricist"].notna()).astype(int)

        isrc = df["isrc"].fillna("")
        df["isrc_country"] = isrc.str.slice(0, 2).replace("", "unknown")
        df["isrc_year"] = _parse_isrc_year(df["isrc"])

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(
            base
            + ["member_tenure_days", "age_clean", "genre_count", "primary_genre"]
            + ["composer_count", "has_composer", "has_lyricist", "isrc_country", "isrc_year"]
        )


def build_preprocessor() -> ColumnTransformer:
    """Numeric columns: median-impute, then standard-scale. Categorical
    columns: most-frequent-impute, then one-hot encode. `sparse_output=True`
    (the default) matters here — this panel is 7.4M rows, and even with
    `artist_name` excluded, dense one-hot output would need several GB
    for no benefit most of these models need.
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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_COLS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )


def build_feature_pipeline() -> Pipeline:
    """InteractionCountEncoder + FeatureEngineer + RareCategoryGrouper +
    preprocessor, without a final estimator.
    """
    return Pipeline(
        steps=[
            ("interaction_counts", InteractionCountEncoder()),
            ("engineer", FeatureEngineer()),
            ("group_rare", RareCategoryGrouper()),
            ("preprocess", build_preprocessor()),
        ]
    )
