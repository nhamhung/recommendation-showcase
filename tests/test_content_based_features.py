"""Tests for feature engineering, using a small synthetic catalog instead
of the real (114k-row) Spotify dataset.
"""

import numpy as np
import pandas as pd

from recommendation_showcase.content_based import config
from recommendation_showcase.content_based.features import FeatureEngineer, build_feature_pipeline


def make_synthetic_catalog(n_tracks: int = 200, seed: int = 0) -> pd.DataFrame:
    """One row per track, indexed by a synthetic `track_id`, with all the
    raw columns `data.split_features_target` expects plus `track_genre`/
    `popularity` for evaluation helpers that need them.
    """
    rng = np.random.default_rng(seed)
    genres = ["pop", "rock", "classical", "edm"]
    df = pd.DataFrame(
        {
            "danceability": rng.uniform(0, 1, n_tracks),
            "energy": rng.uniform(0, 1, n_tracks),
            "loudness": rng.uniform(-60, 0, n_tracks),
            "speechiness": rng.uniform(0, 1, n_tracks),
            "acousticness": rng.uniform(0, 1, n_tracks),
            "instrumentalness": rng.uniform(0, 1, n_tracks),
            "liveness": rng.uniform(0, 1, n_tracks),
            "valence": rng.uniform(0, 1, n_tracks),
            "tempo": rng.uniform(60, 200, n_tracks),
            "key": rng.integers(0, 12, n_tracks),
            "mode": rng.integers(0, 2, n_tracks),
            "time_signature": rng.integers(3, 8, n_tracks),
            "explicit": rng.integers(0, 2, n_tracks).astype(bool),
            "duration_ms": rng.integers(90_000, 360_000, n_tracks),
            "popularity": rng.integers(0, 100, n_tracks),
            "track_genre": rng.choice(genres, n_tracks),
        },
        index=pd.Index([f"track_{i}" for i in range(n_tracks)], name=config.ID_COL),
    )
    return df


def test_feature_engineer_key_encoding_is_circular():
    catalog = make_synthetic_catalog(n_tracks=20)
    engineered = FeatureEngineer().fit_transform(catalog)

    # Pitch class 0 and pitch class 11 are one semitone apart, so their
    # (sin, cos) points must be close on the unit circle...
    row_key0 = engineered[catalog["key"] == 0].iloc[0]
    row_key11 = engineered[catalog["key"] == 11].iloc[0] if (catalog["key"] == 11).any() else None
    if row_key11 is not None:
        dist_adjacent = np.hypot(
            row_key0["key_sin"] - row_key11["key_sin"], row_key0["key_cos"] - row_key11["key_cos"]
        )
        # ...much closer than pitch class 0 to pitch class 6 (the opposite
        # side of the circle, a tritone away).
        row_key6 = engineered[catalog["key"] == 6].iloc[0] if (catalog["key"] == 6).any() else None
        if row_key6 is not None:
            dist_opposite = np.hypot(
                row_key0["key_sin"] - row_key6["key_sin"], row_key0["key_cos"] - row_key6["key_cos"]
            )
            assert dist_adjacent < dist_opposite


def test_feature_engineer_adds_expected_columns():
    catalog = make_synthetic_catalog(n_tracks=20)
    engineered = FeatureEngineer().fit_transform(catalog)

    for col in config.ENGINEERED_NUMERIC_COLS + config.ENGINEERED_CATEGORICAL_COLS:
        assert col in engineered.columns
    assert engineered["mood_quadrant"].isin(
        ["energetic_happy", "energetic_dark", "calm_happy", "calm_dark"]
    ).all()


def test_full_feature_pipeline_runs_end_to_end():
    catalog = make_synthetic_catalog(n_tracks=50)
    X = catalog[
        config.RAW_AUDIO_NUMERIC_COLS
        + [config.KEY_COL]
        + config.RAW_AUDIO_CATEGORICAL_COLS
        + ["duration_ms"]
    ]

    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).all()
