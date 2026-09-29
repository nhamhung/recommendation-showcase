"""Tests for feature engineering, using a small synthetic panel instead
of the real (multi-GB, non-redistributable) KKBox data.
"""

import numpy as np
import pandas as pd

from recommendation_showcase.collaborative import config
from recommendation_showcase.collaborative.features import (
    FeatureEngineer,
    InteractionCountEncoder,
    RareCategoryGrouper,
    build_feature_pipeline,
)


def make_synthetic_panel(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """One row per listening event, with all raw columns
    `data.merge_side_tables` would have produced, plus `target`.
    """
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(30)]
    songs = [f"song_{i}" for i in range(60)]
    artists = [f"artist_{i}" for i in range(15)]

    df = pd.DataFrame(
        {
            config.USER_COL: rng.choice(users, n_rows),
            config.SONG_COL: rng.choice(songs, n_rows),
            "source_system_tab": rng.choice(["explore", "my library", "search"], n_rows),
            "source_screen_name": rng.choice(["Explore", "Local playlist more"], n_rows),
            "source_type": rng.choice(["online-playlist", "local-playlist"], n_rows),
            "song_length": rng.integers(60_000, 400_000, n_rows),
            "genre_ids": rng.choice(["465", "958|465", "1609", None], n_rows),
            "artist_name": rng.choice(artists, n_rows),
            "composer": rng.choice([None, "Some Composer", "A|B"], n_rows),
            "lyricist": rng.choice([None, "Some Lyricist"], n_rows),
            "language": rng.choice([3.0, 52.0, -1.0], n_rows),
            "city": rng.integers(1, 22, n_rows),
            "bd": rng.choice([0, -5, 25, 30, 1051], n_rows),
            "gender": rng.choice([None, "male", "female"], n_rows),
            "registered_via": rng.integers(1, 7, n_rows),
            "registration_init_time": rng.choice([20110820, 20150628, 20130101], n_rows),
            "expiration_date": rng.choice([20170920, 20170622, 20180101], n_rows),
            "name": [f"Track {i}" for i in range(n_rows)],
            "isrc": rng.choice(["TWUM71200043", "QMZSY1600015", None], n_rows),
            "target": rng.integers(0, 2, n_rows),
        }
    )
    return df


def test_interaction_count_encoder_counts_from_training_rows_only():
    panel = make_synthetic_panel()
    encoder = InteractionCountEncoder().fit(panel)
    transformed = encoder.transform(panel)

    # A song appearing 3 times in the fit data must show song_popularity == 3
    # on every row for that song.
    top_song = panel[config.SONG_COL].value_counts().idxmax()
    expected_count = panel[config.SONG_COL].value_counts().max()
    rows = transformed[transformed[config.SONG_COL] == top_song]
    assert (rows["song_popularity"] == expected_count).all()


def test_interaction_count_encoder_unseen_id_gets_zero():
    panel = make_synthetic_panel()
    encoder = InteractionCountEncoder().fit(panel)

    unseen = panel.iloc[[0]].copy()
    unseen[config.SONG_COL] = "never_seen_song"
    unseen[config.USER_COL] = "never_seen_user"
    unseen["artist_name"] = "never_seen_artist"
    transformed = encoder.transform(unseen)

    assert transformed["song_popularity"].iloc[0] == 0
    assert transformed["user_activity_count"].iloc[0] == 0
    assert transformed["artist_song_count"].iloc[0] == 0


def test_feature_engineer_cleans_garbage_ages():
    panel = make_synthetic_panel()
    engineered = FeatureEngineer().fit_transform(panel)

    low, high = config.VALID_AGE_RANGE
    valid_mask = panel["bd"].between(low, high)
    # Garbage ages (0, -5, 1051) must become NaN, not pass through.
    assert engineered.loc[~valid_mask, "age_clean"].isna().all()
    assert (engineered.loc[valid_mask, "age_clean"] == panel.loc[valid_mask, "bd"]).all()


def test_feature_engineer_parses_isrc_year_with_century_pivot():
    panel = make_synthetic_panel(n_rows=10)
    panel["isrc"] = "TWUM71200043"  # 2-digit year "12" -> 2012 (<=17 pivot)
    engineered = FeatureEngineer().fit_transform(panel)
    assert (engineered["isrc_year"] == 2012).all()

    panel["isrc"] = "TWUM79900043"  # 2-digit year "99" -> 1999 (>17 pivot)
    engineered2 = FeatureEngineer().fit_transform(panel)
    assert (engineered2["isrc_year"] == 1999).all()


def test_feature_engineer_counts_pipe_separated_genres():
    panel = make_synthetic_panel(n_rows=5)
    panel["genre_ids"] = ["465", "958|465", None, "1|2|3", ""]
    engineered = FeatureEngineer().fit_transform(panel)
    assert list(engineered["genre_count"]) == [1, 2, 0, 3, 0]


def test_rare_category_grouper_collapses_low_frequency_categories():
    panel = make_synthetic_panel(n_rows=200)
    panel["primary_genre"] = ["common"] * 190 + ["rare_a"] * 5 + ["rare_b"] * 5

    grouper = RareCategoryGrouper(columns=["primary_genre"], min_frequency=0.05).fit(panel)
    transformed = grouper.transform(panel)

    assert (transformed.loc[panel["primary_genre"] == "common", "primary_genre"] == "common").all()
    assert (transformed.loc[panel["primary_genre"].isin(["rare_a", "rare_b"]), "primary_genre"] == "rare").all()


def test_full_feature_pipeline_runs_end_to_end():
    panel = make_synthetic_panel()
    X = panel.drop(columns=["target"])
    y = panel["target"]

    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
