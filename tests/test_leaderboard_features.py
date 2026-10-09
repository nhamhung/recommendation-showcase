"""Tests for the KKBox leaderboard feature pipeline (synthetic data, no download).

Covers the parts where a silent bug would corrupt a submission without any
error: the full-data guard, leakage-free block target encodings, and the
row-order window / gap features.
"""

import numpy as np
import pandas as pd
import pytest

from recommendation_showcase.collaborative import config, data
from recommendation_showcase.collaborative import leaderboard_features as lf


def _panel(msno, target, t=None):
    n = len(msno)
    return pd.DataFrame({
        "msno": msno,
        "song_id": np.arange(n) % 3,
        "artist_name": np.zeros(n, dtype=int),
        "genre_ids": np.zeros(n, dtype=int),
        "source_screen_name": np.zeros(n, dtype=int),
        "source_type": np.zeros(n, dtype=int),
        config.TARGET_COL: target,
        "t": np.arange(n) if t is None else t,
    })


class TestFullDataGuard:
    def test_refuses_demo_sized_data(self, monkeypatch):
        small = pd.DataFrame({"msno": ["a"], "song_id": ["s"], config.TARGET_COL: [1], config.ID_COL: [0]})
        monkeypatch.setattr(data, "load_train", lambda: small.drop(columns=[config.ID_COL]))
        monkeypatch.setattr(data, "load_test", lambda: small.drop(columns=[config.TARGET_COL]))
        with pytest.raises(RuntimeError, match="full KKBox data"):
            lf.load_interactions()

    def test_forces_full_data_flag(self, monkeypatch):
        monkeypatch.delenv("USE_FULL_KAGGLE_DATA", raising=False)
        lf._use_full_data()
        import os
        assert os.environ["USE_FULL_KAGGLE_DATA"] == "true"


class TestBlockTargetEncoding:
    def test_uses_only_earlier_blocks(self):
        # One user, 6 rows, blocks of 2 rows: [0,1] [2,3] [4,5].
        table = _panel(msno=[0] * 6, target=[1, 1, 0, 0, 1, 1])
        out = lf.add_target_encodings(table, np.ones(6, dtype=bool), block_size=2, smoothing=0.0)
        # Block 0 has no earlier labels; block 1 sees block 0 (2 ones); block 2 sees blocks 0-1.
        np.testing.assert_array_equal(out["te_n_msno"].to_numpy(), [0, 0, 2, 2, 4, 4])
        np.testing.assert_allclose(out["te_msno"].to_numpy()[2:], [1.0, 1.0, 0.5, 0.5])

    def test_unavailable_labels_are_never_used(self):
        table = _panel(msno=[0] * 4, target=[1, 1, 0, 0])
        available = np.array([True, True, False, False])  # e.g. validation rows
        out = lf.add_target_encodings(table, available, block_size=1, smoothing=0.0)
        # Rows 2 and 3 see only rows 0-1 (the unavailable row 2 doesn't feed row 3).
        np.testing.assert_array_equal(out["te_n_msno"].to_numpy(), [0, 1, 2, 2])

    def test_keys_are_independent(self):
        table = _panel(msno=[0, 1, 0, 1], target=[1, 0, 1, 0])
        out = lf.add_target_encodings(table, np.ones(4, dtype=bool), block_size=2, smoothing=0.0)
        np.testing.assert_allclose(out["te_msno"].to_numpy()[2:], [1.0, 0.0])


class TestRowOrderFeatures:
    def test_window_counts(self):
        df = pd.DataFrame({"msno": [0, 1, 0, 0, 1], "t": np.arange(5)})
        lf._window_counts(df, ["msno"], "user", windows=(2,))
        np.testing.assert_array_equal(df["user_count_prev_2"], [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(df["user_count_next_2"], [1, 0, 1, 0, 0])

    def test_gap_features(self):
        df = pd.DataFrame({"msno": [0, 1, 0, 0], "t": np.arange(4)})
        lf._gap_features(df, ["msno"], "user")
        np.testing.assert_array_equal(df["user_prev_gap"].fillna(-1), [-1, -1, 2, 1])
        np.testing.assert_array_equal(df["user_next_gap"].fillna(-1), [2, -1, 1, -1])
