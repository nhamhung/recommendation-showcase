"""Feature table for the leaderboard model (separate from the app pipeline).

The app model (`features.py` + `model.py`) is a scikit-learn pipeline built
for explainability: one-hot categories, frequency counts learned from the
training rows only. This module builds a different, much stronger feature
table purely for the Kaggle submission, using three properties of this
competition that the app pipeline deliberately ignores:

1. **IDs are features.** LightGBM can split on `msno`/`song_id`/`artist`
   directly as native categoricals — a user's own taste and a song's own
   replay appeal are the strongest signals in the data.
2. **Rows are in time order.** `train.csv` and then `test.csv` are sorted
   chronologically (test follows train), so row position `t` is a usable
   timestamp. "Replayed within a month" depends on what a user does
   *around* that moment — e.g. how soon they show up again.
3. **Counts over train + test.** Interaction counts are computed over all
   9.9M rows (train and test together), so a test row's features are
   computed exactly the same way as a training row's. No labels are used
   anywhere in this module — it is safe to compute on test.

Everything is integer-coded early (`pd.factorize`) so the full 9.9M-row
table stays within a laptop's RAM.
"""

import os

import numpy as np
import pandas as pd

from . import config, data

# The full competition files. data.py's loaders return small demo tables by
# default (so the Streamlit app starts without a Kaggle download); the
# leaderboard model must never be built from those, so this module forces the
# full data and checks it got it.
FULL_TRAIN_ROWS = 7_377_418
FULL_TEST_ROWS = 2_556_790


def _use_full_data() -> None:
    os.environ["USE_FULL_KAGGLE_DATA"] = "true"

# Columns LightGBM should treat as native categoricals.
CATEGORICAL_FEATURES = [
    "msno", "song_id", "source_system_tab", "source_screen_name", "source_type",
    "artist_name", "composer", "lyricist", "genre_ids", "first_genre", "second_genre", "language",
    "city", "gender", "registered_via", "isrc_country", "isrc_registrant",
    # What the same user played just before / after this row (see _context_features).
    "prev_song_id", "next_song_id", "prev_artist_name", "next_artist_name",
    "prev_source_type", "next_source_type", "prev_source_screen_name", "next_source_screen_name",
]

_STRING_KEYS = [
    "msno", "song_id", "source_system_tab", "source_screen_name", "source_type",
    "artist_name", "composer", "lyricist", "genre_ids", "first_genre", "second_genre", "gender",
    "isrc_country", "isrc_registrant",
]


def _codes(series: pd.Series) -> np.ndarray:
    """Integer codes, -1 for missing (LightGBM reads negative categories as missing)."""
    return pd.factorize(series, use_na_sentinel=True)[0].astype(np.int32)


def _lookup(keys: pd.Series, table: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    """Left-join `columns` of `table` onto `keys` by position (much lighter
    than `DataFrame.merge` on 9.9M string keys)."""
    positions = pd.Index(table[key]).get_indexer(keys)
    out = table[columns].iloc[np.where(positions >= 0, positions, 0)].reset_index(drop=True)
    out.loc[positions < 0, :] = np.nan
    return out


def load_interactions() -> pd.DataFrame:
    """train + test stacked in file order, with `t` (row position = time).

    Always the full competition files — raises if anything smaller (e.g. the
    app's demo tables) comes back, rather than silently building a broken model.
    """
    _use_full_data()
    train = data.load_train()
    test = data.load_test()
    if (len(train), len(test)) != (FULL_TRAIN_ROWS, FULL_TEST_ROWS):
        raise RuntimeError(
            f"Expected the full KKBox data ({FULL_TRAIN_ROWS:,} train / {FULL_TEST_ROWS:,} test rows), "
            f"got {len(train):,} / {len(test):,}. Download the competition files (see the README)."
        )
    train["is_test"] = 0
    test["is_test"] = 1
    test[config.TARGET_COL] = np.nan
    panel = pd.concat([train, test.drop(columns=[config.ID_COL])], ignore_index=True)
    panel["test_id"] = np.concatenate([np.full(len(train), -1), test[config.ID_COL].to_numpy()])
    panel["t"] = np.arange(len(panel), dtype=np.int32)
    return panel


def _group_count(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.groupby(cols, sort=False)[cols[0]].transform("size").to_numpy(np.int32)


def _gap_features(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Rows since this key's previous / until its next occurrence (NaN at the ends),
    plus this row's position within the key's history (0 = first, 1 = last)."""
    grouped = df.groupby(cols, sort=False)["t"]
    df[f"{name}_prev_gap"] = (df["t"] - grouped.shift(1)).astype(np.float32)
    df[f"{name}_next_gap"] = (grouped.shift(-1) - df["t"]).astype(np.float32)
    order = grouped.cumcount()
    total = grouped.transform("size")
    df[f"{name}_position"] = (order / np.maximum(total - 1, 1)).astype(np.float32)


def _window_counts(df: pd.DataFrame, cols: list[str], name: str, windows: tuple[int, ...]) -> None:
    """How many times this key occurs in the `w` rows before / after this row.

    Vectorised with one sort: encode (key, t) as a single sortable int64,
    then each window edge is a `searchsorted` away. "Is this user (or
    song, or user+artist pair) still active over the next stretch of time"
    is close to the definition of the target itself.
    """
    key = df.groupby(cols, sort=False).ngroup().to_numpy(np.int64)
    t = df["t"].to_numpy(np.int64)
    scale = int(t.max()) + 2 * max(windows) + 1
    encoded = key * scale + t + max(windows)
    sorted_encoded = np.sort(encoded)
    here = np.searchsorted(sorted_encoded, encoded)
    for w in windows:
        before = here - np.searchsorted(sorted_encoded, encoded - w)
        after = np.searchsorted(sorted_encoded, encoded + w, side="right") - here - 1
        df[f"{name}_count_prev_{w}"] = before.astype(np.int32)
        df[f"{name}_count_next_{w}"] = after.astype(np.int32)


WINDOWS = (10_000, 100_000, 1_000_000)
# Users and songs also get session-scale windows (a few to a few thousand rows).
SESSION_WINDOWS = (10, 25, 500, 5_000) + WINDOWS

# Calendar dates of the first train row, the train/test boundary and the
# last test row. Row position is mapped piecewise-linearly onto these, since
# test packs ~47 days into 2.56M rows while train spreads ~151 days over 7.38M.
TRAIN_START, TEST_START, TEST_END = (pd.Timestamp(d) for d in ("2016-08-14", "2017-01-12", "2017-02-27"))

# (item column, feature prefix, SVD rank, raw components kept as features).
# Raw components beyond the first few added almost nothing (validation AUC
# +0.0002 for keeping all 32 song components instead of 8), so most are
# dropped to keep the 9.9M-row table within laptop RAM; the dot products stay.
SVD_SPECS = [
    ("song_id", "song", 48, 16),
    ("artist_name", "artist", 32, 8),
    ("genre_ids", "genre", 16, 0),
    ("composer", "composer", 16, 0),
]


def _svd_features(df: pd.DataFrame, item_col: str, name: str, n_components: int, n_kept: int) -> dict:
    """Latent user and item vectors from the user x item co-occurrence
    matrix (train + test, no labels), via truncated SVD.

    Adds the user-item affinity (dot product of the two vectors — "how well
    does this item fit this user's listening pattern") and the first few
    raw components of each side.
    """
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    users = df["msno"].to_numpy()
    items = df[item_col].to_numpy()
    valid = items >= 0
    matrix = csr_matrix(
        (np.ones(valid.sum(), dtype=np.float32), (users[valid], items[valid])),
        shape=(users.max() + 1, items.max() + 1),
    )
    matrix.data = np.log1p(matrix.data)  # damp heavy repeat listeners
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    user_vectors = svd.fit_transform(matrix).astype(np.float32)
    item_vectors = (svd.components_.T * 1.0).astype(np.float32)
    u = user_vectors[users]
    v = np.where(valid[:, None], item_vectors[np.maximum(items, 0)], np.nan)
    columns = {f"svd_{name}_dot": np.einsum("ij,ij->i", u, v).astype(np.float32)}
    for k in range(n_kept):
        columns[f"svd_{name}_user_{k}"] = u[:, k]
        columns[f"svd_{name}_item_{k}"] = v[:, k].astype(np.float32)
    return columns


def _context_features(df: pd.DataFrame) -> None:
    """What the same user played immediately before and after this row:
    song, artist, and where it was played from — as categoricals — plus
    whether the artist / source type match this row's. A run of songs by one
    artist from the user's own library looks very different from a one-off
    radio play in between other things."""
    grouped = df.groupby("msno", sort=False)
    for col in ("song_id", "artist_name", "source_type", "source_screen_name"):
        prev = grouped[col].shift(1)
        nxt = grouped[col].shift(-1)
        df[f"prev_{col}"] = prev.fillna(-1).astype(np.int32)
        df[f"next_{col}"] = nxt.fillna(-1).astype(np.int32)
    for col in ("artist_name", "source_type"):
        df[f"prev_same_{col}"] = (df[f"prev_{col}"] == df[col]).astype(np.int8)
        df[f"next_same_{col}"] = (df[f"next_{col}"] == df[col]).astype(np.int8)


def _calendar_day(t: np.ndarray, n_train: int, n_total: int) -> np.ndarray:
    """Row position -> approximate days since 2017-01-01."""
    origin = pd.Timestamp("2017-01-01")
    start, mid, end = ((d - origin).days for d in (TRAIN_START, TEST_START, TEST_END))
    return np.where(
        t < n_train,
        start + t / max(n_train - 1, 1) * (mid - start),
        mid + (t - n_train) / max(n_total - n_train - 1, 1) * (end - mid),
    ).astype(np.float32)


def build_feature_table() -> pd.DataFrame:
    """The full 9.9M-row table: one row per train/test interaction."""
    _use_full_data()  # songs / members / song_extra_info lookups below must be the full tables too
    df = load_interactions()

    songs = data.load_songs()
    song_cols = ["song_length", "genre_ids", "artist_name", "composer", "lyricist", "language"]
    df[song_cols] = _lookup(df[config.SONG_COL], songs, config.SONG_COL, song_cols).to_numpy()
    del songs

    extra = data.load_song_extra_info()
    isrc = _lookup(df[config.SONG_COL], extra, config.SONG_COL, ["isrc"])["isrc"]
    del extra
    df["isrc_country"] = isrc.str.slice(0, 2)
    df["isrc_registrant"] = isrc.str.slice(2, 5)
    year2 = pd.to_numeric(isrc.str.slice(5, 7), errors="coerce")
    df["isrc_year"] = np.where(year2 > 17, 1900 + year2, 2000 + year2).astype(np.float32)

    members = data.load_members()
    member_cols = ["city", "bd", "gender", "registered_via", "registration_init_time", "expiration_date"]
    df[member_cols] = _lookup(df[config.USER_COL], members, config.USER_COL, member_cols).to_numpy()
    del members

    genres = df["genre_ids"].astype("string").str.split("|")
    df["first_genre"] = genres.str[0]
    df["second_genre"] = genres.str[1]
    del genres
    df["genre_count"] = df["genre_ids"].astype("string").str.count(r"\|").add(1).fillna(0).astype(np.int8)
    df["composer_count"] = df["composer"].astype("string").str.count(r"\||/|,").add(1).fillna(0).astype(np.int8)

    for col in _STRING_KEYS:
        df[col] = _codes(df[col])
    for col in ["song_length", "language", "city", "bd", "registered_via"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    df["language"] = df["language"].fillna(-2).astype(np.int16)
    df["city"] = df["city"].fillna(-1).astype(np.int16)
    df["registered_via"] = df["registered_via"].fillna(-1).astype(np.int16)
    df["bd"] = df["bd"].where(df["bd"].between(*config.VALID_AGE_RANGE)).astype(np.float32)

    reg = pd.to_datetime(df["registration_init_time"].astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    exp = pd.to_datetime(df["expiration_date"].astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    df["registration_year"] = reg.dt.year.astype(np.float32)
    df["membership_days"] = (exp - reg).dt.days.astype(np.float32)
    df["expiration_days"] = (exp - pd.Timestamp("2017-01-01")).dt.days.astype(np.float32)
    registration_days = (reg - pd.Timestamp("2017-01-01")).dt.days.astype(np.float32)
    df = df.drop(columns=["registration_init_time", "expiration_date"])

    n_train = int((df["is_test"] == 0).sum())
    df["day"] = _calendar_day(df["t"].to_numpy(), n_train, len(df))
    df["days_until_expiration"] = (df["expiration_days"] - df["day"]).astype(np.float32)
    df["days_since_registration"] = (df["day"] - registration_days).astype(np.float32)

    # --- Counts over train + test --------------------------------------------
    for cols in (
        ["msno"], ["song_id"], ["artist_name"], ["composer"], ["lyricist"], ["genre_ids"],
        ["language"], ["source_type"], ["source_screen_name"],
        ["msno", "artist_name"], ["msno", "genre_ids"], ["msno", "language"],
        ["msno", "source_type"], ["msno", "source_system_tab"], ["msno", "source_screen_name"],
        ["song_id", "source_type"], ["song_id", "source_system_tab"], ["song_id", "source_screen_name"],
        ["artist_name", "source_type"], ["msno", "isrc_registrant"], ["msno", "isrc_year"],
        ["msno", "first_genre"],
    ):
        df["count_" + "_".join(cols)] = _group_count(df, cols)
    df["user_artist_share"] = (df["count_msno_artist_name"] / df["count_msno"]).astype(np.float32)
    df["user_genre_share"] = (df["count_msno_genre_ids"] / df["count_msno"]).astype(np.float32)
    df["user_source_type_share"] = (df["count_msno_source_type"] / df["count_msno"]).astype(np.float32)
    df["song_source_type_share"] = (df["count_song_id_source_type"] / df["count_song_id"]).astype(np.float32)
    # How typical this row's context / song attributes are *for this user* (and song).
    for key in ("source_system_tab", "source_screen_name", "language", "isrc_registrant", "isrc_year", "first_genre"):
        df[f"user_{key}_share"] = (df[f"count_msno_{key}"] / df["count_msno"]).astype(np.float32)
    for key in ("source_system_tab", "source_screen_name"):
        df[f"song_{key}_share"] = (df[f"count_song_id_{key}"] / df["count_song_id"]).astype(np.float32)
    df["user_mean_song_length"] = df.groupby("msno")["song_length"].transform("mean").astype(np.float32)
    df["artist_song_count"] = df.groupby("artist_name")["song_id"].transform("nunique").astype(np.int32)

    # --- Time (row order) features -------------------------------------------
    _gap_features(df, ["msno"], "user")
    _gap_features(df, ["song_id"], "song")
    _gap_features(df, ["msno", "artist_name"], "user_artist")
    _gap_features(df, ["msno", "source_type"], "user_source")

    _window_counts(df, ["msno"], "user", SESSION_WINDOWS)
    _window_counts(df, ["song_id"], "song", SESSION_WINDOWS)
    _window_counts(df, ["artist_name"], "artist", WINDOWS)
    _window_counts(df, ["msno", "artist_name"], "user_artist", WINDOWS)

    df["user_distinct_songs"] = df.groupby("msno")["song_id"].transform("nunique").astype(np.int32)
    df["user_distinct_artists"] = df.groupby("msno")["artist_name"].transform("nunique").astype(np.int32)
    df["song_distinct_users"] = df.groupby("song_id")["msno"].transform("nunique").astype(np.int32)
    df["user_repeat_ratio"] = (df["count_msno"] / df["user_distinct_songs"]).astype(np.float32)

    for key, name in (("msno", "user"), ("song_id", "song")):
        grouped = df.groupby(key, sort=False)["t"]
        df[f"{name}_plays_so_far"] = grouped.cumcount().astype(np.int32)
        df[f"{name}_t_mean"] = grouped.transform("mean").astype(np.float32)
        df[f"{name}_t_std"] = grouped.transform("std").astype(np.float32)
    df["t_vs_user_mean"] = ((df["t"] - df["user_t_mean"]) / (df["user_t_std"] + 1)).astype(np.float32)

    _context_features(df)

    svd_columns = {}
    for item_col, name, n_components, n_kept in SVD_SPECS:
        svd_columns.update(_svd_features(df, item_col, name, n_components, n_kept))
    return pd.concat([df, pd.DataFrame(svd_columns, index=df.index)], axis=1)


TARGET_ENCODING_KEYS = [
    ["msno"], ["song_id"], ["artist_name"], ["genre_ids"], ["source_screen_name"],
    ["msno", "artist_name"], ["msno", "genre_ids"], ["msno", "source_type"],
    ["msno", "source_screen_name"], ["song_id", "source_type"], ["artist_name", "source_type"],
]


def add_target_encodings(
    table: pd.DataFrame,
    label_available: np.ndarray,
    block_size: int = 1_500_000,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Replay rate of each key over labelled rows from *earlier time blocks*.

    `label_available` marks the rows whose target may be used. Rows are cut
    into blocks of `block_size` consecutive rows, and a row's encoding uses
    only labels from blocks strictly before its own. Why blocks rather than
    "every earlier row": validation/test rows only ever see labels up to a
    cutoff that can be weeks stale (test spans ~2.5M rows after the last
    label). An expanding encoding gives training rows labels from moments
    earlier — far fresher than anything test will have — and the model
    learns to trust a signal it won't get (verified: validation AUC fell
    0.741 -> 0.726). Blocks make training rows exactly as stale as test.

    For validation, pass a mask covering only the fit period; for the
    submission, every training row. Returns new columns (`te_<key>`
    smoothed rate, `te_n_<key>` how many labels it was based on).
    """
    y = np.where(label_available, table[config.TARGET_COL].fillna(0).to_numpy(), 0.0)
    n = label_available.astype(np.float64)
    prior = y.sum() / n.sum()
    block = (table["t"].to_numpy() // block_size).astype(np.int64)
    n_blocks = int(block.max()) + 1
    out = pd.DataFrame(index=table.index)
    for cols in TARGET_ENCODING_KEYS:
        name = "_".join(cols)
        group = table.groupby(cols, sort=False).ngroup().to_numpy(np.int64)
        cell = group * n_blocks + block
        cell_y = np.bincount(cell, weights=y, minlength=(group.max() + 1) * n_blocks)
        cell_n = np.bincount(cell, weights=n, minlength=(group.max() + 1) * n_blocks)
        # Labels from strictly earlier blocks: cumulative over blocks, minus own block.
        cum_y = cell_y.reshape(-1, n_blocks).cumsum(axis=1).ravel() - cell_y
        cum_n = cell_n.reshape(-1, n_blocks).cumsum(axis=1).ravel() - cell_n
        past_y, past_n = cum_y[cell], cum_n[cell]
        out[f"te_{name}"] = ((past_y + smoothing * prior) / (past_n + smoothing)).astype(np.float32)
        out[f"te_n_{name}"] = past_n.astype(np.float32)
    return out


def train_test_frames(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return table[table["is_test"] == 0], table[table["is_test"] == 1]


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in (config.TARGET_COL, "is_test", "test_id")]
