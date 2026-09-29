"""Data loading and side-table merging.

Raw CSVs are not committed to the repo (large, and not ours to
redistribute). Download them first — see the project README for the
`kaggle competitions download` command.

Verified directly (not assumed) that the full join fits comfortably in
memory on a normal laptop: `train.csv` (7,377,418 rows) merged with
`songs.csv`, `members.csv`, and `song_extra_info.csv` takes about 16
seconds and under 1.5GB of RAM.
"""

from pathlib import Path

import pandas as pd

from . import config


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the competition data first — see "
            "the README's 'Get the data' section, e.g.:\n"
            f"  kaggle competitions download -c {config.KAGGLE_COMPETITION} "
            f"-p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_COMPETITION + '.zip')} "
            f"-d {config.DATA_RAW_DIR}\n"
            "  (then extract the six .csv.7z files inside with `7z x` or `py7zr`)"
        )
    return path


def _full_or_demo_path(full_path: Path, demo_path: Path) -> Path:
    """Prefer full local data, falling back to the compact hosted demo."""
    if full_path.exists():
        return full_path
    return _require_file(demo_path)


def using_demo_data() -> bool:
    return not config.TRAIN_CSV.exists() and config.DEMO_TRAIN_CSV.exists()


def load_train() -> pd.DataFrame:
    return pd.read_csv(_full_or_demo_path(config.TRAIN_CSV, config.DEMO_TRAIN_CSV))


def load_test() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.TEST_CSV))


def load_songs() -> pd.DataFrame:
    return pd.read_csv(
        _full_or_demo_path(config.SONGS_CSV, config.DEMO_SONGS_CSV),
        dtype={"genre_ids": "string"},
    )


def load_members() -> pd.DataFrame:
    return pd.read_csv(_full_or_demo_path(config.MEMBERS_CSV, config.DEMO_MEMBERS_CSV))


def load_song_extra_info() -> pd.DataFrame:
    return pd.read_csv(
        _full_or_demo_path(config.SONG_EXTRA_INFO_CSV, config.DEMO_SONG_EXTRA_INFO_CSV)
    )


def merge_side_tables(
    interactions_df: pd.DataFrame,
    songs_df: pd.DataFrame | None = None,
    members_df: pd.DataFrame | None = None,
    song_extra_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-merges song/member/track-name metadata onto an interactions
    frame (`train.csv` or `test.csv`, one row per listening event).
    """
    songs_df = songs_df if songs_df is not None else load_songs()
    members_df = members_df if members_df is not None else load_members()
    song_extra_df = song_extra_df if song_extra_df is not None else load_song_extra_info()

    merged = (
        interactions_df.merge(songs_df, on=config.SONG_COL, how="left")
        .merge(members_df, on=config.USER_COL, how="left")
        .merge(song_extra_df, on=config.SONG_COL, how="left")
    )
    return merged


def load_full_train_panel() -> pd.DataFrame:
    """The training interactions, merged with every side table."""
    return merge_side_tables(load_train())


def load_full_test_panel() -> pd.DataFrame:
    """The competition test interactions, merged with every side table.
    No `target` column — this is what a real submission is scored on.
    """
    return merge_side_tables(load_test())


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled panel into (X, y)."""
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Input frame is missing the target column {config.TARGET_COL!r}")
    y = df[config.TARGET_COL].copy()
    X = df.drop(columns=[config.TARGET_COL])
    return X, y
