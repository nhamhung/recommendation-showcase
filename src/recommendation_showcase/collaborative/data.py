"""Data loading and side-table merging.

Raw CSVs are not committed to the repo (large, and not ours to
redistribute). Download them first — see the project README for the
`kaggle competitions download` command — or let `_require_file` fetch
them automatically via the Kaggle API.

**This one deserves a specific caveat the rest of this portfolio's
auto-download doesn't need**: verified directly, this competition's
four needed files are `.7z`-compressed and total **~360MB compressed**
(train.csv.7z alone is ~106MB, decompressing to ~975MB). That's a real
cost to pay at app startup, not a quick fetch — likely impractical on a
free-tier cloud host with limited disk/RAM or a cold-start timeout.
For this specific sub-project, baking the data into a Docker image
ahead of time (this project's original deployment approach) is the
more realistic choice; auto-download is implemented here for
consistency and for less-constrained deployment targets, not as the
recommended path for this particular dataset's size.

Verified directly (not assumed) that the full join fits comfortably in
memory on a normal laptop: `train.csv` (7,377,418 rows) merged with
`songs.csv`, `members.csv`, and `song_extra_info.csv` takes about 16
seconds and under 1.5GB of RAM.
"""

from pathlib import Path

import pandas as pd

from . import config

# The competition's files are `.7z`-compressed, unlike a plain Kaggle
# dataset's `unzip=True` convenience — each one is fetched and
# decompressed individually.
_KAGGLE_7Z_FILES = ["train.csv", "test.csv", "songs.csv", "members.csv", "song_extra_info.csv"]


def _download_from_kaggle() -> bool:
    """Best-effort automatic fetch via the Kaggle API — see the module
    docstring for why this is a heavier operation here than in this
    portfolio's other projects. Returns whether every needed file
    exists afterward. Silently does nothing (returns False) if the
    `kaggle`/`py7zr` packages aren't installed, no credentials are
    configured, or this account hasn't accepted the competition's rules
    on kaggle.com yet — callers fall back to the manual-download error
    message either way.
    """
    try:
        import py7zr
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        for name in _KAGGLE_7Z_FILES:
            csv_path = config.DATA_RAW_DIR / name
            if csv_path.exists():
                continue
            archive_path = config.DATA_RAW_DIR / f"{name}.7z"
            api.competition_download_file(config.KAGGLE_COMPETITION, f"{name}.7z", path=str(config.DATA_RAW_DIR), quiet=True)
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                archive.extractall(path=config.DATA_RAW_DIR)
            archive_path.unlink(missing_ok=True)
    except Exception:
        return False
    return all((config.DATA_RAW_DIR / name).exists() for name in _KAGGLE_7Z_FILES)


def _require_file(path: Path) -> Path:
    if not path.exists():
        _download_from_kaggle()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, and automatic download via the Kaggle API "
            "didn't produce it either (no credentials configured, the "
            "`kaggle`/`py7zr` packages aren't installed, this account "
            "hasn't accepted the competition's rules on kaggle.com yet, or "
            "the ~360MB download was interrupted — see this module's "
            "docstring for why this one is heavier than this portfolio's "
            "other auto-downloads). Download the competition data manually "
            "instead — see the README's 'Get the data' section, e.g.:\n"
            f"  kaggle competitions download -c {config.KAGGLE_COMPETITION} "
            f"-p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_COMPETITION + '.zip')} "
            f"-d {config.DATA_RAW_DIR}\n"
            "  (then extract the six .csv.7z files inside with `7z x` or `py7zr`)"
        )
    return path


def load_train() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.TRAIN_CSV))


def load_test() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.TEST_CSV))


def load_songs() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.SONGS_CSV))


def load_members() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.MEMBERS_CSV))


def load_song_extra_info() -> pd.DataFrame:
    return pd.read_csv(_require_file(config.SONG_EXTRA_INFO_CSV))


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
