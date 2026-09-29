"""Data loading helpers.

The raw CSV is not committed to the repo (large, and best fetched fresh
from Kaggle rather than duplicated here). Download it first — see the
project README for the `kaggle datasets download` command.
"""

from pathlib import Path

import pandas as pd

from . import config


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first — see the "
            "README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_DATASET.split('/')[-1].lstrip('-') + '.zip')} "
            f"-d {config.DATA_RAW_DIR}"
        )
    return path


def _tracks_path() -> Path:
    """Prefer the full local catalog, falling back to the hosted demo sample."""
    if config.TRACKS_CSV.exists():
        return config.TRACKS_CSV
    return _require_file(config.DEMO_TRACKS_CSV)


def using_demo_data() -> bool:
    return not config.TRACKS_CSV.exists() and config.DEMO_TRACKS_CSV.exists()


def load_tracks() -> pd.DataFrame:
    """Load the full tracks table, deduplicated and indexed by `track_id`.

    The raw CSV has a meaningless leading index column (`Unnamed: 0`) and
    a small number of exact-duplicate `track_id` rows (the same track
    catalogued under more than one genre tag) — the first occurrence is
    kept, since the recommender treats each track as a single point in
    feature space regardless of which genre tag happened to be attached.
    """
    df = pd.read_csv(_tracks_path())
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop_duplicates(subset=config.ID_COL, keep="first")
    return df.set_index(config.ID_COL)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into (X, y) for the genre classifier / recommendation
    evaluation — `y` is `track_genre`, used only as a label/evaluation
    signal, never as a recommender input feature (see `config.py`).
    """
    if config.GENRE_COL not in df.columns:
        raise ValueError(f"Input frame is missing {config.GENRE_COL!r}")
    y = df[config.GENRE_COL].copy()
    feature_cols = (
        config.RAW_AUDIO_NUMERIC_COLS + [config.KEY_COL] + config.RAW_AUDIO_CATEGORICAL_COLS + ["duration_ms"]
    )
    X = df[feature_cols].copy()
    return X, y
