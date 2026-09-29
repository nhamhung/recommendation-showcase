"""Paths, constants, and column schema for the Spotify Tracks dataset.

Dataset: Kaggle's "Spotify Tracks Dataset"
(https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset),
~114k tracks with Spotify's per-track audio features plus genre and
popularity. This is a Kaggle *dataset*, not a competition — there's no
leaderboard or submission file; the deliverable is a working recommender,
not a scored prediction (see `scripts/recommend.py` in place of the other
two projects' `make_submission.py`).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "content_based" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "content_based" / "processed"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"
MODELS_DIR = PROJECT_ROOT / "models"

TRACKS_CSV = DATA_RAW_DIR / "dataset.csv"
DEMO_TRACKS_CSV = DEMO_DATA_DIR / "content_tracks.csv"
MODEL_PATH = MODELS_DIR / "content_based_model.joblib"

# --- Kaggle dataset ------------------------------------------------------

KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"

# --- Keys & labels -------------------------------------------------------

ID_COL = "track_id"
NAME_COLS = ["track_name", "artists", "album_name"]

# Not a supervised target for the recommender itself — it's the
# ground-truth signal used only to *evaluate* recommendation quality
# (precision@k) and as the label for the secondary genre classifier used
# for SHAP interpretability and distance-metric weighting. Never included
# in the recommender's own feature space (see features.py).
GENRE_COL = "track_genre"
RANDOM_SEED = 42

# --- Column schema -----------------------------------------------------
# Spotify's own audio-feature columns, present as-is in the raw CSV.
RAW_AUDIO_NUMERIC_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
RAW_AUDIO_CATEGORICAL_COLS = ["mode", "time_signature", "explicit"]

# `key` (0-11, one of the 12 pitch classes) is circular, not ordinal or a
# plain nominal code — pitch class 11 (B) is one semitone from pitch
# class 0 (C), not "far away" from it. FeatureEngineer replaces it with
# key_sin/key_cos rather than one-hot or leaving it as a raw integer.
KEY_COL = "key"
ENGINEERED_NUMERIC_COLS = ["key_sin", "key_cos", "duration_min"]
ENGINEERED_CATEGORICAL_COLS = ["mood_quadrant"]

# Popularity is deliberately excluded from the recommender's feature
# space: it's a measure of how many people played a track, not a
# property of what the track sounds like — including it would make the
# recommender chase popularity rather than sonic similarity.
POPULARITY_COL = "popularity"

NUMERIC_COLS = RAW_AUDIO_NUMERIC_COLS + ["duration_min", "key_sin", "key_cos"]
CATEGORICAL_COLS = RAW_AUDIO_CATEGORICAL_COLS + ["mood_quadrant"]

# --- Recommendation ------------------------------------------------------

DEFAULT_TOP_K = 10
