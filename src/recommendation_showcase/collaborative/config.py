"""Paths, constants, and column schema for the KKBox Music Recommendation
dataset.

Dataset: Kaggle's "WSDM - KKBox's Music Recommendation Challenge"
(https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge),
2018. Real user listening logs from KKBox, a Taiwan-based music streaming
service: predict whether a user will listen to a given song again within
a month of their first observed listening event for it. Unlike
`music_recommendation` (content-based, Spotify audio features, no user
data at all), this is genuine collaborative filtering — the whole task
is defined by real user/song interaction history.

Schema verified directly against the real downloaded CSVs (not assumed):
`train.csv`/`test.csv` are (msno, song_id, source_system_tab,
source_screen_name, source_type, target); `songs.csv` adds per-song
metadata; `members.csv` adds per-user metadata; `song_extra_info.csv`
adds a track name and ISRC code. `target` is close to balanced (50.4%/
49.6%) — no class-imbalance correction needed, unlike academic_success.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "collaborative" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "collaborative" / "processed"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
SONGS_CSV = DATA_RAW_DIR / "songs.csv"
MEMBERS_CSV = DATA_RAW_DIR / "members.csv"
SONG_EXTRA_INFO_CSV = DATA_RAW_DIR / "song_extra_info.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "sample_submission.csv"
MODEL_PATH = MODELS_DIR / "collaborative_model.joblib"

DEMO_TRAIN_CSV = DEMO_DATA_DIR / "collaborative_train.csv"
DEMO_SONGS_CSV = DEMO_DATA_DIR / "collaborative_songs.csv"
DEMO_MEMBERS_CSV = DEMO_DATA_DIR / "collaborative_members.csv"
DEMO_SONG_EXTRA_INFO_CSV = DEMO_DATA_DIR / "collaborative_song_extra_info.csv"

# --- Kaggle competition ---------------------------------------------------

KAGGLE_COMPETITION = "kkbox-music-recommendation-challenge"

# --- Keys & target -------------------------------------------------------

ID_COL = "id"  # test.csv only — the row id the submission maps predictions to
USER_COL = "msno"
SONG_COL = "song_id"
TARGET_COL = "target"

RANDOM_SEED = 42

# --- Raw column schema (verified against the real CSVs) ------------------

INTERACTION_CONTEXT_COLS = ["source_system_tab", "source_screen_name", "source_type"]
SONG_RAW_COLS = ["song_length", "genre_ids", "artist_name", "composer", "lyricist", "language"]
MEMBER_RAW_COLS = ["city", "bd", "gender", "registered_via", "registration_init_time", "expiration_date"]

# --- Known data-quality issues (verified directly, not assumed) ----------
# `bd` (self-reported age) is garbage for ~58% of members: 0 (the default/
# unset value), negative, or absurd (max observed: 1051). `FeatureEngineer`
# treats anything outside this range as missing rather than a real age.
VALID_AGE_RANGE = (5, 100)

# `gender` is missing for ~58% of members, `composer` for ~47% of songs,
# `lyricist` for ~85% — all three are used only as "is this present"
# flags plus a coarse category, not as rich per-value features, given how
# sparse they are.

# --- Engineered feature names --------------------------------------------
# `artist_name` (222,363 distinct values — verified directly) is far too
# high-cardinality to one-hot encode: `artist_song_count` (a frequency
# encoding — how many catalog songs share this artist, learned from
# training data only) replaces it as a numeric feature instead.
# `primary_genre` (182 distinct values — verified directly) is moderate,
# not huge: `RareCategoryGrouper` (same technique as academic_success)
# handles its long tail before one-hot encoding, rather than needing a
# frequency encoding of its own.

ENGINEERED_NUMERIC_COLS = [
    "member_tenure_days",
    "age_clean",
    "genre_count",
    "composer_count",
    "song_popularity",
    "user_activity_count",
    "artist_song_count",
    "isrc_year",
]
ENGINEERED_CATEGORICAL_COLS = ["isrc_country", "primary_genre", "has_composer", "has_lyricist"]

# High-cardinality categorical columns needing RareCategoryGrouper before
# one-hot encoding (see features.py) — just `primary_genre` here, unlike
# academic_success's several occupation/nationality columns.
RARE_GROUPED_COLS = ["primary_genre"]

CATEGORICAL_COLS = (
    INTERACTION_CONTEXT_COLS
    + ["city", "gender", "registered_via", "language"]
    + ENGINEERED_CATEGORICAL_COLS
)
NUMERIC_COLS = ["song_length"] + ENGINEERED_NUMERIC_COLS
