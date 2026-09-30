"""Cached data/model loaders shared across every page, for both the
content-based (Spotify) and collaborative-filtering (KKBox) sides.

`sys.path` is set up once, in `streamlit_app.py`, before any page module
(including this one) is imported — see its docstring.
"""

import os

import pandas as pd
import streamlit as st

from recommendation_showcase.content_based import data as cb_data
from recommendation_showcase.content_based import interpretability as cb_interpretability
from recommendation_showcase.content_based import model as cb_model
from recommendation_showcase.collaborative import config as coll_config
from recommendation_showcase.collaborative import data as coll_data
from recommendation_showcase.collaborative import interpretability as coll_interpretability
from recommendation_showcase.collaborative import model as coll_model

# The KKBox training panel is 7.4M rows — fine for a one-off script, but
# too slow to recompute live on every page load/rerun. Pages that need
# to demonstrate model comparisons or SHAP work off this fixed subsample
# instead (the same tradeoff academic_success makes for its own
# expensive stacking-ensemble comparison).
COLL_SWEEP_SAMPLE_SIZE = 300_000


def _configure_kaggle_credentials() -> None:
    """Wire Kaggle API credentials from Streamlit secrets into the
    environment variables the `kaggle` package reads, so a deployment
    without a pre-baked Docker image (e.g. Streamlit Community Cloud)
    can fetch the datasets automatically on first load — see each
    sub-package's `data._download_from_kaggle`. Note the KKBox
    (collaborative-filtering) download is ~360MB compressed — likely
    impractical on a constrained free-tier host regardless of whether
    credentials are configured; see `collaborative/data.py`'s module
    docstring. A no-op if real environment variables are already set
    (e.g. running locally) or no `[kaggle]` secret is configured (falls
    back to `~/.kaggle/kaggle.json` if present, or to the
    manual-download error message if not).
    """
    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            return
    except Exception:
        pass
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
    except Exception:
        pass


_configure_kaggle_credentials()


# --- Content-based (Spotify) -----------------------------------------------


@st.cache_resource
def get_cb_recommender():
    return cb_model.load_pipeline()


@st.cache_data
def get_cb_tracks_df() -> pd.DataFrame:
    return cb_data.load_tracks()


@st.cache_data(show_spinner="Computing SHAP values for the genre classifier (first load only)...")
def get_cb_shap_explanation(sample_size: int = 500):
    tracks_df = get_cb_tracks_df()
    X, y = cb_data.split_features_target(tracks_df)
    classifier_pipeline = cb_model.build_classifier_pipeline(cb_model.random_forest_estimator())
    classifier_pipeline.fit(X, y)
    explanation, X_transformed = cb_interpretability.compute_shap_values(
        classifier_pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed


# --- Collaborative filtering (KKBox) ---------------------------------------


@st.cache_resource
def get_coll_pipeline():
    return coll_model.load_pipeline()


@st.cache_data
def get_coll_songs_df() -> pd.DataFrame:
    songs = coll_data.load_songs()
    extra = coll_data.load_song_extra_info()
    return songs.merge(extra, on=coll_config.SONG_COL, how="left").set_index(coll_config.SONG_COL)


@st.cache_data
def get_coll_members_df() -> pd.DataFrame:
    return coll_data.load_members().set_index(coll_config.USER_COL)


@st.cache_data(show_spinner="Loading a training sample (first load only)...")
def get_coll_train_sample(n: int = COLL_SWEEP_SAMPLE_SIZE) -> pd.DataFrame:
    train = coll_data.load_train().sample(n, random_state=coll_config.RANDOM_SEED)
    return coll_data.merge_side_tables(train)


@st.cache_data(show_spinner="Computing SHAP values (first load only)...")
def get_coll_shap_explanation(sample_size: int = 500):
    pipeline = get_coll_pipeline()
    panel = get_coll_train_sample()
    X = panel.drop(columns=[coll_config.TARGET_COL])
    explanation, X_transformed = coll_interpretability.compute_shap_values(
        pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed
