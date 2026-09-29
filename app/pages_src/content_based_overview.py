"""Dataset Overview page: what's in the Spotify Tracks dataset before any
modeling — genre distribution, audio-feature distributions, and how
popularity relates to those features.
"""

import plotly.express as px
import streamlit as st

from . import shared
from recommendation_showcase.content_based import config


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "~114k Spotify tracks, each described by Spotify's own audio-feature "
        "measurements plus a genre tag and a popularity score."
    )

    tracks_df = shared.get_cb_tracks_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracks", f"{len(tracks_df):,}")
    c2.metric("Genres", f"{tracks_df[config.GENRE_COL].nunique()}")
    c3.metric("Artists", f"{tracks_df['artists'].nunique():,}")

    st.subheader("Genre distribution")
    st.caption(
        "Roughly even across genres by design (this dataset was built with "
        "~1,000 tracks sampled per genre) — unlike academic_success's "
        "naturally imbalanced target classes, there's no class-imbalance "
        "correction needed here."
    )
    genre_counts = tracks_df[config.GENRE_COL].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]
    st.plotly_chart(
        px.bar(genre_counts.head(20), x="genre", y="count", title="Top 20 genres by track count"),
        width="stretch",
    )

    st.subheader("Audio feature distributions")
    feature = st.selectbox("Feature", options=config.RAW_AUDIO_NUMERIC_COLS)
    st.plotly_chart(
        px.histogram(tracks_df, x=feature, nbins=50, title=f"Distribution of {feature}"),
        width="stretch",
    )

    st.subheader("Does popularity track any single audio feature?")
    st.caption(
        "Spoiler for the Model Insights page: if it did strongly, popularity "
        "would be a shortcut around building a real content-based "
        "recommender — this scatter is why it isn't included as a feature."
    )
    x_feature = st.selectbox("Feature to compare against popularity", options=config.RAW_AUDIO_NUMERIC_COLS, key="popularity_scatter")
    sample = tracks_df.sample(min(5000, len(tracks_df)), random_state=config.RANDOM_SEED)
    st.plotly_chart(
        px.scatter(sample, x=x_feature, y=config.POPULARITY_COL, opacity=0.3, title=f"Popularity vs. {x_feature}"),
        width="stretch",
    )
