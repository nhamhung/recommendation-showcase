"""Feature Engineering page: what `FeatureEngineer` derives, and why —
made visual rather than just described.
"""

import plotly.express as px
import streamlit as st

from . import shared
from recommendation_showcase.content_based.features import FeatureEngineer


def render():
    st.title("🔧 Feature Engineering")
    st.caption(
        "Three derived features, and why each one matters more than it "
        "might look at first."
    )

    tracks_df = shared.get_cb_tracks_df()
    sample = tracks_df.sample(min(3000, len(tracks_df)), random_state=42)
    engineered = FeatureEngineer().fit_transform(sample)

    st.subheader("1. `key_sin` / `key_cos` — musical key is a circle, not a line")
    st.markdown(
        "Spotify's `key` column is one of 12 pitch classes (0=C, 1=C♯/D♭, "
        "..., 11=B). Treated as a plain integer, key 11 would look *far* "
        "from key 0 — but musically, they're a single semitone apart. "
        "Plotting every track's key as a point on a circle (rather than a "
        "line) makes that adjacency visible: key 11 and key 0 end up "
        "right next to each other."
    )
    fig = px.scatter_polar(
        engineered, r=[1] * len(engineered), theta=engineered["key"] * 30, color=engineered["key"].astype(str),
        title="Each track's key, placed on a 12-hour circle (30° per semitone)",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("2. `duration_min` — a readability transform")
    st.markdown(
        "`duration_ms` (e.g. `210000`) is exact but unreadable. "
        "`duration_min = duration_ms / 60000` doesn't change what the "
        "model learns — it's the same information — but it's what the "
        "notebook's EDA and this app display instead."
    )
    st.plotly_chart(
        px.histogram(engineered, x="duration_min", nbins=50, title="Track duration (minutes)"),
        width="stretch",
    )

    st.subheader("3. `mood_quadrant` — a human-legible summary of valence + energy")
    st.markdown(
        "Russell's circumplex model of affect maps musical *valence* "
        "(positivity — sad to happy) and *energy* (calm to intense) onto "
        "four mood quadrants. A genre classifier can already learn this "
        "distinction from the raw valence/energy floats directly — this "
        "feature exists for **legibility**: 'energetic_happy vs. "
        "calm_dark' is something a person compares two songs by far more "
        "naturally than two decimals."
    )
    fig2 = px.scatter(
        engineered,
        x="valence",
        y="energy",
        color="mood_quadrant",
        opacity=0.5,
        title="Every sampled track's (valence, energy), colored by mood_quadrant",
    )
    st.plotly_chart(fig2, width="stretch")
