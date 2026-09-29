"""Dataset Overview page: what's in the KKBox listening-event log before
any modeling — target balance, interaction context, and the real
data-quality issues this project's feature engineering works around.
"""

import plotly.express as px
import streamlit as st

from . import shared
from recommendation_showcase.collaborative import config


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "Real KKBox listening events: did a member replay a song within "
        "a month of first hearing it?"
    )

    panel = shared.get_coll_train_sample()
    members_df = shared.get_coll_members_df()
    songs_df = shared.get_coll_songs_df()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training interactions", "7,377,418")
    c2.metric("Distinct members", f"{members_df.index.nunique():,}")
    c3.metric("Distinct songs", f"{songs_df.index.nunique():,}")
    c4.metric("Target = replay", f"{panel[config.TARGET_COL].mean():.1%}")
    st.caption(f"Sample shown below: {len(panel):,} rows (see `shared.COLL_SWEEP_SAMPLE_SIZE`).")

    st.subheader("Target balance")
    st.caption(
        "Close to 50/50 — verified directly on the full dataset (50.4%/49.6%). "
        "Unlike academic_success's imbalanced dropout/enrolled/graduate classes, "
        "no resampling is needed here."
    )
    target_counts = panel[config.TARGET_COL].value_counts().rename({0: "No replay", 1: "Replay"})
    st.plotly_chart(px.bar(target_counts, title="Target distribution (sample)"), width="stretch")

    st.subheader("How do members encounter the songs they replay?")
    context_col = st.selectbox("Context column", config.INTERACTION_CONTEXT_COLS)
    grouped = panel.groupby(context_col)[config.TARGET_COL].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 50].sort_values("mean", ascending=False)
    st.plotly_chart(
        px.bar(grouped, x=context_col, y="mean", title=f"Replay rate by {context_col}"),
        width="stretch",
    )

    st.subheader("Known data-quality issues (verified directly, not assumed)")
    q1, q2 = st.columns(2)
    with q1:
        low, high = config.VALID_AGE_RANGE
        garbage_age = (~members_df["bd"].between(low, high)).mean()
        st.metric("Members with unusable `bd` (age)", f"{garbage_age:.0%}")
        st.caption(f"Outside {low}-{high}: 0s, negatives, and a max of 1051.")
    with q2:
        missing_gender = members_df["gender"].isna().mean()
        st.metric("Members with missing `gender`", f"{missing_gender:.0%}")

    q3, q4 = st.columns(2)
    with q3:
        missing_composer = songs_df["composer"].isna().mean()
        st.metric("Songs with missing `composer`", f"{missing_composer:.0%}")
    with q4:
        missing_lyricist = songs_df["lyricist"].isna().mean()
        st.metric("Songs with missing `lyricist`", f"{missing_lyricist:.0%}")
    st.caption(
        "See the Feature Engineering page for how `age_clean`, `has_composer`/"
        "`has_lyricist`, and the rest handle these directly rather than feeding "
        "garbage values straight into the model."
    )
