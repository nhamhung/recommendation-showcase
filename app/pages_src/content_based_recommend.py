"""Recommend page: pick a real track from the catalog, see the
recommender's top-k similar tracks, and compare audio-feature profiles
with a radar chart.

Deliberately a "pick from a list" interface, not a form of 15 raw audio-
feature sliders — the same lesson the academic_success project's Predict
page redesign already applied: recommending from a track someone actually
recognizes is a far more intuitive demo than asking a non-technical user
to guess a plausible "danceability" value.
"""

import plotly.graph_objects as go
import streamlit as st

from . import shared
from recommendation_showcase.content_based import config, data

RADAR_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def _track_label(row) -> str:
    return f"{row['track_name']} — {row['artists']} ({row['track_genre']})"


def _radar_chart(tracks_df, track_ids: list[str], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for track_id, label in zip(track_ids, labels):
        values = tracks_df.loc[track_id, RADAR_FEATURES].tolist()
        fig.add_trace(
            go.Scatterpolar(r=values + values[:1], theta=RADAR_FEATURES + RADAR_FEATURES[:1], name=label)
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig


def render():
    st.title("🎧 Recommend Similar Tracks")
    st.caption(
        "Pick a track you know, and see the tracks whose Spotify audio "
        "features are closest to it — no need to know what 'danceability' "
        "or 'acousticness' actually mean to try it."
    )

    try:
        recommender = shared.get_cb_recommender()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tracks_df = shared.get_cb_tracks_df()

    search = st.text_input("Search for a track or artist", "")
    candidates = tracks_df
    if search:
        mask = tracks_df["track_name"].str.contains(search, case=False, na=False) | tracks_df[
            "artists"
        ].str.contains(search, case=False, na=False)
        candidates = tracks_df[mask]

    if candidates.empty:
        st.warning("No tracks match that search — try a different term.")
        st.stop()

    # Cap the dropdown so a broad search term doesn't render tens of
    # thousands of options; sorted by popularity so the most recognizable
    # matches surface first.
    options = candidates.sort_values(config.POPULARITY_COL, ascending=False).head(200)
    seed_id = st.selectbox(
        "Track", options=options.index, format_func=lambda tid: _track_label(options.loc[tid])
    )

    k = st.slider("How many recommendations?", min_value=3, max_value=20, value=config.DEFAULT_TOP_K)

    if st.button("Recommend", type="primary"):
        seed_row = tracks_df.loc[[seed_id]]
        X_query, _ = data.split_features_target(seed_row)
        recs = recommender.recommend(X_query, k=k, exclude_own_id=True)

        st.subheader(f"Because you picked *{tracks_df.loc[seed_id, 'track_name']}*")
        display = recs.merge(
            tracks_df[["track_name", "artists", config.GENRE_COL, config.POPULARITY_COL]],
            left_on="recommended_id",
            right_index=True,
        )[["rank", "track_name", "artists", config.GENRE_COL, config.POPULARITY_COL, "distance"]]
        st.dataframe(display, hide_index=True, width="stretch")

        top_pick_id = recs.iloc[0]["recommended_id"]
        st.plotly_chart(
            _radar_chart(
                tracks_df,
                [seed_id, top_pick_id],
                [f"Seed: {tracks_df.loc[seed_id, 'track_name']}", f"Top pick: {tracks_df.loc[top_pick_id, 'track_name']}"],
            ),
            width="stretch",
        )

        same_genre_share = (display[config.GENRE_COL] == tracks_df.loc[seed_id, config.GENRE_COL]).mean()
        st.caption(
            f"{same_genre_share:.0%} of these recommendations share the seed track's "
            f"genre ({tracks_df.loc[seed_id, config.GENRE_COL]}) — genre is never given to "
            "the recommender directly, so this is a rough, after-the-fact sanity check, "
            "not something it was optimizing for. See Model Insights for the full "
            "precision@k evaluation."
        )
