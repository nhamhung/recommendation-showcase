"""Interactive content-based recommendation demo."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import shared, ui
from recommendation_showcase.content_based import config, data

RADAR_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
FOCUS_OPTIONS = {
    "Closest sound": "Keep the nearest tracks based only on the model's audio-feature distance.",
    "Stay in the genre": "Keep the closest results that share the selected track's catalog genre.",
    "Genre discovery": "Keep the closest results from other catalog genres for more variety.",
}


def _track_label(row: pd.Series) -> str:
    return f"{row['track_name']} — {row['artists']} ({row['track_genre']})"


def _set_track(track_id: str) -> None:
    st.session_state["cb_seed_id"] = track_id
    st.session_state["cb_search"] = ""
    st.session_state.pop("cb_last_result", None)


def _load_random_track() -> None:
    tracks = shared.get_cb_tracks_df().sort_values(config.POPULARITY_COL, ascending=False).head(200)
    _set_track(str(tracks.sample(1).index[0]))


def _recommendations(recommender, tracks_df: pd.DataFrame, seed_id: str, k: int, focus: str) -> pd.DataFrame:
    """Get model-ranked candidates and optionally apply a transparent discovery filter."""
    seed_row = tracks_df.loc[[seed_id]]
    X_query, _ = data.split_features_target(seed_row)
    pool_k = min(max(k * 8, 60), len(tracks_df) - 1)
    recs = recommender.recommend(X_query, k=pool_k, exclude_own_id=True)
    display = recs.merge(
        tracks_df[["track_name", "artists", config.GENRE_COL, config.POPULARITY_COL]],
        left_on="recommended_id",
        right_index=True,
    )
    seed_genre = str(tracks_df.loc[seed_id, config.GENRE_COL])
    if focus == "Stay in the genre":
        display = display[display[config.GENRE_COL] == seed_genre]
    elif focus == "Genre discovery":
        display = display[display[config.GENRE_COL] != seed_genre]
    display = display.head(k).copy()
    display["rank"] = range(1, len(display) + 1)
    display["same_genre"] = display[config.GENRE_COL].eq(seed_genre)
    return display[
        [
            "rank",
            "recommended_id",
            "track_name",
            "artists",
            config.GENRE_COL,
            config.POPULARITY_COL,
            "distance",
            "same_genre",
        ]
    ]


def _radar_chart(tracks_df: pd.DataFrame, track_ids: list[str], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    colors = ["#4338ca", "#0f766e"]
    for track_id, label, color in zip(track_ids, labels, colors):
        values = tracks_df.loc[track_id, RADAR_FEATURES].astype(float).tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=RADAR_FEATURES + RADAR_FEATURES[:1],
                name=label,
                line=dict(color=color, width=3),
                fill="toself",
                opacity=0.68,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(238,242,255,.45)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#cbd5e1"),
            angularaxis=dict(gridcolor="#cbd5e1"),
        ),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=30, b=55),
    )
    return fig


def render() -> None:
    ui.hero(
        "Interactive content-based recommender",
        "Start with one song. Discover what sounds close.",
        "Choose a familiar track and the model searches its audio profile—not listening history—to find nearby music.",
        ["11k-track demo catalog", "No login needed", "Results in seconds"],
    )

    try:
        recommender = shared.get_cb_recommender()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tracks_df = shared.get_cb_tracks_df()
    featured = tracks_df.sort_values(config.POPULARITY_COL, ascending=False).head(3)

    st.subheader("Choose your starting track")
    st.caption("Use a quick pick, search the catalog, or let the app surprise you.")
    quick_columns = st.columns(4)
    for column, (track_id, row) in zip(quick_columns[:3], featured.iterrows()):
        column.button(
            f"▶ {str(row['track_name'])[:25]}",
            key=f"featured_{track_id}",
            on_click=_set_track,
            args=(str(track_id),),
            width="stretch",
            help=f"{row['artists']} · {row[config.GENRE_COL]}",
        )
    quick_columns[3].button(
        "🎲 Surprise me",
        key="cb_random",
        on_click=_load_random_track,
        width="stretch",
    )

    search = st.text_input(
        "Search by track or artist",
        key="cb_search",
        placeholder="Try: Hold On, Jason Mraz, acoustic…",
    )
    candidates = tracks_df
    if search:
        mask = tracks_df["track_name"].str.contains(search, case=False, na=False) | tracks_df[
            "artists"
        ].str.contains(search, case=False, na=False)
        candidates = tracks_df[mask]

    if candidates.empty:
        st.warning("No catalog matches yet. Try fewer words or search for an artist.")
        st.stop()

    options = candidates.sort_values(config.POPULARITY_COL, ascending=False).head(200)
    if st.session_state.get("cb_seed_id") not in options.index:
        st.session_state["cb_seed_id"] = str(options.index[0])
    seed_id = st.selectbox(
        "Selected track",
        options=options.index,
        format_func=lambda track_id: _track_label(options.loc[track_id]),
        key="cb_seed_id",
    )

    seed = tracks_df.loc[seed_id]
    track_summary = st.columns([2, 1, 1, 1])
    track_summary[0].metric("Selected", str(seed["track_name"]))
    track_summary[0].caption(str(seed["artists"]))
    track_summary[1].metric("Genre", str(seed[config.GENRE_COL]))
    track_summary[2].metric("Energy", f"{float(seed['energy']):.0%}")
    track_summary[3].metric("Popularity", f"{int(seed[config.POPULARITY_COL])}/100")

    controls = st.columns([2, 1])
    focus = controls[0].selectbox(
        "Recommendation mode",
        options=list(FOCUS_OPTIONS),
        help="The model always ranks by audio distance. Genre modes only filter that ranked list afterward.",
    )
    k = controls[1].select_slider("Number of results", options=list(range(3, 13)), value=6)
    st.caption(FOCUS_OPTIONS[focus])

    if st.button("Find my recommendations", type="primary", width="stretch"):
        st.session_state["cb_last_result"] = {"seed_id": seed_id, "focus": focus, "k": k}

    request = st.session_state.get("cb_last_result")
    if not request or request["seed_id"] != seed_id:
        st.info("Choose a mode and press **Find my recommendations** to build your playlist.")
        return

    display = _recommendations(
        recommender,
        tracks_df,
        request["seed_id"],
        request["k"],
        request["focus"],
    )
    if display.empty:
        st.warning("No tracks matched this discovery filter. Try **Closest sound** for the full ranked list.")
        return

    st.divider()
    st.subheader(f"Your matches for “{seed['track_name']}”")
    if len(display) < request["k"]:
        st.caption(f"This catalog contained {len(display)} matches for the selected filter.")

    card_columns = st.columns(min(3, len(display)))
    for column, (_, row) in zip(card_columns, display.head(3).iterrows()):
        with column:
            ui.recommendation_card(
                int(row["rank"]),
                str(row["track_name"]),
                str(row["artists"]),
                str(row[config.GENRE_COL]),
                float(row["distance"]),
            )

    same_genre_share = float(display["same_genre"].mean())
    summary = st.columns(3)
    summary[0].metric("Same catalog genre", f"{same_genre_share:.0%}")
    summary[1].metric("Different artists", f"{display['artists'].nunique()}/{len(display)}")
    summary[2].metric("Median audio distance", f"{display['distance'].median():.2f}", help="Lower means closer.")

    tab_list, tab_profile, tab_method = st.tabs(["Full playlist", "Why the top match?", "How it works"])
    with tab_list:
        table = display.drop(columns=["recommended_id", "same_genre"]).rename(
            columns={
                "rank": "Rank",
                "track_name": "Track",
                "artists": "Artist",
                config.GENRE_COL: "Genre",
                config.POPULARITY_COL: "Popularity",
                "distance": "Audio distance",
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={"Audio distance": st.column_config.NumberColumn(format="%.3f")},
        )
        st.download_button(
            "Download playlist as CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name="recommendation_playlist.csv",
            mime="text/csv",
        )
    with tab_profile:
        top_pick_id = str(display.iloc[0]["recommended_id"])
        st.plotly_chart(
            _radar_chart(
                tracks_df,
                [seed_id, top_pick_id],
                [f"Seed · {seed['track_name']}", f"Top match · {tracks_df.loc[top_pick_id, 'track_name']}"],
            ),
            width="stretch",
        )
        st.caption("Overlapping shapes mean the two tracks have similar Spotify audio-feature profiles.")
    with tab_method:
        st.markdown(
            """
            1. Numeric audio features are cleaned and scaled so one unit cannot dominate the others.
            2. Musical key is represented on a circle so B and C remain close rather than appearing far apart.
            3. A nearest-neighbour model ranks catalog tracks by distance in that processed feature space.

            Genre is **not** given to the recommender. It is shown afterward as an intuitive check and, in the two
            discovery modes, as a transparent display filter over the model-ranked candidates.
            """
        )
