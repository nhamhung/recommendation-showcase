"""Interactive collaborative-filtering replay prediction demo."""

import pandas as pd
import streamlit as st

from . import shared, ui
from recommendation_showcase.collaborative import config, data

SOURCE_SYSTEM_TABS = ["explore", "my library", "search", "discover", "radio", "listen with", "notification", "settings"]
SOURCE_SCREEN_NAMES = ["Explore", "Local playlist more", "Online playlist more", "Radio", "Search", "Discover Feature", "My library"]
SOURCE_TYPES = ["online-playlist", "local-playlist", "local-library", "top-hits-for-artist", "song", "song-based-playlist", "radio"]
CONTEXT_PRESETS = {
    "✨ Discovery": ("discover", "Discover Feature", "song"),
    "❤️ My library": ("my library", "My library", "local-library"),
    "📻 Radio": ("radio", "Radio", "radio"),
}


def _friendly(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def _clear_prediction() -> None:
    st.session_state.pop("coll_last_prediction", None)


def _load_random_pair() -> None:
    st.session_state["picked_msno"] = shared.get_coll_members_df().sample(1).index[0]
    st.session_state["picked_song_id"] = shared.get_coll_songs_df().sample(1).index[0]
    _clear_prediction()


def _set_context(system_tab: str, screen_name: str, source_type: str) -> None:
    st.session_state["coll_system_tab"] = system_tab
    st.session_state["coll_screen_name"] = screen_name
    st.session_state["coll_source_type"] = source_type
    _clear_prediction()


def _valid_age(value) -> str:
    low, high = config.VALID_AGE_RANGE
    if pd.isna(value) or not low <= float(value) <= high:
        return "Not provided"
    return str(int(value))


def render() -> None:
    ui.hero(
        "Interactive collaborative-filtering model",
        "Will this listener come back to the song?",
        "Pair an anonymous KKBox member with a song and predict whether the member will replay it within one month.",
        ["Real listening signals", "Anonymous members", "Replay probability"],
    )

    try:
        pipeline = shared.get_coll_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    members_df = shared.get_coll_members_df()
    songs_df = shared.get_coll_songs_df()
    if "picked_msno" not in st.session_state:
        st.session_state["picked_msno"] = members_df.sample(1, random_state=config.RANDOM_SEED).index[0]
    if "picked_song_id" not in st.session_state:
        st.session_state["picked_song_id"] = songs_df.sample(1, random_state=config.RANDOM_SEED).index[0]

    intro, action = st.columns([3, 1])
    intro.subheader("1. Meet the listener and song")
    intro.caption("The IDs stay anonymous; the cards show only the context available to the model.")
    action.button("🎲 New pair", on_click=_load_random_pair, width="stretch")

    msno = st.session_state["picked_msno"]
    song_id = st.session_state["picked_song_id"]
    member_row = members_df.loc[msno]
    song_row = songs_df.loc[song_id]

    member_card, song_card = st.columns(2)
    with member_card:
        st.markdown("#### 👤 Anonymous listener")
        member_metrics = st.columns(3)
        member_metrics[0].metric("City", int(member_row["city"]))
        member_metrics[1].metric("Age", _valid_age(member_row["bd"]))
        gender = member_row["gender"] if pd.notna(member_row["gender"]) else "Not provided"
        member_metrics[2].metric("Gender", str(gender).title())
        st.caption(f"Registration channel {int(member_row['registered_via'])}")
    with song_card:
        st.markdown("#### 🎵 Candidate song")
        st.markdown(f"**{song_row.get('name', 'Untitled track')}**  \n{song_row['artist_name']}")
        song_metrics = st.columns(3)
        song_metrics[0].metric("Length", f"{float(song_row['song_length']) / 60_000:.1f} min")
        song_metrics[1].metric("Genre IDs", str(song_row["genre_ids"]) if pd.notna(song_row["genre_ids"]) else "Unknown")
        song_metrics[2].metric("Language", str(song_row["language"]) if pd.notna(song_row["language"]) else "Unknown")

    st.subheader("2. Choose how the song was discovered")
    preset_columns = st.columns(3)
    for column, (label, values) in zip(preset_columns, CONTEXT_PRESETS.items()):
        column.button(label, key=f"context_{label}", on_click=_set_context, args=values, width="stretch")

    st.session_state.setdefault("coll_system_tab", SOURCE_SYSTEM_TABS[0])
    st.session_state.setdefault("coll_screen_name", SOURCE_SCREEN_NAMES[0])
    st.session_state.setdefault("coll_source_type", SOURCE_TYPES[0])
    context_columns = st.columns(3)
    source_system_tab = context_columns[0].selectbox(
        "App area",
        SOURCE_SYSTEM_TABS,
        key="coll_system_tab",
        format_func=_friendly,
        help="The broad section of the KKBox app.",
    )
    source_screen_name = context_columns[1].selectbox(
        "Screen",
        SOURCE_SCREEN_NAMES,
        key="coll_screen_name",
        format_func=_friendly,
        help="The screen where the song appeared.",
    )
    source_type = context_columns[2].selectbox(
        "Source",
        SOURCE_TYPES,
        key="coll_source_type",
        format_func=_friendly,
        help="The playlist or recommendation surface that supplied the song.",
    )

    if st.button("Predict replay probability", type="primary", width="stretch"):
        interaction = pd.DataFrame(
            [
                {
                    config.USER_COL: msno,
                    config.SONG_COL: song_id,
                    "source_system_tab": source_system_tab,
                    "source_screen_name": source_screen_name,
                    "source_type": source_type,
                }
            ]
        )
        features = data.merge_side_tables(
            interaction,
            songs_df=songs_df.reset_index(),
            members_df=members_df.reset_index(),
            song_extra_df=songs_df.reset_index()[[config.SONG_COL]],
        )
        probability = float(pipeline.predict_proba(features)[0, 1])
        baseline = float(shared.get_coll_train_sample()[config.TARGET_COL].mean())
        st.session_state["coll_last_prediction"] = {
            "msno": msno,
            "song_id": song_id,
            "probability": probability,
            "baseline": baseline,
            "context": (_friendly(source_system_tab), _friendly(source_screen_name), _friendly(source_type)),
        }

    result = st.session_state.get("coll_last_prediction")
    if not result or result["msno"] != msno or result["song_id"] != song_id:
        st.info("Choose a discovery context and press **Predict replay probability**.")
        return

    probability = result["probability"]
    baseline = result["baseline"]
    st.divider()
    st.subheader("Prediction result")
    result_columns = st.columns([1.2, 1, 1])
    result_columns[0].metric("Replay probability", f"{probability:.1%}")
    result_columns[1].metric("Demo average", f"{baseline:.1%}")
    result_columns[2].metric("Difference", f"{probability - baseline:+.1%}", help="Percentage-point difference from the demo sample average.")
    st.progress(probability, text="Model-estimated chance of a replay within one month")

    if probability >= 0.65:
        st.success("Strong replay signal — this pair and discovery context look promising.")
    elif probability >= 0.5:
        st.success("Replay is slightly more likely than not, but the result is close enough to treat cautiously.")
    elif probability >= 0.35:
        st.warning("The model is uncertain and leans toward no replay.")
    else:
        st.info("Low replay signal — the model expects the listener to move on from this song.")

    context = result["context"]
    st.caption(f"Prediction context: {context[0]} → {context[1]} → {context[2]}")
    with st.expander("What does—and does not—go into this score?"):
        st.markdown(
            """
            The model combines member activity, song and artist popularity, catalog metadata, membership details,
            and the discovery context selected above. It learns patterns across many historical member–song pairs.

            This is a model estimate, not a statement about a person. Member IDs are anonymous, and a probability
            should support ranking or experimentation—not be treated as certainty about individual behaviour.
            """
        )
