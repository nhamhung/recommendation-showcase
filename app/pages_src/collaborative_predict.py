"""Predict page: pick a real member and a real song, choose how they
encountered it, and see the model's predicted replay probability.

Unlike the Spotify project's "pick a song, see similar songs" (a
similarity search) or academic_success's "tweak a student's record" (one
entity, many editable fields), this project's prediction is inherently
about a *pair* — a specific member and a specific song — so the demo
picks both from the real catalog rather than asking anyone to guess
plausible values for either.
"""

import pandas as pd
import streamlit as st

from . import shared
from recommendation_showcase.collaborative import config, data

SOURCE_SYSTEM_TABS = ["explore", "my library", "search", "discover", "radio", "listen with", "notification", "settings"]
SOURCE_SCREEN_NAMES = ["Explore", "Local playlist more", "Online playlist more", "Radio", "Search", "Discover Feature", "My library"]
SOURCE_TYPES = ["online-playlist", "local-playlist", "local-library", "top-hits-for-artist", "song", "song-based-playlist", "radio"]


def _load_random_member():
    members_df = shared.get_coll_members_df()
    msno = members_df.sample(1).index[0]
    st.session_state["picked_msno"] = msno


def _load_random_song():
    songs_df = shared.get_coll_songs_df()
    song_id = songs_df.sample(1).index[0]
    st.session_state["picked_song_id"] = song_id


def render():
    st.title("🎯 Predict a Replay")
    st.caption(
        "Will this member listen to this song again within a month of "
        "first hearing it? Pick a real member and a real song from the "
        "catalog, choose how they encountered it, and predict."
    )

    try:
        pipeline = shared.get_coll_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    members_df = shared.get_coll_members_df()
    songs_df = shared.get_coll_songs_df()

    col1, col2 = st.columns(2)
    with col1:
        st.button("🎲 Load a random member", on_click=_load_random_member, width="stretch")
    with col2:
        st.button("🎲 Load a random song", on_click=_load_random_song, width="stretch")

    if "picked_msno" not in st.session_state:
        st.session_state["picked_msno"] = members_df.sample(1, random_state=config.RANDOM_SEED).index[0]
    if "picked_song_id" not in st.session_state:
        st.session_state["picked_song_id"] = songs_df.sample(1, random_state=config.RANDOM_SEED).index[0]

    msno = st.session_state["picked_msno"]
    song_id = st.session_state["picked_song_id"]

    member_row = members_df.loc[msno]
    song_row = songs_df.loc[song_id]

    st.subheader("Member")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("City code", int(member_row["city"]))
    m2.metric("Age (bd, raw)", int(member_row["bd"]))
    m3.metric("Gender", member_row["gender"] if pd.notna(member_row["gender"]) else "unknown")
    m4.metric("Registered via", int(member_row["registered_via"]))

    st.subheader("Song")
    st.write(f"**{song_row.get('name', song_id)}** — {song_row['artist_name']}")
    s1, s2, s3 = st.columns(3)
    s1.metric("Length (min)", f"{song_row['song_length'] / 60_000:.1f}")
    s2.metric("Genre ID(s)", song_row["genre_ids"] if pd.notna(song_row["genre_ids"]) else "unknown")
    s3.metric("Language code", song_row["language"] if pd.notna(song_row["language"]) else "unknown")

    st.subheader("How did they encounter this song?")
    c1, c2, c3 = st.columns(3)
    source_system_tab = c1.selectbox("source_system_tab", SOURCE_SYSTEM_TABS)
    source_screen_name = c2.selectbox("source_screen_name", SOURCE_SCREEN_NAMES)
    source_type = c3.selectbox("source_type", SOURCE_TYPES)

    if st.button("Predict replay probability", type="primary"):
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
        X = data.merge_side_tables(
            interaction,
            songs_df=songs_df.reset_index(),
            members_df=members_df.reset_index(),
            song_extra_df=songs_df.reset_index()[[config.SONG_COL]],
        )
        proba = pipeline.predict_proba(X)[0, 1]

        st.subheader(f"Predicted replay probability: **{proba:.1%}**")
        st.progress(min(max(proba, 0.0), 1.0))
        if proba >= 0.5:
            st.success("The model expects this member to listen to this song again.")
        else:
            st.info("The model doesn't expect a repeat listen within a month.")
