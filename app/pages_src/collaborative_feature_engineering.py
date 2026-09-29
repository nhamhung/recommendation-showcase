"""Feature Engineering page: what `FeatureEngineer`/`InteractionCountEncoder`
derive, and why — made visual rather than just described.
"""

import plotly.express as px
import streamlit as st

from . import shared
from recommendation_showcase.collaborative import config
from recommendation_showcase.collaborative.features import FeatureEngineer, InteractionCountEncoder


def render():
    st.title("🔧 Feature Engineering")
    st.caption("The real engineering work in this project — turning sparse, messy raw columns into usable signal.")

    panel = shared.get_coll_train_sample(n=50_000)
    X = panel.drop(columns=[config.TARGET_COL])

    counted = InteractionCountEncoder().fit_transform(X)
    engineered = FeatureEngineer().fit_transform(counted)

    st.subheader("1. Frequency encoding for `song_id`/`msno`/`artist_name`")
    st.markdown(
        "`artist_name` has **222,363** distinct values (verified directly) — "
        "far too many to one-hot encode. `InteractionCountEncoder` replaces "
        "identity with a learned frequency instead: how often has this "
        "song/user/artist appeared in *training* data? A brand-new song or "
        "user gets `0` — itself a meaningful signal, not a gap to impute."
    )
    st.plotly_chart(
        px.histogram(counted, x="song_popularity", nbins=50, title="song_popularity distribution (log y)", log_y=True),
        width="stretch",
    )

    st.subheader("2. Cleaning `bd` (self-reported age)")
    low, high = config.VALID_AGE_RANGE
    st.markdown(
        f"58% of members have an unusable `bd` value (0, negative, or as high "
        f"as 1051 — verified directly). Anything outside **{low}-{high}** "
        "becomes `NaN`, imputed downstream by the shared median-imputer "
        "rather than trusted as a real age."
    )
    st.plotly_chart(
        px.histogram(engineered, x="age_clean", nbins=40, title="age_clean (garbage values already removed)"),
        width="stretch",
    )

    st.subheader("3. Presence flags for sparse credit fields")
    st.markdown(
        "`composer` is missing for ~47% of songs, `lyricist` for ~85% "
        "(verified directly) — too sparse to use the actual names as "
        "features, but *whether* a song credits one at all is cheap, "
        "always-available signal."
    )
    flag_counts = engineered[["has_composer", "has_lyricist"]].mean().rename("share_present")
    st.plotly_chart(px.bar(flag_counts, title="Share of songs with a credited composer/lyricist"), width="stretch")

    st.subheader("4. ISRC parsing: country and release year")
    st.markdown(
        "`song_extra_info.isrc` (e.g. `TWUM71200043`) encodes a country code "
        "and a 2-digit year. The year needs a century pivot — this dataset's "
        "tracks span up to ~2017, so a 2-digit year over 17 is assumed 19xx, "
        "otherwise 20xx."
    )
    st.plotly_chart(
        px.histogram(engineered, x="isrc_year", nbins=40, title="Parsed release year"),
        width="stretch",
    )
