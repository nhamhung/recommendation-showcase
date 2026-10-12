"""Recommendation Systems Showcase — multi-page Streamlit app entry point.

Two full recommenders, one app: **content-based** (Spotify audio
features, no user data — `content_based/`) and **collaborative
filtering** (real KKBox listening logs, no audio features at all —
`collaborative/`). Each keeps its full four-page depth (Predict/
Recommend, Overview, Feature Engineering, Model Insights), grouped under
its own section in the sidebar.

Run locally:
    streamlit run app/streamlit_app.py

Or via Docker (from the project root):
    docker build -t recommendation-showcase-app -f app/Dockerfile .
    docker run -p 8501:8501 recommendation-showcase-app
"""

import sys
from pathlib import Path

import streamlit as st

# Must happen before importing any pages_src module — every page (and
# shared.py) imports directly from the recommendation_showcase.content_based/
# .collaborative packages, which live under src/, not on the default path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pages_src import (  # noqa: E402
    collaborative_feature_engineering,
    collaborative_model_insights,
    collaborative_overview,
    collaborative_demo,
    content_based_feature_engineering,
    content_based_model_insights,
    content_based_overview,
    content_demo,
)
from theme import apply_theme  # noqa: E402

st.set_page_config(page_title="Recommendation Systems Showcase", page_icon="🎶", layout="wide")
apply_theme()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <strong>🎶 Recommendation Lab</strong>
          <span>Two datasets. Two recommendation strategies. One interactive comparison.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

pages = {
    "Try the models": [
        st.Page(content_demo.render, title="Find similar songs", icon="🎧", url_path="cb-recommend", default=True),
        st.Page(collaborative_demo.render, title="Predict a replay", icon="🎯", url_path="coll-predict"),
    ],
    "How content-based works": [
        st.Page(content_based_overview.render, title="Spotify dataset", icon="📊", url_path="cb-overview"),
        st.Page(content_based_feature_engineering.render, title="Audio features", icon="🔧", url_path="cb-feature-engineering"),
        st.Page(content_based_model_insights.render, title="Model results", icon="🧠", url_path="cb-model-insights"),
    ],
    "How collaborative works": [
        st.Page(collaborative_overview.render, title="KKBox dataset", icon="📊", url_path="coll-overview"),
        st.Page(collaborative_feature_engineering.render, title="Listening signals", icon="🔧", url_path="coll-feature-engineering"),
        st.Page(collaborative_model_insights.render, title="Model results", icon="🧠", url_path="coll-model-insights"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
