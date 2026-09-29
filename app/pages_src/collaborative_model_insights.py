"""Model Insights page: how the model families compare (AUC, on a
300k-row sample — see shared.py's docstring for why not the full 7.4M
rows live), plus SHAP on the saved LightGBM pipeline.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from recommendation_showcase.collaborative import model
from recommendation_showcase.collaborative.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f"]


@st.cache_data(show_spinner="Sweeping model families on a 300k-row sample (first load only)...")
def _model_sweep() -> pd.DataFrame:
    panel = shared.get_coll_train_sample()
    rows = []
    for name, factory in model.MODEL_FACTORIES.items():
        result = model.holdout_eval(panel, estimator=factory())
        rows.append({"model": name, "auc": result["auc"]})
    return pd.DataFrame(rows).sort_values("auc")


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "How the model families compare on AUC, and — via SHAP — what "
        "the saved LightGBM pipeline actually relies on."
    )

    try:
        shared.get_coll_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Model comparison (AUC, 300k-row holdout)")
    sweep = _model_sweep()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(sweep["model"], sweep["auc"], color=PALETTE[0])
    bars[-1].set_color(PALETTE[3])
    ax.set_xlabel("AUC")
    for bar, value in zip(bars, sweep["auc"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig)
    st.caption(
        "Random Forest is excluded from the *full-dataset* training script by "
        "default — verified directly that it didn't finish fitting on the "
        "full 5.9M-row training split even after 500 seconds, while LightGBM "
        "finished the same job in 91 seconds. The number above is from this "
        "300k-row sample only, where it does finish, for a fair comparison "
        "point — not evidence it would scale to the real dataset."
    )

    st.subheader("What does the saved model actually rely on? (SHAP)")
    st.caption("Computed live from the saved LightGBM pipeline — first load takes a few seconds.")
    explanation, _ = shared.get_coll_shap_explanation(sample_size=300)
    top = top_shap_features(explanation, top_n=15).sort_values("mean_abs_shap")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    is_engineered = top["feature"].str.contains(
        "song_popularity|user_activity_count|artist_song_count|member_tenure|age_clean|genre_count|composer_count|has_composer|has_lyricist|isrc"
    )
    colors = [PALETTE[3] if e else PALETTE[0] for e in is_engineered]
    ax2.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("Top 15 features (orange = engineered)")
    st.pyplot(fig2)
