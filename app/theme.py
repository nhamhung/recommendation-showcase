"""Load the project's shared, accessible visual theme."""

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


def apply_theme() -> None:
    """Inject the base component styling and project-specific palette."""
    app_dir = Path(__file__).resolve().parent
    css = "\n".join((app_dir / name).read_text(encoding="utf-8") for name in ("theme_base.css", "theme.css"))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    pio.templates["recommendation_showcase"] = go.layout.Template(
        layout={
            "font": {"family": "Inter, ui-sans-serif, system-ui", "color": "#334155"},
            "colorway": ["#4338ca", "#0f766e", "#db2777", "#f59e0b", "#64748b"],
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "title": {"font": {"color": "#172554", "size": 18}},
            "xaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1"},
            "yaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1"},
            "legend": {"bgcolor": "rgba(255,255,255,.75)"},
        }
    )
    pio.templates.default = "plotly_white+recommendation_showcase"
