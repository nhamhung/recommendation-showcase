"""Small presentation helpers shared by the two interactive demos."""

from html import escape

import streamlit as st


def hero(eyebrow: str, title: str, description: str, tags: list[str]) -> None:
    """Render an accessible, project-branded page introduction."""
    tag_html = "".join(f'<span class="hero-tag">{escape(tag)}</span>' for tag in tags)
    st.markdown(
        f"""
        <section class="project-hero">
          <div class="hero-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
          <div class="hero-tags">{tag_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def recommendation_card(rank: int, title: str, artist: str, genre: str, distance: float) -> None:
    """Render one compact recommendation card without allowing catalog HTML."""
    st.markdown(
        f"""
        <article class="recommendation-card">
          <div class="recommendation-rank">#{rank}</div>
          <div class="recommendation-title">{escape(str(title))}</div>
          <div class="recommendation-artist">{escape(str(artist))}</div>
          <div class="recommendation-meta">
            <span>{escape(str(genre))}</span>
            <span>distance {distance:.2f}</span>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
