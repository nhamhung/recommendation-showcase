"""Tests for the genre-family consolidation mapping."""

import pandas as pd

from recommendation_showcase.content_based.genre_families import GENRE_TO_FAMILY, to_family


def test_to_family_maps_a_scalar_genre():
    assert to_family("k-pop") == "pop"
    assert to_family("death-metal") == "metal"


def test_to_family_maps_a_series():
    series = pd.Series(["rock", "jazz", "sad"])
    mapped = to_family(series)
    assert list(mapped) == ["rock", "jazz_blues_soul", "mood_context"]


def test_to_family_falls_back_to_the_genre_itself_for_unknown_values():
    assert to_family("some-future-genre-tag") == "some-future-genre-tag"


def test_mood_context_genres_are_kept_separate_from_musical_families():
    mood_tags = ["chill", "comedy", "happy", "kids", "party", "romance", "sad", "sleep", "study", "children"]
    for tag in mood_tags:
        assert GENRE_TO_FAMILY[tag] == "mood_context"
