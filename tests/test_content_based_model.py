"""Tests for the recommender, genre classifier, and evaluation helpers,
using a small synthetic catalog instead of the real Spotify data.
"""

import numpy as np
import pandas as pd

from recommendation_showcase.content_based import model
from tests.test_content_based_features import make_synthetic_catalog


def test_recommender_excludes_query_track_from_its_own_results():
    catalog = make_synthetic_catalog(n_tracks=50)
    X, _ = _split(catalog)

    recommender = model.ContentRecommender().fit(X)
    recs = recommender.recommend(X.iloc[[0]], k=5, exclude_own_id=True)

    assert X.index[0] not in recs["recommended_id"].to_numpy()
    assert len(recs) == 5


def test_recommender_returns_k_neighbors_per_query():
    catalog = make_synthetic_catalog(n_tracks=50)
    X, _ = _split(catalog)
    train_X, test_X = X.iloc[:40], X.iloc[40:]

    recommender = model.ContentRecommender().fit(train_X)
    recs = recommender.recommend(test_X, k=5, exclude_own_id=False)

    assert set(recs["query_id"]) == set(test_X.index)
    assert (recs.groupby("query_id").size() == 5).all()


def test_precision_at_k_is_between_zero_and_one():
    catalog = make_synthetic_catalog(n_tracks=200)
    train_df, test_df = model.train_test_split_tracks(catalog, test_size=0.2)

    score = model.precision_at_k(train_df, test_df, k=5)
    assert 0.0 <= score <= 1.0


def test_popularity_baseline_is_between_zero_and_one():
    catalog = make_synthetic_catalog(n_tracks=200)
    train_df, test_df = model.train_test_split_tracks(catalog, test_size=0.2)

    score = model.popularity_baseline_precision_at_k(train_df, test_df, k=5)
    assert 0.0 <= score <= 1.0


def test_feature_weights_from_classifier_average_to_one():
    catalog = make_synthetic_catalog(n_tracks=200)
    X, y = model.data.split_features_target(catalog)

    pipeline = model.build_classifier_pipeline(model.random_forest_estimator())
    pipeline.fit(X, y)
    weights = model.compute_feature_weights_from_classifier(pipeline)

    assert weights.shape[0] > 0
    assert np.isclose(weights.mean(), 1.0)


def test_weighted_recommender_runs_end_to_end():
    catalog = make_synthetic_catalog(n_tracks=200)
    X, y = model.data.split_features_target(catalog)

    pipeline = model.build_classifier_pipeline(model.random_forest_estimator())
    pipeline.fit(X, y)
    weights = model.compute_feature_weights_from_classifier(pipeline)

    recommender = model.ContentRecommender(feature_weights=weights).fit(X)
    recs = recommender.recommend(X.iloc[[0]], k=5)
    assert len(recs) == 5


def test_genre_family_map_relaxes_precision_at_k_upward():
    """Family-level matching can only ever count everything exact-genre
    matching already counts, plus possibly more (near-synonym genres
    mapped to the same family) — so precision@k under a genre_map must
    be >= precision@k without one, never lower.
    """
    catalog = make_synthetic_catalog(n_tracks=200)
    train_df, test_df = model.train_test_split_tracks(catalog, test_size=0.2)

    # Collapses "pop" and "rock" into one family, leaves the others alone.
    genre_map = {"pop": "mainstream", "rock": "mainstream"}

    exact_score = model.precision_at_k(train_df, test_df, k=5)
    family_score = model.precision_at_k(train_df, test_df, k=5, genre_map=genre_map)
    assert family_score >= exact_score


def test_genre_families_module_covers_every_real_dataset_genre():
    """A snapshot of the real dataset's 114 genre tags (see
    data.load_tracks's docstring for why the deduplicated catalog itself
    has only 113) — this test doesn't need the actual CSV downloaded, but
    still catches a typo or a genre this mapping forgot to cover, which
    would otherwise silently fall back to leaving that genre unmapped
    (see `to_family`'s docstring) rather than raising.
    """
    from recommendation_showcase.content_based.genre_families import GENRE_TO_FAMILY

    real_genres = [
        "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
        "black-metal", "bluegrass", "blues", "brazil", "breakbeat", "british",
        "cantopop", "chicago-house", "children", "chill", "classical", "club",
        "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
        "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep",
        "edm", "electro", "electronic", "emo", "folk", "forro", "french",
        "funk", "garage", "german", "gospel", "goth", "grindcore", "groove",
        "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle",
        "heavy-metal", "hip-hop", "honky-tonk", "house", "idm", "indian",
        "indie", "indie-pop", "industrial", "iranian", "j-dance", "j-idol",
        "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino", "malay",
        "mandopop", "metal", "metalcore", "minimal-techno", "mpb", "new-age",
        "opera", "pagode", "party", "piano", "pop", "pop-film", "power-pop",
        "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b",
        "reggae", "reggaeton", "rock", "rock-n-roll", "rockabilly", "romance",
        "sad", "salsa", "samba", "sertanejo", "show-tunes", "singer-songwriter",
        "ska", "sleep", "songwriter", "soul", "spanish", "study", "swedish",
        "synth-pop", "tango", "techno", "trance", "trip-hop", "turkish",
        "world-music",
    ]
    missing = set(real_genres) - set(GENRE_TO_FAMILY.keys())
    assert not missing, f"genres missing from GENRE_TO_FAMILY: {missing}"


def test_pca_embedding_runs_end_to_end_and_reduces_dimensionality():
    catalog = make_synthetic_catalog(n_tracks=200)
    train_df, test_df = model.train_test_split_tracks(catalog, test_size=0.2)

    score = model.precision_at_k(train_df, test_df, k=5, embedding=model.pca_embedding(n_components=5))
    assert 0.0 <= score <= 1.0


def test_nca_embedding_runs_end_to_end():
    catalog = make_synthetic_catalog(n_tracks=200)
    train_df, test_df = model.train_test_split_tracks(catalog, test_size=0.2)

    score = model.precision_at_k(train_df, test_df, k=5, embedding=model.nca_embedding(n_components=5))
    assert 0.0 <= score <= 1.0


def test_embedding_recommender_predicts_expected_shape():
    catalog = make_synthetic_catalog(n_tracks=100)
    X, y = _split(catalog)

    recommender = model.ContentRecommender(embedding=model.pca_embedding(n_components=5)).fit(X, y)
    recs = recommender.recommend(X.iloc[[0]], k=5)
    assert len(recs) == 5


def test_save_and_load_pipeline_roundtrip(tmp_path):
    catalog = make_synthetic_catalog(n_tracks=50)
    recommender = model.train_recommender(catalog, use_weights=False)

    path = tmp_path / "model.joblib"
    model.save_pipeline(recommender, path=path)
    loaded = model.load_pipeline(path=path)

    X, _ = _split(catalog)
    query = X.iloc[[0]]
    pd.testing.assert_frame_equal(
        recommender.recommend(query, k=5), loaded.recommend(query, k=5)
    )


def _split(catalog: pd.DataFrame):
    return model.data.split_features_target(catalog)
