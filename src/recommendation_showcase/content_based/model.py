"""The recommender itself, a secondary genre classifier used to weight
its distance metric, evaluation (precision@k), and persistence.

There's no ground-truth "correct recommendation" in this dataset — unlike
the other two projects, there's no `y` a recommender is trying to
predict. `track_genre` is never a recommender input (see `features.py`
and `config.py`), but it's used here purely as an *evaluation* signal:
`precision_at_k` holds out a set of tracks, recommends from the
remaining catalog, and measures what fraction of each track's
recommendations share its true genre. Two tracks sharing a genre tag
isn't a perfect proxy for "these actually sound alike," but it's a
far better one than eyeballing a few example outputs, and it's the
standard approach when no real user feedback (clicks, plays, skips)
exists — the same reasoning that motivates the CO2/academic_success
projects' cross-validation, applied to a problem with no labels for the
task itself.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from . import config, data
from .features import build_feature_pipeline


class ContentRecommender(BaseEstimator):
    """Nearest-neighbor search over the shared feature pipeline's output.

    `feature_weights`, if given, multiplies each transformed feature
    column before computing distances — stretching a feature's axis up
    weights it more heavily in "closeness," down weights it less, without
    changing the underlying pipeline. `compute_feature_weights_from_classifier`
    below derives these from a genre classifier's learned feature
    importances, turning "which audio features actually separate genres"
    into "weight those features more heavily when deciding what's similar."

    `embedding`, if given, is a fresh (unfitted) sklearn transformer —
    typically `sklearn.decomposition.PCA` or
    `sklearn.neighbors.NeighborhoodComponentsAnalysis` (see
    `pca_embedding`/`nca_embedding` below) — fit once on the (weighted)
    training matrix and applied after it, mapping into a lower-dimensional
    or better-separated space before the nearest-neighbor search runs.
    PCA ignores `y`; NCA uses it to directly optimize the embedding for
    neighbor quality under the genre labels, rather than relying on
    `feature_weights`' borrowed-from-elsewhere heuristic. Distances are
    always computed in whichever space is actually searched — raw
    weighted features if `embedding` is `None`, the embedding's output
    otherwise.

    `embedding_fit_sample_size`, if given, fits `embedding` on a random
    subsample of that size instead of the full training set — needed for
    NCA specifically: its optimization cost grows roughly quadratically
    with row count, and fitting it on this project's ~72k-row training
    split gets killed by the OS for exhausting memory (verified directly;
    a 5,000-row subsample fits in under 20 seconds). Once fit, `.transform()`
    is just a matrix multiply, so it's still applied to the *full* training
    catalog afterward — only the expensive fitting step is subsampled, not
    the pool of tracks the recommender can actually recommend from. PCA
    doesn't need this (fits on the full set in about a second either way),
    but the parameter works for any embedding.
    """

    def __init__(
        self,
        feature_weights: np.ndarray | None = None,
        embedding: BaseEstimator | None = None,
        embedding_fit_sample_size: int | None = None,
    ):
        self.feature_weights = feature_weights
        self.embedding = embedding
        self.embedding_fit_sample_size = embedding_fit_sample_size

    def _apply_weights(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * self.feature_weights if self.feature_weights is not None else matrix

    def fit(self, X: pd.DataFrame, y=None) -> "ContentRecommender":
        self.pipeline_ = build_feature_pipeline()
        transformed = self._apply_weights(self.pipeline_.fit_transform(X))
        if self.embedding is not None:
            fit_matrix, fit_y = transformed, y
            if self.embedding_fit_sample_size is not None and len(transformed) > self.embedding_fit_sample_size:
                rng = np.random.default_rng(config.RANDOM_SEED)
                sample_idx = rng.choice(len(transformed), size=self.embedding_fit_sample_size, replace=False)
                fit_matrix = transformed[sample_idx]
                fit_y = np.asarray(y)[sample_idx] if y is not None else None
            self.embedding.fit(fit_matrix, fit_y)
            transformed = self.embedding.transform(transformed)
        self.train_ids_ = np.asarray(X.index)
        self.nn_ = NearestNeighbors(metric="euclidean").fit(transformed)
        return self

    def recommend(
        self, X_query: pd.DataFrame, k: int = config.DEFAULT_TOP_K, exclude_own_id: bool = True
    ) -> pd.DataFrame:
        """Returns one row per (query track, recommended track): columns
        `query_id`, `recommended_id`, `rank` (1..k), `distance`. Long
        format — easy to join back onto track metadata for display.

        `exclude_own_id=True` (the app's use case: recommending from the
        same catalog the model was trained on) drops a query track from
        its own recommendation list. Leave it `False` when the query pool
        and training pool are disjoint (e.g. `precision_at_k`'s train/test
        split), where a query track can never appear in its own results
        anyway and asking for one extra neighbor would be wasted work.
        """
        transformed = self._apply_weights(self.pipeline_.transform(X_query))
        if self.embedding is not None:
            transformed = self.embedding.transform(transformed)
        search_k = min(k + 1 if exclude_own_id else k, len(self.train_ids_))
        distances, indices = self.nn_.kneighbors(transformed, n_neighbors=search_k)

        query_ids = np.asarray(X_query.index)
        rows = []
        for row_i, query_id in enumerate(query_ids):
            rank = 0
            for col_j in range(indices.shape[1]):
                candidate_id = self.train_ids_[indices[row_i, col_j]]
                if exclude_own_id and candidate_id == query_id:
                    continue
                rank += 1
                rows.append(
                    {
                        "query_id": query_id,
                        "recommended_id": candidate_id,
                        "rank": rank,
                        "distance": distances[row_i, col_j],
                    }
                )
                if rank >= k:
                    break
        return pd.DataFrame(rows, columns=["query_id", "recommended_id", "rank", "distance"])


# --- Genre classifier (interpretability + distance-metric weighting) -----
# Not the recommender itself — see module docstring. 114 distinct genres
# make this a genuinely hard multiclass problem (macro-F1 will look much
# lower in absolute terms than academic_success's 3-class one); it's
# included for what its learned feature importances reveal, not to be a
# high-scoring classifier in its own right.


def default_classifier() -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=200, random_state=config.RANDOM_SEED, verbosity=-1)


def logistic_estimator() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def random_forest_estimator() -> BaseEstimator:
    """`max_depth=12` isn't just a regularization choice here — it's load
    bearing for SHAP. Left unbounded, trees on this ~72k-row, 113-genre
    training set grow to depth 40-60, and `shap.TreeExplainer`'s cost
    scales with tree depth (roughly quadratically) multiplied by the
    number of classes: 113 genres x depth-50 trees made SHAP fail to
    finish even 20 samples in 3 minutes. Capped at depth 12, the same
    300-sample SHAP call finishes in well under a minute — see
    `interpretability.py`, whose only caller of this factory is exactly
    that SHAP/feature-weighting path.
    """
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1
    )


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Logistic Regression": logistic_estimator,
    "Random Forest": random_forest_estimator,
    "LightGBM": default_classifier,
}


def build_classifier_pipeline(estimator: BaseEstimator | None = None) -> Pipeline:
    pipeline = build_feature_pipeline()
    steps = list(pipeline.steps)
    steps.append(("model", estimator if estimator is not None else default_classifier()))
    return Pipeline(steps=steps)


def compute_feature_weights_from_classifier(fitted_classifier_pipeline: Pipeline) -> np.ndarray:
    """Extracts per-feature importance from a fitted genre classifier,
    aligned to `build_preprocessor()`'s output column order, normalized
    so the mean weight is 1 — a feature the classifier ignores gets
    down-weighted below 1, one it relies on heavily gets weighted above
    1, and a classifier that used every feature equally would leave the
    recommender's distances completely unchanged.
    """
    fitted_model = fitted_classifier_pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
    else:
        # Multiclass LogisticRegression: coef_ is (n_classes, n_features);
        # average the magnitude of a feature's influence across all classes.
        importances = np.abs(fitted_model.coef_).mean(axis=0)
    return importances / importances.mean()


# --- Alternative distance metrics: PCA / NCA embeddings -------------------
# Both are compared empirically against plain and importance-weighted KNN
# via precision@k below, not assumed to be improvements — see the
# report's Discussion for which one, if any, actually wins on this data.


def pca_embedding(n_components: int = 10) -> BaseEstimator:
    """Unsupervised linear dimensionality reduction. With only ~25
    features here (well below where PCA's usual selling points — fighting
    the curse of dimensionality, speeding up distance computation — start
    to matter), the main open question this answers is whether discarding
    the lower-variance components acts as useful denoising or just throws
    away real signal.
    """
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=config.RANDOM_SEED)


def nca_embedding(n_components: int = 10) -> BaseEstimator:
    """Neighborhood Components Analysis: unlike `feature_weights` (which
    reuses an unrelated classifier's feature importances as a heuristic),
    NCA directly learns a linear projection via gradient descent that
    maximizes leave-one-out KNN accuracy under the genre labels — a
    supervised metric learned *for* nearest-neighbor search, not borrowed
    from a different model's objective.
    """
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    return NeighborhoodComponentsAnalysis(n_components=n_components, random_state=config.RANDOM_SEED)


# --- Evaluation: precision@k ----------------------------------------------


def precision_at_k(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = config.DEFAULT_TOP_K,
    feature_weights: np.ndarray | None = None,
    embedding: BaseEstimator | None = None,
    embedding_fit_sample_size: int | None = None,
    genre_map: dict[str, str] | None = None,
) -> float:
    """Fits a recommender on `train_df`'s audio features, recommends `k`
    tracks from that training catalog for every track in `test_df`, and
    returns the average fraction of those recommendations that share the
    query track's true genre. `train_df`/`test_df` must both still have
    `track_genre` — it's read here for evaluation only, never handed to
    the recommender itself.

    `embedding` (see `pca_embedding`/`nca_embedding`) is fit inside
    `ContentRecommender.fit` using `y_train` — NCA uses the genre labels
    to learn its projection; PCA ignores them. `embedding_fit_sample_size`
    is forwarded to `ContentRecommender` — see its docstring; pass e.g.
    `5000` for NCA on this project's real ~72k-row training split.

    `genre_map` (see `genre_families.GENRE_TO_FAMILY`), if given, maps
    both the query's true genre and each recommendation's genre through
    it before comparing — loosening "does the genre match exactly" to
    "does the genre *family* match." This changes what counts as a hit,
    not the recommender itself: the same fitted `ContentRecommender` is
    used either way.
    """
    X_train, y_train = data.split_features_target(train_df)
    X_test, y_test = data.split_features_target(test_df)

    recommender = ContentRecommender(
        feature_weights=feature_weights,
        embedding=embedding,
        embedding_fit_sample_size=embedding_fit_sample_size,
    ).fit(X_train, y_train)
    recs = recommender.recommend(X_test, k=k, exclude_own_id=False)

    recs["true_genre"] = recs["query_id"].map(y_test)
    recs["recommended_genre"] = recs["recommended_id"].map(y_train)
    if genre_map is not None:
        recs["true_genre"] = recs["true_genre"].map(lambda g: genre_map.get(g, g))
        recs["recommended_genre"] = recs["recommended_genre"].map(lambda g: genre_map.get(g, g))
    recs["match"] = recs["true_genre"] == recs["recommended_genre"]
    return float(recs.groupby("query_id")["match"].mean().mean())


def popularity_baseline_precision_at_k(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = config.DEFAULT_TOP_K,
    genre_map: dict[str, str] | None = None,
) -> float:
    """The simplest possible baseline: recommend the same `k` most
    popular tracks in the training catalog to every query, completely
    ignoring the query track's own audio features. Any real content-aware
    recommender needs to beat this — a recommender that just chases
    popularity isn't actually recommending anything. `genre_map` — see
    `precision_at_k` — is applied the same way, for a fair comparison at
    whichever genre granularity is being evaluated.
    """
    top_k_genres = (
        train_df.sort_values(config.POPULARITY_COL, ascending=False)
        .head(k)[config.GENRE_COL]
    )
    test_genres = test_df[config.GENRE_COL]
    if genre_map is not None:
        top_k_genres = top_k_genres.map(lambda g: genre_map.get(g, g))
        test_genres = test_genres.map(lambda g: genre_map.get(g, g))
    match_rates = test_genres.apply(lambda genre: (top_k_genres == genre).mean())
    return float(match_rates.mean())


def train_test_split_tracks(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track-level train/test split, stratified by genre so every genre
    is represented in both the training catalog and the evaluation
    queries (with 114 genres and ~114k tracks, an un-stratified split
    would still usually work out, but stratifying removes the risk).
    """
    return train_test_split(
        df, test_size=test_size, stratify=df[config.GENRE_COL], random_state=random_state
    )


# --- Training the production recommender ----------------------------------


def train_recommender(df: pd.DataFrame, use_weights: bool = True) -> ContentRecommender:
    """Fits the final recommender on the full catalog. If `use_weights`,
    first fits a genre classifier on the same data to derive
    feature-importance weights for the distance metric — the same
    weighting the notebook's precision@k experiment evaluates, applied
    once more to the production model.
    """
    X, y = data.split_features_target(df)
    weights = None
    if use_weights:
        classifier_pipeline = build_classifier_pipeline()
        classifier_pipeline.fit(X, y)
        weights = compute_feature_weights_from_classifier(classifier_pipeline)
    return ContentRecommender(feature_weights=weights).fit(X)


def save_pipeline(recommender: ContentRecommender, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(recommender, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> ContentRecommender:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train the recommender first: `python scripts/train.py`."
        )
    return joblib.load(path)
