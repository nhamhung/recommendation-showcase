"""SHAP-based model interpretability for the genre classifier.

SHAP (SHapley Additive exPlanations) attributes each prediction to
individual feature contributions. This project explains the genre
classifier (see `model.py`'s module docstring for why it exists), not
the recommender itself — a nearest-neighbor search has no "feature
contribution to the prediction" in the SHAP sense, since it isn't
predicting anything; it's the classifier's feature importances that get
reused as the recommender's distance-metric weights, and SHAP is what
makes those importances legible per-genre rather than just one ranked
list.
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted genre-classifier pipeline (from
    `model.build_classifier_pipeline`).

    Returns `(explanation, X_transformed)`: the SHAP explanation (one row
    per sampled input, one column per transformed feature, one "layer"
    per genre for this multiclass model) and the transformed feature
    matrix it was computed against (with real column names).

    `max_samples` subsamples `X` before computing SHAP values: exact tree
    SHAP is fast per-row but still linear in row count.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    X_transformed = pd.DataFrame(
        preprocessing.transform(X), columns=feature_names, index=X.index
    )

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(explanation: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value, averaged across genres
    for this multiclass explanation (i.e. overall importance, not
    per-genre).
    """
    values = explanation.values
    if values.ndim == 3:  # (n_samples, n_features, n_genres)
        importance = abs(values).mean(axis=(0, 2))
    else:  # (n_samples, n_features)
        importance = abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
