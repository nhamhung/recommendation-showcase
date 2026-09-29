"""SHAP-based model interpretability.

Same `shap.TreeExplainer` approach as the other three projects. This is
a binary classifier, not multi-class (academic_success, or this
project's own genre-classifier analogue in the Spotify project) — SHAP
values here are a plain 2D array (samples x features), no per-class
"layer" to average over.
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted tree-based pipeline (from
    `model.build_pipeline`).

    `max_samples` subsamples `X` before computing SHAP values: exact tree
    SHAP is fast per-row but still linear in row count, and this panel
    has millions of rows.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    transformed = preprocessing.transform(X)
    if hasattr(transformed, "toarray"):  # sparse one-hot output — see build_preprocessor
        transformed = transformed.toarray()
    X_transformed = pd.DataFrame(transformed, columns=feature_names, index=X.index)

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(explanation: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value."""
    values = explanation.values
    if values.ndim == 3:  # some binary-classifier explainers still return (n, features, 2)
        importance = abs(values[:, :, -1]).mean(axis=0)
    else:
        importance = abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
