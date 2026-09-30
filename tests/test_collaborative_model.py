"""Tests for the model pipeline and evaluation helpers, using a small
synthetic panel instead of the real KKBox data.
"""

import numpy as np

from recommendation_showcase.collaborative import config, model
from tests.test_collaborative_features import make_synthetic_panel


def test_build_pipeline_predicts_proba_shape():
    panel = make_synthetic_panel(n_rows=300)
    X, y = panel.drop(columns=[config.TARGET_COL]), panel[config.TARGET_COL]

    pipeline = model.build_pipeline(model.logistic_estimator())
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_holdout_eval_returns_auc_in_valid_range():
    panel = make_synthetic_panel(n_rows=500)
    result = model.holdout_eval(panel, estimator=model.logistic_estimator())

    assert 0.0 <= result["auc"] <= 1.0
    assert result["n_train"] + result["n_holdout"] == len(panel)


def test_save_and_load_pipeline_roundtrip(tmp_path):
    panel = make_synthetic_panel(n_rows=300)
    pipeline = model.train_pipeline(panel, estimator=model.logistic_estimator())

    path = tmp_path / "model.joblib"
    model.save_pipeline(pipeline, path=path)
    loaded = model.load_pipeline(path=path)

    X = panel.drop(columns=[config.TARGET_COL])
    np.testing.assert_allclose(pipeline.predict_proba(X), loaded.predict_proba(X))


def test_load_pipeline_missing_file_raises_with_helpful_message(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    try:
        model.load_pipeline(path=missing_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "scripts/train.py" in str(exc)
