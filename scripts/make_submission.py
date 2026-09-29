"""Generate a Kaggle-submittable submission.csv from the saved pipeline.

Unlike the Spotify recommendation project (a dataset with no leaderboard),
this is a real competition — AUC-scored on Kaggle's servers, so the
submission needs predicted *probabilities*, not hard 0/1 labels.
"""

import pandas as pd

from recommendation_showcase.collaborative import config, data, model


def main() -> None:
    test_panel = data.load_full_test_panel()
    pipeline = model.load_pipeline()

    ids = test_panel[config.ID_COL]
    X_test = test_panel.drop(columns=[config.ID_COL])
    proba = pipeline.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: proba})
    submission_path = config.PROJECT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Wrote {len(submission)} predictions to {submission_path}")


if __name__ == "__main__":
    main()
