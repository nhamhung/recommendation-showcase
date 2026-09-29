"""Train the production recommender on the full catalog and save it.

Equivalent to the notebook's modeling steps, without the notebook.
"""

import argparse

from recommendation_showcase.content_based import config, data, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip genre-classifier feature weighting; use plain unweighted Euclidean distance.",
    )
    args = parser.parse_args()

    df = data.load_tracks()
    recommender = model.train_recommender(df, use_weights=not args.no_weights)
    model.save_pipeline(recommender)
    print(f"Trained on {len(df)} tracks. Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
