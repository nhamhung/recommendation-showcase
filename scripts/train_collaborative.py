"""Train the production pipeline on the full training panel and save it.

Equivalent to the notebook's modeling steps, without the notebook.
Verified directly: the full 7.4M-row join + feature engineering + a
LightGBM fit takes under 2 minutes and stays well within a normal
laptop's memory — no subsampling needed.
"""

import argparse

from recommendation_showcase.collaborative import config, data, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(model.MODEL_FACTORIES.keys()),
        default="LightGBM",
        help="Which model family to train (default: LightGBM).",
    )
    args = parser.parse_args()

    panel = data.load_full_train_panel()
    estimator = model.MODEL_FACTORIES[args.model]()
    pipeline = model.train_pipeline(panel, estimator=estimator)
    model.save_pipeline(pipeline)
    print(f"Trained {args.model} on {len(panel)} rows. Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
