#!/usr/bin/env python
"""Build the KKBox leaderboard models and their submission.

Usage:
    # 1. Chronological validation (fit on first 80% of train, score the last 20%)
    python scripts/make_leaderboard_submission.py --model lgbm --validate
    python scripts/make_leaderboard_submission.py --model nn --validate

    # 2. Fit on all of train, predict test
    python scripts/make_leaderboard_submission.py --model lgbm          # the best submission: lr 0.1, 1980 rounds
    python scripts/make_leaderboard_submission.py --model nn

    # 3. Blend: weights learned on the validation predictions, applied to test
    python scripts/blend_leaderboard_submission.py

Every run saves its predictions to
data/collaborative/processed/leaderboard_predictions/<model>_<valid|test>.npy.
A single model's test predictions are also written as a submission CSV.

The feature table (~9.9M rows) is cached at
data/collaborative/processed/leaderboard_features.parquet after the first
build (~3 minutes); pass --rebuild after changing leaderboard_features.py.

Then submit (a late submission — the competition closed in 2017):
    kaggle competitions submit -c kkbox-music-recommendation-challenge -f submission_leaderboard.csv -m "message"
"""

import argparse
import functools
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation_showcase.collaborative import config  # noqa: E402
from recommendation_showcase.collaborative import leaderboard_features as lf  # noqa: E402
from recommendation_showcase.collaborative import leaderboard_model as lm  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

FEATURE_CACHE = config.DATA_PROCESSED_DIR / "leaderboard_features.parquet"
PREDICTIONS_DIR = config.DATA_PROCESSED_DIR / "leaderboard_predictions"


def load_table(rebuild: bool) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not rebuild:
        print(f"Loading cached feature table {FEATURE_CACHE} ...")
        return pd.read_parquet(FEATURE_CACHE)
    print("Building feature table from the raw CSVs (~3 minutes) ...")
    table = lf.build_feature_table()
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURE_CACHE, index=False)
    return table


def save_predictions(name: str, part: str, predictions: np.ndarray) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREDICTIONS_DIR / f"{name}_{part}.npy", predictions.astype(np.float32))
    print(f"Saved {part} predictions to {PREDICTIONS_DIR / f'{name}_{part}.npy'}")


def run_lgbm(split: lm.Split, args) -> dict[str, np.ndarray]:
    params = {"learning_rate": args.learning_rate, "seed": config.RANDOM_SEED + args.seed}
    if args.validate:
        booster = lm.train(split, num_rounds=5000, params=params, log_every=100)
        print(f"Validation AUC (last 20% of train, chronological): "
              f"{booster.best_score['valid_0']['auc']:.5f} at {booster.best_iteration} rounds")
        print(f"Suggested --rounds for the full fit (~25% more data): {int(booster.best_iteration * 1.25)}")
        return {"valid": booster.predict(split.valid[split.features], num_iteration=booster.best_iteration)}
    booster = lm.train(split, num_rounds=args.rounds, params=params)
    return {"test": lm.predict_test(booster, split)[config.TARGET_COL].to_numpy()}


def run_nn(split: lm.Split, args) -> dict[str, np.ndarray]:
    from recommendation_showcase.collaborative import leaderboard_nn

    predictions = leaderboard_nn.train_and_predict(split, epochs=args.epochs, seed=config.RANDOM_SEED + args.seed)
    if not args.validate:
        # Test rows are already in id order (test.csv order == id order).
        order = np.argsort(split.test["test_id"].to_numpy())
        predictions["test"] = predictions["test"][order]
        predictions.pop("valid", None)
    else:
        predictions.pop("test", None)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["lgbm", "nn"], default="lgbm")
    parser.add_argument("--validate", action="store_true", help="Score on the chronological validation split instead of predicting test.")
    parser.add_argument("--rounds", type=int, default=1980,
                        help="LightGBM rounds for the full fit (default: 1980 — the best submission, at learning rate 0.1).")
    parser.add_argument("--learning-rate", type=float, default=lm.PARAMS["learning_rate"],
                        help=f"LightGBM learning rate (default: {lm.PARAMS['learning_rate']}).")
    parser.add_argument("--epochs", type=int, default=2, help="Neural-network epochs (default: 2).")
    parser.add_argument("--seed", type=int, default=0, help="Offset added to the random seed (for seed bagging).")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cached feature table.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "submission_leaderboard.csv")
    args = parser.parse_args()

    started = time.time()
    table = load_table(args.rebuild)
    split = lm.prepare(table, validation=args.validate)
    del table
    gc.collect()
    print(f"{args.model}: {len(split.features)} features; fitting on {len(split.fit):,} rows ...")

    predictions = (run_lgbm if args.model == "lgbm" else run_nn)(split, args)
    name = args.model if args.seed == 0 else f"{args.model}_seed{args.seed}"
    for part, values in predictions.items():
        save_predictions(name, part, values)

    if "test" in predictions:
        ids = np.sort(split.test["test_id"].to_numpy().astype(np.int64))
        pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: predictions["test"]}).to_csv(args.output, index=False)
        print(f"Wrote {len(ids):,} predictions to {args.output}")
    print(f"Done in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
