#!/usr/bin/env python
"""Blend saved leaderboard-model predictions into one submission.

Usage:
    python scripts/blend_leaderboard_submission.py                  # every model with both valid + test predictions
    python scripts/blend_leaderboard_submission.py --models lgbm nn

Weights are chosen on the chronological validation predictions (the last
20% of train), then applied to the test predictions. Members are blended by
*rank*, not probability: AUC only depends on ordering, and LightGBM's and
the network's probabilities aren't calibrated to the same scale.

Validation weights are an estimate: the validation models were fit on 80%
of train and the test models on 100%, so the blend is chosen on slightly
weaker versions of the same models.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation_showcase.collaborative import config  # noqa: E402
from recommendation_showcase.collaborative.leaderboard_model import VALIDATION_FRACTION  # noqa: E402

PREDICTIONS_DIR = config.DATA_PROCESSED_DIR / "leaderboard_predictions"


def ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", help="Prediction names to blend (default: all with valid + test files).")
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid step (default: 0.05).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "submission_blend.csv")
    args = parser.parse_args()

    names = args.models or sorted(
        p.name[: -len("_valid.npy")] for p in PREDICTIONS_DIR.glob("*_valid.npy")
        if (PREDICTIONS_DIR / p.name.replace("_valid", "_test")).exists()
    )
    if len(names) < 2:
        raise SystemExit(f"Need at least two models with valid + test predictions in {PREDICTIONS_DIR}; found {names}")

    target = pd.read_csv(config.TRAIN_CSV, usecols=[config.TARGET_COL])[config.TARGET_COL].to_numpy()
    y_valid = target[int(len(target) * (1 - VALIDATION_FRACTION)):]
    valid = {n: ranks(np.load(PREDICTIONS_DIR / f"{n}_valid.npy")) for n in names}
    test = {n: ranks(np.load(PREDICTIONS_DIR / f"{n}_test.npy")) for n in names}

    for n in names:
        print(f"  {n:<14} validation AUC {roc_auc_score(y_valid, valid[n]):.5f}")

    grid = np.arange(0, 1 + 1e-9, args.step)
    best_auc, best_weights = -1.0, None
    for weights in itertools.product(grid, repeat=len(names) - 1):
        last = 1 - sum(weights)
        if last < -1e-9:
            continue
        w = np.array([*weights, max(last, 0.0)])
        auc = roc_auc_score(y_valid, sum(wi * valid[n] for wi, n in zip(w, names)))
        if auc > best_auc:
            best_auc, best_weights = auc, w
    print("Blend weights: " + ", ".join(f"{n} {w:.2f}" for n, w in zip(names, best_weights)))
    print(f"Blended validation AUC: {best_auc:.5f}")

    blended = sum(w * test[n] for w, n in zip(best_weights, names))
    ids = pd.read_csv(config.TEST_CSV, usecols=[config.ID_COL])[config.ID_COL].to_numpy()
    pd.DataFrame({config.ID_COL: np.sort(ids), config.TARGET_COL: blended}).to_csv(args.output, index=False)
    print(f"Wrote {len(ids):,} blended predictions to {args.output}")


if __name__ == "__main__":
    main()
