"""Neural-network member of the leaderboard ensemble — EXPERIMENTAL.

Not used in any submission: its one validation run was stopped before
finishing, so it has no measured score. Kept as the starting point for the
LightGBM + NN blend (`scripts/blend_leaderboard_submission.py`).

A different inductive bias from LightGBM: every ID column (user, song,
artist, ...) gets a learned embedding, concatenated with the standardised
numeric features from `leaderboard_features`, and fed to a small MLP.
Trees split IDs into groups; embeddings place them in a continuous space
where similar users/songs share statistical strength — which is why
blending the two helps even when the network alone scores lower.

Trains on Apple-silicon GPUs (`mps`) when available, else CUDA, else CPU.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config
from . import leaderboard_features as lf
from .leaderboard_model import Split

EMBEDDING_DIMS = {
    "msno": 32, "song_id": 32, "artist_name": 16, "composer": 8, "lyricist": 8,
    "genre_ids": 8, "first_genre": 4, "source_system_tab": 4, "source_screen_name": 4,
    "source_type": 4, "language": 4, "city": 4, "gender": 2, "registered_via": 2, "isrc_country": 4,
}


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Preprocessor:
    """Fits numeric scaling and categorical index maps on the fit rows only."""

    def fit(self, frame: pd.DataFrame, features: list[str]) -> "Preprocessor":
        self.categorical = [c for c in EMBEDDING_DIMS if c in features]
        self.numeric = [c for c in features if c not in lf.CATEGORICAL_FEATURES]
        values = self._numeric(frame)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0) + 1e-6
        # Categorical codes: shift so missing (-1) becomes 0; codes beyond
        # what the fit rows saw become 0 too ("unknown").
        self.cardinality = {c: int(frame[c].max()) + 2 for c in self.categorical}
        return self

    def _numeric(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.numeric].to_numpy(np.float32)
        # Counts and gaps are heavy-tailed; a signed log keeps them on a sane scale.
        return np.sign(values) * np.log1p(np.abs(values))

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = (self._numeric(frame) - self.mean) / self.std
        # float16 storage halves RAM for the ~6M x 200 matrix; batches are
        # cast back to float32 on the device.
        numeric = np.clip(np.nan_to_num(numeric, nan=0.0), -30, 30).astype(np.float16)
        categorical = np.stack(
            [np.where((frame[c] >= 0) & (frame[c] < self.cardinality[c] - 1), frame[c] + 1, 0)
             for c in self.categorical], axis=1,
        ).astype(np.int64)
        return numeric, categorical


class Network(nn.Module):
    def __init__(self, n_numeric: int, cardinality: dict[str, int], hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, EMBEDDING_DIMS[c]) for c, n in cardinality.items())
        width = n_numeric + sum(EMBEDDING_DIMS[c] for c in cardinality)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.mlp(torch.cat([numeric, *embedded], dim=1)).squeeze(1)


def _predict(net: Network, numeric: np.ndarray, categorical: np.ndarray, dev: torch.device) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(numeric), 65_536):
            logits = net(torch.from_numpy(numeric[i:i + 65_536]).to(dev).float(),
                         torch.from_numpy(categorical[i:i + 65_536]).to(dev))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_and_predict(
    split: Split, epochs: int = 2, batch_size: int = 4096, learning_rate: float = 2e-3,
    seed: int = config.RANDOM_SEED, log=print,
) -> dict[str, np.ndarray]:
    """Fit on `split.fit`; return predictions for `valid` (if present) and `test`.

    Validation AUC is logged after every epoch when a validation frame exists.
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    dev = device()
    prep = Preprocessor().fit(split.fit, split.features)
    fit_numeric, fit_categorical = prep.transform(split.fit)
    y = split.fit[config.TARGET_COL].to_numpy(np.float32)
    valid = prep.transform(split.valid) if split.valid is not None else None

    net = Network(fit_numeric.shape[1], prep.cardinality).to(dev)
    steps = epochs * int(np.ceil(len(y) / batch_size))
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        net.train()
        order = generator.permutation(len(y))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            logits = net(torch.from_numpy(fit_numeric[idx]).to(dev).float(), torch.from_numpy(fit_categorical[idx]).to(dev))
            loss = loss_fn(logits, torch.from_numpy(y[idx]).to(dev))
            loss.backward()
            optimizer.step()
            schedule.step()
        if valid is not None:
            auc = roc_auc_score(split.valid[config.TARGET_COL], _predict(net, *valid, dev))
            log(f"  epoch {epoch + 1}/{epochs}: validation AUC {auc:.5f}")

    predictions = {"test": _predict(net, *prep.transform(split.test), dev)}
    if valid is not None:
        predictions["valid"] = _predict(net, *valid, dev)
    return predictions
