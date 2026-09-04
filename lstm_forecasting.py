"""The notebook's LSTM, lifted into a tested module and given the feature panel.

`Main.ipynb` trains a PyTorch LSTM on a sequence of past volatilities. Two
things were wrong with it and are fixed here rather than reproduced: it
predicted the CURRENT 30-day rolling volatility, whose window overlaps its own
inputs, and it scaled with statistics from the full sample. This module keeps
the architecture (one LSTM layer, two dense layers, ReLU) and changes the
protocol: forward target, training-window scaling only, early stopping on a
validation tail that is the END of the training window rather than a random
split, and a fixed seed so a rerun reproduces the number.

The point of including it is comparability, not hope. The literature is
consistent that neural gains over HAR on index volatility are small, and the
README's baseline chapter already reports ridge as statistically
indistinguishable from HAR. Adding the alternative-data features is the one
change that could plausibly move that, since a sequence model is the natural
place for a slow-moving attention or news series to interact with the
volatility path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def make_sequences(features: np.ndarray, target: np.ndarray, length: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling windows of `length` rows, each labelled by the target at the last row.

    Returns (X, y, positions) where `positions[i]` is the row index of the last
    observation in window i, so a prediction can be put back on the right date.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    n = len(features)
    if n < length:
        return (np.empty((0, length, features.shape[1])), np.empty(0),
                np.empty(0, dtype=int))
    idx = np.arange(length - 1, n)
    windows = np.stack([features[i - length + 1:i + 1] for i in idx])
    return windows, target[idx], idx


@dataclass
class LSTMForecaster:
    """Sequence model over HAR terms plus whatever else is handed to it.

    Parameters are small by design: about 1,100 training rows is not a regime in
    which a large network can be fitted honestly, and a model that needs one to
    beat a four-parameter regression has not shown anything.
    """

    seq_len: int = 10
    hidden: int = 32
    dense: int = 16
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    val_tail: int = 126
    seed: int = 0
    columns_: Optional[list] = None
    _model: object = field(default=None, repr=False)
    _lo: Optional[np.ndarray] = field(default=None, repr=False)
    _span: Optional[np.ndarray] = field(default=None, repr=False)

    def _build(self, n_features: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int, dense: int):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, dense), nn.ReLU(), nn.Linear(dense, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(n_features, self.hidden, self.dense)

    def _scale(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num((values - self._lo) / self._span, nan=0.0,
                             posinf=0.0, neginf=0.0)

    def fit(self, frame: pd.DataFrame, columns: Sequence[str],
            target_col: str = "target") -> "LSTMForecaster":
        import torch
        from torch import nn

        self.columns_ = list(columns)
        train = frame[self.columns_ + [target_col]].dropna()
        if len(train) < self.seq_len + self.val_tail + 50:
            raise ValueError("not enough rows to fit the LSTM")
        values = train[self.columns_].to_numpy(dtype=float)
        # min-max scaling on the TRAINING window only; span guarded so a
        # constant column (the calendar dummies in a quiet stretch) cannot
        # produce a division by zero
        self._lo = values.min(axis=0)
        span = values.max(axis=0) - self._lo
        self._span = np.where(span <= 0, 1.0, span)
        X, y, _ = make_sequences(self._scale(values),
                                 train[target_col].to_numpy(dtype=float), self.seq_len)
        if len(X) <= self.val_tail:
            raise ValueError("not enough sequences to hold out a validation tail")
        cut = len(X) - self.val_tail
        Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]

        torch.manual_seed(self.seed)
        model = self._build(X.shape[2])
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32)
        yva_t = torch.tensor(yva, dtype=torch.float32)

        best, best_state, stale = np.inf, None, 0
        generator = torch.Generator().manual_seed(self.seed)
        n = len(Xtr_t)
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                sel = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss_fn(model(Xtr_t[sel]), ytr_t[sel]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(loss_fn(model(Xva_t), yva_t))
            if val < best - 1e-9:
                best, stale = val, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        import torch

        if self._model is None or self.columns_ is None:
            raise RuntimeError("fit() first")
        block = frame[self.columns_]
        usable = block.dropna()
        out = pd.Series(np.nan, index=frame.index)
        X, _, positions = make_sequences(
            self._scale(usable.to_numpy(dtype=float)),
            np.zeros(len(usable)), self.seq_len)
        if len(X) == 0:
            return out
        with torch.no_grad():
            preds = self._model(torch.tensor(X, dtype=torch.float32)).numpy()
        out.loc[usable.index[positions]] = preds
        return out
