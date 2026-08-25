"""Leakage-free volatility forecasting utilities.

This module is the corrected methodology for the volatility-forecasting
experiments in Main.ipynb. It fixes three issues in the original notebook
pipeline and adds the standard econometric baseline:

1. Forward target. The notebook predicted the CURRENT 30-day rolling
   volatility, whose estimation window overlaps ~29/30 days with the input
   features, so models mostly learn persistence. `forward_realized_vol`
   builds the target from returns strictly AFTER the prediction date.
2. Scaler leakage. The notebook fit MinMaxScaler on the full sample before
   the chronological split. `fit_scaler_train_only` scales with
   train-window statistics only.
3. Missing baseline. Any ML vol forecast must be benchmarked against
   HAR-RV (Corsi 2009); recent comparisons find LSTM gains over HAR-RV on
   index data are modest and often insignificant. `HARRV` implements it
   with plain OLS. `qlike` and `diebold_mariano` provide the standard loss
   and forecast-comparison test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing annualized realized volatility (uses data through t only)."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def forward_realized_vol(returns: pd.Series, horizon: int = 21) -> pd.Series:
    """Annualized realized volatility over the NEXT `horizon` days.

    The value at date t is computed from returns on (t, t+horizon] only,
    so a model trained to predict it from features known at t has a genuine
    forecasting target rather than an overlapping-window echo.
    """
    fwd = returns.rolling(horizon, min_periods=horizon).std().shift(-horizon)
    return fwd * np.sqrt(TRADING_DAYS)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/validation/test split (no shuffling)."""
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def fit_scaler_train_only(
    train: pd.DataFrame, *frames: pd.DataFrame
) -> Tuple[pd.DataFrame, ...]:
    """Min-max scale using TRAIN-window statistics only.

    Fitting any scaler on the full sample leaks test-period ranges into
    training. Returns the scaled train frame followed by the other frames
    scaled with the same (train) parameters.
    """
    lo = train.min()
    hi = train.max()
    span = (hi - lo).replace(0, np.nan)

    def _scale(frame: pd.DataFrame) -> pd.DataFrame:
        return ((frame - lo) / span).fillna(0.0)

    return tuple([_scale(train)] + [_scale(f) for f in frames])


@dataclass
class HARRV:
    """HAR-RV of Corsi (2009): RV_{t+h} ~ RV_t + RV_t(weekly) + RV_t(monthly).

    Regressors at date t use data through t only. Plain OLS via lstsq.
    """

    horizon: int = 21
    coef_: Optional[np.ndarray] = None

    @staticmethod
    def _features(rv: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rv_d": rv,
                "rv_w": rv.rolling(5, min_periods=5).mean(),
                "rv_m": rv.rolling(22, min_periods=22).mean(),
            }
        )

    def fit(self, rv: pd.Series, target: pd.Series) -> "HARRV":
        X = self._features(rv)
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        A = np.column_stack([np.ones(len(frame)), frame[["rv_d", "rv_w", "rv_m"]].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict(self, rv: pd.Series) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit() first")
        X = self._features(rv)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        out = pd.Series(A @ self.coef_, index=rv.index)
        out[X.isna().any(axis=1)] = np.nan
        return out


def qlike(forecast_vol: pd.Series, realized_vol_series: pd.Series) -> float:
    """QLIKE loss on variances; the standard robust loss for vol forecasts.

    QLIKE = mean( r/f - log(r/f) - 1 ), with r, f variances. Lower is better.
    """
    f = (pd.to_numeric(forecast_vol, errors="coerce") ** 2).replace(0, np.nan)
    r = pd.to_numeric(realized_vol_series, errors="coerce") ** 2
    ratio = (r / f).dropna()
    ratio = ratio[ratio > 0]
    if ratio.empty:
        return np.nan
    return float((ratio - np.log(ratio) - 1.0).mean())


def diebold_mariano(
    errors_a: pd.Series, errors_b: pd.Series, lag: Optional[int] = None
) -> Tuple[float, float]:
    """Diebold-Mariano test on loss differentials (squared errors by default
    already applied by caller). Newey-West long-run variance.

    Returns (dm_statistic, two_sided_p_value). Negative statistic means
    model A has lower loss than model B.
    """
    d = (pd.to_numeric(errors_a, errors="coerce") - pd.to_numeric(errors_b, errors="coerce")).dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    if lag is None:
        lag = int(np.floor(n ** (1 / 3)))
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float((dc**2).mean())
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        cov = float((dc.iloc[k:].to_numpy() * dc.iloc[:-k].to_numpy()).mean())
        lrv += 2.0 * w * cov
    if lrv <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / n)
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(stat), float(p)
