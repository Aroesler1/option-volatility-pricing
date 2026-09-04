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
from typing import Optional, Sequence, Tuple

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
    """Time-ordered train/validation/test split (no shuffling).

    Provided for notebook-style single-split workflows, and used by the scaler
    test to build a split whose statistics can be checked. The benchmark
    scripts do NOT use it: `run_vol_benchmark.py`, `run_iv_benchmark.py` and
    `run_intraday_benchmark.py` each run their own expanding-window loop that
    refits every `refit` days, which is a stricter protocol than one fixed
    split because every forecast is made from a model that saw only prior data.
    """
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


# ---------------------------------------------------------------------------
# Path-dependent volatility (Guyon & Lekeufack)
# ---------------------------------------------------------------------------


def tspl_kernel(max_lag: int, alpha: float, delta: float) -> np.ndarray:
    """Time-shifted power-law kernel K(i) = (i + delta)^(-alpha), normalised to sum 1.

    The shift `delta` keeps the kernel finite at lag 0 and controls how sharply
    weight concentrates on recent observations; `alpha` sets the decay, so a
    single kernel spans short and long memory rather than needing two regimes.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    lags = np.arange(max_lag, dtype=float)
    weights = (lags + delta) ** (-alpha)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"degenerate kernel for alpha={alpha}, delta={delta}")
    return weights / total


def _convolve_causal(series: pd.Series, kernel: np.ndarray) -> pd.Series:
    """Weighted sum of the trailing `len(kernel)` values, kernel[0] on the most recent.

    Strictly causal: the value at t uses observations at t, t-1, ... only.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    filled = np.nan_to_num(values, nan=0.0)
    # np.convolve with the reversed kernel gives sum_i kernel[i] * x[t-i]
    conv = np.convolve(filled, kernel[::-1], mode="full")[: len(filled)]
    out = pd.Series(conv, index=series.index)
    out.iloc[: len(kernel) - 1] = np.nan  # windows not yet full
    return out


def pdv_features(
    returns: pd.Series,
    alpha1: float = 0.6,
    delta1: float = 1.0,
    alpha2: float = 0.4,
    delta2: float = 1.0,
    max_lag: int = 252,
) -> pd.DataFrame:
    """The two path-dependent state variables of Guyon & Lekeufack (2023).

    R1 is a weighted sum of past RETURNS: a trend feature, and the channel
    through which the leverage effect enters (falling prices raise vol).
    R2 is a weighted sum of past SQUARED returns: an activity feature whose
    square root has the units of volatility.

    Their central empirical claim is that

        sigma_t ~ beta0 + beta1 * R1_t + beta2 * sqrt(R2_t)

    explains most of the level of index volatility, i.e. volatility is mostly
    path-dependent rather than an exogenous latent process. Both kernels are
    time-shifted power laws so one term carries short and long memory together.
    """
    r = pd.to_numeric(returns, errors="coerce")
    k1 = tspl_kernel(max_lag, alpha1, delta1)
    k2 = tspl_kernel(max_lag, alpha2, delta2)
    r1 = _convolve_causal(r, k1)
    r2 = _convolve_causal(r.pow(2), k2)
    return pd.DataFrame({"R1": r1, "R2": r2, "sqrt_R2": np.sqrt(r2.clip(lower=0.0))})


@dataclass
class PDVModel:
    """Guyon-Lekeufack path-dependent volatility forecaster.

    Kernel shape parameters are held fixed and the three loadings are fitted by
    OLS, which keeps the estimator linear and avoids the identification problems
    of jointly optimising kernels and loadings on a short sample. `fit_kernels`
    performs a coarse grid search over (alpha1, alpha2) when the caller wants it.
    """

    alpha1: float = 0.6
    delta1: float = 1.0
    alpha2: float = 0.4
    delta2: float = 1.0
    max_lag: int = 252
    coef_: Optional[np.ndarray] = None

    def _design(self, returns: pd.Series) -> pd.DataFrame:
        feats = pdv_features(
            returns, self.alpha1, self.delta1, self.alpha2, self.delta2, self.max_lag
        )
        return feats[["R1", "sqrt_R2"]]

    def fit(self, returns: pd.Series, target: pd.Series) -> "PDVModel":
        X = self._design(returns)
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        if len(frame) < 30:
            raise ValueError("not enough overlapping observations to fit PDV")
        A = np.column_stack([np.ones(len(frame)), frame[["R1", "sqrt_R2"]].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit() first")
        X = self._design(returns)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        out = pd.Series(A @ self.coef_, index=returns.index)
        out[X.isna().any(axis=1)] = np.nan
        # volatility cannot be negative; the linear form can dip below zero when
        # the trend term is strongly positive
        return out.clip(lower=1e-6)

    def fit_kernels(
        self,
        returns: pd.Series,
        target: pd.Series,
        alphas1: Sequence[float] = (0.3, 0.6, 1.0, 1.5, 2.0),
        alphas2: Sequence[float] = (0.3, 0.6, 1.0, 1.5, 2.0),
    ) -> "PDVModel":
        """Grid-search kernel decay on QLIKE, then refit loadings.

        Selection is on QLIKE rather than MSE for two reasons. QLIKE is the
        loss the model is evaluated under, and MSE on a right-skewed volatility
        target rewards over-smoothed forecasts that cannot rise during stress.
        The difference is not cosmetic: on SPY the MSE-selected kernel scored a
        QLIKE of 1.7e6 (predictions collapsing toward zero, which QLIKE
        punishes without bound) while the QLIKE-selected kernel scored 0.61.
        """
        best = (np.inf, self.alpha1, self.alpha2)
        for a1 in alphas1:
            for a2 in alphas2:
                trial = PDVModel(a1, self.delta1, a2, self.delta2, self.max_lag)
                try:
                    trial.fit(returns, target)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                pred = trial.predict(returns)
                loss = qlike(pred, target)
                if np.isfinite(loss) and loss < best[0]:
                    best = (loss, a1, a2)
        self.alpha1, self.alpha2 = best[1], best[2]
        return self.fit(returns, target)


@dataclass
class HARQ(HARRV):
    """HARQ of Bollerslev, Patton & Quaedvlieg (2016).

    Realized variance is a GENERATED regressor: it estimates latent integrated
    variance with error, and that error varies over time. HARQ lets the daily
    loading vary with the estimated measurement error, so the model leans on the
    daily term when it is precisely measured and discounts it when it is not.
    Empirically it beats both the jump-augmented HAR and the Patton-Sheppard
    semivariance HAR.

    `rq` is the realized quarticity proxy; sqrt(RQ) scales the daily interaction.
    """

    def fit_q(self, rv: pd.Series, rq: pd.Series, target: pd.Series) -> "HARQ":
        X = self._features(rv)
        X = X.assign(rv_d_q=X["rv_d"] * np.sqrt(pd.to_numeric(rq, errors="coerce")))
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        A = np.column_stack([np.ones(len(frame)),
                             frame[["rv_d", "rv_w", "rv_m", "rv_d_q"]].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict_q(self, rv: pd.Series, rq: pd.Series) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit_q() first")
        X = self._features(rv)
        X = X.assign(rv_d_q=X["rv_d"] * np.sqrt(pd.to_numeric(rq, errors="coerce")))
        A = np.column_stack([np.ones(len(X)),
                             X[["rv_d", "rv_w", "rv_m", "rv_d_q"]].to_numpy()])
        out = pd.Series(A @ self.coef_, index=rv.index)
        out[X.isna().any(axis=1)] = np.nan
        return out.clip(lower=1e-6)



@dataclass
class HARPDV:
    """HAR-RV augmented with the two path-dependent state variables.

    The scientifically interesting question is not "PDV or HAR" but whether
    path-dependence carries information HAR's three horizon averages do not.
    HAR already summarises the recent squared-return path, so R2 is largely
    redundant with it; R1 (signed, trend) is the genuinely new channel and is
    how the leverage effect enters. Nesting both inside HAR isolates that
    increment, and the nesting makes the comparison a fair one.
    """

    alpha1: float = 1.0
    delta1: float = 1.0
    alpha2: float = 0.3
    delta2: float = 1.0
    max_lag: int = 252
    coef_: Optional[np.ndarray] = None

    _COLS = ["rv_d", "rv_w", "rv_m", "R1", "sqrt_R2"]

    def _design(self, rv: pd.Series, returns: pd.Series) -> pd.DataFrame:
        har = HARRV._features(rv)
        pdv = pdv_features(returns, self.alpha1, self.delta1,
                           self.alpha2, self.delta2, self.max_lag)
        return pd.concat([har, pdv[["R1", "sqrt_R2"]]], axis=1)[self._COLS]

    def fit(self, rv: pd.Series, returns: pd.Series, target: pd.Series) -> "HARPDV":
        X = self._design(rv, returns)
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        if len(frame) < 60:
            raise ValueError("not enough overlapping observations to fit HAR-PDV")
        A = np.column_stack([np.ones(len(frame)), frame[self._COLS].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict(self, rv: pd.Series, returns: pd.Series) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit() first")
        X = self._design(rv, returns)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        out = pd.Series(A @ self.coef_, index=rv.index)
        out[X.isna().any(axis=1)] = np.nan
        return out.clip(lower=1e-6)


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, n_boot: int, block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap resampling indices.

    Blocks have geometric length with mean `block_length` and wrap around the
    end of the sample, which keeps the resampled series stationary. Serial
    dependence matters here: QLIKE loss differentials from overlapping h-step
    forecasts are correlated out to ~h-1 lags, and an i.i.d. bootstrap would
    understate their long-run variance exactly the way a too-short
    Newey-West lag does.

    Returns an (n_boot, n) integer array of positions into the original sample.
    """
    if n < 2:
        raise ValueError("need at least 2 observations")
    p = 1.0 / max(float(block_length), 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    start_new = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(start_new[:, t], fresh[:, t], cont)
    return idx


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    n_boot: int = 2000,
    block_length: Optional[float] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Model Confidence Set of Hansen, Lunde & Nason (Econometrica, 2011).

    Pairwise Diebold-Mariano is the wrong tool for ranking many models: with m
    models there are m(m-1)/2 comparisons, every one tested at the same nominal
    size, and the choice of which model is "the benchmark" decides the answer.
    The MCS instead returns a SET that contains the best model(s) with a given
    asymptotic confidence level, and needs no benchmark to be nominated.

    The procedure tests equal predictive ability across the surviving models,
    eliminates the worst when the null is rejected, and repeats. Following
    Clements & Preve (2021), the test statistic is the RANGE statistic of
    Hansen, Lunde & Nason (2003),

        T_R = max_{i,j} |d_bar_ij| / sqrt(var(d_bar_ij)),

    with elimination rule e_R = argmax_i sup_j t_ij, and the null distribution
    obtained by stationary bootstrap.

    Parameters
    ----------
    losses : DataFrame
        One column per model, one row per out-of-sample date; entries are the
        per-observation loss (QLIKE here). Rows with any missing loss are
        dropped so every model is judged on the identical sample.
    alpha : float
        1 - alpha is the confidence level. alpha=0.10 gives the 90% MCS.

    Returns
    -------
    DataFrame indexed by model with columns `loss` (mean loss),
    `mcs_pvalue`, `elimination_order` and `in_mcs`. The MCS p-value is the
    running maximum of the elimination p-values, so the surviving set at level
    alpha is exactly the set of models with mcs_pvalue >= alpha, and the last
    model standing has p-value 1.

    Note that p-values are monotone in ELIMINATION ORDER, not in mean loss.
    The elimination rule is studentized, so a model can be dropped ahead of one
    with a higher mean loss when its loss differentials have a small variance.
    A forecast combination is the usual case: it shares almost all its variance
    with its constituents, so being slightly worse than the best of them is
    measured very precisely and it is eliminated early.
    """
    frame = losses.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    names = list(frame.columns)
    if len(names) < 2:
        raise ValueError("need at least two models")
    values = frame.to_numpy(dtype=float)
    n_obs = len(values)
    if n_obs < 10:
        raise ValueError("need at least 10 common observations")

    if block_length is None:
        block_length = max(2.0, float(n_obs) ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    boot_idx = stationary_bootstrap_indices(n_obs, n_boot, block_length, rng)
    # (n_boot, m) bootstrap replications of each model's mean loss
    boot_means = values[boot_idx].mean(axis=1)

    alive = list(range(len(names)))
    pvalues: dict[str, float] = {}
    order: dict[str, int] = {}
    running = 0.0
    step = 0

    # Run the elimination to a single survivor rather than stopping at alpha:
    # the running-max p-values are what make membership at ANY level readable
    # off one table, which is how the MCS is normally reported.
    while len(alive) > 1:
        d_bar = values[:, alive].mean(axis=0)
        d_ij = d_bar[:, None] - d_bar[None, :]
        boot_d = boot_means[:, alive][:, :, None] - boot_means[:, alive][:, None, :]
        var_ij = ((boot_d - d_ij) ** 2).mean(axis=0)
        np.fill_diagonal(var_ij, np.inf)          # self-comparisons carry no information
        var_ij = np.where(var_ij <= 0, np.inf, var_ij)
        scale = np.sqrt(var_ij)

        t_ij = d_ij / scale
        stat = float(np.nanmax(np.abs(t_ij)))
        boot_stat = np.nanmax(np.abs((boot_d - d_ij) / scale), axis=(1, 2))
        p = float((boot_stat >= stat).mean())
        running = max(running, p)

        worst = int(np.nanargmax(np.nanmax(t_ij, axis=1)))
        step += 1
        pvalues[names[alive[worst]]] = running
        order[names[alive[worst]]] = step
        alive.pop(worst)

    pvalues[names[alive[0]]] = 1.0
    order[names[alive[0]]] = step + 1

    out = pd.DataFrame({
        "loss": frame.mean(),
        "mcs_pvalue": pd.Series(pvalues),
        "elimination_order": pd.Series(order),
    }).loc[names]
    out["in_mcs"] = out["mcs_pvalue"] >= alpha
    return out.sort_values("loss")


# ---------------------------------------------------------------------------
# Clements & Preve (2021) remedies
# ---------------------------------------------------------------------------
#
# Clements & Preve, "A Practical Guide to Harnessing the HAR Volatility Model"
# (JBF 2021; SSRN 3369484). Their point is that standard HAR pairs a
# right-skewed, heteroskedastic dependent variable with OLS, which is close to
# the worst available combination, and that simple remedies systematically beat
# both HAR and HARQ out of sample. On SPX, DJI and DAX they find the WLS and
# transformed-RV schemes are ALWAYS in the 90% MCS.
#
# One unit convention has to be stated because it is easy to get wrong. Clements
# & Preve model realized VARIANCE; this repository models annualized realized
# VOLATILITY throughout, and its QLIKE squares the forecast before scoring. Since
# log(RV) = 2*log(vol), the log-HAR slope coefficients are identical either way
# and only the intercept and the retransformation constant differ. The
# corrections below are therefore applied on the scale the model is estimated
# on, which is volatility.


def clements_preve_weights(
    rv: Optional[pd.Series] = None,
    rq: Optional[pd.Series] = None,
    scheme: str = "rv",
) -> pd.Series:
    """WLS weights from Clements & Preve section 2.3.3, verbatim.

    `rv`   w_t = 1 / RV_t.       Their fourth (nonparametric) scheme, motivated
                                 by sqrt(RQ) being hard to estimate in finite
                                 samples while being strongly correlated with RV.
    `rq`   w_t = 1 / sqrt(RQ_t). Their third scheme, which downweights days on
                                 which RV is imprecisely measured, in the same
                                 spirit as HARQ.

    Both are reported by the paper as always included in the 90% MCS. The two
    schemes they also consider and this does not are WLS_G (weights from a
    GARCH(1,1) fitted to OLS residuals) and WLS_RVhat (weights 1/fitted-RV from
    an OLS HAR).
    """
    if scheme == "rv":
        if rv is None:
            raise ValueError("scheme 'rv' needs rv")
        base = pd.to_numeric(rv, errors="coerce")
    elif scheme == "rq":
        if rq is None:
            raise ValueError("scheme 'rq' needs rq")
        base = np.sqrt(pd.to_numeric(rq, errors="coerce"))
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    base = base.where(base > 0)
    return 1.0 / base


@dataclass
class WLSHARRV(HARRV):
    """HAR-RV estimated by weighted least squares (Clements & Preve 2021).

    Identical regression to `HARRV`; only the estimator changes. Observations
    are weighted by `weights`, so days on which the error is likely to be large
    -- high-volatility days, where RV is both larger and less precisely measured
    -- pull the fit around less. Use `clements_preve_weights` to build them.
    """

    def fit_wls(self, rv: pd.Series, target: pd.Series, weights: pd.Series) -> "WLSHARRV":
        X = self._features(rv)
        frame = pd.concat([X, target.rename("y"),
                           pd.to_numeric(weights, errors="coerce").rename("w")], axis=1).dropna()
        frame = frame[frame["w"] > 0]
        if len(frame) < 60:
            raise ValueError("not enough overlapping observations to fit WLS-HAR")
        A = np.column_stack([np.ones(len(frame)), frame[["rv_d", "rv_w", "rv_m"]].to_numpy()])
        y = frame["y"].to_numpy()
        # WLS as OLS on sqrt(w)-scaled rows: minimising sum_t w_t * resid_t^2
        root = np.sqrt(frame["w"].to_numpy())
        self.coef_, *_ = np.linalg.lstsq(A * root[:, None], y * root, rcond=None)
        return self


@dataclass
class LogHARRV(HARRV):
    """HAR-RV fitted to log RV, retransformed with the lognormal correction.

    The logarithmic transformation is the lambda=0 case of the Box-Cox family
    in Clements & Preve section 2.4; it takes a series whose sample skewness
    exceeds 10 to one with skewness ~0.5, which is what makes OLS a reasonable
    estimator again.

    Retransforming needs care. exp() of the fitted mean of log RV estimates the
    MEDIAN, not the mean, and is biased low. Their equation (8), following
    Proietti and Lutkepohl (2013), adds half the residual variance inside the
    exponent:

        F_t = exp( b0 + b1*log RV_d + b2*log RV_w + b3*log RV_m + sigma_u^2 / 2 )

    `sigma2_` is that residual variance, estimated on the training window only.
    """

    sigma2_: Optional[float] = None

    @staticmethod
    def _log_features(rv: pd.Series) -> pd.DataFrame:
        lrv = np.log(pd.to_numeric(rv, errors="coerce").where(lambda s: s > 0))
        return pd.DataFrame({
            "rv_d": lrv,
            "rv_w": lrv.rolling(5, min_periods=5).mean(),
            "rv_m": lrv.rolling(22, min_periods=22).mean(),
        })

    def fit_log(self, rv: pd.Series, target: pd.Series) -> "LogHARRV":
        X = self._log_features(rv)
        y = np.log(pd.to_numeric(target, errors="coerce").where(lambda s: s > 0))
        frame = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(frame) < 60:
            raise ValueError("not enough overlapping observations to fit log-HAR")
        A = np.column_stack([np.ones(len(frame)), frame[["rv_d", "rv_w", "rv_m"]].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        resid = frame["y"].to_numpy() - A @ self.coef_
        # residual variance with a degrees-of-freedom correction for the 4 fitted
        # parameters; this is the sigma_u^2 of equation (8)
        dof = max(len(frame) - A.shape[1], 1)
        self.sigma2_ = float((resid ** 2).sum() / dof)
        return self

    def predict_log(self, rv: pd.Series, bias_correct: bool = True) -> pd.Series:
        if self.coef_ is None or self.sigma2_ is None:
            raise RuntimeError("fit_log() first")
        X = self._log_features(rv)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        linear = pd.Series(A @ self.coef_, index=rv.index)
        if bias_correct:
            linear = linear + 0.5 * self.sigma2_
        out = np.exp(linear)
        out[X.isna().any(axis=1)] = np.nan
        return out


def mean_combination(forecasts: Sequence[pd.Series]) -> pd.Series:
    """Equal-weighted mean of several forecasts, on their common dates.

    Not a Clements-Preve remedy -- their paper does not consider forecast
    combination -- but the standard first thing to try when several models are
    close, and the "forecast combination puzzle" is that equal weights are hard
    to beat with estimated ones (Stock and Watson 2004, Smith and Wallis 2009).
    Included so the MCS has a combination to accept or reject rather than the
    question going unasked.
    """
    if not len(forecasts):
        raise ValueError("need at least one forecast")
    frame = pd.concat([pd.to_numeric(f, errors="coerce") for f in forecasts], axis=1)
    return frame.mean(axis=1, skipna=False)


# ---------------------------------------------------------------------------
# Semivariance HAR and exogenous-regressor HAR
# ---------------------------------------------------------------------------
#
# UNITS, again, because this module models annualized realized VOLATILITY while
# the papers below are written in realized VARIANCE. The semivariance identity
# RS+ + RS- = RV holds in variance. Its volatility-unit analogue is
#
#     sqrt(252*RS+)^2 + sqrt(252*RS-)^2 = sqrt(252*RV)^2,
#
# so `semivol` below returns sqrt(252*RS) and the two regressors still add up to
# the daily RV regressor in squares. That is a reparameterisation of
# Patton-Sheppard, not a different model: the split between upside and downside
# variation, which is the whole content of SHAR, is preserved exactly.


def semivol(semivariance: pd.Series, trading_days: int = TRADING_DAYS) -> pd.Series:
    """Annualised volatility-unit form of a daily realized semivariance."""
    s = pd.to_numeric(semivariance, errors="coerce")
    return np.sqrt(s.clip(lower=0.0) * trading_days)


@dataclass
class SHARRV(HARRV):
    """Semivariance HAR of Patton & Sheppard (REStat 2015).

    The daily term of HAR is split by the SIGN of the intraday return that
    produced it:

        RV_{t+h} ~ b0 + b_pos * RS+_t + b_neg * RS-_t + b_w * RV_w + b_m * RV_m

    Their finding is that b_neg is large and positive while b_pos is small and
    often NEGATIVE: volatility that arrives on down moves predicts more future
    volatility, and volatility on up moves predicts less. HAR forces the two to
    share one coefficient, which averages the effect away. This is the strongest
    horizon-stable result in the forecasting literature that needs no data
    beyond the price path.
    """

    _COLS = ["rs_pos", "rs_neg", "rv_w", "rv_m"]

    @staticmethod
    def _shar_features(rv: pd.Series, rs_pos: pd.Series, rs_neg: pd.Series) -> pd.DataFrame:
        har = HARRV._features(rv)
        return pd.DataFrame({
            "rs_pos": semivol(rs_pos),
            "rs_neg": semivol(rs_neg),
            "rv_w": har["rv_w"],
            "rv_m": har["rv_m"],
        })

    def fit_shar(self, rv: pd.Series, rs_pos: pd.Series, rs_neg: pd.Series,
                 target: pd.Series) -> "SHARRV":
        X = self._shar_features(rv, rs_pos, rs_neg)
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        if len(frame) < 60:
            raise ValueError("not enough overlapping observations to fit SHAR")
        A = np.column_stack([np.ones(len(frame)), frame[self._COLS].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict_shar(self, rv: pd.Series, rs_pos: pd.Series,
                     rs_neg: pd.Series) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit_shar() first")
        X = self._shar_features(rv, rs_pos, rs_neg)
        A = np.column_stack([np.ones(len(X)), X[self._COLS].to_numpy()])
        out = pd.Series(A @ self.coef_, index=rv.index)
        out[X.isna().any(axis=1)] = np.nan
        return out


@dataclass
class HARX(HARRV):
    """HAR-RV with arbitrary exogenous regressors appended, fitted by OLS.

    One class covers every "HAR plus something" specification in this study:
    HAR-RV-IV is `HARX` with implied variance as the single extra column, the
    marginal-value table is `HARX` run once per feature, and the kitchen-sink
    model is `HARX` with all of them. Keeping them the same estimator means a
    difference between two rows of the results table is a difference in
    INFORMATION, not in fitting machinery.

    `exog` is a DataFrame aligned on the same index as `rv`. Column order is
    stored at fit time so predict cannot silently reorder them.
    """

    exog_cols_: Optional[list] = None

    def _design(self, rv: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
        har = self._features(rv)
        ex = exog.reindex(rv.index)
        if self.exog_cols_ is not None:
            ex = ex[self.exog_cols_]
        return pd.concat([har, ex], axis=1)

    def fit_x(self, rv: pd.Series, exog: pd.DataFrame, target: pd.Series) -> "HARX":
        self.exog_cols_ = list(exog.columns)
        X = self._design(rv, exog)
        frame = pd.concat([X, target.rename("y")], axis=1).dropna()
        if len(frame) < 60:
            raise ValueError("not enough overlapping observations to fit HAR-X")
        A = np.column_stack([np.ones(len(frame)), frame[X.columns].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)
        return self

    def predict_x(self, rv: pd.Series, exog: pd.DataFrame) -> pd.Series:
        if self.coef_ is None or self.exog_cols_ is None:
            raise RuntimeError("fit_x() first")
        X = self._design(rv, exog)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        out = pd.Series(A @ self.coef_, index=rv.index)
        out[X.isna().any(axis=1)] = np.nan
        return out


# ---------------------------------------------------------------------------
# Rolling Model Confidence Set
# ---------------------------------------------------------------------------


def rolling_model_confidence_set(
    losses: pd.DataFrame,
    window: int = 504,
    step: int = 21,
    alpha: float = 0.10,
    n_boot: int = 500,
    seed: int = 0,
    min_obs: int = 200,
) -> pd.DataFrame:
    """MCS membership on a rolling window, so regime dependence is visible.

    One MCS over the whole sample answers "which model was best on average since
    2018", which averages a pandemic, a rate-hiking cycle and two quiet years
    into a single verdict. Running the same test on two-year windows shows
    whether a model's membership is stable or whether it is carried by one
    regime. `window` is in observations, so 504 is roughly two trading years.

    Returns a frame indexed by window end date, one boolean column per model.
    Windows with fewer than `min_obs` common observations are skipped.
    """
    frame = losses.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    rows: dict[pd.Timestamp, pd.Series] = {}
    for end in range(window, len(frame) + 1, step):
        block = frame.iloc[end - window:end]
        if len(block) < min_obs:
            continue
        res = model_confidence_set(block, alpha=alpha, n_boot=n_boot, seed=seed)
        rows[block.index[-1]] = res["in_mcs"].reindex(frame.columns)
    if not rows:
        return pd.DataFrame(columns=frame.columns)
    out = pd.DataFrame(rows).T
    out.index.name = "window_end"
    return out
