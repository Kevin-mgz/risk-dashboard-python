# src/risk_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


ReturnMethod = Literal["simple", "log"]


def compute_returns(prices: pd.Series, method: ReturnMethod = "log") -> pd.Series:
    """
    Compute daily returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.
    method : {"simple", "log"}
        - "simple": r_t = P_t / P_{t-1} - 1
        - "log":    r_t = ln(P_t / P_{t-1})

    Returns
    -------
    pd.Series
        Return series (same index, first value NaN dropped).
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if prices.empty:
        raise ValueError("prices is empty.")
    if (prices <= 0).any():
        # Log returns require strictly positive prices; also a good sanity check in general.
        raise ValueError("prices contains non-positive values.")

    if method == "simple":
        rets = prices.pct_change()
    elif method == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError("method must be 'simple' or 'log'.")

    return rets.dropna()


def rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:

    if window <= 1:
        raise ValueError("window must be > 1.")
    if returns.empty:
        raise ValueError("returns is empty.")

    vol = returns.rolling(window=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.dropna()


def historical_var(returns: pd.Series, level: float = 0.95) -> float:

    if not (0 < level < 1):
        raise ValueError("level must be between 0 and 1.")
    if returns.empty:
        raise ValueError("returns is empty.")

    alpha = 1.0 - level
    q = returns.quantile(alpha)  # typically negative
    return float(max(0.0, -q))


def parametric_var_gaussian(returns: pd.Series, level: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR.

    VaR = -(mu + z_alpha * sigma)  where z_alpha is quantile of Normal(alpha).
    Returns VaR as positive loss magnitude.

    Notes:
    - Good as a comparison/bonus, but historical VaR is the simplest baseline.
    """
    if not (0 < level < 1):
        raise ValueError("level must be between 0 and 1.")
    if returns.empty:
        raise ValueError("returns is empty.")

    # We avoid SciPy dependency by using an approximation via numpy if possible:
    # But numpy doesn't have inverse CDF. So: keep this function optional unless you add scipy.
    raise NotImplementedError(
        "Gaussian VaR needs scipy.stats.norm.ppf. "
        "Either install scipy or remove this function."
    )


def var_to_pnl(notional: float, var_return: float) -> float:
    """
    Convert a VaR expressed in return space to a monetary loss.

    Parameters
    ----------
    notional : float
        Position size (e.g., 10_000 CHF).
    var_return : float
        VaR in return space (positive number like 0.03 = 3%).

    Returns
    -------
    float
        P&L loss (positive number).
    """
    if notional < 0:
        raise ValueError("notional must be >= 0.")
    if var_return < 0:
        raise ValueError("var_return must be >= 0.")
    return float(notional * var_return)


def stress_pnl(notional: float, shock: float = -0.10) -> float:
    """
    Stress scenario P&L for a given shock (e.g., -10%).

    Convention:
    - shock is a return (decimal), negative means down move.
    - output is a POSITIVE loss magnitude.

    Example: notional=10_000, shock=-0.10 => loss=1_000
    """
    if notional < 0:
        raise ValueError("notional must be >= 0.")
    loss = -notional * shock  # if shock is negative => positive loss
    return float(max(0.0, loss))


@dataclass(frozen=True)
class RiskSummary:
    """
    Convenience container for the dashboard KPIs (optional).
    """

    last_price: float
    last_return: float
    rolling_vol: Optional[float]
    var_95: float
    var_95_pnl: float
    stress_pnl_10: float


def build_risk_summary(
    prices: pd.Series,
    notional: float = 10_000.0,
    return_method: ReturnMethod = "log",
    vol_window: int = 20,
    var_level: float = 0.95,
    periods_per_year: int = 252,
) -> RiskSummary:
    """
    Compute a KPI summary for a single asset price series.
    """
    rets = compute_returns(prices, method=return_method)

    vol_series = rolling_volatility(
        rets, window=vol_window, annualize=True, periods_per_year=periods_per_year
    )
    vol_last = float(vol_series.iloc[-1]) if not vol_series.empty else None

    var_r = historical_var(rets, level=var_level)  # positive magnitude
    var_pnl = var_to_pnl(notional, var_r)
    stress = stress_pnl(notional, shock=-0.10)

    return RiskSummary(
        last_price=float(prices.iloc[-1]),
        last_return=float(rets.iloc[-1]),
        rolling_vol=vol_last,
        var_95=float(var_r),
        var_95_pnl=float(var_pnl),
        stress_pnl_10=float(stress),
    )
