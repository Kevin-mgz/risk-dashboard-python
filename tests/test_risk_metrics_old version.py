# tests/test_risk_metrics.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.risk_metrics import (
    compute_returns,
    rolling_volatility,
    historical_var,
    stress_pnl,
    var_to_pnl,
)


def make_price_series(values, start="2024-01-01"):
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="price").astype(float)


def test_compute_returns_log_basic():
    prices = make_price_series([100, 110, 121])  # +10% then +10%
    rets = compute_returns(prices, method="log")

    assert isinstance(rets, pd.Series)
    assert len(rets) == 2  # first NaN dropped

    expected = np.log(pd.Series([110 / 100, 121 / 110])).to_numpy()
    np.testing.assert_allclose(rets.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_compute_returns_raises_on_non_positive_prices():
    prices = make_price_series([100, 0, 105])
    with pytest.raises(ValueError):
        compute_returns(prices, method="log")


def test_rolling_volatility_length_and_non_negative():
    prices = make_price_series([100, 101, 99, 102, 103, 104, 100, 98, 99, 101, 102])
    rets = compute_returns(prices, method="simple")

    window = 5
    vol = rolling_volatility(rets, window=window, annualize=False)

    # rolling std starts producing values after (window) observations in returns
    assert len(vol) == len(rets) - (window - 1)
    assert (vol >= 0).all()


def test_historical_var_95_known_quantile():
    # Construct returns so 5th percentile is exactly -0.10
    # 20 observations: 1 very bad (-10%), 19 small (+1%)
    rets = pd.Series([-0.10] + [0.01] * 19)

    var_95 = historical_var(rets, level=0.95)

    # Our convention: VaR returned as positive loss magnitude
    assert var_95 >= 0
    assert var_95 == pytest.approx(0.10, rel=1e-12, abs=1e-12)


def test_pnl_conversions():
    notional = 10_000

    assert var_to_pnl(notional, 0.02) == pytest.approx(200.0)
    assert stress_pnl(notional, shock=-0.10) == pytest.approx(1000.0)

    # Stress with +10% should not be a loss in our convention => 0
    assert stress_pnl(notional, shock=0.10) == pytest.approx(0.0)
