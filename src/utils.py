# src/utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------- Paths / project constants ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------- Data sanity checks ----------


def validate_price_series(
    prices: pd.Series,
    name: str = "price",
    min_points: int = 50,
) -> pd.Series:
    """
    Basic validation + cleaning for price series.

    Ensures:
    - DatetimeIndex
    - Sorted index
    - Numeric dtype
    - No non-positive prices
    - Enough points

    Returns a cleaned copy (sorted, float, NaNs dropped).
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if prices.empty:
        raise ValueError("prices is empty.")

    s = prices.copy()
    s.name = name

    # Index checks
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a pandas DatetimeIndex.")
    s = s.sort_index()

    # Numeric
    s = pd.to_numeric(s, errors="coerce").astype("float64")

    # Drop NaNs
    s = s.dropna()

    if len(s) < min_points:
        raise ValueError(
            f"Not enough data points after cleaning: {len(s)} < {min_points}"
        )

    if (s <= 0).any():
        raise ValueError("prices contains non-positive values (<= 0).")

    return s


def max_drawdown(prices: pd.Series) -> float:
    """
    Compute maximum drawdown from a price series.
    Returns a negative number (e.g., -0.35 means -35%).
    """
    s = validate_price_series(prices, min_points=2)
    running_max = s.cummax()
    drawdowns = s / running_max - 1.0
    return float(drawdowns.min())


# ---------- Formatting helpers for UI ----------


def fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    """Format decimal returns/vol as percent string."""
    if x is None:
        return "—"
    return f"{x * 100:.{decimals}f}%"


def fmt_chf(x: Optional[float], decimals: int = 0) -> str:
    """Format as CHF amount (simple)."""
    if x is None:
        return "—"
    # thousands separator: 12'345 style (Swiss)
    s = f"{x:,.{decimals}f}"
    s = s.replace(",", "'")
    return f"CHF {s}"


def fmt_float(x: Optional[float], decimals: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f}"


# ---------- Asset definitions (single source of truth) ----------


@dataclass(frozen=True)
class AssetSpec:
    key: str
    ticker: str
    label: str
    kind: str  # "crypto" | "fx" | "index"
    periods_per_year: int = 252


def get_assets() -> dict[str, AssetSpec]:
    """
    Central asset registry used by the app + loader.
    Keeps tickers in one place.
    """
    return {
        "BTC_USD": AssetSpec(
            key="BTC_USD",
            ticker="BTC-USD",
            label="Bitcoin (BTC/USD)",
            kind="crypto",
            periods_per_year=365,  # crypto trades 7/7 (optional choice)
        ),
        "USDCHF": AssetSpec(
            key="USDCHF",
            ticker="USDCHF=X",
            label="USD/CHF",
            kind="fx",
            periods_per_year=252,
        ),
        "EURCHF": AssetSpec(
            key="EURCHF",
            ticker="EURCHF=X",
            label="EUR/CHF",
            kind="fx",
            periods_per_year=252,
        ),
        "SMI": AssetSpec(
            key="SMI",
            ticker="^SSMI",
            label="Swiss Market Index (SMI)",
            kind="index",
            periods_per_year=252,
        ),
        # Fallback global index (useful if SMI ticker has issues)
        "SP500": AssetSpec(
            key="SP500",
            ticker="^GSPC",
            label="S&P 500",
            kind="index",
            periods_per_year=252,
        ),
    }
