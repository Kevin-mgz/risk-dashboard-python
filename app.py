"""
Risk Dashboard — app.py
=======================
A single-page Streamlit dashboard for financial risk monitoring.

Usage (from project root):
    streamlit run app.py

Assumes:
    - Price CSVs already downloaded into data/raw/ via src/data_loader.py
    - src/ is importable (we add project root to sys.path below)

Author: Murengezi Kevin — Risk Dashboard Portfolio Project
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make src.* importable when running from project root ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.risk_metrics import (
    compute_returns,
    historical_var,
    rolling_volatility,
    stress_pnl,
    var_to_pnl,
)

# Optional imports — degrade gracefully if utils is missing or incomplete
try:
    from src.utils import fmt_chf, fmt_pct, get_assets, max_drawdown

    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False


# ── Constants & fallbacks ─────────────────────────────────────────────────────

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

_FALLBACK_ASSETS = {
    "BTC_USD": {
        "ticker": "BTC-USD",
        "label": "Bitcoin (BTC/USD)",
        "kind": "crypto",
        "periods_per_year": 365,
        "filename": "btc_usd.csv",
    },
    "USDCHF": {
        "ticker": "USDCHF=X",
        "label": "USD/CHF",
        "kind": "fx",
        "periods_per_year": 252,
        "filename": "usdchf.csv",
    },
    "EURCHF": {
        "ticker": "EURCHF=X",
        "label": "EUR/CHF",
        "kind": "fx",
        "periods_per_year": 252,
        "filename": "eurchf.csv",
    },
    "SMI": {
        "ticker": "^SSMI",
        "label": "Swiss Market Index (SMI)",
        "kind": "index",
        "periods_per_year": 252,
        "filename": "smi.csv",
    },
    "SP500": {
        "ticker": "^GSPC",
        "label": "S&P 500",
        "kind": "index",
        "periods_per_year": 252,
        "filename": "sp500.csv",
    },
}

TIME_WINDOWS = {
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "20 Years": 20,
    "All Data": None,
}


def _build_asset_registry() -> dict:
    if _HAS_UTILS:
        raw = get_assets()
        return {
            key: {
                "ticker": spec.ticker,
                "label": spec.label,
                "kind": spec.kind,
                "periods_per_year": spec.periods_per_year,
                "filename": f"{key.lower()}.csv",
            }
            for key, spec in raw.items()
        }
    return _FALLBACK_ASSETS


ASSET_REGISTRY = _build_asset_registry()


# ── Formatting helpers ────────────────────────────────────────────────────────


def _fmt_pct(x: float | None, decimals: int = 2) -> str:
    if _HAS_UTILS:
        return fmt_pct(x, decimals)
    if x is None:
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_chf(x: float | None, decimals: int = 0) -> str:
    if _HAS_UTILS:
        return fmt_chf(x, decimals)
    if x is None:
        return "—"
    return f"CHF {f'{x:,.{decimals}f}'.replace(',', chr(39))}"


def _compute_max_drawdown(prices: pd.Series) -> float:
    running_max = prices.cummax()
    return float((prices / running_max - 1.0).min())


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices_from_csv(filepath: str) -> pd.Series:
    """
    Load a yfinance-style CSV and return a clean closing price Series.
    Prioritises 'Adj Close' over 'Close'.
    Cached for 1 hour — time window filtering happens after this call.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Flatten multi-level headers (yfinance sometimes produces these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(c).strip() for c in df.columns]

    # Find the price column (handles "Adj Close BTC-USD" style names too)
    def _find_col(*candidates: str) -> str | None:
        for cand in candidates:
            for col in df.columns:
                if col.lower().startswith(cand):
                    return col
        return None

    price_col = _find_col("adj close", "close")
    if price_col is None:
        raise ValueError(
            f"Could not find Close/Adj Close in {path.name}. "
            f"Columns: {list(df.columns)}"
        )

    prices = (
        df[price_col]
        .rename("price")
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_index()
    )

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    return prices


def _apply_time_window(prices: pd.Series, years: int | None) -> pd.Series:
    """Slice prices to the selected time window. Called after cache."""
    if years is None:
        return prices
    cutoff = prices.index[-1] - pd.DateOffset(years=years)
    return prices[prices.index >= cutoff]


# ── Chart builders ────────────────────────────────────────────────────────────


def _price_chart(prices: pd.Series, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name=label,
            line=dict(color="#2563EB", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:,.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{label} — Price History",
        xaxis_title=None,
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=36, b=0),
        height=320,
        template="plotly_white",
    )
    return fig


def _returns_chart(returns: pd.Series, label: str) -> go.Figure:
    colors = ["#16a34a" if r >= 0 else "#dc2626" for r in returns.values]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=returns.index,
            y=returns.values * 100,
            marker_color=colors,
            name="Daily Return",
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{label} — Daily Returns (%)",
        xaxis_title=None,
        yaxis_title="Return (%)",
        hovermode="x unified",
        bargap=0,
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        template="plotly_white",
    )
    return fig


def _vol_chart(vol: pd.Series, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol.index,
            y=vol.values * 100,
            mode="lines",
            name="Rolling Vol",
            fill="tozeroy",
            line=dict(color="#7c3aed", width=1.5),
            fillcolor="rgba(124,58,237,0.10)",
            hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{label} — Rolling Annualised Volatility (%)",
        xaxis_title=None,
        yaxis_title="Annualised Vol (%)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        template="plotly_white",
    )
    return fig


def _distribution_chart(
    returns: pd.Series, var_r: float, var_level: float, label: str
) -> go.Figure:
    """
    Return distribution histogram with VaR threshold.
    Shows geometrically what VaR means — useful talking point in interviews.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.values * 100,
            nbinsx=80,
            marker_color="#3b82f6",
            opacity=0.75,
            name="Return distribution",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=-var_r * 100,
        line_color="#dc2626",
        line_dash="dash",
        line_width=2,
        annotation_text=f"{int(var_level*100)}% VaR = {var_r*100:.2f}%",
        annotation_position="top right",
        annotation_font_color="#dc2626",
    )
    fig.update_layout(
        title=f"{label} — Return Distribution & VaR",
        xaxis_title="Daily Return (%)",
        yaxis_title="Count",
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        template="plotly_white",
    )
    return fig


# ── KPI rendering ─────────────────────────────────────────────────────────────


def _render_kpis(
    prices: pd.Series,
    returns: pd.Series,
    vol_series: pd.Series,
    var_r: float,
    var_level: float,
    notional: float,
) -> None:
    last_price = float(prices.iloc[-1])
    last_ret = float(returns.iloc[-1])
    last_vol = float(vol_series.iloc[-1]) if not vol_series.empty else None
    var_pnl = var_to_pnl(notional, var_r)
    stress_loss = stress_pnl(notional, shock=-0.10)

    try:
        mdd = max_drawdown(prices) if _HAS_UTILS else _compute_max_drawdown(prices)
    except Exception:
        mdd = None

    row1 = st.columns(4)
    row1[0].metric("Last Price", f"{last_price:,.4f}")
    row1[1].metric(
        "Last Daily Return",
        _fmt_pct(last_ret),
        delta=_fmt_pct(last_ret),
        delta_color="normal",
    )
    row1[2].metric("Rolling Vol (ann.)", _fmt_pct(last_vol))
    row1[3].metric("Max Drawdown", _fmt_pct(mdd))

    row2 = st.columns(3)
    row2[0].metric(
        f"{int(var_level*100)}% VaR (1-day, historical)",
        _fmt_pct(var_r),
        help="Worst expected 1-day loss at given confidence, historical simulation.",
    )
    row2[1].metric(
        f"VaR in CHF (notional {_fmt_chf(notional)})",
        _fmt_chf(var_pnl),
        help="Monetary loss = notional × VaR%",
    )
    row2[2].metric(
        "Stress −10% (CHF)",
        _fmt_chf(stress_loss),
        help="Hypothetical loss if the asset drops 10% overnight.",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _build_sidebar() -> dict:
    st.sidebar.title("⚙️ Parameters")

    # Asset
    asset_options = {v["label"]: k for k, v in ASSET_REGISTRY.items()}
    selected_label = st.sidebar.selectbox("Asset", list(asset_options.keys()))
    asset_key = asset_options[selected_label]
    asset = ASSET_REGISTRY[asset_key]

    # Time window
    selected_window = st.sidebar.selectbox(
        "Time Window",
        list(TIME_WINDOWS.keys()),
        index=2,  # default: 5 Years
        help=(
            "Filter the analysis period. VaR and vol estimates vary significantly "
            "across windows — a key concept in risk model stability."
        ),
    )

    st.sidebar.divider()

    # Return method
    return_method = st.sidebar.selectbox(
        "Return Method",
        ["log", "simple"],
        help=(
            "Log returns are time-additive and the industry default for risk. "
            "Simple returns are more intuitive economically."
        ),
    )

    # Rolling vol window — number_input prevents window=1 which would crash rolling_volatility
    vol_window = st.sidebar.number_input(
        "Rolling Vol Window (days)",
        min_value=5,
        max_value=252,
        value=20,
        step=1,
        help="Number of trading days for the rolling standard deviation.",
    )

    # VaR level
    var_level = st.sidebar.slider(
        "VaR Confidence Level",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        help="0.95 = worst loss not exceeded 95% of the time.",
    )

    # Notional
    notional = st.sidebar.number_input(
        "Notional (CHF)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1_000.0,
        format="%.0f",
        help="Portfolio size in CHF for monetary risk conversion.",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "💡 **Tip:** Run `python src/data_loader.py` to refresh market data."
    )

    return {
        "asset_key": asset_key,
        "asset": asset,
        "years": TIME_WINDOWS[selected_window],
        "return_method": return_method,
        "vol_window": int(vol_window),
        "var_level": float(var_level),
        "notional": float(notional),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Risk Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Risk Dashboard")
    st.caption(
        "Historical market risk analytics — VaR, rolling volatility, stress testing."
    )

    params = _build_sidebar()
    asset = params["asset"]

    # ── Load & filter data ────────────────────────────────────────────────────
    csv_path = RAW_DATA_DIR / asset["filename"]

    with st.spinner(f"Loading {asset['label']}…"):
        try:
            prices_full = load_prices_from_csv(str(csv_path))
        except FileNotFoundError:
            st.error(
                f"**Data file not found:** `{csv_path}`\n\n"
                "Run the data loader first:\n```\npython src/data_loader.py\n```"
            )
            st.stop()
        except ValueError as e:
            st.error(f"**Error reading data:** {e}")
            st.stop()

    # Apply time window AFTER cache (no re-download triggered)
    prices = _apply_time_window(prices_full, params["years"])

    if len(prices) < 50:
        st.warning(
            f"Only {len(prices)} data points in this window — "
            "try a wider time range or re-run the data loader."
        )
        st.stop()

    # ── Compute metrics ───────────────────────────────────────────────────────
    try:
        returns = compute_returns(prices, method=params["return_method"])
    except (ValueError, TypeError) as e:
        st.error(f"**Error computing returns:** {e}")
        st.stop()

    vol_series = rolling_volatility(
        returns,
        window=params["vol_window"],
        annualize=True,
        periods_per_year=asset["periods_per_year"],
    )

    var_r = historical_var(returns, level=params["var_level"])

    # ── Header info bar ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Asset:** {asset['label']}")
    c2.markdown(
        f"**Period:** {prices.index[0].date()} → {prices.index[-1].date()} "
        f"({len(prices):,} days)"
    )
    c3.markdown(f"**Periods/year:** {asset['periods_per_year']}")

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    _render_kpis(
        prices, returns, vol_series, var_r, params["var_level"], params["notional"]
    )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    label = asset["label"]

    st.plotly_chart(_price_chart(prices, label), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(_returns_chart(returns, label), use_container_width=True)
    with col_right:
        st.plotly_chart(_vol_chart(vol_series, label), use_container_width=True)

    st.plotly_chart(
        _distribution_chart(returns, var_r, params["var_level"], label),
        use_container_width=True,
    )

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("📄 Raw price data (last 30 rows)"):
        st.dataframe(
            prices.tail(30).rename("Close Price").to_frame(),
            use_container_width=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Disclaimer:** Educational purposes only — not financial advice. "
        "VaR has well-known limitations (fat tails, non-stationarity, no tail expectation). "
        "Production risk systems complement VaR with Expected Shortfall (CVaR) and scenario analysis."
        # TODO: multi-asset view — correlation matrix, portfolio VaR aggregation
    )


if __name__ == "__main__":
    main()
