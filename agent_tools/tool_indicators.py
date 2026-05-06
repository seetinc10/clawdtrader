#!/usr/bin/env python3
"""
MCP Technical Indicators Service
Provides RSI, VWAP, Bollinger Bands, and Momentum Score tools.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("TechnicalIndicators")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_market(symbol: str) -> str:
    """Return the active market type for the current repo."""
    return "us"


def _workspace_data_path(filename: str, symbol: Optional[str] = None) -> Path:
    """Resolve the JSONL data path for the U.S. dataset."""
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "data" / filename


def _extract_ohlcv(bar: dict, market: str, is_intraday: bool = False) -> Dict[str, float]:
    """Normalize field names into a standard OHLCV dict.

    U.S. intraday uses '1. open' / '4. close'.
    """
    if is_intraday:
        # Intraday (60min) format: "1. open", "4. close"
        return {
            "open": float(bar.get("1. open", 0)),
            "high": float(bar.get("2. high", 0)),
            "low": float(bar.get("3. low", 0)),
            "close": float(bar.get("4. close", 0)),
            "volume": float(bar.get("5. volume", 0)),
        }
    else:
        # Daily format: "1. buy price", "4. sell price"
        return {
            "open": float(bar.get("1. buy price", 0)),
            "high": float(bar.get("2. high", 0)),
            "low": float(bar.get("3. low", 0)),
            "close": float(bar.get("4. sell price", 0)),
            "volume": float(bar.get("5. volume", 0)),
        }


def _load_historical_bars(symbol: str, anchor_date: str, num_periods: int) -> List[Dict[str, float]]:
    """Load *num_periods* OHLCV bars ending at/before *anchor_date*.

    For US stocks the merged.jsonl contains hourly data keyed under
    'Time Series (60min)'.  We aggregate to daily bars (one bar per
    calendar day) so indicators work on a daily timeframe.

    Returns normalised OHLCV dicts sorted oldest-first.
    """
    market = _detect_market(symbol)
    data_path = _workspace_data_path("merged.jsonl", symbol)

    if not data_path.exists():
        return []

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            meta = doc.get("Meta Data", {})
            if meta.get("2. Symbol") != symbol:
                continue

            # Determine which series key to use
            if market == "us":
                series = doc.get("Time Series (60min)", {})
                if not series:
                    continue

                # Aggregate hourly bars into daily bars
                daily_agg: Dict[str, Dict[str, float]] = {}
                for ts_key, bar_data in series.items():
                    date_part = ts_key.split(" ")[0] if " " in ts_key else ts_key
                    ohlcv = _extract_ohlcv(bar_data, market, is_intraday=True)
                    if date_part not in daily_agg:
                        daily_agg[date_part] = {
                            "open": ohlcv["open"],
                            "high": ohlcv["high"],
                            "low": ohlcv["low"],
                            "close": ohlcv["close"],
                            "volume": ohlcv["volume"],
                            "_first_ts": ts_key,
                            "_last_ts": ts_key,
                        }
                    else:
                        agg = daily_agg[date_part]
                        # Track earliest and latest timestamps for open/close
                        if ts_key < agg["_first_ts"]:
                            agg["_first_ts"] = ts_key
                            agg["open"] = ohlcv["open"]
                        if ts_key > agg["_last_ts"]:
                            agg["_last_ts"] = ts_key
                            agg["close"] = ohlcv["close"]
                        agg["high"] = max(agg["high"], ohlcv["high"])
                        agg["low"] = min(agg["low"], ohlcv["low"])
                        agg["volume"] += ohlcv["volume"]

                # Filter dates <= anchor_date, sort, slice
                eligible = sorted(
                    [(d, {k: v for k, v in vals.items() if not k.startswith("_")})
                     for d, vals in daily_agg.items() if d <= anchor_date],
                    key=lambda x: x[0],
                )
                sliced = eligible[-num_periods:]
                return [bar for _, bar in sliced]
            else:
                # Crypto / CN: daily data
                series = doc.get("Time Series (Daily)", {})
                if not series:
                    continue

                eligible = sorted(
                    [(d, _extract_ohlcv(bar, market, is_intraday=False))
                     for d, bar in series.items() if d <= anchor_date],
                    key=lambda x: x[0],
                )
                sliced = eligible[-num_periods:]
                return [bar for _, bar in sliced]

    return []


def _load_intraday_bars_for_day(symbol: str, date_part: str) -> List[Dict[str, float]]:
    """Load all hourly bars for a single calendar day (US stocks only)."""
    data_path = _workspace_data_path("merged.jsonl", symbol)
    if not data_path.exists():
        return []

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            meta = doc.get("Meta Data", {})
            if meta.get("2. Symbol") != symbol:
                continue

            series = doc.get("Time Series (60min)", {})
            bars = []
            for ts_key, bar_data in series.items():
                if ts_key.startswith(date_part):
                    ohlcv = _extract_ohlcv(bar_data, "us", is_intraday=True)
                    ohlcv["_ts"] = ts_key
                    bars.append(ohlcv)
            # Sort oldest-first
            bars.sort(key=lambda b: b["_ts"])
            for b in bars:
                del b["_ts"]
            return bars

    return []


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

def _compute_rsi(symbol: str, date: str) -> Dict[str, Any]:
    """Core RSI computation."""
    bars = _load_historical_bars(symbol, date, 15)  # need 15 closes for 14-period RSI
    if len(bars) < 15:
        return {"error": f"Not enough data for RSI. Need 15 bars, got {len(bars)}.", "symbol": symbol, "date": date}

    closes = [b["close"] for b in bars]
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)

    avg_gain = sum(gains) / 14.0
    avg_loss = sum(losses) / 14.0

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = round(rsi, 2)

    if rsi <= 30:
        signal = "OVERSOLD"
    elif rsi >= 70:
        signal = "OVERBOUGHT"
    else:
        signal = "NEUTRAL"

    return {"symbol": symbol, "date": date, "value": rsi, "signal": signal}


@mcp.tool()
def get_rsi(symbol: str, date: str) -> Dict[str, Any]:
    """Calculate 14-period RSI (Relative Strength Index) for a symbol on a given date.

    Args:
        symbol: Stock/crypto symbol (e.g. 'AAPL', 'BTC-USDT', '600028.SH').
        date: Date in YYYY-MM-DD format.

    Returns:
        Dictionary with RSI value and signal (OVERSOLD / OVERBOUGHT / NEUTRAL).
    """
    return _compute_rsi(symbol, date)


def _compute_vwap(symbol: str, date: str) -> Dict[str, Any]:
    """Core VWAP computation."""
    market = _detect_market(symbol)

    if market == "us":
        # True intraday VWAP for US stocks
        bars = _load_intraday_bars_for_day(symbol, date)
        if not bars:
            return {"error": f"No intraday data found for {symbol} on {date}.", "symbol": symbol, "date": date}
    else:
        # Daily approximation for crypto/CN
        bars = _load_historical_bars(symbol, date, 20)
        if len(bars) < 2:
            return {"error": f"Not enough data for VWAP. Need at least 2 bars, got {len(bars)}.", "symbol": symbol, "date": date}

    # Calculate VWAP: sum(typical_price * volume) / sum(volume)
    cumulative_tpv = 0.0
    cumulative_vol = 0.0
    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        vol = bar["volume"]
        cumulative_tpv += typical_price * vol
        cumulative_vol += vol

    if cumulative_vol == 0:
        return {"error": "Total volume is zero, cannot compute VWAP.", "symbol": symbol, "date": date}

    vwap = cumulative_tpv / cumulative_vol
    current_close = bars[-1]["close"]

    price_vs_vwap_pct = ((current_close - vwap) / vwap) * 100.0 if vwap != 0 else 0.0

    signal = "BULLISH" if current_close > vwap else "BEARISH"

    return {
        "symbol": symbol,
        "date": date,
        "vwap": round(vwap, 4),
        "current_close": round(current_close, 4),
        "price_vs_vwap_pct": round(price_vs_vwap_pct, 4),
        "signal": signal,
    }


@mcp.tool()
def get_vwap(symbol: str, date: str) -> Dict[str, Any]:
    """Calculate VWAP (Volume Weighted Average Price) for a symbol on a given date.

    For US stocks: uses intraday (hourly) bars for the given day.
    For crypto/CN stocks: uses 20-period daily approximation.

    Args:
        symbol: Stock/crypto symbol.
        date: Date in YYYY-MM-DD format.

    Returns:
        Dictionary with VWAP, current close, percentage vs VWAP, and signal.
    """
    return _compute_vwap(symbol, date)


def _compute_bollinger_bands(symbol: str, date: str) -> Dict[str, Any]:
    """Core Bollinger Bands computation."""
    bars = _load_historical_bars(symbol, date, 20)
    if len(bars) < 20:
        return {"error": f"Not enough data for Bollinger Bands. Need 20 bars, got {len(bars)}.", "symbol": symbol, "date": date}

    closes = [b["close"] for b in bars]

    # 20-period SMA
    sma = sum(closes) / 20.0

    # Standard deviation
    variance = sum((c - sma) ** 2 for c in closes) / 20.0
    std_dev = math.sqrt(variance)

    upper_band = sma + 2 * std_dev
    lower_band = sma - 2 * std_dev

    bandwidth_pct = ((upper_band - lower_band) / sma) * 100.0 if sma != 0 else 0.0

    current_close = closes[-1]
    band_range = upper_band - lower_band
    percent_b = ((current_close - lower_band) / band_range) * 100.0 if band_range != 0 else 50.0

    # Signal based on %B
    if percent_b <= 0:
        signal = "OVERSOLD"
    elif percent_b >= 100:
        signal = "OVERBOUGHT"
    elif percent_b <= 20:
        signal = "NEAR_LOWER_BAND"
    elif percent_b >= 80:
        signal = "NEAR_UPPER_BAND"
    else:
        signal = "NEUTRAL"

    return {
        "symbol": symbol,
        "date": date,
        "upper_band": round(upper_band, 4),
        "middle_band": round(sma, 4),
        "lower_band": round(lower_band, 4),
        "bandwidth_pct": round(bandwidth_pct, 4),
        "percent_b": round(percent_b, 4),
        "signal": signal,
    }


@mcp.tool()
def get_bollinger_bands(symbol: str, date: str) -> Dict[str, Any]:
    """Calculate 20-period Bollinger Bands for a symbol on a given date.

    Args:
        symbol: Stock/crypto symbol.
        date: Date in YYYY-MM-DD format.

    Returns:
        Dictionary with upper/middle/lower bands, bandwidth %, %B, and signal.
    """
    return _compute_bollinger_bands(symbol, date)


def _compute_momentum_score(symbol: str, date: str) -> Dict[str, Any]:
    """Core momentum score computation."""
    bars = _load_historical_bars(symbol, date, 21)
    if len(bars) < 21:
        return {"error": f"Not enough data for momentum score. Need 21 bars, got {len(bars)}.", "symbol": symbol, "date": date}

    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    # --- RSI component (use last 15 closes for 14-period RSI) ---
    rsi_closes = closes[-15:]
    gains = []
    losses = []
    for i in range(1, len(rsi_closes)):
        diff = rsi_closes[i] - rsi_closes[i - 1]
        gains.append(diff if diff > 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)

    avg_gain = sum(gains) / 14.0
    avg_loss = sum(losses) / 14.0
    if avg_loss == 0:
        rsi_value = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + rs))

    # RSI score: map 0-100 RSI directly to 0-100 score
    rsi_score = max(0.0, min(100.0, rsi_value))

    # --- Bollinger %B component (last 20 closes) ---
    bb_closes = closes[-20:]
    sma = sum(bb_closes) / 20.0
    variance = sum((c - sma) ** 2 for c in bb_closes) / 20.0
    std_dev = math.sqrt(variance)
    upper = sma + 2 * std_dev
    lower = sma - 2 * std_dev
    band_range = upper - lower
    if band_range != 0:
        percent_b = ((closes[-1] - lower) / band_range) * 100.0
    else:
        percent_b = 50.0
    bb_score = max(0.0, min(100.0, percent_b))

    # --- Volume trend component ---
    vol_recent_5 = sum(volumes[-5:]) / 5.0 if len(volumes) >= 5 else 0.0
    vol_avg_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0.0
    if vol_avg_20 > 0:
        vol_ratio = vol_recent_5 / vol_avg_20
        # Map ratio: 0.5x->0, 1.0x->50, 1.5x->100
        vol_score = max(0.0, min(100.0, (vol_ratio - 0.5) * 100.0))
    else:
        vol_score = 50.0

    # --- Price rate of change component (10-period) ---
    if len(closes) >= 11:
        price_10_ago = closes[-11]
        price_now = closes[-1]
        if price_10_ago != 0:
            roc_pct = ((price_now - price_10_ago) / price_10_ago) * 100.0
        else:
            roc_pct = 0.0
        # Map ROC: -10% -> 0, 0% -> 50, +10% -> 100
        roc_score = max(0.0, min(100.0, (roc_pct + 10.0) * 5.0))
    else:
        roc_score = 50.0

    # --- Composite score ---
    score = (rsi_score * 0.40) + (bb_score * 0.30) + (vol_score * 0.15) + (roc_score * 0.15)
    score = round(max(0.0, min(100.0, score)), 2)

    if score >= 70:
        signal = "STRONG_BULLISH"
    elif score >= 55:
        signal = "BULLISH"
    elif score >= 45:
        signal = "NEUTRAL"
    elif score >= 30:
        signal = "BEARISH"
    else:
        signal = "STRONG_BEARISH"

    return {
        "symbol": symbol,
        "date": date,
        "score": score,
        "signal": signal,
        "components": {
            "rsi": round(rsi_score, 2),
            "bollinger": round(bb_score, 2),
            "volume": round(vol_score, 2),
            "price_roc": round(roc_score, 2),
        },
    }


@mcp.tool()
def get_momentum_score(symbol: str, date: str) -> Dict[str, Any]:
    """Calculate a composite momentum score (0-100) for a symbol on a given date.

    Components:
    - RSI (40%): 14-period RSI value as momentum signal.
    - Bollinger %B (30%): where price sits in the bands.
    - Volume trend (15%): recent 5-period avg vs 20-period avg volume.
    - Price rate of change (15%): % change over last 10 periods.

    Args:
        symbol: Stock/crypto symbol.
        date: Date in YYYY-MM-DD format.

    Returns:
        Dictionary with composite score, signal, and component breakdown.
    """
    return _compute_momentum_score(symbol, date)


@mcp.tool()
def get_all_indicators(symbol: str, date: str) -> Dict[str, Any]:
    """Get all technical indicators (RSI, VWAP, Bollinger Bands, Momentum Score) at once.

    Args:
        symbol: Stock/crypto symbol.
        date: Date in YYYY-MM-DD format.

    Returns:
        Combined dictionary with results from all indicator tools.
    """
    return {
        "symbol": symbol,
        "date": date,
        "rsi": _compute_rsi(symbol, date),
        "vwap": _compute_vwap(symbol, date),
        "bollinger_bands": _compute_bollinger_bands(symbol, date),
        "momentum_score": _compute_momentum_score(symbol, date),
    }


if __name__ == "__main__":
    port = int(os.getenv("INDICATORS_HTTP_PORT", "8004"))
    mcp.run(transport="streamable-http", port=port)
