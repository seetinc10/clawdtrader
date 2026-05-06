"""
Sentiment Temperature Module

Converts the MAGS (Magnificent 7 ETF) or QQQ put/call ratio into an LLM
temperature value (0.0 - 1.0) for dynamic trading behavior.

High put/call ratio (bearish/fear)  -> Low temperature  (conservative, deterministic)
Low put/call ratio (bullish/greed)  -> High temperature (aggressive, exploratory)

Data sources tried in order:
1. CBOE options data for MAGS via Yahoo Finance
2. Fallback to QQQ if MAGS options liquidity is too thin
3. Manual override via config
"""

import json
import math
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# Optional: yfinance handles Yahoo's auth/crumb system automatically
try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

# Cache file to avoid hammering APIs on every trading step
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "sentiment"
_CACHE_FILE = _CACHE_DIR / "temperature_cache.json"
_YF_CACHE_DIR = _CACHE_DIR / "yfinance_cache"

# Mag 7 tickers (for reference / individual fallback)
MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Primary ETF that tracks Mag 7
PRIMARY_ETF = "MAGS"
FALLBACK_ETF = "QQQ"

# Put/Call ratio bounds for normalization
# Typical equity P/C ratio ranges: 0.4 (very bullish) to 1.3 (very bearish)
PC_RATIO_MIN = 0.4   # Below this -> max temperature
PC_RATIO_MAX = 1.3   # Above this -> min temperature

# Temperature output bounds
TEMP_MIN = 0.1   # Floor: never fully deterministic
TEMP_MAX = 0.95  # Ceiling: never fully random


def _ensure_cache_dir():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_yfinance_cache_dir():
    _YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _configure_yfinance_cache():
    """Use a project-local yfinance cache so we avoid corrupt global user caches."""
    if not _HAS_YFINANCE:
        return

    try:
        _ensure_yfinance_cache_dir()
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(_YF_CACHE_DIR))
    except Exception as e:
        print(f"[SentimentTemp] Warning: Could not configure yfinance cache: {e}")


def _is_corrupt_cache_error(error: Exception) -> bool:
    msg = str(error).lower()
    return any(
        text in msg for text in (
            "database disk image is malformed",
            "file is not a database",
            "malformed database schema",
        )
    )


def _reset_yfinance_cache() -> bool:
    """Delete the local yfinance cache so the next request can rebuild it cleanly."""
    try:
        shutil.rmtree(_YF_CACHE_DIR, ignore_errors=True)
        _ensure_yfinance_cache_dir()
        _configure_yfinance_cache()
        return True
    except Exception as e:
        print(f"[SentimentTemp] Warning: Could not reset yfinance cache: {e}")
        return False


def _load_cache() -> Dict[str, Any]:
    """Load cached temperature data."""
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(data: Dict[str, Any]):
    """Save temperature data to cache."""
    _ensure_cache_dir()
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[SentimentTemp] Warning: Could not save cache: {e}")


def _fetch_via_yfinance(ticker: str, max_expirations: int = 4) -> Optional[Dict[str, Any]]:
    """Fetch put/call ratio using the yfinance library (handles Yahoo auth automatically)."""
    if not _HAS_YFINANCE:
        return None

    _configure_yfinance_cache()

    for attempt in range(2):
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options  # list of date strings like ['2026-02-07', ...]
            if not expirations:
                return None

            total_put_oi = 0
            total_call_oi = 0
            total_put_volume = 0
            total_call_volume = 0
            dates_fetched = 0

            for exp_date in expirations[:max_expirations]:
                try:
                    chain = tk.option_chain(exp_date)

                    calls = chain.calls
                    puts = chain.puts

                    if calls is not None and len(calls) > 0:
                        total_call_oi += int(calls["openInterest"].fillna(0).sum())
                        total_call_volume += int(calls["volume"].fillna(0).sum())

                    if puts is not None and len(puts) > 0:
                        total_put_oi += int(puts["openInterest"].fillna(0).sum())
                        total_put_volume += int(puts["volume"].fillna(0).sum())

                    dates_fetched += 1
                except Exception:
                    continue

            if total_call_oi == 0 and total_call_volume == 0:
                return None

            # Prefer volume-based ratio (more current), fall back to OI
            if total_call_volume > 0 and total_put_volume > 0:
                pc_ratio = total_put_volume / total_call_volume
                ratio_type = "volume"
            elif total_call_oi > 0:
                pc_ratio = total_put_oi / total_call_oi
                ratio_type = "open_interest"
            else:
                return None

            return {
                "ticker": ticker,
                "put_call_ratio": round(pc_ratio, 4),
                "ratio_type": ratio_type,
                "total_put_oi": total_put_oi,
                "total_call_oi": total_call_oi,
                "total_put_volume": total_put_volume,
                "total_call_volume": total_call_volume,
                "expirations_aggregated": dates_fetched,
                "expiration_count": len(expirations),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            if attempt == 0 and _is_corrupt_cache_error(e) and _reset_yfinance_cache():
                print(f"[SentimentTemp] yfinance cache was corrupted while fetching {ticker}; reset and retrying.")
                continue
            print(f"[SentimentTemp] yfinance error for {ticker}: {e}")
            return None


def _fetch_via_raw_api(ticker: str, max_expirations: int = 4) -> Optional[Dict[str, Any]]:
    """Fetch put/call ratio via Yahoo Finance raw HTTP API (fallback)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Try multiple Yahoo Finance API endpoints
        urls = [
            f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}",
            f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}",
        ]

        data = None
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    break
            except Exception:
                continue

        if data is None:
            return None

        result = data.get("optionChain", {}).get("result", [])
        if not result:
            return None

        option_data = result[0]
        expirations = option_data.get("expirationDates", [])
        if not expirations:
            return None

        total_put_oi = 0
        total_call_oi = 0
        total_put_volume = 0
        total_call_volume = 0
        dates_fetched = 0

        # Process first expiration from initial response
        options = option_data.get("options", [])
        if options:
            nearest = options[0]
            for call in nearest.get("calls", []):
                total_call_oi += call.get("openInterest", 0)
                total_call_volume += call.get("volume", 0)
            for put in nearest.get("puts", []):
                total_put_oi += put.get("openInterest", 0)
                total_put_volume += put.get("volume", 0)
            dates_fetched += 1

        # Fetch additional expirations
        for exp_ts in expirations[1:max_expirations]:
            try:
                exp_url = f"{urls[0]}?date={exp_ts}"
                exp_resp = requests.get(exp_url, headers=headers, timeout=10)
                if exp_resp.status_code != 200:
                    continue

                exp_data = exp_resp.json()
                exp_result = exp_data.get("optionChain", {}).get("result", [])
                if not exp_result:
                    continue

                exp_options = exp_result[0].get("options", [])
                if not exp_options:
                    continue

                for call in exp_options[0].get("calls", []):
                    total_call_oi += call.get("openInterest", 0)
                    total_call_volume += call.get("volume", 0)
                for put in exp_options[0].get("puts", []):
                    total_put_oi += put.get("openInterest", 0)
                    total_put_volume += put.get("volume", 0)
                dates_fetched += 1

            except Exception:
                continue

        if total_call_oi == 0 and total_call_volume == 0:
            return None

        # Prefer volume-based ratio
        if total_call_volume > 0 and total_put_volume > 0:
            pc_ratio = total_put_volume / total_call_volume
            ratio_type = "volume"
        elif total_call_oi > 0:
            pc_ratio = total_put_oi / total_call_oi
            ratio_type = "open_interest"
        else:
            return None

        return {
            "ticker": ticker,
            "put_call_ratio": round(pc_ratio, 4),
            "ratio_type": ratio_type,
            "total_put_oi": total_put_oi,
            "total_call_oi": total_call_oi,
            "total_put_volume": total_put_volume,
            "total_call_volume": total_call_volume,
            "expirations_aggregated": dates_fetched,
            "expiration_count": len(expirations),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"[SentimentTemp] Raw API error for {ticker}: {e}")
        return None


def fetch_put_call_ratio_yahoo(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch put/call ratio for a ticker. Tries yfinance first, falls back to raw API.

    Returns:
        Dict with 'put_call_ratio', 'total_put_oi', 'total_call_oi', 'ticker'
        or None if data unavailable.
    """
    # Try yfinance first (handles Yahoo's auth/crumb system)
    result = _fetch_via_yfinance(ticker, max_expirations=1)
    if result is not None:
        return result

    # Fallback to raw API
    return _fetch_via_raw_api(ticker, max_expirations=1)


def fetch_put_call_ratio_multi_expiry(ticker: str, max_expirations: int = 4) -> Optional[Dict[str, Any]]:
    """
    Fetch put/call ratio aggregated across multiple near-term expiration dates.
    Tries yfinance first, falls back to raw API.

    Args:
        ticker: Stock/ETF ticker
        max_expirations: How many expiration dates to aggregate (default 4 = ~1 month)

    Returns:
        Dict with aggregated put/call ratio data or None.
    """
    # Try yfinance first
    result = _fetch_via_yfinance(ticker, max_expirations=max_expirations)
    if result is not None:
        return result

    # Fallback to raw API
    return _fetch_via_raw_api(ticker, max_expirations=max_expirations)


def pc_ratio_to_temperature(
    pc_ratio: float,
    invert: bool = False,
    pc_min: float = PC_RATIO_MIN,
    pc_max: float = PC_RATIO_MAX,
    temp_min: float = TEMP_MIN,
    temp_max: float = TEMP_MAX,
) -> float:
    """
    Convert a put/call ratio to an LLM temperature value.

    Default mapping (trend-following):
        High P/C (fear)   -> Low temperature  (conservative)
        Low P/C  (greed)  -> High temperature (aggressive)

    Contrarian mapping (invert=True):
        High P/C (fear)   -> High temperature (look for opportunities)
        Low P/C  (greed)  -> Low temperature  (be cautious at tops)

    Uses a sigmoid curve for smooth, bounded output.

    Args:
        pc_ratio: Raw put/call ratio
        invert: If True, use contrarian mapping
        pc_min: Lower bound of expected P/C range
        pc_max: Upper bound of expected P/C range
        temp_min: Minimum temperature output
        temp_max: Maximum temperature output

    Returns:
        Temperature value between temp_min and temp_max
    """
    # Normalize to 0-1 range (linear)
    normalized = (pc_ratio - pc_min) / (pc_max - pc_min)
    normalized = max(0.0, min(1.0, normalized))

    # Apply sigmoid for smooth transitions at extremes
    # Map normalized [0,1] to sigmoid input [-6, 6] for good curve shape
    sigmoid_input = (normalized * 12.0) - 6.0
    sigmoid_value = 1.0 / (1.0 + math.exp(-sigmoid_input))

    if invert:
        # Contrarian: high P/C -> high temp
        temperature = temp_min + sigmoid_value * (temp_max - temp_min)
    else:
        # Trend-following: high P/C -> low temp
        temperature = temp_max - sigmoid_value * (temp_max - temp_min)

    return round(temperature, 3)


def fetch_vix() -> Optional[float]:
    """Latest CBOE VIX close via yfinance. None on failure."""
    if not _HAS_YFINANCE:
        return None

    _configure_yfinance_cache()

    for attempt in range(2):
        try:
            df = yf.Ticker("^VIX").history(period="2d")
            if df is None or df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception as e:
            if attempt == 0 and _is_corrupt_cache_error(e) and _reset_yfinance_cache():
                print("[SentimentTemp] VIX cache was corrupted; reset local yfinance cache and retrying.")
                continue
            print(f"[SentimentTemp] VIX fetch failed: {e}")
            return None


def vix_to_temperature(vix: float, invert: bool = False) -> float:
    """Map VIX → temperature in [0.1, 0.9].

    Trend-following: low VIX → high temp (bullish), high VIX → low temp (fear).
        VIX 12 → 0.90 (calm)
        VIX 20 → 0.70
        VIX 25 → 0.45
        VIX 30 → 0.20 (panic)
    """
    raw = (30.0 - float(vix)) / 20.0 + 0.2
    temp = max(0.1, min(0.9, raw))
    if invert:
        temp = 1.0 - temp
    return round(temp, 3)


def get_sentiment_temperature(
    use_contrarian: bool = False,
    cache_ttl_minutes: int = 60,
    force_refresh: bool = False,
    vix_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Main entry point: Get the current sentiment-based temperature.

    Tries MAGS first, falls back to QQQ.
    Caches results to avoid excessive API calls.

    Args:
        use_contrarian: If True, use contrarian mapping
        cache_ttl_minutes: How long to use cached data (default 60 min)
        force_refresh: If True, bypass cache

    Returns:
        Dict with:
            - temperature: float (0.1 - 0.95)
            - put_call_ratio: float
            - ticker_used: str (MAGS or QQQ)
            - sentiment_label: str (EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED)
            - mode: str (trend_following or contrarian)
            - timestamp: str
            - source: str (live or cached)
    """
    # Check cache
    if not force_refresh:
        cache = _load_cache()
        if cache:
            cached_ts = cache.get("timestamp")
            if cached_ts:
                try:
                    cached_time = datetime.fromisoformat(cached_ts)
                    if datetime.now() - cached_time < timedelta(minutes=cache_ttl_minutes):
                        # Recalculate temperature with current mode setting
                        pc_ratio = cache.get("put_call_ratio", 0.7)
                        temperature = pc_ratio_to_temperature(pc_ratio, invert=use_contrarian)
                        cache["temperature"] = temperature
                        cache["mode"] = "contrarian" if use_contrarian else "trend_following"
                        cache["source"] = "cached"
                        return cache
                except Exception:
                    pass

    # Try MAGS first (pure Mag 7 exposure)
    pc_data = fetch_put_call_ratio_multi_expiry(PRIMARY_ETF)
    ticker_used = PRIMARY_ETF

    # Check if MAGS has sufficient liquidity (at least 100 total OI)
    if pc_data and (pc_data.get("total_put_oi", 0) + pc_data.get("total_call_oi", 0)) < 100:
        print(f"[SentimentTemp] MAGS options liquidity too thin, falling back to QQQ")
        pc_data = None

    # Fallback to QQQ
    if pc_data is None:
        pc_data = fetch_put_call_ratio_multi_expiry(FALLBACK_ETF)
        ticker_used = FALLBACK_ETF

    if pc_data is None:
        # Complete fallback: return neutral temperature
        print("[SentimentTemp] Could not fetch options data, using neutral temperature 0.5")
        return {
            "temperature": 0.5,
            "put_call_ratio": 0.7,
            "ticker_used": "NONE",
            "sentiment_label": "NEUTRAL",
            "mode": "contrarian" if use_contrarian else "trend_following",
            "timestamp": datetime.now().isoformat(),
            "source": "default_fallback",
            "error": "Could not fetch options data from any source",
        }

    pc_ratio = pc_data["put_call_ratio"]
    pc_temp = pc_ratio_to_temperature(pc_ratio, invert=use_contrarian)

    # Blend in VIX (broad-market fear gauge)
    vix = fetch_vix()
    if vix is not None and 0.0 < vix_weight <= 1.0:
        vix_temp = vix_to_temperature(vix, invert=use_contrarian)
        temperature = round((1.0 - vix_weight) * pc_temp + vix_weight * vix_temp, 3)
    else:
        vix_temp = None
        temperature = pc_temp

    # Classify sentiment using PC ratio (the more granular signal)
    if pc_ratio >= 1.1:
        sentiment = "EXTREME_FEAR"
    elif pc_ratio >= 0.85:
        sentiment = "FEAR"
    elif pc_ratio >= 0.6:
        sentiment = "NEUTRAL"
    elif pc_ratio >= 0.45:
        sentiment = "GREED"
    else:
        sentiment = "EXTREME_GREED"

    # VIX label override on extremes (broad panic outweighs tech-only sentiment)
    if vix is not None:
        if vix >= 30:
            sentiment = "EXTREME_FEAR"
        elif vix >= 25 and sentiment in ("NEUTRAL", "GREED", "EXTREME_GREED"):
            sentiment = "FEAR"

    result = {
        "temperature": temperature,
        "pc_temperature": pc_temp,
        "vix_temperature": vix_temp,
        "vix": round(vix, 2) if vix is not None else None,
        "vix_weight": vix_weight if vix is not None else 0.0,
        "put_call_ratio": pc_ratio,
        "ratio_type": pc_data.get("ratio_type", "unknown"),
        "ticker_used": ticker_used,
        "sentiment_label": sentiment,
        "mode": "contrarian" if use_contrarian else "trend_following",
        "total_put_oi": pc_data.get("total_put_oi", 0),
        "total_call_oi": pc_data.get("total_call_oi", 0),
        "total_put_volume": pc_data.get("total_put_volume", 0),
        "total_call_volume": pc_data.get("total_call_volume", 0),
        "expirations_aggregated": pc_data.get("expirations_aggregated", 1),
        "timestamp": datetime.now().isoformat(),
        "source": "live",
    }

    # Cache the result
    _save_cache(result)

    return result


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Sentiment Temperature Test")
    print("=" * 60)

    result = get_sentiment_temperature(force_refresh=True)

    print(f"\nTicker used:      {result['ticker_used']}")
    print(f"Put/Call Ratio:   {result['put_call_ratio']}")
    print(f"Ratio Type:       {result.get('ratio_type', 'N/A')}")
    print(f"Sentiment:        {result['sentiment_label']}")
    print(f"Temperature:      {result['temperature']}")
    print(f"Mode:             {result['mode']}")
    print(f"Source:           {result['source']}")

    if result.get("total_put_oi"):
        print(f"\nPut OI:           {result['total_put_oi']:,}")
        print(f"Call OI:          {result['total_call_oi']:,}")
        print(f"Put Volume:       {result['total_put_volume']:,}")
        print(f"Call Volume:      {result['total_call_volume']:,}")

    # Show contrarian version
    print("\n--- Contrarian Mode ---")
    result_c = get_sentiment_temperature(use_contrarian=True, force_refresh=True)
    print(f"Temperature:      {result_c['temperature']} (contrarian)")

    # Show the mapping curve
    print("\n--- P/C Ratio -> Temperature Mapping ---")
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        temp_tf = pc_ratio_to_temperature(ratio, invert=False)
        temp_ct = pc_ratio_to_temperature(ratio, invert=True)
        bar_tf = "#" * int(temp_tf * 30)
        print(f"  P/C {ratio:.1f} -> Trend: {temp_tf:.3f} {bar_tf}  |  Contrarian: {temp_ct:.3f}")
