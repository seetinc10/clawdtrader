"""
Open-of-day scanner: rank a universe of symbols by |%change| * relative volume,
filter for liquidity, return the top-N watchlist for the trading session.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any


def scan_top_movers(
    api,
    symbols: List[str],
    top_n: int = 10,
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 20_000_000,
    lookback_days: int = 15,
) -> List[Dict[str, Any]]:
    """Score = |day %change| * (today_volume / 5d_avg_volume). Liquidity-filtered."""
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days * 2 + 5)).date().isoformat()
    try:
        try:
            bars = api.get_bars(symbols, '1Day', start=start, feed='iex').df
        except TypeError:
            bars = api.get_bars(symbols, '1Day', start=start).df
    except Exception as e:
        print(f"[scanner] get_bars failed: {e}")
        return []
    if bars is None or len(bars) == 0:
        print("[scanner] get_bars returned empty dataframe")
        return []

    results = []
    for sym in symbols:
        try:
            if 'symbol' in bars.columns:
                df = bars[bars['symbol'] == sym]
            else:
                df = bars.xs(sym, level=0)
        except Exception:
            continue
        if len(df) < 6:
            continue
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prior5 = df.iloc[-6:-1]

        last_close = float(last['close'])
        prev_close = float(prev['close'])
        if last_close < min_price or prev_close <= 0:
            continue

        avg_dollar_vol = float((prior5['close'] * prior5['volume']).mean())
        if avg_dollar_vol < min_avg_dollar_vol:
            continue

        pct_change = (last_close - prev_close) / prev_close * 100.0
        avg_vol = float(prior5['volume'].mean()) or 1.0
        rel_vol = float(last['volume']) / avg_vol
        score = abs(pct_change) * rel_vol

        results.append({
            'symbol': sym,
            'price': last_close,
            'pct_change': pct_change,
            'rel_vol': rel_vol,
            'avg_dollar_vol': avg_dollar_vol,
            'score': score,
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:top_n]
