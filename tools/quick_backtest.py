"""
Lightweight watchlist backtest helper for the live GUI.

This reuses the existing agent/backtester classes over a short recent window,
then distills the result into:
1. a compact prompt block for live trading decisions
2. a small number of memory insights

The helper intentionally keeps the output compact so the live trader gets a
useful directional prior without flooding memory with raw backtest noise.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.general_tools import write_config_value
from tools.memory_tools import save_strategy_insight


_BACKTEST_LOCK = threading.Lock()


@dataclass
class QuickBacktestResult:
    success: bool
    message: str
    signature: str = ""
    granularity: str = ""
    init_date: str = ""
    end_date: str = ""
    processed_points: int = 0
    trade_count: int = 0
    final_equity: float = 0.0
    return_pct: float = 0.0
    used_symbols: List[str] = field(default_factory=list)
    skipped_symbols: List[str] = field(default_factory=list)
    bullish_symbols: List[str] = field(default_factory=list)
    caution_symbols: List[str] = field(default_factory=list)
    prompt_block: str = ""
    saved_insights: List[str] = field(default_factory=list)
    trade_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _runtime_snapshot() -> Tuple[Path, bool, str]:
    from tools import general_tools

    path = Path(general_tools._resolve_runtime_env_path())
    existed = path.exists()
    original = path.read_text(encoding="utf-8") if existed else ""
    return path, existed, original


def _restore_runtime_snapshot(path: Path, existed: bool, original: str) -> None:
    if existed:
        path.write_text(original, encoding="utf-8")
    elif path.exists():
        path.unlink()


def _extract_time_series(doc: dict) -> Dict[str, dict]:
    for key, value in doc.items():
        if key.startswith("Time Series") and isinstance(value, dict):
            return value
    return {}


def _collect_symbol_timestamps(
    symbols: Sequence[str],
    market: str,
    *,
    require_time: bool,
    merged_file: Optional[Path] = None,
) -> Dict[str, set]:
    from tools.price_tools import get_merged_file_path

    wanted = set(symbols)
    merged_file = merged_file or get_merged_file_path(market)
    timestamps: Dict[str, set] = {symbol: set() for symbol in symbols}

    if not merged_file.exists():
        return timestamps

    with merged_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = doc.get("Meta Data", {})
            symbol = meta.get("2. Symbol")
            if symbol not in wanted:
                continue
            series = _extract_time_series(doc)
            if not series:
                continue
            filtered = {
                ts for ts in series.keys()
                if (" " in ts) == require_time
            }
            timestamps[symbol] = filtered
    return timestamps


def _select_backtest_window(
    symbols: Sequence[str],
    market: str,
    lookback_points: int,
    merged_file: Optional[Path] = None,
) -> Tuple[Optional[str], List[str], List[str], List[str]]:
    """
    Choose a recent backtest window shared by as many watchlist symbols as possible.

    Returns:
        (granularity, timestamps, used_symbols, skipped_symbols)
    """
    needed = max(2, int(lookback_points) + 1)

    for granularity, require_time in (("hour", True), ("day", False)):
        per_symbol = _collect_symbol_timestamps(
            symbols,
            market,
            require_time=require_time,
            merged_file=merged_file,
        )
        available = {
            symbol: timestamps
            for symbol, timestamps in per_symbol.items()
            if timestamps
        }
        if not available:
            continue

        all_timestamps = sorted({ts for timestamps in available.values() for ts in timestamps})
        best_timestamps: List[str] = []
        best_symbols: List[str] = []

        for end_index in range(needed - 1, len(all_timestamps)):
            candidate_timestamps = all_timestamps[end_index - needed + 1:end_index + 1]
            candidate_symbols = [
                symbol for symbol, timestamps in available.items()
                if all(ts in timestamps for ts in candidate_timestamps)
            ]
            if not candidate_symbols:
                continue

            if (
                len(candidate_symbols) > len(best_symbols)
                or (
                    len(candidate_symbols) == len(best_symbols)
                    and candidate_timestamps[-1] > (best_timestamps[-1] if best_timestamps else "")
                )
            ):
                best_timestamps = candidate_timestamps
                best_symbols = candidate_symbols

        if best_timestamps and best_symbols:
            used_symbols = [symbol for symbol in symbols if symbol in set(best_symbols)]
            skipped_symbols = [symbol for symbol in symbols if symbol not in set(best_symbols)]
            return granularity, best_timestamps, used_symbols, skipped_symbols

    return None, [], [], list(symbols)


def _fetch_watchlist_merged_dataset(
    symbols: Sequence[str],
    market: str,
    dataset_path: Path,
) -> Tuple[Optional[Path], List[str], List[str], Optional[str]]:
    """
    Build a temporary merged dataset for the current watchlist.

    This keeps the live GUI independent from a pre-generated repo-wide
    `data/merged.jsonl`, and lets the micro-backtest cover newer scanner names.
    """
    if market != "us":
        return None, [], list(symbols), None

    try:
        from data.get_price_yahoo import fetch_hourly_data
    except Exception as exc:
        return None, [], list(symbols), f"Quick backtest skipped: could not load Yahoo fetcher ({exc})."

    unique_symbols = []
    for symbol in symbols:
        if symbol and symbol not in unique_symbols:
            unique_symbols.append(symbol)

    fetched_docs: List[dict] = []
    fetched_symbols: List[str] = []
    failed_symbols: List[str] = []

    for symbol in unique_symbols:
        try:
            doc = fetch_hourly_data(symbol, days=60)
        except Exception:
            doc = None
        if doc and _extract_time_series(doc):
            fetched_docs.append(doc)
            fetched_symbols.append(symbol)
        else:
            failed_symbols.append(symbol)

    if not fetched_docs:
        return None, [], failed_symbols, "Quick backtest skipped: could not fetch recent watchlist history from Yahoo."

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for doc in fetched_docs:
            handle.write(json.dumps(doc) + "\n")

    return dataset_path, fetched_symbols, failed_symbols, None


def _build_no_history_message(
    *,
    market: str,
    default_dataset_exists: bool,
    fetched_symbols: Sequence[str],
    failed_symbols: Sequence[str],
    fetch_error: Optional[str],
) -> str:
    if fetch_error and not default_dataset_exists:
        return (
            f"{fetch_error} Build local history with "
            f"'cd data && python get_price_yahoo.py' if you want an offline fallback."
        )
    if fetch_error:
        return fetch_error

    if not default_dataset_exists and market == "us":
        return (
            "Quick backtest skipped: no usable watchlist history was available, and "
            "C:\\clawdtrader\\data\\merged.jsonl does not exist yet. "
            "Run 'cd data && python get_price_yahoo.py' to build the local dataset."
        )

    if fetched_symbols:
        return "Quick backtest skipped: fetched watchlist history did not have enough overlapping bars yet."

    if failed_symbols:
        return (
            "Quick backtest skipped: none of the current watchlist symbols had usable recent history "
            f"({', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''})."
        )

    return "Quick backtest skipped: could not find enough watchlist history for a recent replay window."


def _load_position_records(position_file: Path) -> List[dict]:
    if not position_file.exists():
        return []

    records: List[dict] = []
    with position_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _extract_trade_events(position_records: Sequence[dict]) -> List[dict]:
    trades: List[dict] = []
    for record in position_records:
        action_info = record.get("this_action", {})
        action = action_info.get("action")
        symbol = action_info.get("symbol")
        amount = action_info.get("amount")
        if action in ("buy", "sell") and symbol and amount:
            trades.append(
                {
                    "date": record.get("date", ""),
                    "symbol": symbol,
                    "action": action,
                    "amount": int(amount),
                }
            )
    return trades


def _estimate_final_equity(
    positions: Dict[str, float],
    end_date: str,
    market: str,
) -> float:
    from tools.price_tools import get_open_prices

    cash = float(positions.get("CASH", 0.0) or 0.0)
    held_symbols = [symbol for symbol, qty in positions.items() if symbol != "CASH" and float(qty or 0) > 0]
    if not held_symbols:
        return cash

    end_prices = get_open_prices(end_date, held_symbols, market=market)
    equity = cash
    for symbol in held_symbols:
        price = end_prices.get(f"{symbol}_price")
        if price is None:
            continue
        equity += float(positions.get(symbol, 0.0) or 0.0) * float(price)
    return equity


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_bar_snapshot(bar: dict) -> Optional[Dict[str, Any]]:
    price = (
        _safe_float(bar.get("4. close"))
        or _safe_float(bar.get("4. sell price"))
        or _safe_float(bar.get("1. open"))
        or _safe_float(bar.get("1. buy price"))
    )
    if price is None:
        return None

    open_price = (
        _safe_float(bar.get("1. open"))
        or _safe_float(bar.get("1. buy price"))
        or price
    )
    high = _safe_float(bar.get("2. high")) or price
    low = _safe_float(bar.get("3. low")) or price
    volume = int(_safe_float(bar.get("5. volume")) or 0)
    return {
        "price": price,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
    }


def _load_price_snapshots(
    symbols: Sequence[str],
    timestamps: Sequence[str],
    merged_file: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    wanted = set(symbols)
    wanted_timestamps = set(timestamps)
    snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {symbol: {} for symbol in symbols}

    if not merged_file.exists():
        return snapshots

    with merged_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = doc.get("Meta Data", {})
            symbol = meta.get("2. Symbol")
            if symbol not in wanted:
                continue

            series = _extract_time_series(doc)
            if not series:
                continue

            for timestamp in wanted_timestamps:
                bar = series.get(timestamp)
                if not isinstance(bar, dict):
                    continue
                snapshot = _extract_bar_snapshot(bar)
                if snapshot is not None:
                    snapshots[symbol][timestamp] = snapshot

    return snapshots


def _build_portfolio_status_for_prompt(
    cash: float,
    positions: Dict[str, Dict[str, float]],
    prices: Dict[str, Dict[str, Any]],
) -> Tuple[str, float, float]:
    portfolio_value = cash
    lines = [
        "ACCOUNT STATUS:",
        f"- Cash: ${cash:,.2f}",
    ]

    position_lines: List[str] = []
    for symbol, position in positions.items():
        qty = float(position.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        price = float(prices.get(symbol, {}).get("price", 0.0) or 0.0)
        avg_entry = float(position.get("avg_entry", 0.0) or 0.0)
        market_value = qty * price
        portfolio_value += market_value
        pnl = (price - avg_entry) * qty
        pnl_pct = ((price - avg_entry) / avg_entry * 100.0) if avg_entry else 0.0
        position_lines.append(
            f"- {symbol}: {int(qty)} shares @ ${avg_entry:.2f} "
            f"(P/L: ${pnl:,.2f} / {pnl_pct:.1f}%)"
        )

    lines.append(f"- Portfolio Value: ${portfolio_value:,.2f}")
    lines.append(f"- Buying Power: ${cash:,.2f}")
    lines.append("")
    lines.append("CURRENT POSITIONS:")
    lines.extend(position_lines or ["- No positions"])

    return "\n".join(lines), cash, cash


def _apply_trade_decision(
    *,
    decision: Dict[str, Any],
    timestamp: str,
    prices: Dict[str, Dict[str, Any]],
    cash: float,
    positions: Dict[str, Dict[str, float]],
    initial_cash: float,
    max_position_size: float,
    trades: List[dict],
) -> float:
    action = str(decision.get("action", "HOLD") or "HOLD").upper()
    symbol = str(decision.get("symbol", "") or "").upper()
    requested_qty = int(decision.get("quantity", 0) or 0)

    if action not in {"BUY", "SELL"} or symbol not in prices or requested_qty <= 0:
        return cash

    price = float(prices[symbol]["price"])
    existing_position = positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
    current_equity = cash + sum(
        float(position.get("qty", 0.0) or 0.0) * float(prices.get(sym, {}).get("price", 0.0) or 0.0)
        for sym, position in positions.items()
    )
    current_equity = max(current_equity, initial_cash * 0.25)

    if action == "BUY":
        current_value = float(existing_position.get("qty", 0.0) or 0.0) * price
        max_position_value = current_equity * max_position_size
        remaining_value = max(0.0, max_position_value - current_value)
        affordable_qty = int(min(cash, remaining_value) / price) if price > 0 else 0
        quantity = min(requested_qty, affordable_qty)
        if quantity <= 0:
            return cash

        total_qty = float(existing_position.get("qty", 0.0) or 0.0) + quantity
        prior_cost = float(existing_position.get("qty", 0.0) or 0.0) * float(existing_position.get("avg_entry", 0.0) or 0.0)
        new_cost = quantity * price
        positions[symbol] = {
            "qty": total_qty,
            "avg_entry": (prior_cost + new_cost) / total_qty if total_qty else price,
        }
        cash -= quantity * price
        trades.append({"date": timestamp, "symbol": symbol, "action": "buy", "amount": quantity})
        return cash

    held_qty = int(float(existing_position.get("qty", 0.0) or 0.0))
    quantity = min(requested_qty, held_qty)
    if quantity <= 0:
        return cash

    remaining_qty = held_qty - quantity
    cash += quantity * price
    trades.append({"date": timestamp, "symbol": symbol, "action": "sell", "amount": quantity})
    if remaining_qty > 0:
        positions[symbol]["qty"] = float(remaining_qty)
    else:
        positions.pop(symbol, None)
    return cash


def _apply_auto_exits(
    *,
    timestamp: str,
    prices: Dict[str, Dict[str, Any]],
    positions: Dict[str, Dict[str, float]],
    cash: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    trades: List[dict],
) -> float:
    for symbol, position in list(positions.items()):
        qty = int(float(position.get("qty", 0.0) or 0.0))
        avg_entry = float(position.get("avg_entry", 0.0) or 0.0)
        current_price = float(prices.get(symbol, {}).get("price", 0.0) or 0.0)
        if qty <= 0 or avg_entry <= 0 or current_price <= 0:
            continue

        pnl_pct = ((current_price - avg_entry) / avg_entry) * 100.0
        if pnl_pct <= stop_loss_pct or pnl_pct >= take_profit_pct:
            cash += qty * current_price
            trades.append({"date": timestamp, "symbol": symbol, "action": "sell", "amount": qty})
            positions.pop(symbol, None)
    return cash


def _select_replay_timestamps(timestamps: Sequence[str], max_steps: int) -> List[str]:
    if len(timestamps) <= 1:
        return list(timestamps)
    if max_steps <= 0 or len(timestamps) <= max_steps:
        return list(timestamps)

    selected_indexes = []
    denominator = max(1, max_steps - 1)
    span = len(timestamps) - 1
    for step in range(max_steps):
        index = round(step * span / denominator)
        if index not in selected_indexes:
            selected_indexes.append(index)

    if selected_indexes[-1] != len(timestamps) - 1:
        selected_indexes[-1] = len(timestamps) - 1

    return [timestamps[index] for index in selected_indexes]


def _derive_bias_lists(
    trades: Sequence[dict],
    final_positions: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, int]], List[str], List[str]]:
    buy_counts: Counter = Counter()
    sell_counts: Counter = Counter()

    for trade in trades:
        if trade["action"] == "buy":
            buy_counts[trade["symbol"]] += 1
        elif trade["action"] == "sell":
            sell_counts[trade["symbol"]] += 1

    symbols = sorted(set(buy_counts) | set(sell_counts) | {s for s, qty in final_positions.items() if s != "CASH" and qty})
    trade_summary: Dict[str, Dict[str, int]] = {}
    for symbol in symbols:
        trade_summary[symbol] = {
            "buys": buy_counts[symbol],
            "sells": sell_counts[symbol],
            "ended_long": int(float(final_positions.get(symbol, 0.0) or 0.0) > 0),
        }

    bullish = [
        symbol for symbol in symbols
        if float(final_positions.get(symbol, 0.0) or 0.0) > 0 or buy_counts[symbol] > sell_counts[symbol]
    ]
    bullish.sort(
        key=lambda symbol: (
            float(final_positions.get(symbol, 0.0) or 0.0) > 0,
            buy_counts[symbol] - sell_counts[symbol],
            buy_counts[symbol],
        ),
        reverse=True,
    )

    caution = [
        symbol for symbol in symbols
        if sell_counts[symbol] > buy_counts[symbol] or (
            sell_counts[symbol] > 0 and float(final_positions.get(symbol, 0.0) or 0.0) <= 0
        )
    ]
    caution.sort(
        key=lambda symbol: (
            sell_counts[symbol] - buy_counts[symbol],
            sell_counts[symbol],
        ),
        reverse=True,
    )

    return trade_summary, bullish[:3], caution[:3]


def build_prompt_block(result: QuickBacktestResult) -> str:
    if not result.success:
        return ""

    lines = [
        "## QUICK BACKTEST CONTEXT (use as a weak prior, not a hard rule):",
        (
            f"- Window: {result.init_date} -> {result.end_date} "
            f"({result.processed_points} {result.granularity} bars)"
        ),
        (
            f"- Result: ${result.final_equity:,.2f} final equity "
            f"({result.return_pct:+.2f}%) from {result.trade_count} trades"
        ),
    ]

    if result.bullish_symbols:
        lines.append(f"- Bullish bias: {', '.join(result.bullish_symbols)}")
    if result.caution_symbols:
        lines.append(f"- Caution bias: {', '.join(result.caution_symbols)}")
    if result.skipped_symbols:
        lines.append(f"- Skipped (insufficient history): {', '.join(result.skipped_symbols)}")
    if result.trade_count == 0:
        lines.append("- No trades fired in the quick backtest; keep conviction light until live price action confirms.")

    return "\n".join(lines)


def build_memory_insights(result: QuickBacktestResult) -> List[Tuple[str, List[str]]]:
    if not result.success or result.trade_count == 0:
        return []

    insights: List[Tuple[str, List[str]]] = []
    if result.bullish_symbols:
        top = result.bullish_symbols[:2]
        insights.append(
            (
                f"Quick backtest bias: favor {', '.join(top)} on the current scanned watchlist; "
                f"recent {result.granularity}-level replay leaned strongest there.",
                ["backtest", "quick-backtest"] + [symbol.lower() for symbol in top],
            )
        )

    if result.caution_symbols:
        top = result.caution_symbols[:2]
        insights.append(
            (
                f"Quick backtest caution: require stronger confirmation before buying {', '.join(top)}; "
                f"recent watchlist replay was weakest there.",
                ["backtest", "quick-backtest", "risk"] + [symbol.lower() for symbol in top],
            )
        )

    if abs(result.return_pct) >= 1.5:
        tone = "aggressive setups worked" if result.return_pct > 0 else "overall watchlist conditions were weak"
        insights.append(
            (
                f"Quick backtest meta: {tone} over the recent {result.granularity}-level window ({result.return_pct:+.2f}%).",
                ["backtest", "quick-backtest", "meta"],
            )
        )

    return insights[:3]


def _save_backtest_insights(result: QuickBacktestResult) -> List[str]:
    saved: List[str] = []
    for insight, tags in build_memory_insights(result):
        save_strategy_insight(
            insight=insight,
            context={
                "window_start": result.init_date,
                "window_end": result.end_date,
                "return_pct": result.return_pct,
                "trade_count": result.trade_count,
                "signature": result.signature,
            },
            source="backtest",
            tags=tags,
        )
        saved.append(insight)
    return saved


def _run_agent_backtest(
    *,
    symbols: Sequence[str],
    market: str,
    granularity: str,
    init_date: str,
    end_date: str,
    basemodel: str,
    openai_api_key: str,
    openai_base_url: str,
    max_steps: int,
    initial_cash: float,
    signature: str,
    log_path: str,
) -> QuickBacktestResult:
    from agent.base_agent.base_agent import BaseAgent
    from agent.base_agent.base_agent_hour import BaseAgent_Hour

    agent_class = BaseAgent_Hour if granularity == "hour" else BaseAgent

    async def _runner() -> QuickBacktestResult:
        agent = agent_class(
            signature=signature,
            basemodel=basemodel,
            stock_symbols=list(symbols),
            log_path=log_path,
            max_steps=max_steps,
            max_retries=2,
            base_delay=0.5,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            initial_cash=initial_cash,
            init_date=init_date,
            market=market,
            verbose=False,
            sentiment_mode=None,
            fixed_temperature=None,
        )
        await agent.initialize()
        await agent.run_date_range(init_date, end_date)

        position_records = _load_position_records(Path(agent.position_file))
        latest_positions = position_records[-1].get("positions", {}) if position_records else {}
        trades = _extract_trade_events(position_records)
        final_equity = _estimate_final_equity(latest_positions, end_date, market)
        return_pct = ((final_equity - initial_cash) / initial_cash * 100.0) if initial_cash else 0.0
        trade_summary, bullish, caution = _derive_bias_lists(trades, latest_positions)

        result = QuickBacktestResult(
            success=True,
            message=f"Quick backtest completed with {basemodel}.",
            signature=signature,
            granularity=granularity,
            init_date=init_date,
            end_date=end_date,
            processed_points=0,
            trade_count=len(trades),
            final_equity=final_equity,
            return_pct=return_pct,
            used_symbols=list(symbols),
            bullish_symbols=bullish,
            caution_symbols=caution,
            trade_summary=trade_summary,
        )
        result.prompt_block = build_prompt_block(result)
        result.saved_insights = _save_backtest_insights(result)
        return result

    return asyncio.run(_runner())


def _run_prompt_backtest_with_client(
    *,
    client: Any,
    symbols: Sequence[str],
    market: str,
    granularity: str,
    timestamps: Sequence[str],
    basemodel: str,
    initial_cash: float,
    signature: str,
    merged_file: Path,
    max_steps: int,
    max_position_size: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> QuickBacktestResult:
    from tools.live_trading_utils import build_trading_prompt, get_memory_section, request_ai_decision

    price_snapshots = _load_price_snapshots(symbols, timestamps, merged_file)
    replay_timestamps = _select_replay_timestamps(timestamps, max_steps)
    memory_section = get_memory_section(symbols=symbols)

    positions: Dict[str, Dict[str, float]] = {}
    cash = float(initial_cash)
    trades: List[dict] = []

    for timestamp in replay_timestamps:
        current_prices = {
            symbol: price_snapshots[symbol][timestamp]
            for symbol in symbols
            if timestamp in price_snapshots.get(symbol, {})
        }
        if len(current_prices) != len(symbols):
            continue

        cash = _apply_auto_exits(
            timestamp=timestamp,
            prices=current_prices,
            positions=positions,
            cash=cash,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trades=trades,
        )
        portfolio_status, prompt_cash, prompt_buying_power = _build_portfolio_status_for_prompt(
            cash,
            positions,
            current_prices,
        )
        prompt = build_trading_prompt(
            symbols=symbols,
            prices=current_prices,
            portfolio_status=portfolio_status,
            cash=prompt_cash,
            buying_power=prompt_buying_power,
            max_position_size=max_position_size,
            memory_section=memory_section,
            intro=(
                f"You are Clawdbot, running a lightweight quick historical replay for {timestamp}. "
                "Use this snapshot as if it were the live market. This fallback mode is only used when "
                "the full langchain backtester is unavailable."
            ),
            rules=[
                f"You can only trade these stocks: {', '.join(symbols)}",
                f"Maximum position size: {max_position_size*100:.0f}% of portfolio per stock",
                f"Auto stop-loss triggers at {stop_loss_pct:.1f}% and auto take-profit triggers at +{take_profit_pct:.1f}%",
                "You can BUY, SELL, or HOLD",
                "Respond with your best short-term trade for this timestamp only",
            ],
        )
        decision = request_ai_decision(client, basemodel, prompt, log_fn=None)
        cash = _apply_trade_decision(
            decision=decision,
            timestamp=timestamp,
            prices=current_prices,
            cash=cash,
            positions=positions,
            initial_cash=initial_cash,
            max_position_size=max_position_size,
            trades=trades,
        )

    end_prices = {
        symbol: price_snapshots[symbol][timestamps[-1]]
        for symbol in symbols
        if timestamps[-1] in price_snapshots.get(symbol, {})
    }
    final_positions = {"CASH": cash}
    for symbol, position in positions.items():
        qty = float(position.get("qty", 0.0) or 0.0)
        if qty > 0:
            final_positions[symbol] = qty

    final_equity = cash + sum(
        float(position.get("qty", 0.0) or 0.0) * float(end_prices.get(symbol, {}).get("price", 0.0) or 0.0)
        for symbol, position in positions.items()
    )
    return_pct = ((final_equity - initial_cash) / initial_cash * 100.0) if initial_cash else 0.0
    trade_summary, bullish, caution = _derive_bias_lists(trades, final_positions)

    result = QuickBacktestResult(
        success=True,
        message=(
            "Quick backtest completed using lightweight prompt replay with "
            f"{basemodel} because langchain is unavailable."
        ),
        signature=signature,
        granularity=granularity,
        init_date=timestamps[0],
        end_date=timestamps[-1],
        processed_points=max(0, len(timestamps) - 1),
        trade_count=len(trades),
        final_equity=final_equity,
        return_pct=return_pct,
        used_symbols=list(symbols),
        bullish_symbols=bullish,
        caution_symbols=caution,
        trade_summary=trade_summary,
    )
    result.prompt_block = build_prompt_block(result)
    result.saved_insights = _save_backtest_insights(result)
    return result


def _run_prompt_backtest(
    *,
    symbols: Sequence[str],
    market: str,
    granularity: str,
    timestamps: Sequence[str],
    basemodel: str,
    openai_api_key: str,
    openai_base_url: str,
    initial_cash: float,
    signature: str,
    merged_file: Path,
    max_steps: int,
    max_position_size: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> QuickBacktestResult:
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url, timeout=60.0)
    return _run_prompt_backtest_with_client(
        client=client,
        symbols=symbols,
        market=market,
        granularity=granularity,
        timestamps=timestamps,
        basemodel=basemodel,
        initial_cash=initial_cash,
        signature=signature,
        merged_file=merged_file,
        max_steps=max_steps,
        max_position_size=max_position_size,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


def run_quick_backtest(
    *,
    symbols: Sequence[str],
    basemodel: str,
    openai_api_key: str,
    openai_base_url: str,
    market: str = "us",
    lookback_points: int = 6,
    max_steps: int = 4,
    initial_cash: float = 10000.0,
    log_path: str = "./data/quick_backtests",
    max_position_size: float = 0.2,
    stop_loss_pct: float = -3.0,
    take_profit_pct: float = 5.0,
) -> QuickBacktestResult:
    """
    Run a short backtest over the scanned watchlist and return a compact summary.
    """
    cleaned_symbols = [symbol for symbol in symbols if symbol]
    if not cleaned_symbols:
        return QuickBacktestResult(success=False, message="No symbols available for quick backtest.")
    if not openai_api_key:
        return QuickBacktestResult(success=False, message="No API key available for quick backtest.")

    from tools.price_tools import get_merged_file_path

    signature = f"quickbt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    default_merged_file = get_merged_file_path(market)
    temp_dataset_path = Path(log_path) / "_datasets" / f"{signature}-merged.jsonl"
    temp_merged_file, fetched_symbols, failed_symbols, fetch_error = _fetch_watchlist_merged_dataset(
        cleaned_symbols,
        market,
        temp_dataset_path,
    )

    granularity = None
    timestamps: List[str] = []
    used_symbols: List[str] = []
    skipped_symbols: List[str] = list(cleaned_symbols)
    selected_merged_file: Optional[Path] = None

    for candidate_path in (temp_merged_file, default_merged_file if default_merged_file.exists() else None):
        if candidate_path is None:
            continue
        candidate_granularity, candidate_timestamps, candidate_used_symbols, candidate_skipped_symbols = _select_backtest_window(
            cleaned_symbols,
            market,
            lookback_points,
            merged_file=candidate_path,
        )
        if (
            candidate_granularity
            and len(candidate_timestamps) >= 2
            and candidate_used_symbols
        ):
            granularity = candidate_granularity
            timestamps = candidate_timestamps
            used_symbols = candidate_used_symbols
            skipped_symbols = candidate_skipped_symbols
            selected_merged_file = candidate_path
            break

    if not granularity or len(timestamps) < 2 or not used_symbols:
        if temp_dataset_path.exists():
            temp_dataset_path.unlink()
        return QuickBacktestResult(
            success=False,
            message=_build_no_history_message(
                market=market,
                default_dataset_exists=default_merged_file.exists(),
                fetched_symbols=fetched_symbols,
                failed_symbols=failed_symbols,
                fetch_error=fetch_error,
            ),
            skipped_symbols=skipped_symbols,
        )

    runtime_path, existed, original = _runtime_snapshot()

    with _BACKTEST_LOCK:
        try:
            write_config_value("MERGED_PATH", str(selected_merged_file))
            result = _run_agent_backtest(
                symbols=used_symbols,
                market=market,
                granularity=granularity,
                init_date=timestamps[0],
                end_date=timestamps[-1],
                basemodel=basemodel,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                max_steps=max_steps,
                initial_cash=initial_cash,
                signature=signature,
                log_path=log_path,
            )
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("langchain"):
                result = _run_prompt_backtest(
                    symbols=used_symbols,
                    market=market,
                    granularity=granularity,
                    timestamps=timestamps,
                    basemodel=basemodel,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    initial_cash=initial_cash,
                    signature=signature,
                    merged_file=selected_merged_file,
                    max_steps=max_steps,
                    max_position_size=max_position_size,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                )
            else:
                raise
        except Exception as exc:
            return QuickBacktestResult(
                success=False,
                message=f"Quick backtest failed: {exc}",
                signature=signature,
                granularity=granularity or "",
                init_date=timestamps[0] if timestamps else "",
                end_date=timestamps[-1] if timestamps else "",
                used_symbols=used_symbols,
                skipped_symbols=skipped_symbols,
            )
        finally:
            _restore_runtime_snapshot(runtime_path, existed, original)
            if temp_dataset_path.exists():
                temp_dataset_path.unlink()

    result.used_symbols = used_symbols
    result.skipped_symbols = skipped_symbols
    result.processed_points = max(0, len(timestamps) - 1)
    result.prompt_block = build_prompt_block(result)
    return result
