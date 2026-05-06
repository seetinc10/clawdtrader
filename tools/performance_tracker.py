"""
Performance Tracker for Clawdbot Trader
Tracks trade outcomes and automatically generates insights based on results.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.memory_tools import (
    save_strategy_insight,
    save_trade_outcome,
    get_trade_outcomes,
    get_win_rate,
    record_insight_outcome,
    get_relevant_insights,
)


class PerformanceTracker:
    """
    Tracks trade performance and generates insights.
    Integrates with the memory system to learn from past trades.
    """

    # Thresholds for generating insights (D: lowered so most trades teach something)
    WIN_THRESHOLD = 1.0
    LOSS_THRESHOLD = -1.0
    BIG_WIN_THRESHOLD = 3.0
    BIG_LOSS_THRESHOLD = -3.0

    def __init__(self, signature: str):
        """
        Initialize tracker for a specific agent.

        Args:
            signature: Agent signature/name
        """
        self.signature = signature
        self.position_file = project_root / "data" / "agent_data" / signature / "position" / "position.jsonl"
        self.pending_trades: Dict[str, Dict] = {}  # Tracks open positions for outcome analysis
        self._load_pending_trades()

    def _load_pending_trades(self):
        """Load unmatched buy trades from trade_outcomes.jsonl so sells can be matched across restarts.
        Performs quantity-aware FIFO replay: a sell of N shares consumes buy records starting
        from the oldest, partially reducing a buy's remaining amount when needed."""
        try:
            outcomes = get_trade_outcomes(n=10000)
            for o in outcomes:
                symbol = o.get("symbol", "")
                action = o.get("action", "")
                if action == "buy" and o.get("outcome") == "pending":
                    if symbol not in self.pending_trades:
                        self.pending_trades[symbol] = []
                    self.pending_trades[symbol].append({
                        "symbol": symbol,
                        "action": "buy",
                        "amount": o.get("amount", 0),
                        "price": o.get("entry_price", 0),
                        "reasoning": o.get("reasoning", ""),
                        "timestamp": o.get("timestamp", ""),
                    })
                elif action == "sell" and o.get("outcome") in ("win", "loss", "neutral"):
                    # Quantity-aware replay: consume sell amount from oldest buys
                    sell_remaining = o.get("amount", 0)
                    if symbol in self.pending_trades:
                        while sell_remaining > 0 and self.pending_trades.get(symbol):
                            oldest = self.pending_trades[symbol][0]
                            if oldest["amount"] <= sell_remaining:
                                sell_remaining -= oldest["amount"]
                                self.pending_trades[symbol].pop(0)
                            else:
                                oldest["amount"] -= sell_remaining
                                sell_remaining = 0
                        if not self.pending_trades.get(symbol):
                            self.pending_trades.pop(symbol, None)
            pending_count = sum(len(v) for v in self.pending_trades.values())
            if pending_count:
                print(f"[Memory] Loaded {pending_count} pending buy trades from history")
        except Exception as e:
            print(f"[Memory] Could not load pending trades: {e}")

    def record_trade(
        self,
        symbol: str,
        action: str,
        amount: int,
        price: float,
        reasoning: str = ""
    ) -> Dict[str, Any]:
        """
        Record a new trade and track for outcome analysis.

        Args:
            symbol: Stock symbol
            action: "buy" or "sell"
            amount: Number of shares
            price: Execution price
            reasoning: Why this trade was made

        Returns:
            Trade record
        """
        trade = {
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "price": price,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

        if action == "buy":
            # Track this as a pending position to evaluate later
            if symbol not in self.pending_trades:
                self.pending_trades[symbol] = []
            self.pending_trades[symbol].append(trade)

            # Save to memory as pending
            save_trade_outcome(
                symbol=symbol,
                action=action,
                amount=amount,
                entry_price=price,
                reasoning=reasoning,
                outcome="pending"
            )

        elif action == "sell":
            # Evaluate the outcome of this sell
            self._evaluate_sell(symbol, amount, price, reasoning)

        return trade

    def _evaluate_sell(
        self,
        symbol: str,
        amount: int,
        sell_price: float,
        reasoning: str
    ):
        """Evaluate a sell trade against previous buys using quantity-aware FIFO.

        Matches the sell quantity against multiple buy records, partially consuming
        buy amounts when needed. Each matched portion gets its own outcome record
        with correct weighted entry price and amount.
        """
        if symbol not in self.pending_trades or not self.pending_trades[symbol]:
            # Selling without tracked buy - just record it
            save_trade_outcome(
                symbol=symbol,
                action="sell",
                amount=amount,
                entry_price=0,
                exit_price=sell_price,
                reasoning=reasoning,
                outcome="neutral"
            )
            return

        sell_remaining = amount
        # Accumulate weighted entry price for insight generation
        total_cost = 0.0
        total_matched = 0

        while sell_remaining > 0 and self.pending_trades.get(symbol):
            buy_trade = self.pending_trades[symbol][0]
            buy_price = buy_trade["price"]
            buy_amount = buy_trade["amount"]

            # How much of this buy record does the current sell consume?
            matched = min(sell_remaining, buy_amount)
            sell_remaining -= matched
            total_cost += buy_price * matched
            total_matched += matched

            # Calculate profit/loss for this matched portion
            profit_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price else 0

            if profit_pct >= self.WIN_THRESHOLD:
                outcome = "win"
            elif profit_pct <= self.LOSS_THRESHOLD:
                outcome = "loss"
            else:
                outcome = "neutral"

            save_trade_outcome(
                symbol=symbol,
                action="sell",
                amount=matched,
                entry_price=buy_price,
                exit_price=sell_price,
                profit_pct=profit_pct,
                reasoning=reasoning,
                outcome=outcome
            )

            if matched >= buy_amount:
                # Fully consumed this buy record
                self.pending_trades[symbol].pop(0)
            else:
                # Partially consumed - reduce the buy's remaining amount
                buy_trade["amount"] -= matched

        # Clean up empty list
        if not self.pending_trades.get(symbol):
            self.pending_trades.pop(symbol, None)

        # Generate insight based on weighted average entry price
        if total_matched > 0:
            avg_entry = total_cost / total_matched
            overall_pct = ((sell_price - avg_entry) / avg_entry) * 100 if avg_entry else 0
            if overall_pct >= self.WIN_THRESHOLD:
                overall_outcome = "win"
            elif overall_pct <= self.LOSS_THRESHOLD:
                overall_outcome = "loss"
            else:
                overall_outcome = "neutral"
            first_buy = {"price": avg_entry, "reasoning": reasoning}
            self._generate_trade_insight(symbol, overall_pct, overall_outcome, first_buy, reasoning)

            # E: credit/blame the insights that were active when this trade was decided
            if overall_outcome in ("win", "loss"):
                try:
                    active = get_relevant_insights(symbols=[symbol], n=10)
                    record_insight_outcome(
                        insight_ids=[e["id"] for e in active if "id" in e],
                        won=(overall_outcome == "win"),
                    )
                except Exception as e:
                    print(f"[tracker] insight outcome update failed: {e}")

        # If sell_remaining > 0, record the unmatched portion as neutral
        if sell_remaining > 0:
            save_trade_outcome(
                symbol=symbol,
                action="sell",
                amount=sell_remaining,
                entry_price=0,
                exit_price=sell_price,
                reasoning=reasoning,
                outcome="neutral"
            )

    def _generate_trade_insight(
        self,
        symbol: str,
        profit_pct: float,
        outcome: str,
        buy_trade: Dict,
        sell_reasoning: str
    ):
        """Generate insights from notable trade outcomes."""
        buy_reasoning = buy_trade.get("reasoning", "")

        # D: emit insights for any meaningful win/loss; tag with magnitude so
        # the prompt formatter can surface "big" vs "small" appropriately.
        if profit_pct >= self.WIN_THRESHOLD:
            mag = "big-win" if profit_pct >= self.BIG_WIN_THRESHOLD else "win"
            insight = f"Profitable {symbol} trade (+{profit_pct:.1f}%)"
            if buy_reasoning:
                insight += f" - thesis worked: {buy_reasoning[:100]}"
            save_strategy_insight(
                insight=insight,
                context={
                    "symbol": symbol,
                    "profit_pct": profit_pct,
                    "buy_reasoning": buy_reasoning,
                    "sell_reasoning": sell_reasoning,
                },
                source="performance",
                tags=[mag, symbol.lower()],
            )
        elif profit_pct <= self.LOSS_THRESHOLD:
            mag = "big-loss" if profit_pct <= self.BIG_LOSS_THRESHOLD else "loss"
            insight = f"Loss on {symbol} ({profit_pct:.1f}%)"
            if buy_reasoning:
                insight += f" - avoid: {buy_reasoning[:100]}"
            save_strategy_insight(
                insight=insight,
                context={
                    "symbol": symbol,
                    "profit_pct": profit_pct,
                    "buy_reasoning": buy_reasoning,
                    "sell_reasoning": sell_reasoning,
                },
                source="performance",
                tags=[mag, symbol.lower(), "lesson"],
            )

    def analyze_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze performance for a specific symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Performance statistics for the symbol
        """
        outcomes = get_trade_outcomes(n=1000, symbol=symbol)

        if not outcomes:
            return {"symbol": symbol, "trades": 0, "message": "No trade history"}

        wins = [o for o in outcomes if o.get("outcome") == "win"]
        losses = [o for o in outcomes if o.get("outcome") == "loss"]

        avg_win = sum(o.get("profit_pct", 0) for o in wins) / len(wins) if wins else 0
        avg_loss = sum(o.get("profit_pct", 0) for o in losses) / len(losses) if losses else 0

        return {
            "symbol": symbol,
            "total_trades": len(outcomes),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "expectancy": (len(wins) * avg_win + len(losses) * avg_loss) / len(outcomes) if outcomes else 0
        }

    def get_best_and_worst_symbols(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Get best and worst performing symbols.

        Args:
            n: Number of symbols to return for each category

        Returns:
            Tuple of (best_symbols, worst_symbols)
        """
        all_outcomes = get_trade_outcomes(n=1000)

        # Group by symbol
        symbol_stats = {}
        for outcome in all_outcomes:
            symbol = outcome.get("symbol")
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"wins": 0, "losses": 0, "total_pct": 0, "trades": 0}

            symbol_stats[symbol]["trades"] += 1

            if outcome.get("outcome") == "win":
                symbol_stats[symbol]["wins"] += 1
            elif outcome.get("outcome") == "loss":
                symbol_stats[symbol]["losses"] += 1

            symbol_stats[symbol]["total_pct"] += outcome.get("profit_pct", 0) or 0

        # Calculate average returns
        results = []
        for symbol, stats in symbol_stats.items():
            if stats["trades"] >= 2:  # At least 2 trades to be meaningful
                avg_return = stats["total_pct"] / stats["trades"]
                win_rate = stats["wins"] / (stats["wins"] + stats["losses"]) * 100 if (stats["wins"] + stats["losses"]) > 0 else 0
                results.append({
                    "symbol": symbol,
                    "avg_return": avg_return,
                    "win_rate": win_rate,
                    "trades": stats["trades"]
                })

        # Sort by average return
        results.sort(key=lambda x: x["avg_return"], reverse=True)

        best = results[:n]
        worst = results[-n:][::-1] if len(results) >= n else results[::-1]

        return best, worst

    def generate_performance_summary(self) -> str:
        """Generate a text summary of overall performance."""
        stats = get_win_rate()
        best, worst = self.get_best_and_worst_symbols(3)

        lines = [
            f"=== Performance Summary for {self.signature} ===",
            f"",
            f"Overall Stats:",
            f"  Total trades: {stats['total_trades']}",
            f"  Win rate: {stats['win_rate']:.1f}%",
            f"  Wins: {stats['wins']} | Losses: {stats['losses']}",
            f""
        ]

        if best:
            lines.append("Best Performing Symbols:")
            for b in best:
                lines.append(f"  {b['symbol']}: {b['avg_return']:+.1f}% avg ({b['trades']} trades)")
            lines.append("")

        if worst:
            lines.append("Worst Performing Symbols:")
            for w in worst:
                lines.append(f"  {w['symbol']}: {w['avg_return']:+.1f}% avg ({w['trades']} trades)")

        return "\n".join(lines)


def analyze_position_history(signature: str) -> List[Dict]:
    """
    Analyze position history from position.jsonl file.
    Useful for backtesting analysis.

    Args:
        signature: Agent signature

    Returns:
        List of analyzed trades
    """
    position_file = project_root / "data" / "agent_data" / signature / "position" / "position.jsonl"

    if not position_file.exists():
        return []

    positions = []
    with open(position_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positions.append(json.loads(line))

    # Track position changes to identify trades
    trades = []
    prev_positions = {}

    for record in positions:
        current_positions = record.get("positions", {})
        date = record.get("date", "")
        action_info = record.get("this_action", {})

        for symbol, qty in current_positions.items():
            if symbol == "CASH":
                continue

            prev_qty = prev_positions.get(symbol, 0)

            if qty > prev_qty:
                # Buy detected
                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "action": "buy",
                    "amount": qty - prev_qty,
                    "action_info": action_info
                })
            elif qty < prev_qty:
                # Sell detected
                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "action": "sell",
                    "amount": prev_qty - qty,
                    "action_info": action_info
                })

        prev_positions = current_positions.copy()

    return trades


if __name__ == "__main__":
    # Demo/test the performance tracker
    print("=== Performance Tracker Test ===\n")

    tracker = PerformanceTracker("test-agent")

    # Simulate some trades
    print("Recording test trades...")

    tracker.record_trade(
        symbol="NVDA",
        action="buy",
        amount=10,
        price=100.0,
        reasoning="Strong AI sector momentum"
    )

    tracker.record_trade(
        symbol="NVDA",
        action="sell",
        amount=10,
        price=108.0,  # 8% gain
        reasoning="Taking profits after rally"
    )

    tracker.record_trade(
        symbol="AAPL",
        action="buy",
        amount=20,
        price=150.0,
        reasoning="iPhone sales expected to beat"
    )

    tracker.record_trade(
        symbol="AAPL",
        action="sell",
        amount=20,
        price=142.0,  # -5.3% loss
        reasoning="Cut losses on disappointing guidance"
    )

    # Show summary
    print("\n" + tracker.generate_performance_summary())

    # Show symbol analysis
    print("\n--- NVDA Analysis ---")
    nvda_stats = tracker.analyze_symbol_performance("NVDA")
    print(json.dumps(nvda_stats, indent=2))
