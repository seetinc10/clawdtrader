"""
Memory Tools for Clawdbot Trader
Provides persistent memory for strategy learnings, insights, and trade outcomes.
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Cross-platform file locking
@contextmanager
def _file_lock(filepath: Path):
    """Cross-platform advisory file lock for serializing access to a file."""
    lock_path = filepath.parent / (filepath.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            # msvcrt.locking needs a non-zero length; lock 1 byte
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield fh
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = DATA_DIR / "memory"

# Memory files
STRATEGY_MEMORY_FILE = MEMORY_DIR / "strategy_insights.jsonl"
TRADE_OUTCOMES_FILE = MEMORY_DIR / "trade_outcomes.jsonl"
CONVERSATION_MEMORY_FILE = MEMORY_DIR / "conversations.jsonl"


def _ensure_memory_dir():
    """Ensure memory directory exists."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STRATEGY INSIGHTS MEMORY
# =============================================================================

def save_strategy_insight(
    insight: str,
    context: Optional[Dict[str, Any]] = None,
    source: str = "manual",
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Save a strategy learning/insight to memory.

    Args:
        insight: The strategy insight or learning
        context: Optional context (symbol, prices, conditions, etc.)
        source: Source of insight ("manual", "chat", "performance", "backtest")
        tags: Optional tags for categorization

    Returns:
        The saved entry
    """
    _ensure_memory_dir()

    with _file_lock(STRATEGY_MEMORY_FILE):
        entry = {
            "id": _get_next_id_unlocked(STRATEGY_MEMORY_FILE),
            "timestamp": datetime.now().isoformat(),
            "insight": insight,
            "context": context or {},
            "source": source,
            "tags": tags or [],
            "active": True,
            # Scoring fields (E): track how this insight performed when followed
            "referenced": 0,      # how many decisions cited it
            "wins_after": 0,      # closed trades that won after referencing
            "losses_after": 0,    # closed trades that lost after referencing
        }

        with open(STRATEGY_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Memory] Saved insight: {insight[:50]}...")
    return entry


def get_strategy_insights(
    n: int = 20,
    active_only: bool = True,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve strategy insights from memory.

    Args:
        n: Maximum number of insights to return (most recent first)
        active_only: Only return active insights
        tags: Filter by tags (any match)
        source: Filter by source

    Returns:
        List of insight entries
    """
    if not STRATEGY_MEMORY_FILE.exists():
        return []

    insights = []
    with open(STRATEGY_MEMORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)

                    # Apply filters
                    if active_only and not entry.get("active", True):
                        continue
                    if source and entry.get("source") != source:
                        continue
                    if tags and not any(t in entry.get("tags", []) for t in tags):
                        continue

                    insights.append(entry)
                except json.JSONDecodeError:
                    continue

    # Return most recent first
    return insights[-n:][::-1]


def format_insights_for_prompt(n: int = 15) -> str:
    """
    Format insights as a string for injection into agent prompts.

    Args:
        n: Maximum number of insights to include

    Returns:
        Formatted string of insights
    """
    insights = get_strategy_insights(n=n)

    if not insights:
        return "No previous strategy insights recorded yet."

    lines = []
    for i, entry in enumerate(insights, 1):
        insight = entry.get("insight", "")
        source = entry.get("source", "unknown")
        tags = entry.get("tags", [])

        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"{i}. {insight}{tag_str} (from {source})")

    return "\n".join(lines)


def deactivate_insight(insight_id: int, reason: str = "") -> bool:
    """
    Deactivate an insight that proved to be wrong.

    Args:
        insight_id: The ID of the insight to deactivate
        reason: Reason for deactivation

    Returns:
        True if successful
    """
    if not STRATEGY_MEMORY_FILE.exists():
        return False

    with _file_lock(STRATEGY_MEMORY_FILE):
        lines = []
        found = False

        with open(STRATEGY_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("id") == insight_id:
                        entry["active"] = False
                        entry["deactivated_at"] = datetime.now().isoformat()
                        entry["deactivation_reason"] = reason
                        found = True
                    lines.append(json.dumps(entry, ensure_ascii=False))

        if found:
            with open(STRATEGY_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    return found


# =============================================================================
# TRADE OUTCOMES MEMORY
# =============================================================================

def save_trade_outcome(
    symbol: str,
    action: str,
    amount: int,
    entry_price: float,
    exit_price: Optional[float] = None,
    profit_pct: Optional[float] = None,
    reasoning: str = "",
    outcome: str = "pending"  # "pending", "win", "loss", "neutral"
) -> Dict[str, Any]:
    """
    Save a trade outcome for learning.

    Args:
        symbol: Stock/crypto symbol
        action: "buy" or "sell"
        amount: Number of shares/units
        entry_price: Price at entry
        exit_price: Price at exit (if closed)
        profit_pct: Profit/loss percentage
        reasoning: Why this trade was made
        outcome: Trade outcome category

    Returns:
        The saved entry
    """
    _ensure_memory_dir()

    with _file_lock(TRADE_OUTCOMES_FILE):
        entry = {
            "id": _get_next_id_unlocked(TRADE_OUTCOMES_FILE),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_pct": profit_pct,
            "reasoning": reasoning,
            "outcome": outcome
        }

        with open(TRADE_OUTCOMES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def get_trade_outcomes(
    n: int = 50,
    symbol: Optional[str] = None,
    outcome: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve trade outcomes.

    Args:
        n: Maximum number of outcomes to return
        symbol: Filter by symbol
        outcome: Filter by outcome ("win", "loss", "neutral")

    Returns:
        List of trade outcome entries
    """
    if not TRADE_OUTCOMES_FILE.exists():
        return []

    outcomes = []
    with open(TRADE_OUTCOMES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)

                    if symbol and entry.get("symbol") != symbol:
                        continue
                    if outcome and entry.get("outcome") != outcome:
                        continue

                    outcomes.append(entry)
                except json.JSONDecodeError:
                    continue

    return outcomes[-n:]


def get_win_rate(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate win rate from trade outcomes.

    Args:
        symbol: Optional symbol to filter by

    Returns:
        Dict with win rate statistics
    """
    outcomes = get_trade_outcomes(n=1000, symbol=symbol)

    if not outcomes:
        return {"win_rate": 0, "total_trades": 0, "wins": 0, "losses": 0}

    wins = sum(1 for o in outcomes if o.get("outcome") == "win")
    losses = sum(1 for o in outcomes if o.get("outcome") == "loss")
    total = wins + losses

    return {
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "total_trades": len(outcomes),
        "wins": wins,
        "losses": losses,
        "neutral": sum(1 for o in outcomes if o.get("outcome") == "neutral"),
        "pending": sum(1 for o in outcomes if o.get("outcome") == "pending")
    }


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

def save_conversation(
    messages: List[Dict[str, str]],
    summary: str = "",
    insights_extracted: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Save a strategy chat conversation.

    Args:
        messages: List of conversation messages
        summary: Summary of the conversation
        insights_extracted: List of insights extracted from this conversation

    Returns:
        The saved entry
    """
    _ensure_memory_dir()

    with _file_lock(CONVERSATION_MEMORY_FILE):
        entry = {
            "id": _get_next_id_unlocked(CONVERSATION_MEMORY_FILE),
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "summary": summary,
            "insights_extracted": insights_extracted or []
        }

        with open(CONVERSATION_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def get_recent_conversations(n: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent conversation summaries.

    Args:
        n: Number of conversations to return

    Returns:
        List of conversation entries
    """
    if not CONVERSATION_MEMORY_FILE.exists():
        return []

    conversations = []
    with open(CONVERSATION_MEMORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return conversations[-n:]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_next_id_unlocked(filepath: Path) -> int:
    """Get next available ID for a memory file. Caller must hold the lock."""
    if not filepath.exists():
        return 1

    max_id = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    max_id = max(max_id, entry.get("id", 0))
                except json.JSONDecodeError:
                    continue

    return max_id + 1


def _get_next_id(filepath: Path) -> int:
    """Get next available ID for a memory file (thread/process safe)."""
    with _file_lock(filepath):
        return _get_next_id_unlocked(filepath)


def get_memory_stats() -> Dict[str, Any]:
    """Get statistics about stored memories."""
    stats = {
        "strategy_insights": 0,
        "trade_outcomes": 0,
        "conversations": 0,
        "win_rate": get_win_rate()
    }

    if STRATEGY_MEMORY_FILE.exists():
        with open(STRATEGY_MEMORY_FILE, "r", encoding="utf-8") as f:
            stats["strategy_insights"] = sum(1 for line in f if line.strip())

    if TRADE_OUTCOMES_FILE.exists():
        with open(TRADE_OUTCOMES_FILE, "r", encoding="utf-8") as f:
            stats["trade_outcomes"] = sum(1 for line in f if line.strip())

    if CONVERSATION_MEMORY_FILE.exists():
        with open(CONVERSATION_MEMORY_FILE, "r", encoding="utf-8") as f:
            stats["conversations"] = sum(1 for line in f if line.strip())

    return stats


def clear_strategy_insights() -> bool:
    """
    Clear all strategy insights.

    Returns:
        True if cleared successfully
    """
    if STRATEGY_MEMORY_FILE.exists():
        STRATEGY_MEMORY_FILE.unlink()
        return True
    return False


def clear_all_memory(confirm: bool = False) -> bool:
    """
    Clear all memory files. USE WITH CAUTION.

    Args:
        confirm: Must be True to actually clear

    Returns:
        True if cleared
    """
    if not confirm:
        print("Warning: Set confirm=True to actually clear all memory")
        return False

    for filepath in [STRATEGY_MEMORY_FILE, TRADE_OUTCOMES_FILE, CONVERSATION_MEMORY_FILE]:
        if filepath.exists():
            filepath.unlink()
            print(f"Deleted: {filepath}")

    return True


# =============================================================================
# QUICK ACCESS FOR PROMPTS
# =============================================================================

def get_memory_context_for_prompt(symbols: Optional[List[str]] = None) -> str:
    """
    Comprehensive memory context for prompt injection.

    If `symbols` is given, retrieval is filtered + ranked by relevance to those
    tickers (A + F). Per-symbol win-rate stats are also injected (B).
    """
    lines = []

    # ---- Strategy insights (relevance-ranked if we have symbols) ----
    if symbols:
        relevant = get_relevant_insights(symbols=symbols, n=10)
    else:
        relevant = get_strategy_insights(n=10)

    lines.append("## LEARNED STRATEGY INSIGHTS:")
    if relevant:
        for i, e in enumerate(relevant, 1):
            score = _insight_quality(e)
            tag_str = f" [{', '.join(e.get('tags', []))}]" if e.get("tags") else ""
            qual_str = f" ({e.get('wins_after', 0)}W/{e.get('losses_after', 0)}L)" if e.get("referenced", 0) else ""
            lines.append(f"{i}. {e.get('insight','')}{tag_str}{qual_str}")
    else:
        lines.append("(none yet)")
    lines.append("")

    # ---- Per-symbol stats (B) ----
    if symbols:
        stats_block = format_symbol_stats(symbols)
        if stats_block:
            lines.append("## PER-SYMBOL TRACK RECORD:")
            lines.append(stats_block)
            lines.append("")

    # ---- Aggregate win rate ----
    win_stats = get_win_rate()
    if win_stats["total_trades"] > 0:
        lines.append("## OVERALL PERFORMANCE:")
        lines.append(
            f"- Win rate: {win_stats['win_rate']:.1f}%  "
            f"({win_stats['wins']}W / {win_stats['losses']}L, "
            f"{win_stats['total_trades']} total)"
        )
        lines.append("")

    # ---- Recent losses ----
    recent_losses = get_trade_outcomes(n=5, outcome="loss")
    if recent_losses:
        lines.append("## RECENT LOSSES TO LEARN FROM:")
        for loss in recent_losses[-3:]:
            sym = loss.get("symbol", "?")
            pct = loss.get("profit_pct", 0) or 0
            reason = (loss.get("reasoning", "") or "")[:60]
            lines.append(f"- {sym}: {pct:.1f}% loss - {reason}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# RELEVANCE RANKING (A + F)
# =============================================================================

_STOP = {"the","a","an","is","of","to","in","on","and","or","for","at","by","with",
         "from","this","that","be","as","it","its","was","were","are","but","if",
         "than","then","too","not","no","do","does"}


def _tokens(text: str) -> set:
    """Lowercase alphanumeric token set, stopwords removed."""
    if not text:
        return set()
    out = set()
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if tok not in _STOP and len(tok) > 1:
                    out.add(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if tok not in _STOP and len(tok) > 1:
            out.add(tok)
    return out


def _insight_quality(entry: Dict[str, Any]) -> float:
    """E: quality score from track record. 0.5 prior, Beta-like update."""
    w = entry.get("wins_after", 0)
    l = entry.get("losses_after", 0)
    return (w + 1) / (w + l + 2)


def _relevance_score(entry: Dict[str, Any], symbol_set: set, query_tokens: set) -> float:
    """
    Combined score for ranking insights for a decision context.
      - Tag match against current symbols (A)            : +3.0 each
      - Token overlap (Jaccard) with query/insight (F)   : up to +2.0
      - Quality from past outcomes (E)                   : 0..+1.0
      - Recency bonus (newer ranked higher)              : tiny tiebreaker
    """
    tags = {t.lower() for t in entry.get("tags", [])}
    insight_text = entry.get("insight", "")
    insight_tokens = _tokens(insight_text)

    score = 0.0
    score += 3.0 * len(tags & symbol_set)

    if query_tokens and insight_tokens:
        inter = len(query_tokens & insight_tokens)
        union = len(query_tokens | insight_tokens) or 1
        score += 2.0 * (inter / union)

    score += _insight_quality(entry)

    # recency tiebreaker (id is monotonic)
    score += 0.001 * entry.get("id", 0)
    return score


def get_relevant_insights(
    symbols: List[str],
    query: str = "",
    n: int = 10,
    pool: int = 200,
) -> List[Dict[str, Any]]:
    """Pick the most relevant active insights for a decision context."""
    pool_insights = get_strategy_insights(n=pool, active_only=True)
    if not pool_insights:
        return []

    symbol_set = {s.lower() for s in symbols}
    query_tokens = _tokens(" ".join(symbols) + " " + (query or ""))

    scored = [(_relevance_score(e, symbol_set, query_tokens), e) for e in pool_insights]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:n]]


# =============================================================================
# PER-SYMBOL STATS (B)
# =============================================================================

def get_symbol_track_record(symbol: str) -> Dict[str, Any]:
    """Aggregate stats for a single symbol from closed trades."""
    outcomes = get_trade_outcomes(n=1000, symbol=symbol)
    closed = [o for o in outcomes if o.get("outcome") in ("win", "loss", "neutral")]
    wins = [o for o in closed if o.get("outcome") == "win"]
    losses = [o for o in closed if o.get("outcome") == "loss"]
    avg_pct = (
        sum((o.get("profit_pct") or 0) for o in closed) / len(closed)
        if closed else 0.0
    )
    return {
        "symbol": symbol,
        "wins": len(wins),
        "losses": len(losses),
        "neutral": len([o for o in closed if o.get("outcome") == "neutral"]),
        "total_closed": len(closed),
        "avg_profit_pct": avg_pct,
    }


def format_symbol_stats(symbols: List[str]) -> str:
    """Compact per-symbol track record block. Empty string if no history."""
    rows = []
    for sym in symbols:
        s = get_symbol_track_record(sym)
        if s["total_closed"] > 0:
            rows.append(
                f"- {sym}: {s['wins']}W/{s['losses']}L  "
                f"avg {s['avg_profit_pct']:+.2f}%"
            )
    return "\n".join(rows)


# =============================================================================
# INSIGHT SCORING / DECAY (E)
# =============================================================================

def record_insight_outcome(
    insight_ids: List[int],
    won: bool,
    auto_deactivate_after: int = 3,
) -> None:
    """Update referenced/wins_after/losses_after for the given insights.
    Auto-deactivate insights with N losses and zero wins."""
    if not insight_ids or not STRATEGY_MEMORY_FILE.exists():
        return
    id_set = set(insight_ids)
    with _file_lock(STRATEGY_MEMORY_FILE):
        out_lines = []
        with open(STRATEGY_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    out_lines.append(line.rstrip("\n"))
                    continue
                if e.get("id") in id_set:
                    e["referenced"] = e.get("referenced", 0) + 1
                    if won:
                        e["wins_after"] = e.get("wins_after", 0) + 1
                    else:
                        e["losses_after"] = e.get("losses_after", 0) + 1
                    if (
                        e.get("active", True)
                        and e.get("losses_after", 0) >= auto_deactivate_after
                        and e.get("wins_after", 0) == 0
                    ):
                        e["active"] = False
                        e["deactivated_at"] = datetime.now().isoformat()
                        e["deactivation_reason"] = (
                            f"auto: {e['losses_after']}L/{e['wins_after']}W after referencing"
                        )
                out_lines.append(json.dumps(e, ensure_ascii=False))
        with open(STRATEGY_MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")


# =============================================================================
# REFLECTION (C)
# =============================================================================

def reflect_on_session(
    openai_client,
    model: str,
    signature: str = "live-trader-gui",
    lookback_hours: int = 24,
) -> List[str]:
    """Ask the LLM to distill recent trades into 1-3 portable rules.
    Saves each as a strategy insight tagged 'reflection'. Returns the list."""
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    outcomes = get_trade_outcomes(n=200)
    recent = []
    for o in outcomes:
        try:
            ts = datetime.fromisoformat(o.get("timestamp", ""))
            if ts >= cutoff:
                recent.append(o)
        except Exception:
            continue
    if not recent:
        return []

    summary_lines = []
    for o in recent[-40:]:
        summary_lines.append(
            f"- {o.get('action','?').upper()} {o.get('symbol','?')} x{o.get('amount','?')} "
            f"@ ${o.get('entry_price','?')} → {o.get('outcome','pending')} "
            f"({(o.get('profit_pct') or 0):+.2f}%) :: {(o.get('reasoning','') or '')[:80]}"
        )
    trades_text = "\n".join(summary_lines)

    prompt = (
        "You are reviewing a trading session. From the trade log below, extract "
        "1 to 3 SHORT, ACTIONABLE rules a trading agent can apply going forward. "
        "Each rule must be concrete (mention symbol, condition, or pattern) and one line. "
        "Reply with ONLY the rules, one per line, no numbering, no preamble.\n\n"
        f"TRADES:\n{trades_text}\n"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            timeout=120,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[reflect] LLM call failed: {e}")
        return []

    rules = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
    rules = [r for r in rules if 8 < len(r) < 240][:3]

    saved = []
    for r in rules:
        tag_syms = [s.lower() for s in {o.get("symbol", "") for o in recent} if s]
        save_strategy_insight(
            insight=r,
            context={"trade_count": len(recent)},
            source="reflection",
            tags=["reflection"] + tag_syms[:5],
        )
        saved.append(r)
    return saved


if __name__ == "__main__":
    # Test the memory system
    print("=== Memory Tools Test ===\n")

    # Save a test insight
    save_strategy_insight(
        "Tech stocks tend to dip on Monday mornings - good buying opportunity",
        context={"observation_period": "2024-2025"},
        source="manual",
        tags=["timing", "tech"]
    )

    # Show stats
    stats = get_memory_stats()
    print(f"\nMemory Stats: {json.dumps(stats, indent=2)}")

    # Show formatted insights
    print("\n" + "="*50)
    print("Formatted insights for prompt:")
    print(format_insights_for_prompt())
