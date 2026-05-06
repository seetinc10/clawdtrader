"""
Shared helpers for the live trading CLI and GUI entry points.
"""

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tools.memory_tools import get_memory_context_for_prompt
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False


Decision = Dict[str, Any]
PriceMap = Dict[str, Dict[str, Any]]
LogFn = Optional[Callable[[str, str], None]]


def _log(log_fn: LogFn, message: str, tag: str = "info") -> None:
    if log_fn is not None:
        log_fn(message, tag)


def fetch_current_prices(api: Any, symbols: Sequence[str], log_fn: LogFn = None) -> PriceMap:
    """Fetch the latest OHLCV snapshot for the requested symbols."""
    prices: PriceMap = {}
    try:
        bars = api.get_latest_bars(list(symbols))
        for symbol in symbols:
            if symbol in bars:
                prices[symbol] = {
                    "price": float(bars[symbol].c),
                    "open": float(bars[symbol].o),
                    "high": float(bars[symbol].h),
                    "low": float(bars[symbol].l),
                    "volume": int(bars[symbol].v),
                }
    except Exception as e:
        _log(log_fn, f"Error getting prices: {e}", "error")
    return prices


def format_portfolio_status(account: Any, positions: Sequence[Any]) -> Tuple[str, float, float]:
    """Render current account and position state into the prompt format."""
    status = f"""
ACCOUNT STATUS:
- Cash: ${float(account.cash):,.2f}
- Portfolio Value: ${float(account.portfolio_value):,.2f}
- Buying Power: ${float(account.buying_power):,.2f}

CURRENT POSITIONS:
"""
    if positions:
        for pos in positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            status += (
                f"- {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                f"(P/L: ${pnl:,.2f} / {pnl_pct:.1f}%)\n"
            )
    else:
        status += "- No positions\n"

    return status, float(account.cash), float(account.buying_power)


def get_portfolio_status(api: Any) -> Tuple[str, float, float]:
    """Load account and positions from Alpaca and format them for the LLM."""
    account = api.get_account()
    positions = api.list_positions()
    return format_portfolio_status(account, positions)


def get_memory_section(symbols: Optional[Iterable[str]] = None, log_fn: LogFn = None) -> str:
    """Return the formatted memory block for a prompt, or an empty string."""
    if not _HAS_MEMORY:
        return ""

    try:
        symbol_list = list(symbols) if symbols else None
        memory_context = get_memory_context_for_prompt(symbols=symbol_list)
        if memory_context and "No previous strategy insights" not in memory_context:
            return f"""
STRATEGY MEMORY (Apply these learnings!):
{memory_context}
"""
    except Exception as e:
        _log(log_fn, f"Memory load error: {e}", "warning")
    return ""


def build_trading_prompt(
    *,
    symbols: Sequence[str],
    prices: PriceMap,
    portfolio_status: str,
    cash: float,
    buying_power: float,
    max_position_size: float,
    memory_section: str = "",
    intro: str,
    rules: Sequence[str],
    examples: Optional[Sequence[str]] = None,
) -> str:
    """Build the shared trading prompt used by the live trader surfaces."""
    price_info = "CURRENT MARKET PRICES:\n"
    for symbol, data in prices.items():
        change = ((data["price"] - data["open"]) / data["open"]) * 100
        price_info += f"- {symbol}: ${data['price']:.2f} (Today: {change:+.2f}%)\n"

    prompt = (
        f"{intro}\n"
        f"{memory_section}"
        f"{portfolio_status}\n"
        f"{price_info}\n\n"
        f"Available cash: ${cash:,.2f}\n"
        f"Buying power: ${buying_power:,.2f}\n\n"
        "RULES:\n"
    )

    for idx, rule in enumerate(rules, start=1):
        prompt += f"{idx}. {rule}\n"

    prompt += (
        "\nRespond with EXACTLY ONE JSON object (no other text):\n"
        '{"action": "BUY" or "SELL" or "HOLD", "symbol": "TICKER", "quantity": NUMBER, "reason": "brief reason"}\n'
    )

    if examples:
        prompt += "\nExamples:\n"
        for example in examples:
            prompt += f"{example}\n"

    return prompt


def parse_ai_decision(content: str, log_fn: LogFn = None) -> Decision:
    """Parse a model response into the canonical trading decision dict."""
    content = (content or "").strip()
    if not content:
        _log(log_fn, "Empty response from AI", "warning")
        return {"action": "HOLD", "symbol": "", "quantity": 0, "reason": "Empty response from AI"}

    if "{" in content and "}" in content:
        start = content.find("{")
        depth = 0
        end = start
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        json_str = content[start:end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            cleaned = json_str.replace("\n", " ").replace("\r", "")
            action_match = re.search(r'"action"\s*:\s*"(\w+)"', cleaned, re.IGNORECASE)
            symbol_match = re.search(r'"symbol"\s*:\s*"(\w*)"', cleaned, re.IGNORECASE)
            qty_match = re.search(r'"quantity"\s*:\s*(\d+)', cleaned, re.IGNORECASE)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)

            if action_match:
                return {
                    "action": action_match.group(1),
                    "symbol": symbol_match.group(1) if symbol_match else "",
                    "quantity": int(qty_match.group(1)) if qty_match else 0,
                    "reason": reason_match.group(1) if reason_match else "Parsed from response",
                }
            _log(log_fn, f"Bad JSON: {cleaned[:80]}...", "warning")

    _log(log_fn, f"Response: {content[:100]}...", "warning")
    return {"action": "HOLD", "symbol": "", "quantity": 0, "reason": "Could not parse response"}


def request_ai_decision(
    client: Any,
    model: str,
    prompt: str,
    *,
    temperature: Optional[float] = None,
    log_fn: LogFn = None,
    reasoner_max_tokens: int = 2000,
    default_max_tokens: int = 200,
) -> Decision:
    """Call the model and parse its JSON trading decision."""
    model_name = model.lower()
    if "reasoner" in model_name:
        messages = [
            {
                "role": "user",
                "content": "You are a professional stock trader. Respond with valid JSON only.\n\n" + prompt,
            }
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a professional stock trader. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": reasoner_max_tokens if "reasoner" in model_name else default_max_tokens,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature

    _log(log_fn, f"Waiting for {model}...", "info")
    response = client.chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    content = message.content

    if not content and hasattr(message, "reasoning_content") and message.reasoning_content:
        content = message.reasoning_content

    return parse_ai_decision(content or "", log_fn=log_fn)


def enforce_position_limit(
    api: Any,
    *,
    action: str,
    symbol: str,
    quantity: int,
    prices: Optional[PriceMap],
    max_position_size: float,
    log_fn: LogFn = None,
) -> Optional[int]:
    """Adjust or reject a BUY that would exceed the max position size."""
    if action.upper() != "BUY" or not prices or symbol not in prices:
        return quantity

    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            return quantity

        current_value = 0.0
        for pos in api.list_positions():
            if pos.symbol == symbol:
                current_value = float(pos.market_value)
                break

        proposed_value = current_value + quantity * prices[symbol]["price"]
        position_pct = proposed_value / portfolio_value
        if position_pct <= max_position_size:
            return quantity

        max_qty = int((max_position_size * portfolio_value - current_value) / prices[symbol]["price"])
        if max_qty <= 0:
            _log(
                log_fn,
                f"REJECTED: {symbol} already at max position size ({position_pct*100:.1f}% > {max_position_size*100}%)",
                "warning",
            )
            return None

        _log(
            log_fn,
            f"WARNING: Reducing {symbol} quantity from {quantity} to {max_qty} to stay under {max_position_size*100}% limit",
            "warning",
        )
        return max_qty
    except Exception as e:
        _log(log_fn, f"Warning: Could not check position size: {e}", "warning")
        return quantity
