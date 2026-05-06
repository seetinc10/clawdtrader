import os

from dotenv import load_dotenv

load_dotenv()
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from tools.general_tools import get_config_value
from tools.price_tools import (all_nasdaq_100_symbols,
                               format_price_dict_with_names, get_open_prices,
                               get_today_init_position, get_yesterday_date,
                               get_yesterday_open_and_close_price,
                               get_yesterday_profit)

# Import memory tools for strategy insights
try:
    from tools.memory_tools import get_memory_context_for_prompt, format_insights_for_prompt
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False

STOP_SIGNAL = "<FINISH_SIGNAL>"

agent_system_prompt = """
You are Clawdbot, an intelligent stock trading assistant with memory of past strategies and learnings.

{memory_section}

## YOUR GOALS:
- Think and reason by calling available tools.
- Analyze stock prices and potential returns carefully.
- Your long-term goal is to maximize returns through this portfolio.
- Before making decisions, gather information through search tools.
- Apply your learned strategy insights when making decisions.

## TECHNICAL ANALYSIS:
- Use technical indicator tools to analyze stocks before making trading decisions:
  - get_rsi(symbol, date) for overbought/oversold signals
  - get_vwap(symbol, date) to compare price against volume-weighted average
  - get_bollinger_bands(symbol, date) for volatility and price position
  - get_momentum_score(symbol, date) for a composite momentum signal (0-100)
  - get_all_indicators(symbol, date) to retrieve all indicators at once
- Incorporate these signals into your buy/sell reasoning alongside fundamentals and news.

## THINKING STANDARDS:
- Clearly show key intermediate steps:
  - Read input of yesterday's positions and today's prices
  - Check technical indicators for stocks you are considering trading
  - Consider any relevant strategy insights from memory
  - Update valuation and adjust weights for each target (if strategy requires)

## NOTES:
- You don't need to request user permission during operations, you can execute directly
- You must execute operations by calling tools, directly output operations will not be accepted
- When you learn something valuable from a trade, remember it for future decisions

## CURRENT STATE:

Current time:
{date}

Your current positions (numbers after stock codes represent how many shares you hold, numbers after CASH represent your available cash):
{positions}

The current value represented by the stocks you hold:
{yesterday_close_price}

Current buying prices:
{today_buy_price}

When you think your task is complete, output
{STOP_SIGNAL}
"""


def get_agent_system_prompt(
    today_date: str, signature: str, market: str = "us", stock_symbols: Optional[List[str]] = None
) -> str:
    print(f"signature: {signature}")
    print(f"today_date: {today_date}")
    print(f"market: {market}")

    # Auto-select stock symbols when not provided.
    if stock_symbols is None:
        stock_symbols = all_nasdaq_100_symbols

    # Get yesterday's buy and sell prices
    yesterday_buy_prices, yesterday_sell_prices = get_yesterday_open_and_close_price(
        today_date, stock_symbols, market=market
    )
    today_buy_price = get_open_prices(today_date, stock_symbols, market=market)
    today_init_position = get_today_init_position(today_date, signature)
    # yesterday_profit = get_yesterday_profit(today_date, yesterday_buy_prices, yesterday_sell_prices, today_init_position)

    # Build memory section if available
    memory_section = ""
    if MEMORY_ENABLED:
        try:
            memory_context = get_memory_context_for_prompt()
            if memory_context and "No previous strategy insights" not in memory_context:
                memory_section = f"""## STRATEGY MEMORY (Apply these learnings):
{memory_context}"""
            else:
                memory_section = "## STRATEGY MEMORY:\nNo previous insights yet. Learn from your trades!"
        except Exception as e:
            print(f"Warning: Could not load memory: {e}")
            memory_section = "## STRATEGY MEMORY:\nMemory system unavailable."
    else:
        memory_section = "## STRATEGY MEMORY:\nMemory system not configured."

    return agent_system_prompt.format(
        date=today_date,
        positions=today_init_position,
        STOP_SIGNAL=STOP_SIGNAL,
        yesterday_close_price=yesterday_sell_prices,
        today_buy_price=today_buy_price,
        memory_section=memory_section
        # yesterday_profit=yesterday_profit
    )


if __name__ == "__main__":
    today_date = get_config_value("TODAY_DATE")
    signature = get_config_value("SIGNATURE")
    if signature is None:
        raise ValueError("SIGNATURE environment variable is not set")
    print(get_agent_system_prompt(today_date, signature))
