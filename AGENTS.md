# Clawdbot Trader - Development Notes

## Project Summary

Clawdbot Trader is an autonomous AI trading system for U.S. stocks. It uses LLM models such as GPT, Claude, DeepSeek, Gemini, and Qwen to drive paper trading through Alpaca and local backtesting against Nasdaq-100 data.

This workspace has been cleaned up to the active U.S.-stock path only. The older A-share and crypto agents, configs, scripts, and datasets are no longer part of the repo.

## Key Files

| File | Purpose |
|------|---------|
| `live_trader_gui.py` | Main GUI app with scanner, sentiment, risk rails, and strategy chat |
| `live_trader.py` | Command-line paper trader |
| `strategy_chat.py` | Memory-aware strategy discussion tool |
| `main.py` | U.S. backtesting entry point |
| `START_TRADER.bat` | Windows launcher for the GUI |
| `START_STRATEGY_CHAT.bat` | Windows launcher for strategy chat |
| `data/get_price_yahoo.py` | Refreshes U.S. Yahoo Finance price data |

## Shared Trading Helpers

| File | Purpose |
|------|---------|
| `tools/live_trading_utils.py` | Shared prompt construction, response parsing, position limits, and portfolio formatting |
| `tools/scanner.py` | Builds the session watchlist from the larger AI-themed stock universe |
| `tools/quick_backtest.py` | Runs the watchlist quick backtest used by the GUI before live trading begins |

## Memory System Files

| File | Purpose |
|------|---------|
| `tools/memory_tools.py` | Save and load strategy insights |
| `tools/performance_tracker.py` | Track live trade outcomes and win rates |
| `data/memory/strategy_insights.jsonl` | Stored strategy learnings |
| `data/memory/trade_outcomes.jsonl` | Historical trade results |
| `data/memory/conversations.jsonl` | Saved strategy conversations |

## Sentiment Temperature Files

| File | Purpose |
|------|---------|
| `tools/sentiment_temperature.py` | Fetches options sentiment plus VIX context and maps it to LLM temperature |
| `agent_tools/tool_sentiment_temperature.py` | MCP service exposing sentiment tools |
| `data/sentiment/temperature_cache.json` | Cached sentiment values |

## MCP Services

| Port | Service | Script | Tools |
|------|---------|--------|-------|
| 8000 | Math | `tool_math.py` | Basic math operations |
| 8001 | Search/News | `tool_alphavantage_news.py` | News sentiment |
| 8002 | Trade | `tool_trade.py` | `buy()`, `sell()` |
| 8003 | Price | `tool_get_price_local.py` | Local OHLCV lookups |
| 8004 | Indicators | `tool_indicators.py` | RSI, VWAP, Bollinger Bands, momentum |
| 8006 | Sentiment | `tool_sentiment_temperature.py` | Market sentiment temperature tools |

Start all services:

```bash
python agent_tools/start_mcp_services.py
```

## Agent Classes

| Agent | Market | Symbols | Frequency | Notes |
|-------|--------|---------|-----------|-------|
| `BaseAgent` | U.S. stocks | Nasdaq-100 | Daily | Standard backtest agent |
| `BaseAgent_Hour` | U.S. stocks | Nasdaq-100 | Hourly | Intraday backtest agent |

## Trading Logic

1. Load positions, memory, and performance context.
2. In the GUI, scan the AI stock universe and pick the top watchlist for the session.
3. Run a short quick backtest on that watchlist and inject the result into prompt context.
4. Fetch current prices, account status, and market sentiment.
5. Ask the selected LLM for a JSON `BUY`, `SELL`, or `HOLD` decision.
6. Enforce deterministic risk rules before submitting any paper trade.
7. Save outcomes back into memory and performance tracking.

## GUI Components

- title bar
- status bar
- P&L bar
- sentiment bar
- AI provider selector
- START / PAUSE / STOP controls
- trading console tab
- strategy chat tab

## Dependencies

```text
alpaca-trade-api
openai
python-dotenv
yfinance
langchain
langchain-openai
langchain-mcp-adapters
fastmcp
requests
pandas
tkinter
```

## Quick Commands

```bash
python live_trader_gui.py
python -u live_trader.py
python strategy_chat.py
python strategy_chat.py --stats
cd data && python get_price_yahoo.py
python main.py configs/default_day_config.json
python tools/sentiment_temperature.py
python -m unittest discover -s tests -v
```
