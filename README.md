# Clawdbot Trader

Clawdbot Trader is an AI-assisted U.S. stock trading project built around two workflows:

- live paper trading through Alpaca in a desktop GUI or CLI
- backtesting Nasdaq-100 strategies with local position logs and MCP tools

This local workspace has been trimmed to the active U.S.-stock path. The older A-share and crypto agents, configs, scripts, and datasets are no longer part of the repo.

## Important Disclaimer

This project is for educational and research purposes only.

It is not financial, investment, legal, tax, or trading advice. Nothing in this repository, its prompts, logs, strategy notes, model outputs, or example trades should be treated as a recommendation to buy, sell, or hold any security.

All trading involves risk, including the risk of losing money. You are solely responsible for how you use this code, any market data, and any broker integrations. Review, test, and paper trade carefully before considering any real-money use.

## What This Build Includes

- GUI live trader with `START`, `PAUSE`, and `STOP`
- scanner-driven session watchlist from a broader AI stock universe
- deterministic risk rails in the GUI:
  - stop-loss
  - take-profit
  - daily kill switch
  - cooldown after stop-outs
  - force-flat before the close
- quick backtest on the scanned watchlist before live decisions begin
- persistent memory with strategy insights, trade outcomes, and conversation history
- sentiment-aware temperature using MAGS/QQQ options data plus VIX
- shared live-trading helpers so the CLI and GUI stay aligned on prompt building, parsing, and sizing
- unit tests plus GitHub Actions for the shared trading, sentiment, memory, and quick-backtest layers

## Main Entry Points

| File | Purpose |
|------|---------|
| `live_trader_gui.py` | Main GUI app for paper trading |
| `live_trader.py` | CLI paper trader |
| `strategy_chat.py` | Strategy discussion and memory management |
| `main.py` | U.S. backtesting entry point |
| `START_TRADER.bat` | Windows launcher for the GUI |
| `START_STRATEGY_CHAT.bat` | Windows launcher for strategy chat |

## How the Live Trader Works

1. Load memory and recent performance context.
2. In the GUI, scan the broader AI-focused universe and rank a top-10 watchlist for the session.
3. Run a short quick backtest on that watchlist and summarize the result into prompt context and memory.
4. Fetch account status, prices, positions, and market sentiment.
5. Ask the selected model for a JSON trading decision.
6. Enforce deterministic guardrails before submitting any Alpaca paper order.
7. Record outcomes back into memory and performance tracking.

### Active GUI Features

- model/provider switcher
- real-time console log
- strategy chat tab
- sentiment gauge
- trade count and session P&L
- safe Tkinter UI updates from worker threads

## Backtesting

The repo still includes the U.S. backtester and hourly U.S. agent:

- `BaseAgent`
- `BaseAgent_Hour`

Backtests use local price data, MCP services, and JSONL position logs. `main.py` is now U.S.-only and uses the Nasdaq-100 symbol set.

## Memory and Learning

Memory lives in `data/memory/`:

- `strategy_insights.jsonl`
- `trade_outcomes.jsonl`
- `conversations.jsonl`

The live trader uses memory in three ways:

- injects relevant insights into prompts
- tracks how symbols and ideas have performed recently
- can save compact quick-backtest takeaways and post-trade learnings

## Sentiment Temperature

`tools/sentiment_temperature.py` maps market fear and greed into model temperature.

Inputs:

- MAGS / QQQ put-call signal
- VIX context
- cached Yahoo data with local cache recovery if the cache becomes corrupted

Modes:

- `trend_following`
- `contrarian`
- fixed override with `fixed_temperature`

## MCP Services

| Port | Service | Script |
|------|---------|--------|
| 8000 | Math | `tool_math.py` |
| 8001 | Search / News | `tool_alphavantage_news.py` |
| 8002 | Trade | `tool_trade.py` |
| 8003 | Price | `tool_get_price_local.py` |
| 8004 | Indicators | `tool_indicators.py` |
| 8006 | Sentiment | `tool_sentiment_temperature.py` |

Start them with:

```bash
python agent_tools/start_mcp_services.py
```

## Installation

```bash
pip install -r requirements.txt
```

Required environment values:

```env
OPENAI_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPHAADVANTAGE_API_KEY=...

MATH_HTTP_PORT=8000
SEARCH_HTTP_PORT=8001
TRADE_HTTP_PORT=8002
GETPRICE_HTTP_PORT=8003
INDICATORS_HTTP_PORT=8004
SENTIMENT_HTTP_PORT=8006
```

## Common Commands

```bash
# GUI trader
python live_trader_gui.py

# CLI trader
python -u live_trader.py

# Strategy chat
python strategy_chat.py

# Save an insight directly
python strategy_chat.py --insight "Avoid chasing extended semiconductor breakouts"

# Memory stats
python strategy_chat.py --stats

# Refresh U.S. price data
cd data && python get_price_yahoo.py

# U.S. backtest
python main.py configs/default_day_config.json

# Sentiment module smoke test
python tools/sentiment_temperature.py

# Frontend cache refresh
python scripts/precompute_frontend_cache.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The current test suite covers:

- shared live-trading helpers
- memory ranking and performance tracking
- sentiment and VIX behavior
- quick backtest helpers and fallback paths

## Repo Layout

```text
agent/
  base_agent/
    base_agent.py
    base_agent_hour.py

agent_tools/
  start_mcp_services.py
  tool_alphavantage_news.py
  tool_get_price_local.py
  tool_indicators.py
  tool_math.py
  tool_sentiment_temperature.py
  tool_trade.py

configs/
  default_config.json
  default_day_config.json
  default_hour_config.json

data/
  memory/
  quick_backtests/

docs/
  config.yaml
  data/

tests/

tools/
  live_trading_utils.py
  memory_tools.py
  performance_tracker.py
  quick_backtest.py
  scanner.py
  sentiment_temperature.py
```

## Notes

- The quick backtest follows the GUI model selection, so if the GUI is set to DeepSeek, quick backtest uses DeepSeek too.
- If the LangChain backtest path is unavailable in the live Python environment, quick backtest falls back to a lightweight prompt-replay mode instead of failing.
- Sentiment fetching now uses a local Yahoo cache area and can recover automatically from malformed cache databases.
