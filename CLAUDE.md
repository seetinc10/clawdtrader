# Clawdbot Trader - Claude Notes

## Current Scope

This repo is now focused on the active U.S.-stock trading workflow:

- live paper trading with Alpaca
- strategy chat and persistent memory
- U.S. backtesting with `BaseAgent` and `BaseAgent_Hour`
- GUI watchlist scanning, quick backtest biasing, and sentiment-aware prompting

The older A-share and crypto trading paths have been removed from the working repo.

## Primary Files

| File | Purpose |
|------|---------|
| `live_trader_gui.py` | Main GUI trader |
| `live_trader.py` | CLI trader |
| `strategy_chat.py` | Strategy chat and insight management |
| `main.py` | U.S. backtesting runner |
| `tools/live_trading_utils.py` | Shared live prompt and sizing helpers |
| `tools/quick_backtest.py` | GUI quick backtest support |
| `tools/sentiment_temperature.py` | Sentiment and VIX temperature mapping |

## Active MCP Services

| Port | Service |
|------|---------|
| 8000 | Math |
| 8001 | Search / News |
| 8002 | Trade |
| 8003 | Price |
| 8004 | Indicators |
| 8006 | Sentiment |

Start them with:

```bash
python agent_tools/start_mcp_services.py
```

## Useful Commands

```bash
python live_trader_gui.py
python -u live_trader.py
python strategy_chat.py
python main.py configs/default_day_config.json
python -m unittest discover -s tests -v
```

## Notes

- The GUI scanner curates a session watchlist from the broader AI stock universe.
- Quick backtest follows the currently selected GUI model and falls back gracefully if the heavier LangChain path is unavailable.
- Sentiment now uses local Yahoo cache recovery to avoid malformed cache database failures.
