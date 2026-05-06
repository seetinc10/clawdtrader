# Configurations

The repo now keeps only the active U.S.-stock backtest configs.

## Available Configs

| File | Purpose |
|------|---------|
| `default_config.json` | Shared default values |
| `default_day_config.json` | Daily U.S. stock backtest |
| `default_hour_config.json` | Hourly U.S. stock backtest |

## Run Examples

```bash
python main.py configs/default_day_config.json
python main.py configs/default_hour_config.json
```

## Notes

- `main.py` is U.S.-only.
- The active backtest agents are `BaseAgent` and `BaseAgent_Hour`.
- Backtest logs and positions are written under `data/agent_data/`.
