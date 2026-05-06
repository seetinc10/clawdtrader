# Frontend Caching

The frontend cache is now U.S.-market only.

## What It Does

`scripts/precompute_frontend_cache.py` reads the static docs data and writes:

- `docs/data/us_cache.json`

This cache lets the docs frontend load precomputed portfolio histories and the QQQ benchmark without recalculating everything in the browser.

## Inputs

- `docs/config.yaml`
- `docs/data/agent_data/*/position/position.jsonl`
- `docs/data/Adaily_prices_QQQ.json`
- `docs/data/daily_prices_*.json`

## Regenerate

```bash
python scripts/precompute_frontend_cache.py
```

## Output Shape

The generated cache includes:

- `version`
- `generatedAt`
- `market`
- `agentsData`

Each agent entry contains:

- `positions`
- `assetHistory`
- `initialValue`
- `currentValue`
- `return`
