#!/usr/bin/env python3
"""
Pre-compute the U.S. frontend cache used by the static docs site.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys

import yaml


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).parent.parent
DOCS_DATA_DIR = ROOT / "docs" / "data"


def load_config() -> dict:
    config_path = ROOT / "docs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_data_version_hash(market_config: dict) -> str:
    """Hash position file timestamps so the frontend can invalidate stale cache."""
    hash_obj = hashlib.md5()
    data_dir = market_config.get("data_dir", "agent_data")
    position_files = sorted((DOCS_DATA_DIR / data_dir).glob("*/position/position.jsonl"))

    timestamps = []
    for position_file in position_files:
        timestamps.append(f"{position_file.name}:{position_file.stat().st_mtime}")

    hash_obj.update("|".join(timestamps).encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def load_position_data(agent_folder: str, market_config: dict) -> list[dict]:
    """Load every saved position snapshot for an agent."""
    data_dir = market_config.get("data_dir", "agent_data")
    position_file = DOCS_DATA_DIR / data_dir / agent_folder / "position" / "position.jsonl"
    if not position_file.exists():
        return []

    positions: list[dict] = []
    with open(position_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                positions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: failed to parse a position line for {agent_folder}: {exc}")

    return positions


def load_price_data(symbol: str) -> dict | None:
    """Load hourly price data when available, otherwise fall back to daily data."""
    hourly_file = DOCS_DATA_DIR / f"Ahourly_prices_{symbol}.json"
    daily_file = DOCS_DATA_DIR / f"daily_prices_{symbol}.json"
    price_file = hourly_file if hourly_file.exists() else daily_file
    if not price_file.exists():
        return None

    try:
        with open(price_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"Warning: failed to load price data for {symbol}: {exc}")
        return None

    return data.get("Time Series (60min)") or data.get("Time Series (Daily)")


def get_closing_price(symbol: str, date: str, price_data: dict[str, dict | None]) -> float | None:
    """Get a symbol close for a specific timestamp."""
    prices = price_data.get(symbol)
    if not prices or date not in prices:
        return None

    value = prices[date].get("4. close") or prices[date].get("4. sell price", 0)
    return float(value) if value else None


def calculate_asset_value(position: dict, date: str, price_data: dict[str, dict | None]) -> float:
    """Calculate a portfolio value from saved holdings and a price snapshot."""
    total_value = position["positions"].get("CASH", 0)
    for symbol, shares in position["positions"].items():
        if symbol == "CASH" or shares <= 0:
            continue

        price = get_closing_price(symbol, date, price_data)
        if price:
            total_value += shares * price

    return total_value


def process_agent_data(agent_config: dict, market_config: dict) -> dict | None:
    """Build the cache payload for one U.S. agent."""
    agent_folder = agent_config["folder"]
    print(f"  Processing {agent_folder}...")

    positions = load_position_data(agent_folder, market_config)
    if not positions:
        print(f"    No positions found for {agent_folder}")
        return None

    all_symbols = set()
    for position in positions:
        all_symbols.update(symbol for symbol in position["positions"].keys() if symbol != "CASH")

    price_data = {symbol: load_price_data(symbol) for symbol in all_symbols}

    positions_by_timestamp = {}
    for position in positions:
        timestamp = position["date"]
        if timestamp not in positions_by_timestamp or position["id"] > positions_by_timestamp[timestamp]["id"]:
            positions_by_timestamp[timestamp] = position

    unique_positions = sorted(positions_by_timestamp.values(), key=lambda item: (item["date"], item["id"]))

    asset_history = []
    for position in unique_positions:
        timestamp = position["date"]
        asset_history.append(
            {
                "date": timestamp,
                "value": calculate_asset_value(position, timestamp, price_data),
                "id": position["id"],
                "action": position.get("this_action"),
            }
        )

    if not asset_history:
        print(f"    No valid asset history for {agent_folder}")
        return None

    result = {
        "name": agent_folder,
        "positions": positions,
        "assetHistory": asset_history,
        "initialValue": asset_history[0]["value"],
        "currentValue": asset_history[-1]["value"],
        "return": ((asset_history[-1]["value"] - asset_history[0]["value"]) / asset_history[0]["value"] * 100),
    }

    print(f"    {len(positions)} positions, {len(asset_history)} data points")
    return result


def process_benchmark(market_config: dict, agents_data: dict[str, dict]) -> dict | None:
    """Build the QQQ benchmark series aligned to the agent date range."""
    print("  Processing QQQ benchmark...")

    benchmark_path = DOCS_DATA_DIR / market_config.get("benchmark_file", "Adaily_prices_QQQ.json")
    if not benchmark_path.exists():
        print("    QQQ benchmark file not found")
        return None

    with open(benchmark_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    time_series = data.get("Time Series (60min)") or data.get("Time Series (Daily)")
    if not time_series:
        print("    No time series data in QQQ benchmark")
        return None

    initial_value = 100000
    start_date_filter = None
    end_date_filter = None

    for agent_data in agents_data.values():
        history = agent_data.get("assetHistory", [])
        if not history:
            continue

        initial_value = history[0]["value"]
        agent_start = history[0]["date"]
        agent_end = history[-1]["date"]
        start_date_filter = agent_start if not start_date_filter or agent_start < start_date_filter else start_date_filter
        end_date_filter = agent_end if not end_date_filter or agent_end > end_date_filter else end_date_filter

    asset_history = []
    benchmark_start_price = None

    for date in sorted(time_series.keys()):
        if start_date_filter and date < start_date_filter:
            continue
        if end_date_filter and date > end_date_filter:
            continue

        close_price = float(time_series[date].get("4. close") or time_series[date].get("4. sell price", 0))
        if benchmark_start_price is None:
            benchmark_start_price = close_price

        benchmark_return = (close_price - benchmark_start_price) / benchmark_start_price
        asset_history.append(
            {
                "date": date,
                "value": initial_value * (1 + benchmark_return),
                "id": f"qqq-{date}",
                "action": None,
            }
        )

    if not asset_history:
        print("    No benchmark history generated")
        return None

    result = {
        "name": market_config.get("benchmark_display_name", "QQQ Invesco"),
        "positions": [],
        "assetHistory": asset_history,
        "initialValue": initial_value,
        "currentValue": asset_history[-1]["value"],
        "return": ((asset_history[-1]["value"] - asset_history[0]["value"]) / asset_history[0]["value"] * 100),
        "currency": "USD",
    }

    print(f"    {len(asset_history)} data points")
    return result


def generate_cache(market_id: str, market_config: dict) -> dict:
    """Generate the U.S. cache file consumed by the docs frontend."""
    print("=" * 60)
    print(f"Generating cache for {market_id.upper()} market")
    print("=" * 60)

    version = get_data_version_hash(market_config)
    print(f"Version hash: {version}")

    agents_data: dict[str, dict] = {}
    for agent_config in market_config.get("agents", []):
        if not agent_config.get("enabled", True):
            continue
        result = process_agent_data(agent_config, market_config)
        if result:
            agents_data[agent_config["folder"]] = result

    benchmark_data = process_benchmark(market_config, agents_data)
    if benchmark_data:
        agents_data[benchmark_data["name"]] = benchmark_data

    cache = {
        "version": f"v5_{version}",
        "generatedAt": datetime.now().isoformat(),
        "market": market_id,
        "agentsData": agents_data,
    }

    output_path = DOCS_DATA_DIR / f"{market_id}_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)

    print(f"Cache generated: {output_path}")
    print(f"  Version: {cache['version']}")
    print(f"  Agents: {len(agents_data)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    return cache


def main() -> None:
    print("=" * 60)
    print("Pre-computing Frontend Cache")
    print("=" * 60)

    config = load_config()
    market_id = "us"
    market_config = config.get("markets", {}).get(market_id)
    if not market_config:
        raise SystemExit("No U.S. market configuration found in docs/config.yaml")

    generate_cache(market_id, market_config)

    print("=" * 60)
    print("Cache generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
