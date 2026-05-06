#!/usr/bin/env python3
"""
Visualize rolling performance metrics for the U.S. backtest agents.
"""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("notebook", font_scale=1.1)

AGENT_MAPPING = {
    "deepseek-chat-v3.1": "DeepSeek-v3.1",
    "MiniMax-M2": "MiniMax-M2",
    "claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "gpt-5": "GPT-5",
    "qwen3-max": "Qwen3-Max",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
}

AGENT_COLORS = {
    "DeepSeek-v3.1": "#FF6B6B",
    "MiniMax-M2": "#4ECDC4",
    "Claude-3.7-Sonnet": "#45B7D1",
    "GPT-5": "#96CEB4",
    "Qwen3-Max": "#FFEAA7",
    "Gemini-2.5-Flash": "#DFE6E9",
}

METRICS = [
    ("CR", "Cumulative Return (%)", "CR"),
    ("SR", "Sortino Ratio", "SR"),
    ("Vol", "Volatility (%)", "Vol"),
    ("MDD", "Maximum Drawdown (%)", "MDD"),
]


def load_portfolio_data(agent_dir: Path) -> pd.DataFrame | None:
    """Load an agent portfolio history if it exists."""
    portfolio_file = agent_dir / "position" / "portfolio_values.csv"
    if not portfolio_file.exists():
        return None

    df = pd.read_csv(portfolio_file)
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_rolling_metrics(df: pd.DataFrame, is_hourly: bool = True) -> pd.DataFrame:
    """Calculate expanding CR, Sortino, volatility, and drawdown series."""
    df = df.copy()
    df["returns"] = df["total_value"].pct_change()

    initial_value = df["total_value"].iloc[0]
    df["CR"] = (df["total_value"] - initial_value) / initial_value * 100

    periods_per_year = 252 * 6.5 if is_hourly else 252
    min_periods = 10 if is_hourly else 3

    sortino_ratios = []
    volatilities = []

    for i in range(len(df)):
        if i < min_periods:
            sortino_ratios.append(np.nan)
        else:
            returns_so_far = df["returns"].iloc[1 : i + 1].dropna()
            if len(returns_so_far) < min_periods:
                sortino_ratios.append(np.nan)
            else:
                negative_returns = returns_so_far[returns_so_far < 0]
                if len(negative_returns) > 0:
                    downside_std = max(negative_returns.std(), 0.0001)
                    sortino = (returns_so_far.mean() / downside_std) * np.sqrt(periods_per_year)
                    sortino_ratios.append(np.clip(sortino, -20, 20))
                else:
                    sortino_ratios.append(20 if returns_so_far.mean() > 0 else 0)

        if i < 2:
            volatilities.append(np.nan)
        else:
            returns_so_far = df["returns"].iloc[1 : i + 1].dropna()
            if len(returns_so_far) < 2:
                volatilities.append(np.nan)
            else:
                volatilities.append(returns_so_far.std() * np.sqrt(periods_per_year) * 100)

    df["SR"] = sortino_ratios
    df["Vol"] = volatilities

    cumulative = (1 + df["returns"].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    df["MDD"] = (cumulative - running_max) / running_max * 100
    return df


def get_agent_date_range(agent_data_dir: Path) -> tuple[str, str] | None:
    """Extract the first available date range from the agent data directory."""
    if not agent_data_dir.exists():
        return None

    for agent_dir in agent_data_dir.iterdir():
        if not agent_dir.is_dir():
            continue

        portfolio_file = agent_dir / "position" / "portfolio_values.csv"
        if not portfolio_file.exists():
            continue

        df = pd.read_csv(portfolio_file)
        if df.empty:
            continue

        start_date = df["date"].iloc[0].split(" ")[0]
        end_date = df["date"].iloc[-1].split(" ")[0]
        return (start_date, end_date)

    return None


def load_baseline_data(
    baseline_file: Path, is_hourly: bool = True, date_range: tuple[str, str] | None = None
) -> pd.DataFrame | None:
    """Load the QQQ benchmark and compute the same rolling metrics."""
    with open(baseline_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    time_series = data.get("Time Series (60min)") or data.get("Time Series (Daily)")
    if not time_series:
        return None

    all_dates = sorted(time_series.keys())
    if date_range:
        start_date, end_date = date_range
        dates = [date for date in all_dates if start_date <= date <= end_date]
    else:
        dates = all_dates

    if len(dates) < 2:
        return None

    prices = [float(time_series[date].get("4. close") or time_series[date].get("4. sell price", 0)) for date in dates]
    df = pd.DataFrame({"date": pd.to_datetime(dates), "price": prices})
    df["total_value"] = df["price"] / df["price"].iloc[0] * 10000
    return calculate_rolling_metrics(df, is_hourly=is_hourly)


def plot_single_metric(
    agent_data: dict[str, pd.DataFrame],
    baseline_data: pd.DataFrame | None,
    metric_key: str,
    ylabel: str,
    title: str,
    output_file: Path,
) -> None:
    """Render a single-metric figure."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for agent_name, df in agent_data.items():
        display_name = AGENT_MAPPING.get(agent_name, agent_name)
        color = AGENT_COLORS.get(display_name)
        plot_df = df[["date", metric_key]].dropna()
        ax.plot(plot_df["date"], plot_df[metric_key], label=display_name, linewidth=3.0, alpha=0.8, color=color)

    if baseline_data is not None:
        plot_df = baseline_data[["date", metric_key]].dropna()
        ax.plot(plot_df["date"], plot_df[metric_key], label="QQQ", linewidth=3.5, linestyle="--", color="black", alpha=0.7)

    ax.set_xlabel("Time", fontsize=20, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    ax.set_title(f"U.S. Market (NASDAQ-100) - {title}", fontsize=22, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14, framealpha=0.7)
    ax.tick_params(axis="x", rotation=45, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_market_metrics(agent_data: dict[str, pd.DataFrame], baseline_data: pd.DataFrame | None, output_file: Path) -> None:
    """Render a four-panel summary figure."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for index, (metric_key, ylabel, title) in enumerate(METRICS):
        ax = axes[index]

        for agent_name, df in agent_data.items():
            display_name = AGENT_MAPPING.get(agent_name, agent_name)
            color = AGENT_COLORS.get(display_name)
            plot_df = df[["date", metric_key]].dropna()
            ax.plot(plot_df["date"], plot_df[metric_key], label=display_name, linewidth=2, alpha=0.8, color=color)

        if baseline_data is not None:
            plot_df = baseline_data[["date", metric_key]].dropna()
            ax.plot(plot_df["date"], plot_df[metric_key], label="QQQ", linewidth=2.5, linestyle="--", color="black", alpha=0.7)

        ax.set_xlabel("Time", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def load_us_agent_metrics() -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Load all supported U.S. agent metrics and the QQQ benchmark."""
    us_data_dir = Path("data/agent_data")
    agent_data: dict[str, pd.DataFrame] = {}

    if not us_data_dir.exists():
        return agent_data, None

    print("=" * 70)
    print("PROCESSING U.S. MARKET VISUALIZATIONS")
    print("=" * 70)

    for agent_dir in sorted(us_data_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        agent_name = agent_dir.name
        if agent_name not in AGENT_MAPPING:
            continue

        print(f"Loading {agent_name}...")
        df = load_portfolio_data(agent_dir)
        if df is None:
            continue

        agent_data[agent_name] = calculate_rolling_metrics(df, is_hourly=True)
        print(f"  {agent_name}: {len(df)} time points")

    qqq_file = Path("data/daily_prices_QQQ.json")
    baseline_data = None
    if qqq_file.exists():
        date_range = get_agent_date_range(us_data_dir)
        baseline_data = load_baseline_data(qqq_file, is_hourly=True, date_range=date_range)
        if baseline_data is not None:
            print(f"  QQQ: {len(baseline_data)} time points")

    return agent_data, baseline_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize U.S. trading metrics over time")
    parser.add_argument("--separate-plots", action="store_true", help="Save each metric as a separate PDF")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    agent_data, baseline_data = load_us_agent_metrics()
    if not agent_data:
        print("No U.S. agent data found.")
        return

    if args.separate_plots:
        market_prefix = "us_market"
        for metric_key, ylabel, title in METRICS:
            plot_single_metric(agent_data, baseline_data, metric_key, ylabel, title, output_dir / f"{market_prefix}_{metric_key.lower()}_metrics.pdf")
    else:
        plot_market_metrics(agent_data, baseline_data, output_dir / "us_market_metrics.pdf")

    print("=" * 70)
    print(f"All plots saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
