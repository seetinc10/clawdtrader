import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.quick_backtest import (
    QuickBacktestResult,
    _derive_bias_lists,
    _select_backtest_window,
    _build_no_history_message,
    _run_prompt_backtest_with_client,
    build_memory_insights,
    build_prompt_block,
)


class QuickBacktestHelperTests(unittest.TestCase):
    class _FakeResponse:
        def __init__(self, content):
            self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]

    class _FakeClient:
        def __init__(self, contents):
            self._contents = list(contents)
            self.calls = []
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = self._contents.pop(0) if self._contents else '{"action": "HOLD", "symbol": "", "quantity": 0, "reason": "done"}'
            return QuickBacktestHelperTests._FakeResponse(content)

    def test_derive_bias_lists_prefers_longs_and_sell_pressure(self):
        trades = [
            {"symbol": "NVDA", "action": "buy", "amount": 3},
            {"symbol": "NVDA", "action": "buy", "amount": 2},
            {"symbol": "TSLA", "action": "sell", "amount": 1},
            {"symbol": "TSLA", "action": "sell", "amount": 1},
            {"symbol": "AVGO", "action": "buy", "amount": 1},
            {"symbol": "AVGO", "action": "sell", "amount": 1},
        ]
        positions = {"NVDA": 5, "TSLA": 0, "AVGO": 0, "CASH": 10_000}

        summary, bullish, caution = _derive_bias_lists(trades, positions)

        self.assertEqual(summary["NVDA"]["buys"], 2)
        self.assertIn("NVDA", bullish)
        self.assertEqual(caution[0], "TSLA")

    def test_build_prompt_block_includes_bias_and_skipped_symbols(self):
        result = QuickBacktestResult(
            success=True,
            message="ok",
            granularity="hour",
            init_date="2026-05-01 10:30:00",
            end_date="2026-05-01 15:30:00",
            processed_points=5,
            trade_count=4,
            final_equity=10_125.0,
            return_pct=1.25,
            bullish_symbols=["NVDA", "AVGO"],
            caution_symbols=["TSLA"],
            skipped_symbols=["NBIS"],
        )

        block = build_prompt_block(result)

        self.assertIn("Bullish bias: NVDA, AVGO", block)
        self.assertIn("Caution bias: TSLA", block)
        self.assertIn("Skipped (insufficient history): NBIS", block)

    def test_build_memory_insights_creates_compact_backtest_rules(self):
        result = QuickBacktestResult(
            success=True,
            message="ok",
            granularity="hour",
            trade_count=5,
            return_pct=-2.1,
            bullish_symbols=["NVDA", "AVGO"],
            caution_symbols=["TSLA", "AMZN"],
        )

        insights = build_memory_insights(result)
        texts = [text for text, _ in insights]

        self.assertTrue(any("favor NVDA, AVGO" in text for text in texts))
        self.assertTrue(any("before buying TSLA, AMZN" in text for text in texts))
        self.assertTrue(any("watchlist conditions were weak" in text for text in texts))

    def test_select_backtest_window_uses_largest_recent_subset(self):
        with TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "merged.jsonl"
            docs = [
                {
                    "Meta Data": {"2. Symbol": "AAA"},
                    "Time Series (60min)": {
                        "2026-05-01 10:30:00": {},
                        "2026-05-01 11:30:00": {},
                        "2026-05-01 12:30:00": {},
                    },
                },
                {
                    "Meta Data": {"2. Symbol": "BBB"},
                    "Time Series (60min)": {
                        "2026-05-01 10:30:00": {},
                        "2026-05-01 11:30:00": {},
                        "2026-05-01 12:30:00": {},
                    },
                },
                {
                    "Meta Data": {"2. Symbol": "CCC"},
                    "Time Series (60min)": {
                        "2026-05-01 11:30:00": {},
                        "2026-05-01 12:30:00": {},
                    },
                },
            ]
            with merged_path.open("w", encoding="utf-8") as handle:
                for doc in docs:
                    handle.write(json.dumps(doc) + "\n")

            granularity, timestamps, used_symbols, skipped_symbols = _select_backtest_window(
                ["AAA", "BBB", "CCC"],
                "us",
                lookback_points=2,
                merged_file=merged_path,
            )

        self.assertEqual(granularity, "hour")
        self.assertEqual(timestamps, ["2026-05-01 10:30:00", "2026-05-01 11:30:00", "2026-05-01 12:30:00"])
        self.assertEqual(used_symbols, ["AAA", "BBB"])
        self.assertEqual(skipped_symbols, ["CCC"])

    def test_build_no_history_message_mentions_missing_local_dataset(self):
        message = _build_no_history_message(
            market="us",
            default_dataset_exists=False,
            fetched_symbols=[],
            failed_symbols=["IREN", "NBIS"],
            fetch_error="Quick backtest skipped: could not fetch recent watchlist history from Yahoo.",
        )

        self.assertIn("Build local history with 'cd data && python get_price_yahoo.py'", message)

    def test_prompt_backtest_fallback_replays_and_returns_summary(self):
        with TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "merged.jsonl"
            docs = [
                {
                    "Meta Data": {"2. Symbol": "AAA"},
                    "Time Series (60min)": {
                        "2026-05-01 10:30:00": {
                            "1. open": "10.0",
                            "2. high": "10.5",
                            "3. low": "9.9",
                            "4. close": "10.0",
                            "5. volume": "1000",
                        },
                        "2026-05-01 11:30:00": {
                            "1. open": "10.2",
                            "2. high": "11.2",
                            "3. low": "10.1",
                            "4. close": "11.0",
                            "5. volume": "1200",
                        },
                        "2026-05-01 12:30:00": {
                            "1. open": "11.1",
                            "2. high": "12.2",
                            "3. low": "11.0",
                            "4. close": "12.0",
                            "5. volume": "1400",
                        },
                    },
                }
            ]
            with merged_path.open("w", encoding="utf-8") as handle:
                for doc in docs:
                    handle.write(json.dumps(doc) + "\n")

            client = self._FakeClient([
                '{"action": "BUY", "symbol": "AAA", "quantity": 5, "reason": "enter"}',
                '{"action": "SELL", "symbol": "AAA", "quantity": 5, "reason": "exit"}',
            ])
            result = _run_prompt_backtest_with_client(
                client=client,
                symbols=["AAA"],
                market="us",
                granularity="hour",
                timestamps=[
                    "2026-05-01 10:30:00",
                    "2026-05-01 11:30:00",
                    "2026-05-01 12:30:00",
                ],
                basemodel="deepseek-reasoner",
                initial_cash=1000.0,
                signature="quickbt-test",
                merged_file=merged_path,
                max_steps=2,
                max_position_size=0.5,
                stop_loss_pct=-3.0,
                take_profit_pct=5.0,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.trade_count, 2)
        self.assertGreater(result.final_equity, 1000.0)
        self.assertIn("lightweight prompt replay", result.message)
        self.assertIn("deepseek-reasoner", result.message)
        self.assertTrue(client.calls)
        self.assertTrue(all(call.get("model") == "deepseek-reasoner" for call in client.calls))


if __name__ == "__main__":
    unittest.main()
