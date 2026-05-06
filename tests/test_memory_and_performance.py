import sys
import shutil
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import memory_tools, performance_tracker


class MemorySandboxTestCase(unittest.TestCase):
    def setUp(self):
        scratch_root = PROJECT_ROOT / "tests" / ".tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.root = scratch_root / f"case-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

        self._memory_originals = {
            "DATA_DIR": memory_tools.DATA_DIR,
            "MEMORY_DIR": memory_tools.MEMORY_DIR,
            "STRATEGY_MEMORY_FILE": memory_tools.STRATEGY_MEMORY_FILE,
            "TRADE_OUTCOMES_FILE": memory_tools.TRADE_OUTCOMES_FILE,
            "CONVERSATION_MEMORY_FILE": memory_tools.CONVERSATION_MEMORY_FILE,
        }
        self._tracker_project_root = performance_tracker.project_root

        memory_tools.DATA_DIR = self.root
        memory_tools.MEMORY_DIR = self.root / "memory"
        memory_tools.STRATEGY_MEMORY_FILE = memory_tools.MEMORY_DIR / "strategy_insights.jsonl"
        memory_tools.TRADE_OUTCOMES_FILE = memory_tools.MEMORY_DIR / "trade_outcomes.jsonl"
        memory_tools.CONVERSATION_MEMORY_FILE = memory_tools.MEMORY_DIR / "conversations.jsonl"

        performance_tracker.project_root = self.root

    def tearDown(self):
        for name, value in self._memory_originals.items():
            setattr(memory_tools, name, value)
        performance_tracker.project_root = self._tracker_project_root
        shutil.rmtree(self.root, ignore_errors=True)


class MemoryRankingTests(MemorySandboxTestCase):
    def test_relevant_insights_prefer_symbol_and_positive_track_record(self):
        strong = memory_tools.save_strategy_insight(
            "AAPL breakouts work best after a calm open",
            source="manual",
            tags=["aapl", "entry"],
        )
        weak = memory_tools.save_strategy_insight(
            "TSLA chases are noisy and usually fade",
            source="manual",
            tags=["tsla", "lesson"],
        )

        memory_tools.record_insight_outcome([strong["id"]], won=True)
        ranked = memory_tools.get_relevant_insights(symbols=["AAPL"], query="AAPL breakout setup", n=2)

        self.assertEqual(ranked[0]["id"], strong["id"])
        self.assertEqual(ranked[1]["id"], weak["id"])

    def test_repeated_losses_auto_deactivate_insight(self):
        entry = memory_tools.save_strategy_insight(
            "Average down on every dip",
            source="manual",
            tags=["risk"],
        )

        memory_tools.record_insight_outcome([entry["id"]], won=False, auto_deactivate_after=2)
        memory_tools.record_insight_outcome([entry["id"]], won=False, auto_deactivate_after=2)

        insights = memory_tools.get_strategy_insights(n=5, active_only=False)
        stored = next(item for item in insights if item["id"] == entry["id"])
        self.assertFalse(stored["active"])
        self.assertIn("auto:", stored["deactivation_reason"])


class PerformanceTrackerTests(MemorySandboxTestCase):
    def test_fifo_sell_matching_tracks_partial_consumption(self):
        tracker = performance_tracker.PerformanceTracker("test-live")
        tracker.record_trade("AAPL", "buy", 5, 100.0, "buy the dip")
        tracker.record_trade("AAPL", "buy", 5, 110.0, "follow-through")
        tracker.record_trade("AAPL", "sell", 6, 120.0, "trim winner")

        outcomes = memory_tools.get_trade_outcomes(n=10, symbol="AAPL")
        sells = [item for item in outcomes if item["action"] == "sell"]

        self.assertEqual([item["amount"] for item in sells], [5, 1])
        self.assertTrue(all(item["outcome"] == "win" for item in sells))
        self.assertEqual(tracker.pending_trades["AAPL"][0]["amount"], 4)

        performance_insights = memory_tools.get_strategy_insights(n=10, source="performance")
        self.assertTrue(any("AAPL" in item["insight"] for item in performance_insights))


if __name__ == "__main__":
    unittest.main()
