import sys
import shutil
import types
import unittest
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

from tools import sentiment_temperature as st


class SentimentTemperatureTests(unittest.TestCase):
    def setUp(self):
        scratch_root = PROJECT_ROOT / "tests" / ".tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = scratch_root / f"case-{uuid.uuid4().hex}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.original_cache_dir = st._CACHE_DIR
        self.original_cache_file = st._CACHE_FILE
        self.original_yf_cache_dir = st._YF_CACHE_DIR
        self.original_fetch_pc = st.fetch_put_call_ratio_multi_expiry
        self.original_fetch_vix = st.fetch_vix
        self.original_has_yfinance = st._HAS_YFINANCE
        self.original_yf = getattr(st, "yf", None)

        st._CACHE_DIR = self.cache_dir
        st._CACHE_FILE = self.cache_dir / "temperature_cache.json"
        st._YF_CACHE_DIR = self.cache_dir / "yfinance_cache"

    def tearDown(self):
        st._CACHE_DIR = self.original_cache_dir
        st._CACHE_FILE = self.original_cache_file
        st._YF_CACHE_DIR = self.original_yf_cache_dir
        st.fetch_put_call_ratio_multi_expiry = self.original_fetch_pc
        st.fetch_vix = self.original_fetch_vix
        st._HAS_YFINANCE = self.original_has_yfinance
        if self.original_yf is not None:
            st.yf = self.original_yf
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_pc_ratio_mapping_changes_direction_for_contrarian_mode(self):
        greedy = st.pc_ratio_to_temperature(0.5, invert=False)
        fearful = st.pc_ratio_to_temperature(1.0, invert=False)
        greedy_contrarian = st.pc_ratio_to_temperature(0.5, invert=True)
        fearful_contrarian = st.pc_ratio_to_temperature(1.0, invert=True)

        self.assertGreater(greedy, fearful)
        self.assertLess(greedy_contrarian, fearful_contrarian)

    def test_cached_result_recomputes_temperature_for_requested_mode(self):
        st._save_cache(
            {
                "put_call_ratio": 1.05,
                "ticker_used": "MAGS",
                "sentiment_label": "FEAR",
                "timestamp": datetime.now().isoformat(),
                "source": "live",
            }
        )

        result = st.get_sentiment_temperature(use_contrarian=True, cache_ttl_minutes=60)

        self.assertEqual(result["source"], "cached")
        self.assertEqual(result["mode"], "contrarian")
        self.assertEqual(result["temperature"], st.pc_ratio_to_temperature(1.05, invert=True))

    def test_returns_neutral_fallback_when_no_sentiment_sources_work(self):
        st.fetch_put_call_ratio_multi_expiry = lambda ticker: None
        st.fetch_vix = lambda: None

        result = st.get_sentiment_temperature(force_refresh=True)

        self.assertEqual(result["source"], "default_fallback")
        self.assertEqual(result["temperature"], 0.5)
        self.assertEqual(result["ticker_used"], "NONE")

    def test_vix_mapping_changes_direction_for_contrarian_mode(self):
        calm = st.vix_to_temperature(12, invert=False)
        panic = st.vix_to_temperature(30, invert=False)
        calm_contrarian = st.vix_to_temperature(12, invert=True)
        panic_contrarian = st.vix_to_temperature(30, invert=True)

        self.assertGreater(calm, panic)
        self.assertLess(calm_contrarian, panic_contrarian)

    def test_vix_blends_into_live_temperature_and_can_override_label(self):
        st.fetch_put_call_ratio_multi_expiry = lambda ticker: {
            "put_call_ratio": 0.5,
            "ratio_type": "volume",
            "total_put_oi": 1000,
            "total_call_oi": 2000,
            "total_put_volume": 100,
            "total_call_volume": 200,
            "expirations_aggregated": 2,
        }
        st.fetch_vix = lambda: 31.0

        result = st.get_sentiment_temperature(force_refresh=True, vix_weight=0.5)

        pc_temp = st.pc_ratio_to_temperature(0.5, invert=False)
        vix_temp = st.vix_to_temperature(31.0, invert=False)
        self.assertEqual(result["pc_temperature"], pc_temp)
        self.assertEqual(result["vix_temperature"], vix_temp)
        self.assertEqual(result["temperature"], round((pc_temp + vix_temp) / 2.0, 3))
        self.assertEqual(result["sentiment_label"], "EXTREME_FEAR")
        self.assertEqual(result["vix"], 31.0)

    def test_fetch_vix_recovers_from_corrupted_yfinance_cache(self):
        class FakeSeries:
            def __init__(self, values):
                self.iloc = values

        class FakeFrame:
            empty = False

            def __getitem__(self, key):
                self._last_key = key
                return FakeSeries([19.25, 20.5])

        class FakeTicker:
            calls = 0

            def history(self, period="2d"):
                FakeTicker.calls += 1
                if FakeTicker.calls == 1:
                    raise Exception("database disk image is malformed")
                return FakeFrame()

        class FakeYF:
            def __init__(self):
                self.cache_locations = []

            def set_tz_cache_location(self, path):
                self.cache_locations.append(path)

            def Ticker(self, symbol):
                return FakeTicker()

        fake_yf = FakeYF()
        st._HAS_YFINANCE = True
        st.yf = fake_yf

        value = st.fetch_vix()

        self.assertEqual(value, 20.5)
        self.assertGreaterEqual(FakeTicker.calls, 2)
        self.assertTrue(fake_yf.cache_locations)


if __name__ == "__main__":
    unittest.main()
