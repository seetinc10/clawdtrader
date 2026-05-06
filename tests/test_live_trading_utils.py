import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.live_trading_utils import enforce_position_limit, parse_ai_decision


class _FakeAccount:
    def __init__(self, portfolio_value):
        self.portfolio_value = portfolio_value


class _FakePosition:
    def __init__(self, symbol, market_value):
        self.symbol = symbol
        self.market_value = market_value


class _FakeApi:
    def __init__(self, portfolio_value, positions):
        self._account = _FakeAccount(portfolio_value)
        self._positions = positions

    def get_account(self):
        return self._account

    def list_positions(self):
        return self._positions


class LiveTradingUtilsTests(unittest.TestCase):
    def test_parse_ai_decision_salvages_json_from_wrapper_text(self):
        content = 'Decision follows:\n{"action":"BUY","symbol":"AAPL","quantity":3,"reason":"setup confirmed"}\nDone.'

        decision = parse_ai_decision(content)

        self.assertEqual(decision["action"], "BUY")
        self.assertEqual(decision["symbol"], "AAPL")
        self.assertEqual(decision["quantity"], 3)

    def test_enforce_position_limit_trims_buy_to_remaining_capacity(self):
        api = _FakeApi(10_000, [_FakePosition("AAPL", 1_500)])
        prices = {"AAPL": {"price": 100.0}}

        adjusted = enforce_position_limit(
            api,
            action="BUY",
            symbol="AAPL",
            quantity=10,
            prices=prices,
            max_position_size=0.20,
        )

        self.assertEqual(adjusted, 5)


if __name__ == "__main__":
    unittest.main()
