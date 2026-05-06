#!/usr/bin/env python3
"""
Clawdbot Live Trader - Real-time AI Trading with Alpaca
Uses GPT to analyze markets and execute trades in real-time
Now with MEMORY - learns from past trades and strategy insights!
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows encoding and buffering
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

load_dotenv()

import alpaca_trade_api as tradeapi
from openai import OpenAI

# Import memory tools
try:
    from tools.memory_tools import (
        get_memory_stats
    )
    from tools.performance_tracker import PerformanceTracker
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False
    print("Warning: Memory tools not available. Running without memory.")

from tools.live_trading_utils import (
    build_trading_prompt,
    enforce_position_limit,
    fetch_current_prices,
    get_memory_section,
    get_portfolio_status,
    request_ai_decision,
)

# Configuration
SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]  # Stocks to trade
TRADE_INTERVAL = 60  # Seconds between trading decisions
MAX_POSITION_SIZE = 0.2  # Max 20% of portfolio in one stock


class LiveTrader:
    def __init__(self):
        # Initialize Alpaca
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Initialize OpenAI
        self.openai = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        )

        self.model = "gpt-4o-mini"

        # Initialize performance tracker for memory
        if MEMORY_ENABLED:
            self.tracker = PerformanceTracker("live-trader")
            print("Memory system enabled - learning from trades!")
        else:
            self.tracker = None

    def log(self, message, tag='info'):
        """Simple logger used by shared trading helpers."""
        print(message)

    def get_account(self):
        """Get account info"""
        return self.api.get_account()

    def get_positions(self):
        """Get current positions"""
        return self.api.list_positions()

    def get_current_prices(self, symbols):
        """Get current prices for symbols"""
        return fetch_current_prices(self.api, symbols, log_fn=self.log)

    def get_market_status(self):
        """Check if market is open"""
        clock = self.api.get_clock()
        return clock.is_open

    def format_portfolio_status(self):
        """Format current portfolio for AI"""
        return get_portfolio_status(self.api)

    def get_ai_decision(self, prices, portfolio_status, cash, buying_power):
        """Get trading decision from AI with memory context"""
        memory_section = get_memory_section(symbols=SYMBOLS if MEMORY_ENABLED else None, log_fn=self.log)
        prompt = build_trading_prompt(
            symbols=SYMBOLS,
            prices=prices,
            portfolio_status=portfolio_status,
            cash=cash,
            buying_power=buying_power,
            max_position_size=MAX_POSITION_SIZE,
            memory_section=memory_section,
            intro="You are Clawdbot, an AI stock trader with memory of past strategies and learnings.",
            rules=[
                f"You can only trade these stocks: {', '.join(SYMBOLS)}",
                f"Maximum position size: {MAX_POSITION_SIZE*100}% of portfolio per stock",
                "Consider risk management - don't go all-in",
                "You can BUY, SELL, or HOLD",
                "IMPORTANT: Apply any relevant strategy insights from your memory!",
            ],
            examples=[
                '{"action": "BUY", "symbol": "AAPL", "quantity": 5, "reason": "Strong momentum, diversifying portfolio"}',
                '{"action": "SELL", "symbol": "TSLA", "quantity": 3, "reason": "Taking profits after 10% gain"}',
                '{"action": "HOLD", "symbol": "", "quantity": 0, "reason": "Market uncertain, preserving capital"}',
            ],
        )

        try:
            return request_ai_decision(
                self.openai,
                self.model,
                prompt,
                temperature=0.3,
                log_fn=self.log,
            )
        except Exception as e:
            self.log(f"AI Error: {e}", 'error')
            return {"action": "HOLD", "symbol": "", "quantity": 0, "reason": f"Error: {e}"}

    def execute_trade(self, decision, prices):
        """Execute the trading decision and track for learning"""
        action = decision.get('action', 'HOLD').upper()
        symbol = decision.get('symbol', '')
        quantity = int(decision.get('quantity', 0))
        reason = decision.get('reason', '')

        if action == 'HOLD' or quantity <= 0:
            self.log(f"  Decision: HOLD - {reason}")
            return None

        # Position size enforcement for buys
        quantity = enforce_position_limit(
            self.api,
            action=action,
            symbol=symbol,
            quantity=quantity,
            prices=prices,
            max_position_size=MAX_POSITION_SIZE,
            log_fn=self.log,
        )
        if quantity is None:
            return None

        try:
            if action == 'BUY':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                self.log(f"  BUY ORDER: {quantity} {symbol} - {reason}")

                # Track for memory/learning
                if self.tracker and symbol in prices:
                    self.tracker.record_trade(
                        symbol=symbol,
                        action="buy",
                        amount=quantity,
                        price=prices[symbol]['price'],
                        reasoning=reason
                    )
                    self.log("  [Memory] Tracking buy for future analysis")

                return order

            elif action == 'SELL':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                self.log(f"  SELL ORDER: {quantity} {symbol} - {reason}")

                # Track for memory/learning
                if self.tracker and symbol in prices:
                    self.tracker.record_trade(
                        symbol=symbol,
                        action="sell",
                        amount=quantity,
                        price=prices[symbol]['price'],
                        reasoning=reason
                    )
                    self.log("  [Memory] Trade outcome recorded")

                return order

        except Exception as e:
            self.log(f"  Trade Error: {e}", 'error')
            return None

    def run(self):
        """Main trading loop"""
        print("=" * 60)
        print("  CLAWDBOT LIVE TRADER")
        print("  Real-time AI Trading with Alpaca Paper Trading")
        print("  Now with MEMORY - Learning from every trade!")
        print("=" * 60)

        # Check account
        account = self.get_account()
        print(f"\nAccount Status: {account.status}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Trading Symbols: {', '.join(SYMBOLS)}")
        print(f"Trade Interval: {TRADE_INTERVAL} seconds")

        # Show memory status
        if MEMORY_ENABLED:
            try:
                stats = get_memory_stats()
                print(f"\nMemory Status:")
                print(f"  Strategy insights: {stats['strategy_insights']}")
                print(f"  Tracked trades: {stats['trade_outcomes']}")
                if stats['win_rate']['total_trades'] > 0:
                    print(f"  Historical win rate: {stats['win_rate']['win_rate']:.1f}%")
            except Exception as e:
                print(f"  Could not load memory stats: {e}")

        print("\n" + "=" * 60)

        trade_count = 0

        while True:
            try:
                now = datetime.now()
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Checking market...")

                # Check market status
                if not self.get_market_status():
                    print("  Market is CLOSED. Waiting...")
                    time.sleep(60)
                    continue

                print("  Market is OPEN")

                # Get current data
                prices = self.get_current_prices(SYMBOLS)
                if not prices:
                    print("  Could not get prices. Waiting...")
                    time.sleep(30)
                    continue

                portfolio_status, cash, buying_power = self.format_portfolio_status()

                # Display current prices
                print("\n  Current Prices:")
                for symbol, data in prices.items():
                    change = ((data['price'] - data['open']) / data['open']) * 100
                    print(f"    {symbol}: ${data['price']:.2f} ({change:+.2f}%)")

                # Get AI decision
                print("\n  Asking AI for trading decision...")
                decision = self.get_ai_decision(prices, portfolio_status, cash, buying_power)

                # Execute trade (pass prices for memory tracking)
                order = self.execute_trade(decision, prices)
                if order:
                    trade_count += 1
                    print(f"  Total trades today: {trade_count}")

                # Show updated portfolio
                print("\n  Portfolio Update:")
                positions = self.get_positions()
                if positions:
                    for pos in positions:
                        print(f"    {pos.symbol}: {pos.qty} shares (${float(pos.market_value):,.2f})")
                else:
                    print("    No positions")

                account = self.get_account()
                print(f"    Cash: ${float(account.cash):,.2f}")
                print(f"    Total Value: ${float(account.portfolio_value):,.2f}")

                # Wait for next interval
                print(f"\n  Waiting {TRADE_INTERVAL} seconds until next check...")
                time.sleep(TRADE_INTERVAL)

            except KeyboardInterrupt:
                print("\n\nStopping trader...")
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(30)


def main():
    print("\nInitializing Clawdbot Live Trader...")

    # Verify credentials
    required = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'OPENAI_API_KEY']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"Error: Missing environment variables: {missing}")
        return

    trader = LiveTrader()

    # Test connection
    try:
        account = trader.get_account()
        print(f"Connected to Alpaca! Account: {account.status}")
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        return

    trader.run()


if __name__ == "__main__":
    main()
