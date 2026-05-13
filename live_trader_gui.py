#!/usr/bin/env python3
"""
Clawdbot Live Trader GUI - Real-time AI Trading with Alpaca
Console window with pause/resume functionality
Now with integrated Strategy Chat!
"""

import os
import sys
import time
import re
import threading
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, ttk

load_dotenv()

import alpaca_trade_api as tradeapi
from openai import OpenAI
from tools.live_trading_utils import (
    enforce_sell_quantity,
    enforce_position_limit,
    enforce_risk_budget,
    extract_trade_theme,
    fetch_current_prices as fetch_live_prices,
    get_memory_section,
    is_non_strategy_exit_reason,
    get_portfolio_status as load_portfolio_status,
    request_role_based_ai_decision,
    validate_setup_scorecard,
    validate_sympathy_trade_confirmation,
)
from tools.alpaca_news import build_watchlist_news_prompt, fetch_news
from tools.price_tools import all_sp_500_symbols
from tools.live_indicators import (
    format_indicators_for_prompt,
    get_atr_for_symbol,
    get_indicator_snapshots,
)
from tools.risk_rails import atr_bracket_prices

# Import memory tools
try:
    from tools.memory_tools import (
        save_strategy_insight,
        format_insights_for_prompt,
        get_memory_stats,
        get_memory_context_for_prompt,
        save_conversation,
        clear_strategy_insights
    )
    from tools.performance_tracker import PerformanceTracker
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False

# Import sentiment temperature
try:
    from tools.sentiment_temperature import get_sentiment_temperature
    SENTIMENT_ENABLED = True
except ImportError:
    SENTIMENT_ENABLED = False

# Configuration
SCANNER_UNIVERSE_NAME = "S&P 500"
SYMBOLS = list(all_sp_500_symbols)
TRADE_INTERVAL = 60                  # legacy fallback (kept for any external readers)
HUNT_INTERVAL_SECONDS = 60           # fast cycle while hunting fresh setups
MONITOR_INTERVAL_SECONDS = 60        # status-only refresh after hunt window (no LLM cost; just polls Alpaca)
HUNT_WINDOW_MINUTES = 15             # max hunt time from session start before auto-latch to monitor
MAX_POSITION_SIZE = 0.2
MAX_THEME_EXPOSURE = 0.4
MAX_RISK_PER_TRADE = 0.005
REENTRY_COOLDOWN_MINUTES = 15
SYMPATHY_MIN_INTRADAY_PCT = 0.75
SYMPATHY_NEAR_HIGH_TOLERANCE_PCT = 0.5
SYMPATHY_MIN_RELVOL = 1.0

# Risk guardrails (deterministic - fire even if the LLM hallucinates)
STOP_LOSS_PCT = -3.0          # auto-sell a position when unrealized P/L hits this
TAKE_PROFIT_PCT = 5.0         # auto-sell at this gain
DAILY_KILL_PCT = -2.0         # pause trading if portfolio drops this much intraday
COOLDOWN_MINUTES = 30         # block re-buys on a symbol for N min after stop-out
FORCE_FLAT_ENABLED = True     # flatten all positions before close
FORCE_FLAT_HHMM_ET = "15:55"  # ET time to flatten

# Quick watchlist backtest
QUICK_BACKTEST_ENABLED = True
QUICK_BACKTEST_LOOKBACK_POINTS = 6
QUICK_BACKTEST_MAX_STEPS = 4
QUICK_BACKTEST_INITIAL_CASH = 10_000.0
QUICK_BACKTEST_LOG_PATH = "./data/quick_backtests"

# Live indicators injected into the LLM prompt (RSI / SMA-slope / ATR / VWAP per watchlist symbol)
ENABLE_LIVE_INDICATORS = True

# Broker-side ATR-scaled bracket orders on every BUY (replaces a polled-only stop)
USE_BRACKET_ORDERS = True
BROKER_ATR_STOP_MULT = 1.5      # stop distance = 1.5x ATR(14)
BROKER_ATR_TP_MULT = 2.5        # take-profit distance = 2.5x ATR(14)

# AI Provider configurations
AI_PROVIDERS = {
    "DeepSeek (deepseek-reasoner)": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_API_BASE",
        "base_url_default": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner"
    },
    "DeepSeek (deepseek-chat)": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_API_BASE",
        "base_url_default": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "OpenAI (gpt-4o-mini)": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_API_BASE",
        "base_url_default": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    },
    "OpenAI (gpt-4o)": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_API_BASE",
        "base_url_default": "https://api.openai.com/v1",
        "model": "gpt-4o"
    },
    "xAI (grok-2)": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_API_BASE",
        "base_url_default": "https://api.x.ai/v1",
        "model": "grok-2-latest"
    },
    "xAI (grok-3)": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_API_BASE",
        "base_url_default": "https://api.x.ai/v1",
        "model": "grok-3"
    },
    "xAI (grok-3-mini)": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_API_BASE",
        "base_url_default": "https://api.x.ai/v1",
        "model": "grok-3-mini"
    },
    "xAI (grok-4)": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_API_BASE",
        "base_url_default": "https://api.x.ai/v1",
        "model": "grok-4"
    }
}


class TradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clawdbot Live Trader")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1a1a2e')

        # State
        self.paused = False
        self.running = False
        self.trade_count = 0
        self._trade_lock = threading.Lock()

        # Chat state
        self.chat_messages = []
        self.chat_processing = False

        # Initialize APIs
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Performance tracker for memory
        if MEMORY_ENABLED:
            self.tracker = PerformanceTracker("live-trader-gui")
        else:
            self.tracker = None

        # Curated watchlist (populated by scanner on first Start)
        self.watchlist = None
        self.quick_backtest_result = None
        self.quick_backtest_prompt_block = ""
        self.quick_backtest_running = False
        self.quick_backtest_thread = None
        self.news_prompt_block = ""
        self.watchlist_scan_map = {}
        self.watchlist_news_symbols = set()

        # Risk-guardrail session state
        self.session_start_value = None      # portfolio value when Start was clicked
        self.session_started_at = None       # wall-clock when Start was clicked (for hunt window)
        self.in_monitor_mode = False         # one-way latch: HUNT -> MONITOR (stays for the session)
        self.cooldowns = {}                  # symbol -> datetime when cooldown ends
        self.reentry_cooldowns = {}          # symbol -> datetime when re-entries are allowed
        self.session_entry_themes = {}       # symbol -> extracted trade theme for this session
        self.kill_switch_tripped = False     # daily DD kill switch
        self.force_flat_done_today = False   # has end-of-day flatten run yet

        # Indicator + bracket state
        self.indicator_snapshots = {}        # symbol -> IndicatorSnapshot (refreshed each loop)
        self.bracketed_symbols = set()       # symbols currently wrapped in a live broker bracket
        self.processed_fill_ids = set()      # closed-order IDs already reconciled (broker-side bracket fills)

        # AI Provider setup - select first available provider
        self.openai = None
        self.model = None
        self.current_provider = None
        self.openai_api_key = None
        self.openai_base_url = None

        # Try to find an available provider
        for provider_name in AI_PROVIDERS.keys():
            config = AI_PROVIDERS[provider_name]
            if os.getenv(config["api_key_env"]):
                self.current_provider = provider_name
                self.set_ai_provider(provider_name)
                break

        self.setup_gui()

    def set_ai_provider(self, provider_name):
        """Set the AI provider and reinitialize the client"""
        if provider_name not in AI_PROVIDERS:
            return False

        config = AI_PROVIDERS[provider_name]
        api_key = os.getenv(config["api_key_env"])
        base_url = os.getenv(config["base_url_env"], config["base_url_default"])

        if not api_key:
            if hasattr(self, 'console'):
                self.log(f"Error: {config['api_key_env']} not found in environment", 'error')
            return False

        self.openai = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.model = config["model"]
        self.current_provider = provider_name

        if hasattr(self, 'model_label'):
            self.model_label.config(text=f"Model: {self.model}")

        # Update chat tab AI label
        if hasattr(self, 'chat_ai_label'):
            self.chat_ai_label.config(text=f"AI: {self.model}")

        if hasattr(self, 'console'):
            self.log(f"Switched to: {provider_name} (model: {self.model})", 'info')

        # Also log in chat if available
        if hasattr(self, 'chat_display'):
            self.chat_log(f"AI Provider changed to: {self.model}", 'system')

        return True

    def on_provider_change(self, event=None):
        """Handle AI provider dropdown change"""
        selected = self.provider_var.get()
        if selected != self.current_provider:
            success = self.set_ai_provider(selected)
            if not success:
                # Revert to previous selection
                self.provider_var.set(self.current_provider)

    def setup_gui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(fill='x', padx=10, pady=10)

        title = tk.Label(title_frame, text="CLAWDBOT LIVE TRADER",
                        font=('Consolas', 20, 'bold'), fg='#00ff88', bg='#1a1a2e')
        title.pack()

        subtitle_text = "Real-time AI Trading with Alpaca Paper Trading"
        if MEMORY_ENABLED:
            subtitle_text += " | Memory Enabled"
        subtitle = tk.Label(title_frame, text=subtitle_text,
                           font=('Consolas', 10), fg='#888888', bg='#1a1a2e')
        subtitle.pack()

        # Status bar
        status_frame = tk.Frame(self.root, bg='#16213e')
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_label = tk.Label(status_frame, text="Status: STOPPED",
                                     font=('Consolas', 12, 'bold'), fg='#ff6b6b', bg='#16213e')
        self.status_label.pack(side='left', padx=10, pady=5)

        self.portfolio_label = tk.Label(status_frame, text="Portfolio: $0.00",
                                        font=('Consolas', 12), fg='#4ecdc4', bg='#16213e')
        self.portfolio_label.pack(side='right', padx=10, pady=5)

        self.cash_label = tk.Label(status_frame, text="Cash: $0.00",
                                   font=('Consolas', 12), fg='#ffe66d', bg='#16213e')
        self.cash_label.pack(side='right', padx=10, pady=5)

        # P&L and Win Rate bar
        pl_frame = tk.Frame(self.root, bg='#16213e')
        pl_frame.pack(fill='x', padx=10, pady=2)

        self.daily_pl_label = tk.Label(pl_frame, text="Daily P/L: $0.00",
                                        font=('Consolas', 12, 'bold'), fg='#00ff88', bg='#16213e')
        self.daily_pl_label.pack(side='left', padx=10, pady=5)

        self.winrate_label = tk.Label(pl_frame, text="Win Rate: --",
                                       font=('Consolas', 12), fg='#4ecdc4', bg='#16213e')
        self.winrate_label.pack(side='right', padx=10, pady=5)

        # Sentiment Temperature indicator
        sentiment_frame = tk.Frame(self.root, bg='#16213e')
        sentiment_frame.pack(fill='x', padx=10, pady=2)

        temp_icon = tk.Label(sentiment_frame, text="SENTIMENT",
                             font=('Consolas', 9, 'bold'), fg='#888888', bg='#16213e')
        temp_icon.pack(side='left', padx=(10, 5), pady=5)

        self.sentiment_label = tk.Label(sentiment_frame, text="--",
                                         font=('Consolas', 12, 'bold'), fg='#888888', bg='#16213e')
        self.sentiment_label.pack(side='left', padx=5, pady=5)

        self.temp_value_label = tk.Label(sentiment_frame, text="Temp: --",
                                          font=('Consolas', 11), fg='#888888', bg='#16213e')
        self.temp_value_label.pack(side='left', padx=10, pady=5)

        self.pc_ratio_label = tk.Label(sentiment_frame, text="P/C: --",
                                        font=('Consolas', 11), fg='#888888', bg='#16213e')
        self.pc_ratio_label.pack(side='left', padx=10, pady=5)

        self.temp_bar_canvas = tk.Canvas(sentiment_frame, width=200, height=20,
                                          bg='#0f0f23', highlightthickness=1,
                                          highlightbackground='#333333')
        self.temp_bar_canvas.pack(side='left', padx=10, pady=5)

        self.sentiment_source_label = tk.Label(sentiment_frame, text="",
                                                font=('Consolas', 9), fg='#555555', bg='#16213e')
        self.sentiment_source_label.pack(side='right', padx=10, pady=5)

        # Initial sentiment fetch
        if SENTIMENT_ENABLED:
            threading.Thread(target=self.update_sentiment_display, daemon=True).start()

        # AI Provider selector
        ai_frame = tk.Frame(self.root, bg='#1a1a2e')
        ai_frame.pack(fill='x', padx=10, pady=5)

        ai_label = tk.Label(ai_frame, text="AI Provider:",
                           font=('Consolas', 11), fg='#ffffff', bg='#1a1a2e')
        ai_label.pack(side='left', padx=5)

        self.provider_var = tk.StringVar(value=self.current_provider)
        self.provider_dropdown = ttk.Combobox(ai_frame, textvariable=self.provider_var,
                                               values=list(AI_PROVIDERS.keys()),
                                               state='readonly', width=25,
                                               font=('Consolas', 10))
        self.provider_dropdown.pack(side='left', padx=5)
        self.provider_dropdown.bind('<<ComboboxSelected>>', self.on_provider_change)

        # Style the combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='#16213e', background='#16213e',
                       foreground='#00ff88', arrowcolor='#00ff88')

        self.model_label = tk.Label(ai_frame, text=f"Model: {self.model}",
                                    font=('Consolas', 10), fg='#4ecdc4', bg='#1a1a2e')
        self.model_label.pack(side='left', padx=15)

        # Control buttons
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(fill='x', padx=10, pady=10)

        self.start_btn = tk.Button(btn_frame, text="START", command=self.start_trading,
                                   font=('Consolas', 12, 'bold'), bg='#00ff88', fg='#000000',
                                   width=12, height=2, cursor='hand2')
        self.start_btn.pack(side='left', padx=5)

        self.pause_btn = tk.Button(btn_frame, text="PAUSE", command=self.toggle_pause,
                                   font=('Consolas', 12, 'bold'), bg='#ffe66d', fg='#000000',
                                   width=12, height=2, cursor='hand2', state='disabled')
        self.pause_btn.pack(side='left', padx=5)

        self.stop_btn = tk.Button(btn_frame, text="STOP", command=self.stop_trading,
                                  font=('Consolas', 12, 'bold'), bg='#ff6b6b', fg='#000000',
                                  width=12, height=2, cursor='hand2', state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        # Trade count
        self.trade_label = tk.Label(btn_frame, text="Trades: 0",
                                    font=('Consolas', 12), fg='#ffffff', bg='#1a1a2e')
        self.trade_label.pack(side='right', padx=10)

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1a1a2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#16213e', foreground='#00ff88',
                       padding=[15, 8], font=('Consolas', 11, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#0f0f23')],
                 foreground=[('selected', '#00ff88')])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # === TRADING TAB ===
        trading_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(trading_frame, text='  Trading  ')

        self.console = scrolledtext.ScrolledText(trading_frame,
                                                  font=('Consolas', 10),
                                                  bg='#0f0f23', fg='#00ff88',
                                                  insertbackground='#00ff88',
                                                  wrap='word')
        self.console.pack(fill='both', expand=True)

        # Configure tags for colors
        self.console.tag_configure('info', foreground='#00ff88')
        self.console.tag_configure('warning', foreground='#ffe66d')
        self.console.tag_configure('error', foreground='#ff6b6b')
        self.console.tag_configure('trade', foreground='#4ecdc4')
        self.console.tag_configure('price', foreground='#ffffff')
        self.console.tag_configure('header', foreground='#00ff88', font=('Consolas', 10, 'bold'))

        self.log("=" * 60, 'header')
        self.log("  CLAWDBOT LIVE TRADER INITIALIZED", 'header')
        self.log("=" * 60, 'header')
        self.log(f"AI Provider: {self.current_provider}", 'info')
        self.log(
            f"Universe: {SCANNER_UNIVERSE_NAME} ({len(SYMBOLS)} symbols, will curate top-10 watchlist on Start)",
            'info'
        )
        self.log(f"Trade Interval: {TRADE_INTERVAL} seconds", 'info')
        if MEMORY_ENABLED:
            self.log("Memory System: ENABLED", 'info')
        self.log("Click START to begin trading\n", 'info')

        # === STRATEGY CHAT TAB ===
        self.setup_chat_tab()

        # Update account info
        self.update_account_display()

        # Start background account updater (every 5 seconds)
        self.account_update_thread = threading.Thread(target=self.account_update_loop, daemon=True)
        self.account_update_thread.start()

    def _run_on_ui_thread(self, callback, *args, **kwargs):
        """Marshal widget updates back to Tk's main thread."""
        try:
            if threading.current_thread() is threading.main_thread():
                callback(*args, **kwargs)
            else:
                self.root.after(0, lambda: callback(*args, **kwargs))
        except (RuntimeError, tk.TclError):
            # The window may be closing while a background thread is still running.
            pass

    def _format_market_time(self, value):
        """Render Alpaca clock timestamps safely for status/log messages."""
        if value is None:
            return "?"
        try:
            if hasattr(value, "astimezone"):
                return value.astimezone().strftime('%I:%M %p')
            if hasattr(value, "strftime"):
                return value.strftime('%I:%M %p')
        except Exception:
            pass
        return str(value)

    def _append_console_log(self, timestamp, message, tag):
        self.console.insert('end', f"[{timestamp}] {message}\n", tag)
        self.console.see('end')

    def log(self, message, tag='info'):
        """Log message to console"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._run_on_ui_thread(self._append_console_log, timestamp, message, tag)

    def _render_account_display(self, cash, portfolio, total_pl, winners, losers):
        self.cash_label.config(text=f"Cash: ${cash:,.2f}")
        self.portfolio_label.config(text=f"Portfolio: ${portfolio:,.2f}")

        if winners is None or losers is None or total_pl is None:
            self.daily_pl_label.config(text="Daily P/L: $0.00", fg='#888888')
            self.winrate_label.config(text="Win Rate: -- (no positions)")
            return

        total = winners + losers
        win_rate = (winners / total * 100) if total > 0 else 0
        sign = "+" if total_pl >= 0 else ""
        pl_color = '#00ff88' if total_pl >= 0 else '#ff6b6b'
        self.daily_pl_label.config(text=f"Daily P/L: {sign}${total_pl:,.2f}", fg=pl_color)
        self.winrate_label.config(text=f"Win Rate: {win_rate:.0f}% ({winners}W/{losers}L)")

    def update_account_display(self):
        """Update account info display including P&L and win rate"""
        try:
            account = self.api.get_account()
            cash = float(account.cash)
            portfolio = float(account.portfolio_value)

            # Update daily P/L and win rate
            positions = self.api.list_positions()
            if positions:
                total_pl = sum(float(p.unrealized_pl) for p in positions)
                winners = sum(1 for p in positions if float(p.unrealized_pl) > 0)
                losers = sum(1 for p in positions if float(p.unrealized_pl) < 0)
            else:
                total_pl = None
                winners = None
                losers = None

            self._run_on_ui_thread(
                self._render_account_display,
                cash,
                portfolio,
                total_pl,
                winners,
                losers,
            )
        except Exception:
            pass  # Silently fail for background updates

    def account_update_loop(self):
        """Background loop to update account info every 5 seconds"""
        consecutive_failures = 0
        sentiment_counter = 0
        while True:
            try:
                self.update_account_display()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    try:
                        self.root.after(0, lambda: self.status_label.config(
                            text="Status: API ERROR", fg='#ff6b6b'))
                    except Exception:
                        pass
            # Update sentiment every ~60 seconds (12 * 5s ticks)
            sentiment_counter += 1
            if sentiment_counter >= 12 and SENTIMENT_ENABLED:
                sentiment_counter = 0
                try:
                    self.update_sentiment_display()
                except Exception:
                    pass
            time.sleep(5)

    def update_sentiment_display(self):
        """Fetch sentiment temperature and update the GUI indicator"""
        if not SENTIMENT_ENABLED:
            return
        try:
            data = get_sentiment_temperature(force_refresh=True)
            temp = data.get("temperature", 0.5)
            pc_ratio = data.get("put_call_ratio", 0.7)
            sentiment = data.get("sentiment_label", "NEUTRAL")
            regime_used = data.get("regime_used") or data.get("ticker_used", "?")
            vix_value = data.get("vix")
            source = data.get("source", "?")

            # Color mapping based on sentiment
            colors = {
                "EXTREME_FEAR": "#ff0000",
                "FEAR": "#ff6b6b",
                "NEUTRAL": "#ffe66d",
                "GREED": "#00cc66",
                "EXTREME_GREED": "#00ff88",
            }
            color = colors.get(sentiment, "#888888")

            # Bar gradient: red (left/cold) -> yellow (mid) -> green (right/hot)
            def temp_to_bar_color(t):
                if t < 0.5:
                    r = 255
                    g = int(255 * (t / 0.5))
                    b = 0
                else:
                    r = int(255 * ((1.0 - t) / 0.5))
                    g = 255
                    b = 0
                return f"#{r:02x}{g:02x}{b:02x}"

            bar_color = temp_to_bar_color(temp)

            # Schedule GUI updates on main thread
            def _update():
                self.sentiment_label.config(text=sentiment, fg=color)
                self.temp_value_label.config(text=f"Temp: {temp:.3f}", fg=color)
                vix_text = "--" if vix_value is None else f"{float(vix_value):.2f}"
                self.pc_ratio_label.config(text=f"P/C: {pc_ratio:.3f}  VIX: {vix_text}", fg='#ffffff')
                self.sentiment_source_label.config(text=f"{regime_used} | {source}")

                # Draw temperature bar
                self.temp_bar_canvas.delete("all")
                bar_width = int(temp * 196)
                # Background
                self.temp_bar_canvas.create_rectangle(2, 2, 198, 18, fill='#1a1a2e', outline='')
                # Filled portion
                if bar_width > 0:
                    self.temp_bar_canvas.create_rectangle(2, 2, 2 + bar_width, 18,
                                                           fill=bar_color, outline='')
                # Tick marks at 0.25, 0.5, 0.75
                for tick in [0.25, 0.5, 0.75]:
                    x = 2 + int(tick * 196)
                    self.temp_bar_canvas.create_line(x, 2, x, 18, fill='#444444', width=1)

            self.root.after(0, _update)

        except Exception as e:
            self.root.after(0, lambda: self.sentiment_label.config(text="ERROR", fg='#ff6b6b'))

    def start_trading(self):
        """Start the trading loop"""
        self.running = True
        self.paused = False
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        self.provider_dropdown.config(state='disabled')
        self.status_label.config(text="Status: RUNNING", fg='#00ff88')

        self.log("\n" + "=" * 40, 'header')
        self.log("  TRADING STARTED", 'header')
        self.log("=" * 40 + "\n", 'header')

        threading.Thread(target=self._bootstrap_trading_session, daemon=True).start()

    def _bootstrap_trading_session(self):
        """Run startup prep before the live trading loop begins."""
        # Run scanner once per session to curate the watchlist.
        if self.watchlist is None:
            self._run_scanner()
            self._refresh_watchlist_news()
            self._start_quick_backtest()
        else:
            if not self.news_prompt_block:
                self._refresh_watchlist_news()
            if self.quick_backtest_result is None and not self.quick_backtest_running:
                self._start_quick_backtest()

        self._wait_for_quick_backtest(timeout_seconds=90)

        # Snapshot starting equity for the daily-DD kill switch.
        try:
            self.session_start_value = float(self.api.get_account().portfolio_value)
            self.session_started_at = datetime.now()
            self.in_monitor_mode = False
            self.kill_switch_tripped = False
            self.force_flat_done_today = False
            self.cooldowns.clear()
            self.reentry_cooldowns.clear()
            self.session_entry_themes.clear()
            # Seed the fill-reconciler with current closed-order IDs so we only act on NEW fills from here on
            try:
                seed_orders = self.api.list_orders(status='closed', limit=100)
                self.processed_fill_ids = {o.id for o in seed_orders if hasattr(o, 'id') and o.id}
                self.log(
                    f"Broker fill reconciler seeded with {len(self.processed_fill_ids)} prior closed order(s).",
                    'info'
                )
            except Exception as exc:
                self.processed_fill_ids = set()
                self.log(f"Could not seed broker fill reconciler: {exc}", 'warning')
            self.log(
                f"Session start equity: ${self.session_start_value:,.2f}  "
                f"| stop-loss {STOP_LOSS_PCT}%  take-profit +{TAKE_PROFIT_PCT}%  "
                f"daily kill {DAILY_KILL_PCT}%  cooldown {COOLDOWN_MINUTES}m  "
                f"re-entry {REENTRY_COOLDOWN_MINUTES}m  "
                f"risk/trade {MAX_RISK_PER_TRADE*100:.2f}%  "
                f"theme cap {MAX_THEME_EXPOSURE*100:.0f}%  "
                f"force-flat {FORCE_FLAT_HHMM_ET if FORCE_FLAT_ENABLED else 'OFF'} ET",
                'info'
            )
        except Exception as e:
            self.log(f"Could not snapshot starting equity: {e}", 'warning')

        self.trade_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trade_thread.start()

    def _wait_for_quick_backtest(self, timeout_seconds: int = 90):
        """Hold the first live trade until the startup backtest finishes or times out."""
        if not QUICK_BACKTEST_ENABLED or not self.quick_backtest_running:
            return

        deadline = time.time() + max(1, int(timeout_seconds))
        self.log("Waiting for quick backtest before the first live decision...", 'info')

        while self.running and self.quick_backtest_running and time.time() < deadline:
            time.sleep(0.25)

        if self.quick_backtest_running:
            self.log(
                f"Quick backtest is still running after {timeout_seconds}s; proceeding without it for now.",
                'warning'
            )
        elif self.quick_backtest_prompt_block.strip():
            self.log("Startup quick backtest is ready. Live trading will include that context.", 'info')

    def _run_scanner(self):
        """Scan the full universe and pick the top-10 movers as today's watchlist."""
        try:
            from tools.scanner import scan_top_movers
            self.log(f"Scanning {SCANNER_UNIVERSE_NAME} universe ({len(SYMBOLS)} symbols)...", 'info')
            top = scan_top_movers(self.api, SYMBOLS, top_n=10)
            if top:
                self.watchlist = [r['symbol'] for r in top]
                self.watchlist_scan_map = {
                    r['symbol']: {
                        "price": r.get("price"),
                        "pct_change": r.get("pct_change"),
                        "rel_vol": r.get("rel_vol"),
                        "score": r.get("score"),
                    }
                    for r in top if r.get("symbol")
                }
                self.log(f"Watchlist: {', '.join(self.watchlist)}", 'info')
                for r in top:
                    self.log(
                        f"  {r['symbol']:6s} ${r['price']:>8.2f}  "
                        f"{r['pct_change']:+6.2f}%  relvol={r['rel_vol']:.2f}x  "
                        f"score={r['score']:.1f}",
                        'info'
                    )
            else:
                self.log("Scanner returned nothing - falling back to the first 10 S&P 500 symbols.", 'warning')
                self.watchlist = SYMBOLS[:10]
                self.watchlist_scan_map = {}
        except Exception as e:
            self.log(f"Scanner failed: {e} - using the first 10 S&P 500 symbols.", 'error')
            self.watchlist = SYMBOLS[:10]
            self.watchlist_scan_map = {}

    def _start_quick_backtest(self):
        """Run a short watchlist-only backtest in the background."""
        if not QUICK_BACKTEST_ENABLED or not self.watchlist:
            return
        if self.quick_backtest_thread and self.quick_backtest_thread.is_alive():
            return
        if not self.model or not self.openai_api_key:
            self.log("Quick backtest skipped: no active AI provider/API key.", 'warning')
            return

        self.quick_backtest_result = None
        self.quick_backtest_prompt_block = ""
        self.quick_backtest_running = True
        self.log(
            f"Starting quick backtest on today's {len(self.watchlist)}-name watchlist "
            f"({QUICK_BACKTEST_LOOKBACK_POINTS} recent points, {QUICK_BACKTEST_MAX_STEPS} max steps)...",
            'info'
        )
        self.log(f"Quick backtest model: {self.model}", 'info')
        self.quick_backtest_thread = threading.Thread(target=self._run_quick_backtest, daemon=True)
        self.quick_backtest_thread.start()

    def _refresh_watchlist_news(self):
        """Fetch and summarize recent Alpaca/Benzinga news for the scanned watchlist."""
        if not self.watchlist:
            self.news_prompt_block = ""
            self.watchlist_news_symbols = set()
            return

        self.news_prompt_block = ""
        self.watchlist_news_symbols = set()
        self.log(
            f"Fetching Alpaca news confirmation for today's watchlist ({len(self.watchlist)} symbols)...",
            'info'
        )
        try:
            articles = fetch_news(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                self.watchlist,
                limit=20,
                lookback_hours=36,
                include_content=False,
            )
            watchlist_set = {symbol.upper() for symbol in self.watchlist}
            self.watchlist_news_symbols = {
                str(symbol).strip().upper()
                for article in articles
                for symbol in (article.get("symbols") or [])
                if isinstance(symbol, str) and str(symbol).strip().upper() in watchlist_set
            }
            self.news_prompt_block = build_watchlist_news_prompt(self.watchlist, articles)
            article_count = len(articles)
            self.log(f"Startup news check complete: {article_count} Alpaca article(s) fetched.", 'info')
            if self.news_prompt_block:
                self.log(self.news_prompt_block.rstrip(), 'info')
        except Exception as e:
            self.news_prompt_block = ""
            self.watchlist_news_symbols = set()
            self.log(f"Startup news check failed: {e}", 'warning')

    def _run_quick_backtest(self):
        try:
            from tools.quick_backtest import run_quick_backtest

            result = run_quick_backtest(
                symbols=self.watchlist or [],
                basemodel=self.model,
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
                market="us",
                lookback_points=QUICK_BACKTEST_LOOKBACK_POINTS,
                max_steps=QUICK_BACKTEST_MAX_STEPS,
                initial_cash=QUICK_BACKTEST_INITIAL_CASH,
                log_path=QUICK_BACKTEST_LOG_PATH,
                max_position_size=MAX_POSITION_SIZE,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
                max_risk_fraction=MAX_RISK_PER_TRADE,
            )
            self.quick_backtest_result = result

            if not result.success:
                self.quick_backtest_prompt_block = ""
                self.log(result.message, 'warning')
                return

            self.quick_backtest_prompt_block = result.prompt_block
            if result.message and result.message != f"Quick backtest completed with {self.model}.":
                self.log(result.message, 'info')
            self.log("Quick backtest complete. Live AI will use this as a weak prior:", 'info')
            if result.prompt_block:
                self.log(result.prompt_block, 'info')
            if result.saved_insights:
                self.log(f"Saved {len(result.saved_insights)} quick-backtest memory insight(s):", 'info')
                for insight in result.saved_insights:
                    self.log(f"  - {insight}", 'info')
                self.update_memory_stats()
        except Exception as e:
            self.quick_backtest_prompt_block = ""
            self.log(f"Quick backtest error: {e}", 'error')
        finally:
            self.quick_backtest_running = False

    def _compose_memory_section(self, symbols):
        """Combine persistent memory with quick-backtest context when available."""
        memory_section = get_memory_section(symbols=symbols, log_fn=self.log)
        blocks = []
        if memory_section.strip():
            blocks.append(memory_section.rstrip())
        if self.news_prompt_block.strip():
            blocks.append(self.news_prompt_block.rstrip())
        if self.quick_backtest_prompt_block.strip():
            blocks.append(self.quick_backtest_prompt_block.rstrip())
        tech_block = self._compose_technical_section()
        if tech_block.strip():
            blocks.append(tech_block.rstrip())
        if not blocks:
            return ""
        return "\n\n".join(blocks) + "\n"

    def _refresh_indicator_snapshots(self, symbols):
        """Pull RSI/SMA/ATR/VWAP for the watchlist. Cached on the instance per cycle."""
        if not ENABLE_LIVE_INDICATORS or not symbols:
            self.indicator_snapshots = {}
            return
        try:
            self.indicator_snapshots = get_indicator_snapshots(
                self.api,
                list(symbols),
                include_intraday=True,
                log_fn=self.log,
            )
        except Exception as exc:
            self.log(f"Indicator fetch failed: {exc}", 'warning')
            self.indicator_snapshots = {}

    def _compose_technical_section(self):
        """Format cached indicator snapshots for prompt injection. Empty if none cached."""
        if not self.indicator_snapshots:
            return ""
        return format_indicators_for_prompt(self.indicator_snapshots)

    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="RESUME", bg='#00ff88')
            self.status_label.config(text="Status: PAUSED", fg='#ffe66d')
            self.log("\n*** AI TRADING PAUSED ***\n", 'warning')
        else:
            self.pause_btn.config(text="PAUSE", bg='#ffe66d')
            self.status_label.config(text="Status: RUNNING", fg='#00ff88')
            self.log("\n*** AI TRADING RESUMED ***\n", 'info')

    def stop_trading(self):
        """Stop trading and liquidate all positions"""
        self.running = False
        self.paused = False
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='disabled', text="PAUSE", bg='#ffe66d')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: LIQUIDATING...", fg='#ff6b6b')

        self.log("\n" + "=" * 40, 'error')
        self.log("  STOPPING - LIQUIDATING ALL POSITIONS", 'error')
        self.log("=" * 40, 'error')

        # Liquidate in separate thread to not freeze GUI
        threading.Thread(target=self.liquidate_all_positions, daemon=True).start()
        # Run end-of-session reflection (C) - distill today's trades into rules
        if MEMORY_ENABLED and self.openai and self.model:
            threading.Thread(target=self._run_reflection, daemon=True).start()

    def _run_reflection(self):
        """End-of-session reflection: ask the LLM to distill rules from today's trades."""
        try:
            from tools.memory_tools import reflect_on_session
            self.log("Reflecting on today's trades...", 'info')
            rules = reflect_on_session(self.openai, self.model)
            if rules:
                self.log(f"Saved {len(rules)} reflection insight(s):", 'info')
                for r in rules:
                    self.log(f"  - {r}", 'info')
            else:
                self.log("No trades to reflect on (or reflection produced no rules).", 'warning')
        except Exception as e:
            self.log(f"Reflection failed: {e}", 'error')

    def liquidate_all_positions(self):
        """Sell all positions"""
        try:
            positions = self.api.list_positions()

            if not positions:
                self.log("No positions to liquidate", 'info')
            else:
                for pos in positions:
                    try:
                        symbol = pos.symbol
                        qty = float(pos.qty)  # Keep as float for fractional shares
                        self.log(f"Selling {qty} {symbol}...", 'trade')

                        sell_price = float(pos.current_price)
                        # Cancel any open GTC bracket legs first so they don't
                        # double-fire after we manually flatten the position.
                        self._cancel_bracket_legs(symbol)
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=str(qty),  # Pass as string to Alpaca for fractional share support
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        self.log(f"  SOLD {qty} {symbol}", 'trade')
                        with self._trade_lock:
                            self.trade_count += 1

                        # Track sell for win/loss calculation
                        if self.tracker:
                            self.tracker.record_trade(symbol, "sell", qty,
                                                      sell_price, "Manual liquidation")
                        self.bracketed_symbols.discard(symbol)

                    except Exception as e:
                        self.log(f"  Error selling {pos.symbol}: {e}", 'error')

            # Wait for orders to settle
            time.sleep(2)

            # Update display
            self.update_account_display()
            self.update_trade_count_display()

            self.log("\n*** ALL POSITIONS LIQUIDATED ***", 'warning')
            self.log("*** TRADING STOPPED ***\n", 'error')

        except Exception as e:
            self.log(f"Liquidation error: {e}", 'error')

        finally:
            self._run_on_ui_thread(self._set_stopped_state)

    def get_current_prices(self):
        """Get current prices for the curated watchlist (falls back to full universe)."""
        symbols = self.watchlist or SYMBOLS
        return fetch_live_prices(self.api, symbols, log_fn=self.log)

    def get_portfolio_status(self):
        """Get portfolio status"""
        return load_portfolio_status(self.api)

    def get_ai_decision(self, prices, portfolio_status, cash, buying_power):
        """Get AI trading decision with memory context"""
        memory_section = self._compose_memory_section(self.watchlist or SYMBOLS)
        return request_role_based_ai_decision(
            self.openai,
            self.model,
            symbols=self.watchlist or SYMBOLS,
            prices=prices,
            portfolio_status=portfolio_status,
            cash=cash,
            buying_power=buying_power,
            max_position_size=MAX_POSITION_SIZE,
            memory_section=memory_section,
            intro="You are Clawdbot, an AI stock trader with memory of past strategies. The rules below are enforced automatically, so do not fight them.",
            rules=[
                f"You can only trade these stocks: {', '.join(self.watchlist or SYMBOLS)}",
                "Long-only book: SELL only to reduce or close an existing long position, never open a short.",
                f"Maximum position size: {MAX_POSITION_SIZE*100}% of portfolio per stock (oversized buys are auto-trimmed)",
                f"Risk per trade is capped at {MAX_RISK_PER_TRADE*100:.2f}% of portfolio using the stop-loss distance",
                f"Stop-loss is automatic at {STOP_LOSS_PCT}% - do not issue a SELL just to cut a loser early; the system handles it",
                f"Take-profit is automatic at +{TAKE_PROFIT_PCT}% - do not sell just to lock a small winner; the system handles it",
                f"After a stop-out, that symbol is on a {COOLDOWN_MINUTES}-minute cooldown and buys will be rejected",
                "Recent live setup scorecards can pause weak patterns; paused BUY setups will be rejected automatically",
                f"Daily kill switch flattens everything if portfolio is down {DAILY_KILL_PCT}% on the day",
                f"Force-flat at {FORCE_FLAT_HHMM_ET} ET - do not open new positions in the last 30 minutes",
                (
                    "Sympathy trades need direct target confirmation: if your reason cites another ticker, "
                    "the target symbol itself must be green, near highs, and have its own catalyst or strong volume."
                ),
                (
                    "Read the TECHNICAL CONTEXT block (RSI, SMA5-vs-SMA20 slope, ATR%, price-vs-VWAP, trend label) "
                    "before BUY/HOLD: prefer BULL/BULL_TREND with RSI < 70 and price above VWAP; "
                    "be cautious of BEAR/BEAR_TREND or OVERBOUGHT readings."
                ),
                "You can BUY, SELL, or HOLD",
                "Apply any relevant strategy insights from your memory!",
            ],
            log_fn=self.log,
        )

    # ------------------------------------------------------------------
    # Risk guardrails
    # ------------------------------------------------------------------
    def _now_et(self):
        """Current time in US/Eastern. Falls back to local if zoneinfo missing."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            return datetime.now()

    def _on_cooldown(self, symbol: str) -> bool:
        until = self.cooldowns.get(symbol)
        if not until:
            return False
        if datetime.now() >= until:
            self.cooldowns.pop(symbol, None)
            return False
        return True

    def _on_reentry_cooldown(self, symbol: str) -> bool:
        until = self.reentry_cooldowns.get(symbol)
        if not until:
            return False
        if datetime.now() >= until:
            self.reentry_cooldowns.pop(symbol, None)
            return False
        return True

    def _reconcile_broker_fills(self):
        """Detect bracket-leg fills at Alpaca and record them with the real exit price.

        We poll closed orders, identify SELL fills with order_type in
        {stop, stop_limit, limit, trailing_stop} on symbols we've bracketed,
        and:
          - record the exit to PerformanceTracker at the actual filled_avg_price
          - set a stop-out cooldown ONLY if it was a stop-loss leg (not take-profit)
          - clear the bracketed_symbols entry
        Dedup by order ID via self.processed_fill_ids.
        """
        try:
            orders = self.api.list_orders(status='closed', limit=100, direction='desc')
        except Exception as exc:
            self.log(f"Broker fill reconcile failed: {exc}", 'warning')
            return

        seen_now = []
        for order in orders or []:
            order_id = getattr(order, 'id', None)
            if not order_id:
                continue
            seen_now.append(order_id)
            if order_id in self.processed_fill_ids:
                continue

            status = (getattr(order, 'status', '') or '').lower()
            side = (getattr(order, 'side', '') or '').lower()
            order_type = (getattr(order, 'order_type', '') or getattr(order, 'type', '') or '').lower()
            symbol = getattr(order, 'symbol', '')

            # Only filled SELLs on bracket-leg order types interest us
            if status != 'filled' or side != 'sell' or order_type not in (
                'stop', 'stop_limit', 'limit', 'trailing_stop'
            ):
                self.processed_fill_ids.add(order_id)
                continue

            if not symbol or symbol not in self.bracketed_symbols:
                # Either we never bracketed it (manual sell already recorded)
                # or it's not ours. Either way, don't double-record.
                self.processed_fill_ids.add(order_id)
                continue

            try:
                fill_price = float(getattr(order, 'filled_avg_price', 0) or 0)
                qty = int(float(getattr(order, 'filled_qty', 0) or 0))
            except (TypeError, ValueError):
                self.processed_fill_ids.add(order_id)
                continue
            if qty <= 0 or fill_price <= 0:
                self.processed_fill_ids.add(order_id)
                continue

            leg_type = "stop-loss" if order_type in ('stop', 'stop_limit', 'trailing_stop') else "take-profit"
            self.log(
                f"BROKER {leg_type.upper()} FILL: {qty} {symbol} @ ${fill_price:.2f} (recording exit)",
                'trade'
            )
            if self.tracker:
                try:
                    self.tracker.record_trade(symbol, "sell", qty, fill_price, f"broker {leg_type} fill")
                    self.log("[Memory] Recorded broker bracket exit", 'info')
                except Exception as exc:
                    self.log(f"[Memory] Could not record broker exit for {symbol}: {exc}", 'warning')

            self.bracketed_symbols.discard(symbol)
            self.session_entry_themes.pop(symbol, None)

            # Only the stop-loss leg should set the harsher 30m cooldown.
            # Take-profit is a *good* exit; we just rely on the 15m re-entry cooldown.
            if leg_type == "stop-loss":
                self.cooldowns[symbol] = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
                self.log(f"Cooldown {COOLDOWN_MINUTES}m on {symbol} after broker stop-loss fill", 'warning')

            with self._trade_lock:
                self.trade_count += 1
            self.update_trade_count_display()
            self.processed_fill_ids.add(order_id)

        # Keep the dedup set bounded
        if len(self.processed_fill_ids) > 1000:
            keep = set(seen_now[:500])
            self.processed_fill_ids = keep

    def _get_theme_exposure(self, candidate_symbol: str, candidate_value: float, candidate_reason: str) -> tuple[Optional[str], float]:
        """Estimate current-session exposure for the candidate's theme."""
        theme = extract_trade_theme(candidate_symbol, candidate_reason)
        if not theme:
            return None, 0.0

        current_value = 0.0
        try:
            positions = self.api.list_positions()
        except Exception:
            positions = []

        for pos in positions:
            if self.session_entry_themes.get(pos.symbol) == theme:
                try:
                    current_value += abs(float(pos.market_value))
                except Exception:
                    continue

        return theme, current_value + max(0.0, candidate_value)

    def _cancel_bracket_legs(self, symbol: str):
        """Best-effort cancel of any open stop/limit child orders for `symbol`.

        With GTC bracket parents, the stop and take-profit legs persist beyond
        the session. Before any manual SELL or guardrail flatten we cancel them
        explicitly so the broker can't fire them on a position we just closed.
        Alpaca usually auto-cancels OCO siblings, but doing it ourselves makes
        the timing deterministic.
        """
        try:
            open_orders = self.api.list_orders(status='open', limit=100)
        except Exception:
            return
        for order in open_orders or []:
            try:
                if getattr(order, 'symbol', None) != symbol:
                    continue
                order_type = (getattr(order, 'order_type', '') or
                              getattr(order, 'type', '') or '').lower()
                if order_type not in ('stop', 'stop_limit', 'limit', 'trailing_stop'):
                    continue
                self.api.cancel_order(order.id)
            except Exception:
                continue

    def _flatten_position(self, pos, reason: str, prices=None):
        """Market-sell an entire position and tag it for memory."""
        try:
            qty = abs(int(float(pos.qty)))
            if qty <= 0:
                return
            # Cancel any live GTC bracket legs first to avoid double-fire.
            self._cancel_bracket_legs(pos.symbol)
            self.api.submit_order(
                symbol=pos.symbol, qty=qty, side='sell',
                type='market', time_in_force='day'
            )
            self.log(f"GUARDRAIL SELL {qty} {pos.symbol} - {reason}", 'trade')
            with self._trade_lock:
                self.trade_count += 1
            self.update_trade_count_display()
            if self.tracker:
                price = float(pos.current_price) if hasattr(pos, "current_price") else 0.0
                if prices and pos.symbol in prices:
                    price = prices[pos.symbol]['price']
                self.tracker.record_trade(pos.symbol, "sell", qty, price, reason)
            self.session_entry_themes.pop(pos.symbol, None)
            self.bracketed_symbols.discard(pos.symbol)
        except Exception as e:
            self.log(f"Guardrail flatten failed for {pos.symbol}: {e}", 'error')

    def _apply_guardrails(self, prices=None) -> bool:
        """Pre-LLM safety pass. Returns True if the loop should skip the LLM
        (kill switch tripped or end-of-day flatten just ran)."""
        # Reconcile any bracket fills the broker has filled since last cycle BEFORE
        # we read positions — that way the cooldowns + memory updates are in place.
        self._reconcile_broker_fills()

        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
        except Exception as e:
            self.log(f"Guardrail account fetch failed: {e}", 'error')
            return False

        portfolio_value = float(account.portfolio_value)

        # 1) Force-flat near the close
        if FORCE_FLAT_ENABLED and not self.force_flat_done_today:
            now_et = self._now_et()
            try:
                hh, mm = [int(x) for x in FORCE_FLAT_HHMM_ET.split(":")]
                cutoff = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if now_et >= cutoff and now_et.hour < 16 and positions:
                    self.log(
                        f"FORCE-FLAT @ {FORCE_FLAT_HHMM_ET} ET - closing "
                        f"{len(positions)} positions",
                        'warning'
                    )
                    for p in positions:
                        self._flatten_position(p, "force-flat (EOD)", prices)
                    self.force_flat_done_today = True
                    self.kill_switch_tripped = True  # no more trading today
                    return True
            except Exception as e:
                self.log(f"Force-flat clock error: {e}", 'warning')

        # 2) Daily drawdown kill switch
        if self.session_start_value and self.session_start_value > 0 and not self.kill_switch_tripped:
            dd_pct = (portfolio_value - self.session_start_value) / self.session_start_value * 100.0
            if dd_pct <= DAILY_KILL_PCT:
                self.log(
                    f"KILL SWITCH: portfolio {dd_pct:+.2f}% on the day "
                    f"(threshold {DAILY_KILL_PCT}%) - flattening and pausing.",
                    'error'
                )
                for p in positions:
                    self._flatten_position(p, f"daily kill switch ({dd_pct:+.2f}%)", prices)
                self.kill_switch_tripped = True
                return True

        if self.kill_switch_tripped:
            return True

        # 3) Per-position stop-loss / take-profit
        for p in positions:
            try:
                pnl_pct = float(p.unrealized_plpc) * 100.0
            except Exception:
                continue
            if pnl_pct <= STOP_LOSS_PCT:
                self._flatten_position(p, f"stop-loss {pnl_pct:+.2f}%", prices)
                self.cooldowns[p.symbol] = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
                self.log(f"Cooldown {COOLDOWN_MINUTES}m on {p.symbol}", 'warning')
            elif pnl_pct >= TAKE_PROFIT_PCT:
                self._flatten_position(p, f"take-profit {pnl_pct:+.2f}%", prices)

        return False

    def execute_trade(self, decision, prices=None):
        """Execute trade and track for memory"""
        action = decision.get('action', 'HOLD').upper()
        symbol = decision.get('symbol', '')
        quantity = int(decision.get('quantity', 0))
        reason = decision.get('reason', '')

        if action == 'HOLD' or quantity <= 0:
            self.log(f"Decision: HOLD - {reason}", 'info')
            return None

        quantity = enforce_sell_quantity(
            self.api,
            action=action,
            symbol=symbol,
            quantity=quantity,
            log_fn=self.log,
        )
        if quantity is None:
            return None

        # Cooldown enforcement (after a stop-out)
        if action == 'BUY' and self._on_cooldown(symbol):
            mins_left = max(0, int((self.cooldowns[symbol] - datetime.now()).total_seconds() / 60))
            self.log(f"REJECTED: {symbol} on cooldown ({mins_left}m left)", 'warning')
            return None

        if action == 'BUY' and self._on_reentry_cooldown(symbol):
            mins_left = max(0, int((self.reentry_cooldowns[symbol] - datetime.now()).total_seconds() / 60))
            self.log(f"REJECTED: {symbol} re-entry cooldown ({mins_left}m left)", 'warning')
            return None

        if action == 'BUY':
            allowed, rejection_reason = validate_sympathy_trade_confirmation(
                symbol=symbol,
                reason=reason,
                prices=prices,
                candidate_symbols=self.watchlist or SYMBOLS,
                news_symbols=self.watchlist_news_symbols,
                scanner_snapshot=self.watchlist_scan_map,
                min_intraday_change_pct=SYMPATHY_MIN_INTRADAY_PCT,
                near_high_tolerance_pct=SYMPATHY_NEAR_HIGH_TOLERANCE_PCT,
                min_rel_vol=SYMPATHY_MIN_RELVOL,
            )
            if not allowed:
                self.log(rejection_reason or f"REJECTED: {symbol} lacks direct target confirmation.", 'warning')
                return None

            allowed, rejection_reason = validate_setup_scorecard(
                symbol=symbol,
                reason=reason,
                active_symbols=self.watchlist or SYMBOLS,
                log_fn=self.log,
            )
            if not allowed:
                self.log(rejection_reason or f"REJECTED: {symbol} setup is paused by recent live results.", 'warning')
                return None

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
        quantity = enforce_risk_budget(
            self.api,
            action=action,
            symbol=symbol,
            quantity=quantity,
            prices=prices,
            stop_loss_pct=STOP_LOSS_PCT,
            max_risk_fraction=MAX_RISK_PER_TRADE,
            log_fn=self.log,
        )
        if quantity is None:
            return None

        try:
            if action == 'BUY':
                if prices and symbol in prices:
                    try:
                        account = self.api.get_account()
                        portfolio_value = float(account.portfolio_value)
                        candidate_value = quantity * float(prices[symbol]['price'])
                        theme, theme_value = self._get_theme_exposure(symbol, candidate_value, reason)
                        if theme and portfolio_value > 0:
                            theme_pct = theme_value / portfolio_value
                            if theme_pct > MAX_THEME_EXPOSURE:
                                self.log(
                                    f"REJECTED: {symbol} would push {theme} exposure to "
                                    f"{theme_pct*100:.1f}% (cap {MAX_THEME_EXPOSURE*100:.0f}%)",
                                    'warning',
                                )
                                return None
                    except Exception as e:
                        self.log(f"Theme exposure check failed: {e}", 'warning')

                order = self._submit_buy_with_bracket(symbol, quantity, prices, reason)
                if order is None:
                    return None
                with self._trade_lock:
                    self.trade_count += 1
                self.update_trade_count_display()
                self.reentry_cooldowns[symbol] = datetime.now() + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
                theme = extract_trade_theme(symbol, reason)
                if theme:
                    self.session_entry_themes[symbol] = theme

                # Track for memory
                if self.tracker and prices and symbol in prices:
                    self.tracker.record_trade(symbol, "buy", quantity,
                                              prices[symbol]['price'], reason)
                    self.log("[Memory] Tracking buy", 'info')

                return order

            elif action == 'SELL':
                # Cancel live GTC bracket legs before manual exit so the broker
                # doesn't fire the stop or take-profit on a flat position.
                self._cancel_bracket_legs(symbol)
                order = self.api.submit_order(
                    symbol=symbol, qty=quantity, side='sell',
                    type='market', time_in_force='day'
                )
                self.log(f"SELL ORDER: {quantity} {symbol} - {reason}", 'trade')
                with self._trade_lock:
                    self.trade_count += 1
                self.update_trade_count_display()

                # Track for memory
                if self.tracker and prices and symbol in prices:
                    self.tracker.record_trade(symbol, "sell", quantity,
                                              prices[symbol]['price'], reason)
                    self.log("[Memory] Trade outcome recorded", 'info')
                self.session_entry_themes.pop(symbol, None)
                self.bracketed_symbols.discard(symbol)

                return order

        except Exception as e:
            self.log(f"Trade Error: {e}", 'error')
            return None

    def _submit_buy_with_bracket(self, symbol, quantity, prices, reason):
        """Submit a BUY wrapped in an ATR-scaled broker bracket; fall back to plain market on rejection."""
        entry_price = 0.0
        if prices and symbol in prices:
            try:
                entry_price = float(prices[symbol]['price'])
            except Exception:
                entry_price = 0.0

        bracket_kwargs = None
        if USE_BRACKET_ORDERS and entry_price > 0:
            atr = get_atr_for_symbol(self.indicator_snapshots, symbol)
            levels = atr_bracket_prices(
                entry_price=entry_price,
                atr=atr,
                stop_atr_mult=BROKER_ATR_STOP_MULT,
                take_profit_atr_mult=BROKER_ATR_TP_MULT,
                fallback_stop_pct=abs(STOP_LOSS_PCT),
                fallback_take_profit_pct=abs(TAKE_PROFIT_PCT),
            )
            bracket_kwargs = {
                "order_class": "bracket",
                "stop_loss": {"stop_price": levels["stop_price"]},
                "take_profit": {"limit_price": levels["take_profit_price"]},
            }
            self.log(
                f"BUY BRACKET: {quantity} {symbol} @ ~${entry_price:.2f}  "
                f"stop=${levels['stop_price']:.2f}  tp=${levels['take_profit_price']:.2f}  "
                f"basis={levels['basis']}  reason={reason}",
                'trade'
            )

        # Bracket legs use GTC so the stop + take-profit persist overnight and
        # protect the position when the bot isn't running. The fallback
        # plain-market BUY (no bracket) stays time_in_force='day' since polled
        # rails are the only protection there and we want it to fully fill.
        parent_tif = 'gtc' if bracket_kwargs else 'day'
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=quantity, side='buy',
                type='market', time_in_force=parent_tif,
                **(bracket_kwargs or {})
            )
            if bracket_kwargs:
                self.bracketed_symbols.add(symbol)
            else:
                self.log(f"BUY ORDER: {quantity} {symbol} - {reason}", 'trade')
            return order
        except Exception as exc:
            if not bracket_kwargs:
                self.log(f"BUY error for {symbol}: {exc}", 'error')
                return None
            self.log(
                f"Bracket rejected for {symbol} ({exc}); falling back to plain market BUY (polled stop will still fire)",
                'warning'
            )
            try:
                order = self.api.submit_order(
                    symbol=symbol, qty=quantity, side='buy',
                    type='market', time_in_force='day'
                )
                self.log(f"BUY ORDER (fallback, no bracket): {quantity} {symbol}", 'trade')
                return order
            except Exception as exc2:
                self.log(f"BUY fallback also failed for {symbol}: {exc2}", 'error')
                return None

    def update_trade_count_display(self):
        """Refresh the trade count label safely from any thread."""
        self._run_on_ui_thread(self.trade_label.config, text=f"Trades: {self.trade_count}")

    def _set_stopped_state(self):
        self.start_btn.config(state='normal')
        self.provider_dropdown.config(state='readonly')
        self.status_label.config(text="Status: STOPPED", fg='#ff6b6b')

    def safe_update_status(self, text, fg):
        """Thread-safe status label update"""
        self._run_on_ui_thread(self.status_label.config, text=text, fg=fg)

    def _check_mode_transition(self):
        """One-way latch: HUNT runs the full HUNT_WINDOW_MINUTES from session start,
        then flips to MONITOR for the rest of the session. The bracket-fire does
        NOT shorten the hunt window -- we want the full 15min to find quality setups.
        Force-flat / STOP / re-Start are the only ways out."""
        if self.in_monitor_mode or self.session_started_at is None:
            return
        elapsed_min = (datetime.now() - self.session_started_at).total_seconds() / 60.0
        if elapsed_min < HUNT_WINDOW_MINUTES:
            return
        self.in_monitor_mode = True
        play_count = len(self.bracketed_symbols)
        if play_count:
            names = ", ".join(sorted(self.bracketed_symbols))
            tail = f"{play_count} play(s) bracketed: {names}"
        else:
            tail = "no plays bracketed"
        cycle_label = (
            f"{MONITOR_INTERVAL_SECONDS}s" if MONITOR_INTERVAL_SECONDS < 120
            else f"{MONITOR_INTERVAL_SECONDS // 60}m"
        )
        self.log(
            f"=== MODE: MONITOR ({cycle_label} cycles, no LLM) - "
            f"hunt window ended ({elapsed_min:.1f}m), {tail} ===",
            'header'
        )

    def _log_monitor_report(self, prices):
        """Status-only pass: P&L and distance to stop/TP for every open position.
        Skips the LLM entirely. Brackets are doing the work; we just report."""
        try:
            positions = self.api.list_positions()
        except Exception as exc:
            self.log(f"Monitor: could not fetch positions ({exc})", 'warning')
            return
        if not positions:
            self.log("Monitor: no open positions.", 'info')
            return

        # Pull working orders once and bucket by symbol so we can show live bracket levels.
        # NOTE: Alpaca holds the stop leg of a bracket in status='held' (not 'open') until
        # the OCO releases it. We query status='all' and filter out clearly-finished states.
        working_states = {'new', 'accepted', 'pending_new', 'accepted_for_bidding',
                          'partially_filled', 'held', 'replaced', 'pending_replace'}
        legs_by_symbol = {}
        try:
            for o in (self.api.list_orders(status='all', limit=200) or []):
                sym = getattr(o, 'symbol', None)
                status = (getattr(o, 'status', '') or '').lower()
                if sym and status in working_states:
                    legs_by_symbol.setdefault(sym, []).append(o)
        except Exception as exc:
            self.log(f"Monitor: could not fetch orders ({exc})", 'warning')

        self.log("Monitor report:", 'header')
        for pos in positions:
            sym = pos.symbol
            try:
                qty = float(pos.qty)
                entry = float(pos.avg_entry_price)
                last = float((prices or {}).get(sym, {}).get('price') or pos.current_price)
            except Exception:
                self.log(f"  {sym}: could not read position fields", 'warning')
                continue
            pnl_pct = (last - entry) / entry * 100.0 if entry else 0.0

            stop_price = None
            tp_price = None
            for o in legs_by_symbol.get(sym, []):
                otype = (getattr(o, 'order_type', '') or getattr(o, 'type', '') or '').lower()
                if otype in ('stop', 'stop_limit') and getattr(o, 'stop_price', None):
                    try:
                        stop_price = float(o.stop_price)
                    except Exception:
                        pass
                elif otype == 'limit' and getattr(o, 'limit_price', None):
                    try:
                        tp_price = float(o.limit_price)
                    except Exception:
                        pass

            stop_txt = (
                f"stop ${stop_price:.2f} ({(stop_price - last) / last * 100:+.2f}%)"
                if stop_price else "stop: not on broker"
            )
            tp_txt = (
                f"tp ${tp_price:.2f} ({(tp_price - last) / last * 100:+.2f}%)"
                if tp_price else "tp: not on broker"
            )
            color = 'info' if pnl_pct >= 0 else 'error'
            self.log(
                f"  {sym}: {qty:g} @ ${entry:.2f} -> ${last:.2f} ({pnl_pct:+.2f}%) | {stop_txt} | {tp_txt}",
                color
            )

    def trading_loop(self):
        """Main trading loop.

        Two modes:
        - HUNT    : full LLM pipeline at HUNT_INTERVAL_SECONDS (60s) for the first
                    HUNT_WINDOW_MINUTES (15m) after Start, hunting for setups.
        - MONITOR : status-only at MONITOR_INTERVAL_SECONDS (60s, no LLM) after the
                    hunt window ends, until STOP or session end.

        One-way latch: HUNT -> MONITOR after 15m elapsed. Force-flat and broker fill
        reconciliation still tick in monitor mode via _apply_guardrails."""
        while self.running:
            try:
                if self.paused:
                    time.sleep(1)
                    continue

                # Check market status
                clock = self.api.get_clock()
                if not clock.is_open:
                    try:
                        open_str = self._format_market_time(clock.next_open)
                    except Exception:
                        open_str = "?"
                    self.safe_update_status(f"Status: WAITING - Market opens {open_str}", '#ffe66d')
                    self.log(f"Market is CLOSED. Opens at {open_str}. Waiting...", 'warning')
                    for _ in range(60):
                        if not self.running:
                            return
                        time.sleep(1)
                    continue

                self._check_mode_transition()
                interval = MONITOR_INTERVAL_SECONDS if self.in_monitor_mode else HUNT_INTERVAL_SECONDS
                mode_tag = "MONITOR" if self.in_monitor_mode else "HUNT"

                self.safe_update_status(f"Status: TRADING ({mode_tag})", '#00ff88')
                self.log(f"\n--- {mode_tag} cycle ---", 'header')

                prices = self.get_current_prices()
                if not prices:
                    self.log("Could not get prices. Waiting...", 'warning')
                    time.sleep(30)
                    continue

                # Pre-LLM guardrails (stop-loss / take-profit / kill switch / force-flat
                # / broker fill reconciliation). Runs in BOTH modes so legs and force-flat
                # still tick when we're sitting back.
                skip_llm = self._apply_guardrails(prices)
                portfolio_status, cash, buying_power = self.get_portfolio_status()
                self.update_account_display()

                if self.in_monitor_mode:
                    # Status only -- no LLM, no new entries. Brackets are the strategy now.
                    self._log_monitor_report(prices)
                elif skip_llm:
                    self.log("Guardrail tripped - skipping LLM this cycle.", 'warning')
                else:
                    # HUNT mode: full pipeline
                    self.log("Current Prices:", 'price')
                    for symbol, data in prices.items():
                        change = ((data['price'] - data['open']) / data['open']) * 100
                        color = 'info' if change >= 0 else 'error'
                        self.log(f"  {symbol}: ${data['price']:.2f} ({change:+.2f}%)", color)

                    self._refresh_indicator_snapshots(self.watchlist or list(prices.keys()))
                    tech_block = self._compose_technical_section()
                    if tech_block:
                        self.log(tech_block, 'info')

                    self.log("\nAsking AI for decision...", 'info')
                    decision = self.get_ai_decision(prices, portfolio_status, cash, buying_power)
                    self.execute_trade(decision, prices)
                    self.update_account_display()

                    # Re-check transition (only flips if the 15-min hunt window has elapsed)
                    self._check_mode_transition()
                    if self.in_monitor_mode:
                        interval = MONITOR_INTERVAL_SECONDS

                self.log(f"\nWaiting {interval}s until next check...", 'info')
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                self.log(f"Error: {e}", 'error')
                time.sleep(30)

    # === STRATEGY CHAT METHODS ===

    def setup_chat_tab(self):
        """Setup the Strategy Chat tab"""
        chat_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(chat_frame, text='  Strategy Chat  ')

        # Stats bar (memory + AI provider)
        stats_frame = tk.Frame(chat_frame, bg='#16213e')
        stats_frame.pack(fill='x', padx=5, pady=5)

        self.memory_stats_label = tk.Label(stats_frame, text="Memory: Loading...",
                                           font=('Consolas', 10), fg='#4ecdc4', bg='#16213e')
        self.memory_stats_label.pack(side='left', padx=10, pady=5)

        self.chat_ai_label = tk.Label(stats_frame, text=f"AI: {self.model}",
                                      font=('Consolas', 10), fg='#00ff88', bg='#16213e')
        self.chat_ai_label.pack(side='right', padx=10, pady=5)

        # Quick action buttons
        btn_frame = tk.Frame(chat_frame, bg='#1a1a2e')
        btn_frame.pack(fill='x', padx=5, pady=5)

        tk.Button(btn_frame, text="Show Insights", command=self.show_insights,
                 font=('Consolas', 10), bg='#4ecdc4', fg='#000000',
                 cursor='hand2').pack(side='left', padx=3)

        tk.Button(btn_frame, text="Show Stats", command=self.show_stats,
                 font=('Consolas', 10), bg='#ffe66d', fg='#000000',
                 cursor='hand2').pack(side='left', padx=3)

        tk.Button(btn_frame, text="Clear Chat", command=self.clear_chat,
                 font=('Consolas', 10), bg='#ff6b6b', fg='#000000',
                 cursor='hand2').pack(side='left', padx=3)

        tk.Button(btn_frame, text="Clear Insights", command=self.clear_insights,
                 font=('Consolas', 10), bg='#e74c3c', fg='#ffffff',
                 cursor='hand2').pack(side='left', padx=3)

        # Input frame at bottom
        input_frame = tk.Frame(chat_frame, bg='#1a1a2e')
        input_frame.pack(fill='x', side='bottom', padx=10, pady=10)

        self.send_btn = tk.Button(input_frame, text="SEND", command=self.send_chat_message,
                                  font=('Consolas', 12, 'bold'), bg='#00ff88', fg='#000000',
                                  width=10, cursor='hand2')
        self.send_btn.pack(side='right', padx=(10, 0))

        self.chat_input = tk.Entry(input_frame, font=('Consolas', 12),
                                   bg='#16213e', fg='#ffffff',
                                   insertbackground='#00ff88')
        self.chat_input.pack(side='left', fill='x', expand=True, ipady=8)
        self.chat_input.bind('<Return>', self.send_chat_message)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame,
                                                       font=('Consolas', 10),
                                                       bg='#0f0f23', fg='#ffffff',
                                                       insertbackground='#00ff88',
                                                       wrap='word')
        self.chat_display.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure chat tags
        self.chat_display.tag_configure('user', foreground='#4ecdc4')
        self.chat_display.tag_configure('bot', foreground='#00ff88')
        self.chat_display.tag_configure('system', foreground='#ffe66d')
        self.chat_display.tag_configure('saved', foreground='#ff6b6b', font=('Consolas', 10, 'italic'))

        # Welcome message
        self.chat_log("Welcome to Clawdbot Strategy Chat!", 'system')
        self.chat_log(f"Using AI: {self.model}", 'system')
        self.chat_log("Discuss trading strategies and I'll remember your insights.", 'system')
        self.chat_log("Type a message below to start chatting.\n", 'system')

        # Update memory stats
        self.update_memory_stats()

    def chat_log(self, message, tag='bot'):
        """Log message to chat display"""
        timestamp = datetime.now().strftime('%H:%M')
        prefix = ""
        if tag == 'user':
            prefix = f"[{timestamp}] You: "
        elif tag == 'bot':
            prefix = f"[{timestamp}] Clawdbot: "
        elif tag == 'system':
            prefix = f"[{timestamp}] "
        elif tag == 'saved':
            prefix = f"[{timestamp}] [SAVED] "

        self._run_on_ui_thread(self._append_chat_log, prefix, message, tag)

    def _append_chat_log(self, prefix, message, tag):
        self.chat_display.insert('end', f"{prefix}{message}\n", tag)
        self.chat_display.see('end')

    def _set_memory_stats_text(self, text):
        self.memory_stats_label.config(text=text)

    def _get_live_win_rate(self):
        """Calculate win rate from live Alpaca positions (profitable vs losing)."""
        try:
            positions = self.api.list_positions()
            if not positions:
                return None, 0, 0, 0.0
            winners = sum(1 for p in positions if float(p.unrealized_pl) > 0)
            losers = sum(1 for p in positions if float(p.unrealized_pl) < 0)
            total = winners + losers
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            rate = (winners / total * 100) if total > 0 else 0.0
            return rate, winners, losers, total_pl
        except Exception:
            return None, 0, 0, 0.0

    def update_memory_stats(self):
        """Update memory stats display"""
        try:
            parts = []
            if MEMORY_ENABLED:
                stats = get_memory_stats()
                parts.append(f"Insights: {stats['strategy_insights']}")
                closed_wins = stats['win_rate']['wins']
                closed_losses = stats['win_rate']['losses']
                if closed_wins + closed_losses > 0:
                    parts.append(f"Closed W/L: {stats['win_rate']['win_rate']:.1f}%")

            live_rate, winners, losers, total_pl = self._get_live_win_rate()
            if live_rate is not None and (winners + losers) > 0:
                pl_color = "+" if total_pl >= 0 else ""
                parts.append(f"Positions: {winners}W/{losers}L ({live_rate:.0f}%)")
                parts.append(f"P/L: {pl_color}${total_pl:,.2f}")

            if parts:
                stats_text = " | ".join(parts)
            else:
                stats_text = "No positions open"
        except Exception as e:
            stats_text = f"Stats Error: {e}"

        self._run_on_ui_thread(self._set_memory_stats_text, stats_text)

    def show_insights(self):
        """Show current strategy insights"""
        if not MEMORY_ENABLED:
            self.chat_log("Memory system not available.", 'system')
            return

        self.chat_log("\n--- Current Strategy Insights ---", 'system')
        insights = format_insights_for_prompt(n=10)
        self.chat_log(insights, 'bot')
        self.chat_log("---\n", 'system')

    def show_stats(self):
        """Show memory and live portfolio statistics"""
        self.chat_log("\n--- Portfolio Performance ---", 'system')

        # Live Alpaca positions
        try:
            positions = self.api.list_positions()
            account = self.api.get_account()
            if positions:
                winners = 0
                losers = 0
                total_pl = 0.0
                for p in positions:
                    pl = float(p.unrealized_pl)
                    pl_pct = float(p.unrealized_plpc) * 100
                    total_pl += pl
                    if pl > 0:
                        winners += 1
                    elif pl < 0:
                        losers += 1
                    sign = "+" if pl >= 0 else ""
                    self.chat_log(f"  {p.symbol}: {p.qty} shares | {sign}${pl:,.2f} ({sign}{pl_pct:.1f}%)", 'bot')
                total = winners + losers
                live_rate = (winners / total * 100) if total > 0 else 0
                sign = "+" if total_pl >= 0 else ""
                self.chat_log(f"  Total P/L: {sign}${total_pl:,.2f}", 'bot')
                self.chat_log(f"  Win rate: {live_rate:.0f}% ({winners}W / {losers}L of {len(positions)} positions)", 'bot')
                self.chat_log(f"  Portfolio: ${float(account.portfolio_value):,.2f} | Cash: ${float(account.cash):,.2f}", 'bot')
            else:
                self.chat_log("  No open positions", 'bot')
        except Exception as e:
            self.chat_log(f"  Could not fetch positions: {e}", 'bot')

        # Closed trade stats from memory
        if MEMORY_ENABLED:
            stats = get_memory_stats()
            closed_wins = stats['win_rate']['wins']
            closed_losses = stats['win_rate']['losses']
            if closed_wins + closed_losses > 0:
                self.chat_log(f"\n  Closed trades: {stats['win_rate']['win_rate']:.1f}% win rate ({closed_wins}W / {closed_losses}L)", 'bot')
            self.chat_log(f"  Strategy insights: {stats['strategy_insights']}", 'bot')

        self.chat_log("---\n", 'system')
        self.update_memory_stats()

    def clear_insights(self):
        """Clear all strategy insights with confirmation"""
        if not MEMORY_ENABLED:
            self.chat_log("Memory system not available.", 'system')
            return

        from tkinter import messagebox
        if messagebox.askyesno("Clear Insights",
                               "Are you sure you want to delete all strategy insights?\nThis cannot be undone."):
            if clear_strategy_insights():
                self.chat_log("All strategy insights cleared.", 'system')
            else:
                self.chat_log("No insights to clear.", 'system')
            self.update_memory_stats()

    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.delete('1.0', 'end')
        self.chat_messages = []
        self.chat_log("Chat cleared. Starting fresh.\n", 'system')

    def send_chat_message(self, event=None):
        """Send a chat message"""
        message = self.chat_input.get().strip()
        if not message or self.chat_processing:
            return

        self.chat_input.delete(0, 'end')
        self.chat_log(message, 'user')

        # Process in background thread
        self.chat_processing = True
        self.send_btn.config(state='disabled', text="...")
        threading.Thread(target=self.process_chat_message, args=(message,), daemon=True).start()

    def process_chat_message(self, message):
        """Process chat message with AI"""
        try:
            # Build system prompt with memory context
            memory_context = ""
            if MEMORY_ENABLED:
                memory_context = get_memory_context_for_prompt()

            system_prompt = f"""You are Clawdbot's Strategy Advisor - an expert AI trading strategist.

Your role is to help improve and refine trading strategies through conversation.

## CURRENT MEMORY STATE:
{memory_context}

## YOUR CAPABILITIES:
1. Analyze proposed strategy changes - discuss pros, cons, and risks
2. Suggest improvements based on past performance
3. Help formalize trading rules
4. Review and critique current approaches

## IMPORTANT:
- When the user proposes a valuable insight or strategy rule, save it by outputting:
  <SAVE_INSIGHT>the insight to save</SAVE_INSIGHT>
- Be specific and actionable in your advice
- Keep responses concise (2-3 paragraphs max)
"""

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (last 10 messages)
            for msg in self.chat_messages[-10:]:
                messages.append(msg)

            # Add current message
            messages.append({"role": "user", "content": message})
            self.chat_messages.append({"role": "user", "content": message})

            # Call AI
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            reply = response.choices[0].message.content

            # Extract and save any insights
            insights_saved = []
            pattern = r'<SAVE_INSIGHT(?:\s+tags="([^"]*)")?\s*>(.*?)</SAVE_INSIGHT>'
            matches = re.findall(pattern, reply, re.DOTALL)

            for tags_str, insight in matches:
                tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
                if MEMORY_ENABLED:
                    save_strategy_insight(insight=insight.strip(), tags=tags, source="chat")
                    insights_saved.append(insight.strip())

            # Clean response for display
            display_reply = re.sub(pattern, '', reply, flags=re.DOTALL).strip()

            # Update UI from main thread
            self.root.after(0, lambda: self._display_chat_response(display_reply, insights_saved))

            # Save to conversation history
            self.chat_messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            self.root.after(0, lambda: self.chat_log(f"Error: {str(e)[:100]}", 'system'))

        finally:
            self.root.after(0, self._reset_chat_input)

    def _display_chat_response(self, reply, insights_saved):
        """Display chat response (called from main thread)"""
        self.chat_log(reply, 'bot')

        for insight in insights_saved:
            self.chat_log(f"Insight saved: {insight[:60]}...", 'saved')

        if insights_saved:
            self.update_memory_stats()

    def _reset_chat_input(self):
        """Reset chat input state (called from main thread)"""
        self.chat_processing = False
        self.send_btn.config(state='normal', text="Send")

    def run(self):
        """Start GUI"""
        self.root.mainloop()


def main():
    # Verify credentials
    required_alpaca = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing_alpaca = [k for k in required_alpaca if not os.getenv(k)]
    if missing_alpaca:
        print(f"Error: Missing Alpaca credentials: {missing_alpaca}")
        return

    # Check for at least one AI provider
    has_openai = os.getenv('OPENAI_API_KEY')
    has_deepseek = os.getenv('DEEPSEEK_API_KEY')
    if not has_openai and not has_deepseek:
        print("Error: No AI API key found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY")
        return

    app = TradingGUI()
    app.run()


if __name__ == "__main__":
    main()
