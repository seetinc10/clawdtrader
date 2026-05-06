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
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, ttk

load_dotenv()

import alpaca_trade_api as tradeapi
from openai import OpenAI
from tools.live_trading_utils import (
    build_trading_prompt,
    enforce_position_limit,
    fetch_current_prices as fetch_live_prices,
    get_memory_section,
    get_portfolio_status as load_portfolio_status,
    request_ai_decision,
)

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
SYMBOLS = [
    # Mag 7
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Compute semis
    "AMD", "AVGO", "TSM", "ARM", "QCOM", "INTC",
    # Memory + semicap
    "MU", "ASML", "LRCX", "AMAT", "KLAC",
    # Analog / power semis
    "ON", "ADI", "TXN",
    # AI networking / connectivity silicon
    "ANET", "MRVL", "ALAB", "CRDO",
    # Optical transceivers
    "COHR", "LITE",
    # AI servers / OEMs
    "SMCI", "DELL", "HPE", "IBM",
    # Enterprise AI / cloud software
    "ORCL", "CRM", "NOW", "SNOW", "MDB", "PLTR",
    # AI ops + edge + cyber
    "DDOG", "NET", "CRWD", "PANW", "ZS",
    # AI-native apps
    "ADBE", "INTU", "RBLX",
    # Power & data-center physical
    "VRT", "ETN", "GEV",
    # IPP / nuclear utilities feeding hyperscalers
    "CEG", "VST", "TLN",
    # Data-center REITs
    "DLR", "EQIX",
    # Pure-play AI cloud
    "CRWV", "NBIS",
    # AI HPC hosting (ex-miners)
    "IREN", "APLD",
    # Robotics / physical AI
    "ISRG", "SYM", "TER",
    # China AI ADRs
    "BABA", "BIDU", "PDD",
    # AI advertising / training-data
    "TTD", "APP", "RDDT",
]
TRADE_INTERVAL = 60
MAX_POSITION_SIZE = 0.2

# Risk guardrails (deterministic â€” fire even if the LLM hallucinates)
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

        # Risk-guardrail session state
        self.session_start_value = None      # portfolio value when Start was clicked
        self.cooldowns = {}                  # symbol -> datetime when cooldown ends
        self.kill_switch_tripped = False     # daily DD kill switch
        self.force_flat_done_today = False   # has end-of-day flatten run yet

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
        self.log(f"Universe: {len(SYMBOLS)} symbols (will curate top-10 watchlist on Start)", 'info')
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
            ticker = data.get("ticker_used", "?")
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
                self.pc_ratio_label.config(text=f"P/C: {pc_ratio:.3f}", fg='#ffffff')
                self.sentiment_source_label.config(text=f"{ticker} | {source}")

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

        # Run scanner once per session to curate the watchlist
        if self.watchlist is None:
            self._run_scanner()
            self._start_quick_backtest()
        elif self.quick_backtest_result is None and not self.quick_backtest_running:
            self._start_quick_backtest()

        # Snapshot starting equity for the daily-DD kill switch
        try:
            self.session_start_value = float(self.api.get_account().portfolio_value)
            self.kill_switch_tripped = False
            self.force_flat_done_today = False
            self.cooldowns.clear()
            self.log(
                f"Session start equity: ${self.session_start_value:,.2f}  "
                f"| stop-loss {STOP_LOSS_PCT}%  take-profit +{TAKE_PROFIT_PCT}%  "
                f"daily kill {DAILY_KILL_PCT}%  cooldown {COOLDOWN_MINUTES}m  "
                f"force-flat {FORCE_FLAT_HHMM_ET if FORCE_FLAT_ENABLED else 'OFF'} ET",
                'info'
            )
        except Exception as e:
            self.log(f"Could not snapshot starting equity: {e}", 'warning')

        # Start trading thread
        self.trade_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trade_thread.start()

    def _run_scanner(self):
        """Scan the full universe and pick the top-10 movers as today's watchlist."""
        try:
            from tools.scanner import scan_top_movers
            self.log(f"Scanning {len(SYMBOLS)}-symbol universe...", 'info')
            top = scan_top_movers(self.api, SYMBOLS, top_n=10)
            if top:
                self.watchlist = [r['symbol'] for r in top]
                self.log(f"Watchlist: {', '.join(self.watchlist)}", 'info')
                for r in top:
                    self.log(
                        f"  {r['symbol']:6s} ${r['price']:>8.2f}  "
                        f"{r['pct_change']:+6.2f}%  relvol={r['rel_vol']:.2f}x  "
                        f"score={r['score']:.1f}",
                        'info'
                    )
            else:
                self.log("Scanner returned nothing â€” falling back to first 10 symbols.", 'warning')
                self.watchlist = SYMBOLS[:10]
        except Exception as e:
            self.log(f"Scanner failed: {e} â€” using first 10 symbols.", 'error')
            self.watchlist = SYMBOLS[:10]

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
        if self.quick_backtest_prompt_block:
            if memory_section.strip():
                return f"{memory_section.rstrip()}\n\n{self.quick_backtest_prompt_block}\n"
            return f"{self.quick_backtest_prompt_block}\n"
        return memory_section

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
        # Run end-of-session reflection (C) â€” distill today's trades into rules
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
                                                      sell_price, "Liquidation on STOP")

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
        prompt = build_trading_prompt(
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
                f"Maximum position size: {MAX_POSITION_SIZE*100}% of portfolio per stock (oversized buys are auto-trimmed)",
                f"Stop-loss is automatic at {STOP_LOSS_PCT}% - do not issue a SELL just to cut a loser early; the system handles it",
                f"Take-profit is automatic at +{TAKE_PROFIT_PCT}% - do not sell just to lock a small winner; the system handles it",
                f"After a stop-out, that symbol is on a {COOLDOWN_MINUTES}-minute cooldown and buys will be rejected",
                f"Daily kill switch flattens everything if portfolio is down {DAILY_KILL_PCT}% on the day",
                f"Force-flat at {FORCE_FLAT_HHMM_ET} ET - do not open new positions in the last 30 minutes",
                "You can BUY, SELL, or HOLD",
                "Apply any relevant strategy insights from your memory!",
            ],
        )

        try:
            return request_ai_decision(self.openai, self.model, prompt, log_fn=self.log)
        except Exception as e:
            self.log(f"AI Error: {str(e)[:100]}", 'error')
            return {"action": "HOLD", "symbol": "", "quantity": 0, "reason": f"Error: {e}"}

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

    def _flatten_position(self, pos, reason: str, prices=None):
        """Market-sell an entire position and tag it for memory."""
        try:
            qty = abs(int(float(pos.qty)))
            if qty <= 0:
                return
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
        except Exception as e:
            self.log(f"Guardrail flatten failed for {pos.symbol}: {e}", 'error')

    def _apply_guardrails(self, prices=None) -> bool:
        """Pre-LLM safety pass. Returns True if the loop should skip the LLM
        (kill switch tripped or end-of-day flatten just ran)."""
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
                        f"FORCE-FLAT @ {FORCE_FLAT_HHMM_ET} ET â€” closing "
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
                    f"(threshold {DAILY_KILL_PCT}%) â€” flattening and pausing.",
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

        # Cooldown enforcement (after a stop-out)
        if action == 'BUY' and self._on_cooldown(symbol):
            mins_left = max(0, int((self.cooldowns[symbol] - datetime.now()).total_seconds() / 60))
            self.log(f"REJECTED: {symbol} on cooldown ({mins_left}m left)", 'warning')
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

        try:
            if action == 'BUY':
                order = self.api.submit_order(
                    symbol=symbol, qty=quantity, side='buy',
                    type='market', time_in_force='day'
                )
                self.log(f"BUY ORDER: {quantity} {symbol} - {reason}", 'trade')
                with self._trade_lock:
                    self.trade_count += 1
                self.update_trade_count_display()

                # Track for memory
                if self.tracker and prices and symbol in prices:
                    self.tracker.record_trade(symbol, "buy", quantity,
                                              prices[symbol]['price'], reason)
                    self.log("[Memory] Tracking buy", 'info')

                return order

            elif action == 'SELL':
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

                return order

        except Exception as e:
            self.log(f"Trade Error: {e}", 'error')
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

    def trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check if paused
                if self.paused:
                    time.sleep(1)
                    continue

                # Check market status
                clock = self.api.get_clock()
                if not clock.is_open:
                    try:
                        next_open = clock.next_open
                        open_str = next_open.astimezone().strftime('%I:%M %p')
                    except Exception:
                        open_str = "?"
                    self.safe_update_status(f"Status: WAITING - Market opens {open_str}", '#ffe66d')
                    self.log(f"Market is CLOSED. Opens at {open_str}. Waiting...", 'warning')
                    # Interruptible sleep so STOP works immediately
                    for _ in range(60):
                        if not self.running:
                            return
                        time.sleep(1)
                    continue

                self.safe_update_status("Status: TRADING", '#00ff88')
                self.log("\n--- Checking market ---", 'header')

                # Get prices
                prices = self.get_current_prices()
                if not prices:
                    self.log("Could not get prices. Waiting...", 'warning')
                    time.sleep(30)
                    continue

                # Display prices
                self.log("Current Prices:", 'price')
                for symbol, data in prices.items():
                    change = ((data['price'] - data['open']) / data['open']) * 100
                    color = 'info' if change >= 0 else 'error'
                    self.log(f"  {symbol}: ${data['price']:.2f} ({change:+.2f}%)", color)

                # Pre-LLM guardrails: stop-loss / take-profit / kill switch / force-flat
                skip_llm = self._apply_guardrails(prices)

                # Get portfolio (after guardrail flattens, if any)
                portfolio_status, cash, buying_power = self.get_portfolio_status()
                self.update_account_display()

                if skip_llm:
                    self.log("Guardrail tripped â€” skipping LLM this cycle.", 'warning')
                    self.log(f"\nWaiting {TRADE_INTERVAL}s until next check...", 'info')
                    for _ in range(TRADE_INTERVAL):
                        if not self.running:
                            break
                        time.sleep(1)
                    continue

                # Get AI decision
                self.log("\nAsking AI for decision...", 'info')
                decision = self.get_ai_decision(prices, portfolio_status, cash, buying_power)

                # Execute (pass prices for memory tracking)
                self.execute_trade(decision, prices)

                # Update display
                self.update_account_display()

                # Wait
                self.log(f"\nWaiting {TRADE_INTERVAL}s until next check...", 'info')

                # Interruptible sleep
                for _ in range(TRADE_INTERVAL):
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
