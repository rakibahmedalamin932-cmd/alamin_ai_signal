"""
ALAMIN AI SIGNAL BOT
Strategy: EMA 9/21 + RSI 14 + MACD + ADX 14 + ATR + candle confirmation
Backtesting: historical one-candle expiry, win rate, wins/losses and payout-adjusted P/L.

IMPORTANT:
- This is an experimental educational signal bot, NOT a guarantee of profitable trades.
- Public Yahoo Finance 1-minute history is limited and may differ from broker/Quotex prices.
- OTC prices are not reliably available from Yahoo Finance; OTC pairs are intentionally excluded.
- Signals are based on completed candles. Backtest uses the next candle close as expiry.
"""

import os
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# -------------------- CONFIG --------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Yahoo Finance symbols. Add/remove symbols as needed.
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
}

INTERVAL = os.getenv("INTERVAL", "1m")
PERIOD = os.getenv("PERIOD", "7d")  # Yahoo 1-minute history is limited.
EXPIRY_CANDLES = 1
PAYOUT = float(os.getenv("PAYOUT", "0.80"))  # 0.80 = 80% profit on a win
STAKE = float(os.getenv("STAKE", "1.00"))    # Backtest accounting only
MIN_SCORE = int(os.getenv("MIN_SCORE", "4"))
SIGNAL_INTERVAL_SECONDS = 60
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("alamin-signal")

# Avoid duplicate signal messages for the same completed candle.
LAST_SENT = {}


# -------------------- MARKET DATA --------------------
def download_candles(symbol: str, period: str = PERIOD, interval: str = INTERVAL):
    """Download OHLC candles from Yahoo Finance."""
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No market data returned for {symbol}")

    # yfinance can return MultiIndex columns depending on version/symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} for {symbol}")

    df = df[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 80:
        raise ValueError(f"Not enough candles for {symbol}: {len(df)}")
    return df


# -------------------- INDICATORS & STRATEGY --------------------
def add_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_safe = df["ATR"].replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    df["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().fillna(0)
    df["PLUS_DI"] = plus_di.fillna(0)
    df["MINUS_DI"] = minus_di.fillna(0)

    df["ATR_PCT"] = (df["ATR"] / close.abs().replace(0, float("nan")) * 100).fillna(0)
    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["BULL_CANDLE"] = df["Close"] > df["Open"]
    df["BEAR_CANDLE"] = df["Close"] < df["Open"]
    df["AVG_BODY"] = df["BODY"].rolling(20).mean()
    return df


def evaluate_row(row: pd.Series):
    """
    Returns CALL/PUT or None plus a transparent explanation.
    Filters are conservative; a signal is not guaranteed to win.
    """
    required = ["EMA9", "EMA21", "RSI", "MACD_HIST", "ADX", "ATR", "AVG_BODY"]
    if any(pd.isna(row.get(k)) for k in required):
        return None, 0, "Indicator warm-up"

    # Avoid very weak trend conditions and tiny/abnormally large candles.
    if row["ADX"] < 15:
        return None, 0, "ADX too weak"
    if row["ATR"] <= 0 or row["AVG_BODY"] <= 0:
        return None, 0, "Invalid volatility"
    if row["BODY"] > row["ATR"] * 1.8:
        return None, 0, "Large candle filter"

    call_score = 0
    put_score = 0

    # Trend
    if row["EMA9"] > row["EMA21"]:
        call_score += 1
    elif row["EMA9"] < row["EMA21"]:
        put_score += 1

    # RSI momentum: avoid chasing extreme readings.
    if 52 <= row["RSI"] <= 68:
        call_score += 1
    elif 32 <= row["RSI"] <= 48:
        put_score += 1

    # MACD momentum
    if row["MACD_HIST"] > 0:
        call_score += 1
    elif row["MACD_HIST"] < 0:
        put_score += 1

    # Directional strength
    if row["PLUS_DI"] > row["MINUS_DI"]:
        call_score += 1
    elif row["MINUS_DI"] > row["PLUS_DI"]:
        put_score += 1

    # Candle confirmation
    if bool(row["BULL_CANDLE"]):
        call_score += 1
    elif bool(row["BEAR_CANDLE"]):
        put_score += 1

    # Require multiple confirmations and a score difference.
    if call_score >= MIN_SCORE and call_score >= put_score + 2:
        return "CALL", call_score, "Bullish trend + momentum confirmation"
    if put_score >= MIN_SCORE and put_score >= call_score + 2:
        return "PUT", put_score, "Bearish trend + momentum confirmation"
    return None, max(call_score, put_score), "No clear confirmation"


def build_signal(df: pd.DataFrame):
    """Use last completed candle; the final downloaded candle may still be forming."""
    if len(df) < 3:
        return None
    # Drop the last candle to reduce the risk of using an incomplete candle.
    data = add_indicators(df.iloc[:-1].copy())
    row = data.iloc[-1]
    direction, score, reason = evaluate_row(row)
    if direction is None:
        return None
    return {
        "direction": direction,
        "score": score,
        "reason": reason,
        "candle_time": data.index[-1],
        "close": float(row["Close"]),
        "rsi": float(row["RSI"]),
        "adx": float(row["ADX"]),
        "ema9": float(row["EMA9"]),
        "ema21": float(row["EMA21"]),
    }


# -------------------- BACKTEST --------------------
def run_backtest(symbol: str, period: str = PERIOD, interval: str = INTERVAL):
    """
    Basic historical test: signal at candle close, outcome at next candle close.
    This is a simplified binary-style test, not broker execution simulation.
    """
    raw = download_candles(symbol, period=period, interval=interval)
    df = add_indicators(raw)
    wins = losses = ties = trades = 0
    pnl = 0.0

    # Signal on bar i; evaluate direction from bar i to bar i+1 close.
    # Uses only indicators calculated through bar i (no future indicator data).
    for i in range(80, len(df) - EXPIRY_CANDLES):
        row = df.iloc[i]
        direction, _, _ = evaluate_row(row)
        if direction is None:
            continue

        entry = float(df["Close"].iloc[i])
        exit_price = float(df["Close"].iloc[i + EXPIRY_CANDLES])
        trades += 1

        if exit_price == entry:
            ties += 1
            continue

        is_win = (direction == "CALL" and exit_price > entry) or (
            direction == "PUT" and exit_price < entry
        )
        if is_win:
            wins += 1
            pnl += STAKE * PAYOUT
        else:
            losses += 1
            pnl -= STAKE

    decisive = wins + losses
    win_rate = (wins / decisive * 100) if decisive else 0.0
    return {
        "symbol": symbol,
        "candles": len(df),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "pnl": pnl,
        "payout": PAYOUT,
        "stake": STAKE,
    }


def format_backtest(result: dict, pair: str) -> str:
    return (
        f"📊 BACKTEST REPORT — {pair}\n"
        f"Market data: Yahoo Finance ({INTERVAL}, {PERIOD})\n"
        f"Expiry model: {EXPIRY_CANDLES} candle(s)\n"
        f"Total signals: {result['trades']}\n"
        f"WIN: {result['wins']} | LOSS: {result['losses']} | TIE: {result['ties']}\n"
        f"Win Rate (ties excluded): {result['win_rate']:.2f}%\n"
        f"Estimated P/L: {result['pnl']:+.2f} stake units\n"
        f"Payout assumed: {PAYOUT:.0%}\n\n"
        "⚠️ Historical test only; real broker prices, payout, slippage and OTC data may differ."
    )


# -------------------- TELEGRAM COMMANDS --------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 ALAMIN AI SIGNAL BOT\n\n"
        "Commands:\n"
        "/signals — latest signals\n"
        "/backtest — test all configured pairs\n"
        "/backtest EURUSD — test one pair\n"
        "/status — bot configuration\n\n"
        "⚠️ Educational signals only. No accuracy or profit is guaranteed."
    )
    await update.effective_message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚙️ ALAMIN BOT STATUS\n"
        f"Pairs: {', '.join(PAIRS.keys())}\n"
        f"Interval: {INTERVAL}\n"
        f"Backtest period: {PERIOD}\n"
        f"Minimum score: {MIN_SCORE}/5\n"
        f"Assumed payout: {PAYOUT:.0%}\n"
        f"Backtest stake unit: {STAKE:g}\n"
        f"Telegram chat configured: {'Yes' if CHAT_ID else 'No'}"
    )
    await update.effective_message.reply_text(text)


async def send_latest_signals(application: Application, target_chat=None):
    target = target_chat or CHAT_ID
    if not target:
        log.warning("TELEGRAM_CHAT_ID is not set; signals cannot be sent.")
        return

    sent_any = False
    for pair, symbol in PAIRS.items():
        try:
            raw = download_candles(symbol)
            signal = build_signal(raw)
            if not signal:
                continue

            candle_key = str(signal["candle_time"])
            if LAST_SENT.get(pair) == candle_key:
                continue

            # A message is sent only after a completed candle passes filters.
            msg = (
                f"ALAMIN SIGNAL BOT\n"
                f"Non Mtg Signal\n\n"
                f"{pair} (REAL MARKET)\n"
                f"SIGNAL: {signal['direction']}\n"
                f"TIMEFRAME: {INTERVAL}\n"
                f"CONFIRMATION: {signal['score']}/5\n"
                f"RSI: {signal['rsi']:.1f} | ADX: {signal['adx']:.1f}\n"
                f"EMA 9/21: {signal['ema9']:.5f} / {signal['ema21']:.5f}\n"
                f"Completed candle: {signal['candle_time']}\n\n"
                f"Reason: {signal['reason']}\n"
                "⚠️ Experimental signal; verify on your broker before any trade."
            )
            await application.bot.send_message(chat_id=target, text=msg)
            LAST_SENT[pair] = candle_key
            sent_any = True
        except Exception as exc:
            log.exception("Signal check failed for %s: %s", pair, exc)

    return sent_any


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Checking latest completed candles...")
    await send_latest_signals(context.application, target_chat=update.effective_chat.id)


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Running historical backtest. Please wait...")
    args = context.args
    if args:
        query = args[0].upper().replace("/", "").replace("-", "")
        matches = [(pair, sym) for pair, sym in PAIRS.items()
                   if pair.replace("/", "") == query or sym.replace("=X", "") == query]
        if not matches:
            await update.effective_message.reply_text(
                "Unknown pair. Try /backtest EURUSD or /backtest GBPUSD."
            )
            return
        selected = matches
    else:
        selected = list(PAIRS.items())

    for pair, symbol in selected:
        try:
            result = run_backtest(symbol)
            await update.effective_message.reply_text(format_backtest(result, pair))
        except Exception as exc:
            log.exception("Backtest failed for %s", pair)
            await update.effective_message.reply_text(
                f"Backtest failed for {pair}: {exc}\n"
                "Check data availability and try again later."
            )


async def scheduled_signal_job(context: ContextTypes.DEFAULT_TYPE):
    await send_latest_signals(context.application)


async def post_init(application: Application):
    # JobQueue is available when python-telegram-bot[job-queue] is installed.
    if application.job_queue is not None:
        application.job_queue.run_repeating(
            scheduled_signal_job,
            interval=SIGNAL_INTERVAL_SECONDS,
            first=10,
            name="signal_loop",
        )
        log.info("Scheduled signal loop started.")
    else:
        log.error("JobQueue missing. Install python-telegram-bot[job-queue].")


def main():
    if not BOT_TOKEN:
        raise SystemExit(
            "Missing TELEGRAM_BOT_TOKEN environment variable. "
            "Set it in GitHub Secrets; never hardcode your token."
        )

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    log.info("ALAMIN AI SIGNAL BOT starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
