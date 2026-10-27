
import os
import json
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ==================================================
# ALAMIN SIGNAL BOT - EUR/JPY REAL MARKET ONLY
# ==================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

SYMBOL = "EURJPY=X"       # Yahoo Finance EUR/JPY
PAIR_NAME = "EUR/JPY"
TIMEFRAME = "1m"
EXPIRY_MINUTES = 1

CHECK_INTERVAL = 20       # seconds
PAYOUT = float(os.getenv("PAYOUT", "0.80"))
MIN_SCORE = 6
STATE_FILE = "bot_state.json"

TZ = ZoneInfo("Asia/Dhaka")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ALAMIN")

DEFAULT_STATE = {
    "last_signal_candle": "",
    "pending": [],
    "history": [],
    "total_profit": 0.0,
    "total_loss": 0.0,
    "wins": 0,
    "losses": 0,
}


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in DEFAULT_STATE.items():
            data.setdefault(key, value)
        return data
    except (OSError, json.JSONDecodeError):
        return DEFAULT_STATE.copy()


state = load_state()


def save_state():
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, STATE_FILE)


def calculate_indicators(df):
    """Calculate indicators without an extra TA library."""
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA 9 / 21 / 50
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD 12 / 26 / 9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # Bollinger Bands 20 / 2
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_UPPER"] = middle + 2 * std
    df["BB_LOWER"] = middle - 2 * std

    # Stochastic 14 / 3
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    denominator = (highest - lowest).replace(0, float("nan"))
    df["STO_K"] = 100 * (close - lowest) / denominator
    df["STO_D"] = df["STO_K"].rolling(3).mean()

    # ATR and ADX 14
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    atr_safe = df["ATR"].replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr_safe

    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    # Recent support / resistance
    df["SUPPORT"] = low.rolling(20).min().shift(1)
    df["RESISTANCE"] = high.rolling(20).max().shift(1)

    return df


def fetch_market_data():
    """Fetch Yahoo Finance minute candles for EUR/JPY only."""
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(
        period="1d",
        interval="1m",
        auto_adjust=False,
        prepost=False,
    )

    if df is None or df.empty or len(df) < 100:
        raise ValueError("Not enough EUR/JPY minute candles")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Exclude the newest candle because it may still be forming.
    df = df.iloc[:-1]

    if len(df) < 80:
        raise ValueError("Not enough completed candles")

    # Ignore stale data rather than sending an outdated signal.
    latest = df.index[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    if (now_utc - latest).total_seconds() > 180:
        raise ValueError("EUR/JPY market data is stale")

    return calculate_indicators(df)


def analyse_market(df):
    """Return CALL or PUT only when the filter threshold passes."""
    row = df.iloc[-1]
    prev = df.iloc[-2]

    required = [
        "EMA9", "EMA21", "EMA50", "RSI",
        "MACD", "MACD_SIGNAL", "BB_UPPER",
        "BB_LOWER", "STO_K", "STO_D",
        "ATR", "ADX", "PLUS_DI", "MINUS_DI",
        "SUPPORT", "RESISTANCE",
    ]

    if any(pd.isna(row[key]) for key in required):
        return None

    close = float(row["Close"])
    open_price = float(row["Open"])
    candle_range = float(row["High"] - row["Low"])

    if close <= 0 or float(row["ATR"]) <= 0:
        return None

    # Skip extremely small or unusually large candles.
    if candle_range < float(row["ATR"]) * 0.10:
        return None
    if candle_range > float(row["ATR"]) * 2.5:
        return None

    call = 0
    put = 0

    # EMA trend
    if row["EMA9"] > row["EMA21"] > row["EMA50"]:
        call += 2
    elif row["EMA9"] < row["EMA21"] < row["EMA50"]:
        put += 2

    # RSI
    if 52 <= row["RSI"] <= 68:
        call += 1
    elif 32 <= row["RSI"] <= 48:
        put += 1

    # MACD
    if row["MACD"] > row["MACD_SIGNAL"]:
        call += 1
    elif row["MACD"] < row["MACD_SIGNAL"]:
        put += 1

    # Stochastic momentum
    if row["STO_K"] > row["STO_D"] and row["STO_K"] < 80:
        call += 1
    elif row["STO_K"] < row["STO_D"] and row["STO_K"] > 20:
        put += 1

    # ADX trend strength
    if row["ADX"] >= 20:
        if row["PLUS_DI"] > row["MINUS_DI"]:
            call += 1
        elif row["MINUS_DI"] > row["PLUS_DI"]:
            put += 1

    # Candle confirmation
    if close > open_price and close > float(prev["Close"]):
        call += 1
    elif close < open_price and close < float(prev["Close"]):
        put += 1

    # Avoid buying directly into resistance or selling into support.
    support = float(row["SUPPORT"])
    resistance = float(row["RESISTANCE"])
    atr = float(row["ATR"])

    near_resistance = resistance > close and resistance - close < atr * 0.25
    near_support = support < close and close - support < atr * 0.25

    if near_resistance:
        call = 0
    if near_support:
        put = 0

    if call >= MIN_SCORE and call > put:
        return "CALL", call, put

    if put >= MIN_SCORE and put > call:
        return "PUT", call, put

    # No message is sent when the setup is weak.
    return None


def format_time(timestamp):
    return timestamp.tz_convert(TZ).strftime("%H:%M")


def get_stats():
    total = state["wins"] + state["losses"]
    win_rate = (state["wins"] / total * 100) if total else 0.0
    net = state["total_profit"] - state["total_loss"]

    return (
        f"WIN : {state['wins']}\n"
        f"LOSS : {state['losses']}\n"
        f"WIN RATE : {win_rate:.2f}%\n"
        f"TOTAL PROFIT : ${state['total_profit']:.2f}\n"
        f"TOTAL LOSS : ${state['total_loss']:.2f}\n"
        f"NET P/L : ${net:.2f}"
    )


async def send_signal(application, df, signal):
    direction, call_score, put_score = signal

    candle_time = df.index[-1]
    candle_id = candle_time.isoformat()

    if state["last_signal_candle"] == candle_id:
        return

    entry_time = candle_time + pd.Timedelta(minutes=1)
    closing_time = entry_time + pd.Timedelta(minutes=EXPIRY_MINUTES)

    message = (
        "ALAMIN SIGNAL BOT\n\n"
        "NON MTG SIGNAL\n"
        f"PAIR : {PAIR_NAME} (REAL MARKET)\n"
        f"DIRECTION : {direction}\n"
        f"ENTRY TIME : {format_time(entry_time)}\n"
        f"CLOSING TIME : {format_time(closing_time)}\n"
        f"TIMEFRAME : {TIMEFRAME}\n"
        f"CONFIRMATION SCORE : {max(call_score, put_score)}"
    )

    await application.bot.send_message(
        chat_id=CHAT_ID,
        text=message,
    )

    state["last_signal_candle"] = candle_id
    state["pending"].append({
        "id": candle_id,
        "direction": direction,
        "entry_reference": float(df.iloc[-1]["Close"]),
        "entry_time": entry_time.isoformat(),
        "expiry_time": closing_time.isoformat(),
        "status": "PENDING",
    })
    save_state()


async def settle_results(application, df):
    """Estimate outcomes using subsequent Yahoo minute-candle closes."""
    changed = False

    for item in state["pending"]:
        if item["status"] != "PENDING":
            continue

        expiry = pd.Timestamp(item["expiry_time"]).tz_convert("UTC")
        eligible = df[df.index >= expiry]

        if eligible.empty:
            continue

        # Use the first available completed candle at/after expiry.
        result_price = float(eligible.iloc[0]["Close"])
        entry_price = float(item["entry_reference"])
        direction = item["direction"]

        if result_price == entry_price:
            result = "DRAW"
        elif direction == "CALL":
            result = "WIN" if result_price > entry_price else "LOSS"
        else:
            result = "WIN" if result_price < entry_price else "LOSS"

        item["status"] = result
        item["result_price"] = result_price
        item["result_time"] = eligible.index[0].isoformat()

        if result == "WIN":
            state["wins"] += 1
            state["total_profit"] += PAYOUT
        elif result == "LOSS":
            state["losses"] += 1
            state["total_loss"] += 1.0

        state["history"].append(item.copy())
        changed = True

        result_message = (
            f"{PAIR_NAME} (REAL MARKET)\n"
            f"RESULT : {result}\n\n"
            f"{get_stats()}\n\n"
            "ALAMIN SIGNAL BOT\n"
            "Note: Result is estimated from Yahoo candle data."
        )

        await application.bot.send_message(
            chat_id=CHAT_ID,
            text=result_message,
        )

    if changed:
        state["pending"] = [
            x for x in state["pending"]
            if x["status"] == "PENDING"
        ]
        # Keep state file from growing without limit.
        state["history"] = state["history"][-500:]
        save_state()


async def scan_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_market_data()

        # Resolve due outcomes before creating a new signal.
        await settle_results(context.application, df)

        signal = analyse_market(df)
        if signal:
            await send_signal(context.application, df, signal)

    except Exception as exc:
        log.warning("Market scan failed: %s", exc)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ALAMIN SIGNAL BOT চালু আছে।\n"
        "Market: EUR/JPY REAL MARKET only\n"
        "Timeframe: 1 minute\n\n"
        "Commands:\n"
        "/status - Win rate ও Profit/Loss\n"
        "/backtest - সীমিত historical test\n"
        "/backtest EUR/JPY - একই পেয়ারের test"
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"ALAMIN SIGNAL BOT\n"
        f"PAIR : {PAIR_NAME} (REAL MARKET)\n\n"
        f"{get_stats()}"
    )


async def backtest_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    # This is a simple historical indicator test, not a broker-execution
    # simulation. Yahoo's historical candles may differ from broker prices.
    try:
        df = yf.Ticker(SYMBOL).history(
            period="7d",
            interval="1m",
            auto_adjust=False,
            prepost=False,
        )

        if df is None or df.empty:
            await update.message.reply_text(
                "EUR/JPY historical data পাওয়া যায়নি।"
            )
            return

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = calculate_indicators(df)

        wins = 0
        losses = 0
        trades = 0

        # Use a future candle as a rough expiry proxy.
        for i in range(80, len(df) - 2):
            window = df.iloc[:i + 1]
            signal = analyse_market(window)
            if not signal:
                continue

            direction = signal[0]
            entry = float(df.iloc[i]["Close"])
            exit_price = float(df.iloc[i + 2]["Close"])

            if direction == "CALL":
                won = exit_price > entry
            else:
                won = exit_price < entry

            trades += 1
            if won:
                wins += 1
            else:
                losses += 1

        win_rate = (wins / trades * 100) if trades else 0.0

        await update.message.reply_text(
            "EUR/JPY REAL MARKET — HISTORICAL TEST\n"
            f"Trades : {trades}\n"
            f"Wins : {wins}\n"
            f"Losses : {losses}\n"
            f"Win Rate : {win_rate:.2f}%\n\n"
            "এটি আনুমানিক candle-based test; "
            "broker execution বা লাভের নিশ্চয়তা নয়।"
        )

    except Exception as exc:
        log.exception("Backtest failed")
        await update.message.reply_text(
            f"Backtest error: {exc}"
        )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN secret missing")
    if not CHAT_ID:
        raise RuntimeError("CHAT_ID secret missing")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("backtest", backtest_command))

    if app.job_queue is None:
        raise RuntimeError(
            'Install python-telegram-bot[job-queue] in requirements.txt'
        )

    app.job_queue.run_repeating(
        scan_job,
        interval=CHECK_INTERVAL,
        first=5,
        name="eurjpy_only_scanner",
    )

    log.info("Starting ALAMIN BOT for EUR/JPY only")
    app.run_polling()


if __name__ == "__main__":
    main()
