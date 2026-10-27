
import os
import json
import logging
from pathlib import Path
from datetime import timedelta
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
# ALAMIN SIGNAL BOT
# ONLY EUR/JPY - REAL MARKET
# SIGNALS ONLY - NO RESULT MESSAGES
# ==================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

SYMBOL = "EURJPY=X"
PAIR = "EUR/JPY"
TIMEFRAME = "1 MINUTE"

BDT = ZoneInfo("Asia/Dhaka")
SCAN_INTERVAL = 20
MIN_SCORE = 6

STATE_FILE = Path("bot_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("ALAMIN_SIGNAL")


def load_state():
    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return {
                "last_signal_candle": data.get(
                    "last_signal_candle", ""
                )
            }
    except (OSError, ValueError, TypeError):
        return {"last_signal_candle": ""}


state = load_state()


def save_state():
    temp = STATE_FILE.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(state, file)
    os.replace(temp, STATE_FILE)


def fetch_data():
    """
    Fetch EUR/JPY minute candles.
    Yahoo Finance is a data source, not the broker's
    guaranteed live execution feed.
    """
    df = yf.Ticker(SYMBOL).history(
        period="1d",
        interval="1m",
        auto_adjust=False,
        prepost=False,
    )

    if df is None or df.empty:
        raise ValueError("EUR/JPY data unavailable")

    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Do not analyze the newest, potentially incomplete candle.
    if len(df) > 1:
        df = df.iloc[:-1]

    if len(df) < 100:
        raise ValueError("Not enough completed candles")

    latest = df.index[-1]
    now = pd.Timestamp.now(tz="UTC")

    # Never issue signals from stale data.
    if (now - latest).total_seconds() > 180:
        raise ValueError("EUR/JPY data is stale")

    return df


def add_indicators(df):
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
    spread = (highest - lowest).replace(0, float("nan"))

    df["STO_K"] = 100 * (close - lowest) / spread
    df["STO_D"] = df["STO_K"].rolling(3).mean()

    # ATR 14
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

    # ADX 14
    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr = df["ATR"].replace(0, float("nan"))

    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr

    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14, adjust=False
    ).mean() / atr

    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # Support / Resistance
    df["SUPPORT"] = low.rolling(20).min().shift(1)
    df["RESISTANCE"] = high.rolling(20).max().shift(1)

    return df


def analyze(df):
    row = df.iloc[-1]
    prev = df.iloc[-2]

    required = [
        "EMA9", "EMA21", "EMA50", "RSI",
        "MACD", "MACD_SIGNAL", "BB_UPPER",
        "BB_LOWER", "STO_K", "STO_D", "ATR",
        "ADX", "PLUS_DI", "MINUS_DI",
        "SUPPORT", "RESISTANCE",
    ]

    if any(pd.isna(row[x]) for x in required):
        return None

    close = float(row["Close"])
    open_price = float(row["Open"])
    candle_range = float(row["High"] - row["Low"])
    atr = float(row["ATR"])

    if close <= 0 or atr <= 0:
        return None

    # Avoid tiny and unusually large candles.
    if candle_range < atr * 0.10:
        return None
    if candle_range > atr * 2.5:
        return None

    call = 0
    put = 0

    # EMA trend
    if row["EMA9"] > row["EMA21"] > row["EMA50"]:
        call += 2
    elif row["EMA9"] < row["EMA21"] < row["EMA50"]:
        put += 2

    # RSI momentum
    if 52 <= row["RSI"] <= 68:
        call += 1
    elif 32 <= row["RSI"] <= 48:
        put += 1

    # MACD
    if row["MACD"] > row["MACD_SIGNAL"]:
        call += 1
    elif row["MACD"] < row["MACD_SIGNAL"]:
        put += 1

    # Stochastic
    if row["STO_K"] > row["STO_D"] and row["STO_K"] < 80:
        call += 1
    elif row["STO_K"] < row["STO_D"] and row["STO_K"] > 20:
        put += 1

    # ADX directional confirmation
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

    # Support / resistance proximity filter
    support = float(row["SUPPORT"])
    resistance = float(row["RESISTANCE"])

    if resistance > close and resistance - close < atr * 0.25:
        call = 0

    if support < close and close - support < atr * 0.25:
        put = 0

    if call >= MIN_SCORE and call > put:
        return "CALL"

    if put >= MIN_SCORE and put > call:
        return "PUT"

    # No weak-setup messages.
    return None


def build_signal_message(direction, candle_time):
    # Yahoo candle timestamps are treated as candle opening times.
    # Entry is scheduled for the next minute boundary.
    entry_utc = candle_time + timedelta(minutes=1)
    closing_utc = entry_utc + timedelta(minutes=1)

    entry_bdt = entry_utc.tz_convert(BDT).strftime("%H:%M")
    closing_bdt = closing_utc.tz_convert(BDT).strftime("%H:%M")

    return (
        "ALAMIN SIGNAL BOT\n\n"
        "Non Mtg Signal\n\n"
        f"PAIR : {PAIR} ( REAL MARKET )\n"
        f"SIGNAL : {direction}\n"
        f"ENTRY TIME : {entry_bdt} (BDT)\n"
        f"CLOSING TIME : {closing_bdt} (BDT)\n"
        f"TIME : 1 MINUTE\n\n"
        "ALAMIN SIGNAL"
    )


async def scan_market(context: ContextTypes.DEFAULT_TYPE):
    try:
        # This function requests exactly one symbol.
        raw = fetch_data()
        df = add_indicators(raw)

        candle_time = df.index[-1]
        candle_id = candle_time.isoformat()

        # Do not send another signal for the same candle.
        if state["last_signal_candle"] == candle_id:
            return

        direction = analyze(df)

        if direction not in ("CALL", "PUT"):
            return

        message = build_signal_message(direction, candle_time)

        await context.application.bot.send_message(
            chat_id=CHAT_ID,
            text=message,
        )

        # Save only after Telegram accepts the message.
        state["last_signal_candle"] = candle_id
        save_state()

        logger.info(
            "Signal sent: %s %s at %s",
            PAIR,
            direction,
            candle_time.isoformat(),
        )

    except Exception:
        logger.exception("EUR/JPY scan failed")


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "ALAMIN SIGNAL BOT is running.\n"
        "Market: EUR/JPY REAL MARKET only.\n"
        "Timeframe: 1 minute.\n"
        "Signals: CALL / PUT only.\n"
        "Result messages: Disabled."
    )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN secret is missing")

    if not CHAT_ID:
        raise RuntimeError("CHAT_ID secret is missing")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))

    if app.job_queue is None:
        raise RuntimeError(
            "Install python-telegram-bot[job-queue] "
            "in requirements.txt"
        )

    app.job_queue.run_repeating(
        scan_market,
        interval=SCAN_INTERVAL,
        first=5,
        name="EURJPY_ONLY",
    )

    logger.info("Starting ALAMIN: EUR/JPY only, no result messages")
    app.run_polling()


if __name__ == "__main__":
    main()
