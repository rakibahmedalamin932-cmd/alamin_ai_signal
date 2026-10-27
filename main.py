"""
ALAMIN AI SIGNAL BOT
- Compact Telegram signal format
- EMA 9/21 + RSI + MACD + ADX + ATR + candle confirmation
- Historical backtest command: /backtest and /backtest EURUSD
- Automatic one-candle result message, exactly once per signal ID while state persists
- Win rate and profit/loss calculated from completed signals and configured payout

Important:
* Experimental tool, not financial advice and no accuracy/profit is guaranteed.
* Yahoo Finance data is not the same as a broker feed and does not provide reliable OTC prices.
* This bot stores state in state.json. GitHub-hosted runners may discard local files between runs;
  for continuous, duplicate-safe operation use a persistent host or persist state.json between runs.
* A signal is generated from a completed candle; result is checked against the next candle close.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
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

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
}

INTERVAL = os.getenv("INTERVAL", "1m")
PERIOD = os.getenv("PERIOD", "7d")
PAYOUT = float(os.getenv("PAYOUT", "0.80"))  # 0.80 means 80% net profit on a win
STAKE = float(os.getenv("STAKE", "1.00"))    # result accounting currency/unit
MIN_SCORE = int(os.getenv("MIN_SCORE", "4"))
SIGNAL_LOOP_SECONDS = int(os.getenv("SIGNAL_LOOP_SECONDS", "30"))
STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("alamin-signal")

# state schema:
# {
#   "last_signal_candle": {"EUR/USD": "timestamp"},
#   "pending": {"signal_id": {...}},
#   "completed": {"signal_id": {...}}
# }
STATE = {"last_signal_candle": {}, "pending": {}, "completed": {}}
STATE_LOCK = asyncio.Lock()


def load_state():
    global STATE
    try:
        if STATE_FILE.exists():
            loaded = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                STATE = {
                    "last_signal_candle": loaded.get("last_signal_candle", {}),
                    "pending": loaded.get("pending", {}),
                    "completed": loaded.get("completed", {}),
                }
    except Exception:
        log.exception("Could not load state file; starting with empty state.")
    for key in ("last_signal_candle", "pending", "completed"):
        if not isinstance(STATE.get(key), dict):
            STATE[key] = {}


def save_state():
    """Atomic local state write to reduce corruption risk."""
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ---------------- MARKET DATA ----------------
def download_candles(symbol: str, period: str = PERIOD, interval: str = INTERVAL):
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"OHLC columns missing for {symbol}")
    df = df[cols].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 80:
        raise ValueError(f"Not enough candles for {symbol}: {len(df)}")
    return df


# ---------------- INDICATORS ----------------
def add_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_HIST"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_safe = df["ATR"].replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    df["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().fillna(0)
    df["PLUS_DI"] = plus_di.fillna(0)
    df["MINUS_DI"] = minus_di.fillna(0)

    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["AVG_BODY"] = df["BODY"].rolling(20).mean()
    return df


def evaluate_row(row: pd.Series):
    needed = ["EMA9", "EMA21", "RSI", "MACD_HIST", "ADX", "ATR", "AVG_BODY"]
    if any(pd.isna(row.get(k)) for k in needed):
        return None, 0

    if row["ADX"] < 15 or row["ATR"] <= 0 or row["AVG_BODY"] <= 0:
        return None, 0
    if row["BODY"] > row["ATR"] * 1.8:
        return None, 0

    call = put = 0
    if row["EMA9"] > row["EMA21"]:
        call += 1
    elif row["EMA9"] < row["EMA21"]:
        put += 1

    if 52 <= row["RSI"] <= 68:
        call += 1
    elif 32 <= row["RSI"] <= 48:
        put += 1

    if row["MACD_HIST"] > 0:
        call += 1
    elif row["MACD_HIST"] < 0:
        put += 1

    if row["PLUS_DI"] > row["MINUS_DI"]:
        call += 1
    elif row["MINUS_DI"] > row["PLUS_DI"]:
        put += 1

    if row["Close"] > row["Open"]:
        call += 1
    elif row["Close"] < row["Open"]:
        put += 1

    if call >= MIN_SCORE and call >= put + 2:
        return "CALL", call
    if put >= MIN_SCORE and put >= call + 2:
        return "PUT", put
    return None, max(call, put)


def build_latest_signal(raw: pd.DataFrame):
    # Use completed candles only: discard the most recent potentially-forming candle.
    df = add_indicators(raw.iloc[:-1].copy())
    if len(df) < 80:
        return None
    row = df.iloc[-1]
    direction, score = evaluate_row(row)
    if not direction:
        return None
    return {
        "direction": direction,
        "score": int(score),
        "candle_time": str(df.index[-1]),
        "entry_price": float(row["Close"]),
    }


# ---------------- RESULTS / REPORTS ----------------
def money(value: float) -> str:
    return f"{value:+.2f}"


def stats():
    completed = list(STATE["completed"].values())
    wins = sum(1 for x in completed if x["result"] == "WIN")
    losses = sum(1 for x in completed if x["result"] == "LOSS")
    ties = sum(1 for x in completed if x["result"] == "TIE")
    decisive = wins + losses
    win_rate = wins / decisive * 100 if decisive else 0.0
    pnl = sum(float(x["pnl"]) for x in completed)
    return wins, losses, ties, win_rate, pnl, len(completed)


def signal_message(pair: str, direction: str) -> str:
    return (
        "ALAMIN SIGNAL BOT\n\n"
        "Non Mtg Signal\n\n"
        f"PAIR : {pair} ( REAL MARKET )\n"
        f"SIGNAL : {direction}\n"
        "TIME : 1 MINUTE\n\n"
        "ALAMIN SIGNAL"
    )


def result_message(pair: str, result: str, pnl: float) -> str:
    wins, losses, ties, win_rate, total_pnl, total = stats()
    profit = max(pnl, 0.0)
    loss = abs(min(pnl, 0.0))
    return (
        f"{pair} ( REAL MARKET )\n\n"
        f"RESULT : {result}\n"
        f"PROFIT : ${profit:.2f}\n"
        f"LOSS : ${loss:.2f}\n"
        f"TRADE P/L : {money(pnl)}\n"
        f"TOTAL P/L : {money(total_pnl)}\n"
        f"WIN RATE : {win_rate:.2f}% ({wins}W/{losses}L)\n\n"
        "ALAMIN SIGNAL"
    )


async def resolve_pending(application: Application, pair: str, symbol: str, raw: pd.DataFrame):
    """Resolve pending signals when a later completed candle is available."""
    pending_items = [
        (sid, item) for sid, item in STATE["pending"].items()
        if item.get("pair") == pair
    ]
    if not pending_items:
        return

    # Current latest completed candle; compare close with stored entry.
    completed_df = raw.iloc[:-1]
    if completed_df.empty:
        return
    latest_time = str(completed_df.index[-1])
    latest_close = float(completed_df["Close"].iloc[-1])

    for signal_id, item in pending_items:
        # Do not resolve on the entry candle; wait for a later completed candle.
        if latest_time == item.get("candle_time"):
            continue
        # Require timestamp to be later than the entry candle.
        try:
            if pd.Timestamp(latest_time) <= pd.Timestamp(item["candle_time"]):
                continue
        except Exception:
            continue

        entry = float(item["entry_price"])
        direction = item["direction"]
        if latest_close == entry:
            result = "TIE"
            pnl = 0.0
        else:
            won = (direction == "CALL" and latest_close > entry) or (
                direction == "PUT" and latest_close < entry
            )
            if won:
                result, pnl = "WIN", STAKE * PAYOUT
            else:
                result, pnl = "LOSS", -STAKE

        # Persist completed state before sending: prevents duplicate result messages on restart.
        STATE["completed"][signal_id] = {
            **item,
            "result": result,
            "pnl": round(pnl, 8),
            "result_candle_time": latest_time,
            "result_price": latest_close,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        del STATE["pending"][signal_id]
        save_state()

        try:
            await application.bot.send_message(
                chat_id=item["chat_id"],
                text=result_message(pair, result, pnl),
            )
        except Exception:
            # The trade remains completed and won't be sent twice. Check logs if Telegram fails.
            log.exception("Could not send result message for %s", signal_id)


async def check_pairs(application: Application, target_chat=None):
    target = target_chat or CHAT_ID
    if not target:
        log.warning("TELEGRAM_CHAT_ID not configured.")
        return

    for pair, symbol in PAIRS.items():
        try:
            raw = download_candles(symbol)
            await resolve_pending(application, pair, symbol, raw)

            signal = build_latest_signal(raw)
            if not signal:
                continue

            # Stable ID from pair + completed candle timestamp.
            signal_id = f"{pair}|{signal['candle_time']}"
            if signal_id in STATE["pending"] or signal_id in STATE["completed"]:
                continue
            if STATE["last_signal_candle"].get(pair) == signal["candle_time"]:
                continue

            item = {
                "pair": pair,
                "symbol": symbol,
                "direction": signal["direction"],
                "score": signal["score"],
                "candle_time": signal["candle_time"],
                "entry_price": signal["entry_price"],
                "chat_id": str(target),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            # Save before sending so repeated polling cannot create duplicate signals.
            STATE["pending"][signal_id] = item
            STATE["last_signal_candle"][pair] = signal["candle_time"]
            save_state()

            try:
                await application.bot.send_message(
                    chat_id=target,
                    text=signal_message(pair, signal["direction"]),
                )
            except Exception:
                # Remove pending record if signal was not delivered, allowing retry.
                STATE["pending"].pop(signal_id, None)
                STATE["last_signal_candle"].pop(pair, None)
                save_state()
                raise
        except Exception as exc:
            log.exception("Could not process %s: %s", pair, exc)


# ---------------- COMMANDS ----------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "ALAMIN SIGNAL BOT\n\n"
        "/signals - check signals now\n"
        "/backtest - historical test for all pairs\n"
        "/backtest EURUSD - test one pair\n"
        "/stats - total completed result statistics\n"
        "/status - configuration\n\n"
        "Experimental signals only. No accuracy or profit is guaranteed."
    )


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Checking completed candles...")
    await check_pairs(context.application, target_chat=update.effective_chat.id)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wins, losses, ties, rate, pnl, total = stats()
    await update.effective_message.reply_text(
        "ALAMIN SIGNAL — TOTAL RESULT\n\n"
        f"TOTAL RESULT : {total}\n"
        f"WIN : {wins}\n"
        f"LOSS : {losses}\n"
        f"TIE : {ties}\n"
        f"WIN RATE : {rate:.2f}%\n"
        f"TOTAL PROFIT/LOSS : {money(pnl)}\n\n"
        "ALAMIN SIGNAL"
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "ALAMIN BOT STATUS\n"
        f"Pairs: {', '.join(PAIRS.keys())}\n"
        f"Interval: {INTERVAL}\n"
        f"Period: {PERIOD}\n"
        f"Minimum score: {MIN_SCORE}/5\n"
        f"Payout assumption: {PAYOUT:.0%}\n"
        f"Stake per result: ${STAKE:.2f}\n"
        f"State file: {STATE_FILE}\n"
        "Note: use persistent storage for state.json to preserve duplicate protection."
    )


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Running backtest. Please wait...")
    selected = list(PAIRS.items())
    if context.args:
        query = context.args[0].upper().replace("/", "").replace("-", "")
        selected = [
            (pair, symbol) for pair, symbol in PAIRS.items()
            if pair.replace("/", "") == query or symbol.replace("=X", "").upper() == query
        ]
        if not selected:
            await update.effective_message.reply_text(
                "Unknown pair. Try /backtest EURUSD, /backtest GBPUSD, /backtest USDJPY."
            )
            return

    for pair, symbol in selected:
        try:
            raw = download_candles(symbol)
            df = add_indicators(raw)
            wins = losses = ties = trades = 0
            pnl = 0.0
            for i in range(80, len(df) - 1):
                direction, _ = evaluate_row(df.iloc[i])
                if not direction:
                    continue
                trades += 1
                entry = float(df["Close"].iloc[i])
                exit_price = float(df["Close"].iloc[i + 1])
                if entry == exit_price:
                    ties += 1
                elif (direction == "CALL" and exit_price > entry) or (
                    direction == "PUT" and exit_price < entry
                ):
                    wins += 1
                    pnl += STAKE * PAYOUT
                else:
                    losses += 1
                    pnl -= STAKE
            rate = wins / (wins + losses) * 100 if wins + losses else 0.0
            await update.effective_message.reply_text(
                f"BACKTEST — {pair}\n\n"
                f"TOTAL SIGNALS : {trades}\n"
                f"WIN : {wins} | LOSS : {losses} | TIE : {ties}\n"
                f"WIN RATE : {rate:.2f}%\n"
                f"ESTIMATED P/L : {money(pnl)}\n"
                f"PAYOUT ASSUMED : {PAYOUT:.0%}\n\n"
                "Historical approximation only; not a broker-execution result."
            )
        except Exception as exc:
            log.exception("Backtest failed for %s", pair)
            await update.effective_message.reply_text(f"Backtest failed for {pair}: {exc}")


async def scheduled_check(context: ContextTypes.DEFAULT_TYPE):
    await check_pairs(context.application)


async def post_init(application: Application):
    if application.job_queue is None:
        log.error("JobQueue unavailable. Install python-telegram-bot[job-queue].")
        return
    application.job_queue.run_repeating(
        scheduled_check,
        interval=max(20, SIGNAL_LOOP_SECONDS),
        first=10,
        name="signal-check",
    )


def main():
    if not BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN environment variable.")
    load_state()
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("backtest", backtest_command))
    log.info("ALAMIN AI SIGNAL BOT started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
