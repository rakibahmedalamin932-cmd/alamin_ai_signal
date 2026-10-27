import os
import json
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
TZ = ZoneInfo("Asia/Dhaka")
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Comma-separated Twelve Data symbols, e.g. EUR/USD,GBP/USD,EUR/JPY
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "EUR/JPY,EUR/USD,GBP/USD,USD/JPY"
).split(",") if p.strip()]
SIGNAL_BY = "RAKIB AHMED ALAMIN"
STATE_FILE = os.getenv("STATE_FILE", "signal_state.json")
CANDLE_COUNT = 250
MIN_SCORE = 5  # Out of 8; tune only after backtesting
REQUEST_TIMEOUT = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("wins", 0)
    state.setdefault("losses", 0)
    state.setdefault("pending", [])
    state.setdefault("sent_keys", [])
    return state


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def telegram(method, **kwargs):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    response = requests.post(url, data=kwargs, timeout=REQUEST_TIMEOUT)
    try:
        data = response.json()
    except ValueError:
        data = {"description": response.text[:500]}
    if not response.ok or not data.get("ok"):
        raise RuntimeError(f"Telegram API error: HTTP {response.status_code}: {data}")
    return data.get("result")


def send_text(text):
    return telegram("sendMessage", chat_id=CHAT_ID, text=text)


def fetch_candles(pair):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": "1min",
        "outputsize": CANDLE_COUNT,
        "timezone": "Asia/Dhaka",
        "apikey": API_KEY,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data error for {pair}: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(f"Not enough candles for {pair}: {len(df)}")
    return df


def add_indicators(df):
    d = df.copy()
    close, high, low = d["close"], d["high"], d["low"]

    # EMAs
    for period in (9, 21, 50, 200):
        d[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD(12,26,9)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    d["macd"] = macd
    d["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Stochastic (14,3)
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    d["stoch_k"] = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # Bollinger Bands (20, 2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    d["bb_mid"], d["bb_upper"], d["bb_lower"] = mid, mid + 2*std, mid - 2*std

    # ATR and ADX (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / d["atr"].replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / d["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    d["plus_di"], d["minus_di"] = plus_di, minus_di

    # Supertrend (ATR 10, multiplier 3)
    hl2 = (high + low) / 2
    tr10 = tr.ewm(alpha=1/10, adjust=False).mean()
    upper = hl2 + 3 * tr10
    lower = hl2 - 3 * tr10
    st_dir = np.ones(len(d), dtype=int)
    for i in range(1, len(d)):
        if close.iloc[i] > upper.iloc[i-1]:
            st_dir[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]
    d["supertrend_dir"] = st_dir

    # Simple market structure and candle features
    d["resistance"] = high.shift(1).rolling(20).max()
    d["support"] = low.shift(1).rolling(20).min()
    d["body"] = (close - d["open"]).abs()
    d["range"] = (high - low).replace(0, np.nan)
    d["bull_engulf"] = (
        (close > d["open"]) & (d["open"] < d["close"].shift(1)) &
        (close > d["open"].shift(1)) & (d["close"].shift(1) < d["open"].shift(1))
    )
    d["bear_engulf"] = (
        (close < d["open"]) & (d["open"] > d["close"].shift(1)) &
        (close < d["open"].shift(1)) & (d["close"].shift(1) > d["open"].shift(1))
    )
    # Pin bars: long wick relative to candle body
    upper_wick = high - pd.concat([close, d["open"]], axis=1).max(axis=1)
    lower_wick = pd.concat([close, d["open"]], axis=1).min(axis=1) - low
    d["bull_pin"] = (lower_wick > 2*d["body"]) & (lower_wick > upper_wick)
    d["bear_pin"] = (upper_wick > 2*d["body"]) & (upper_wick > lower_wick)
    return d


def get_signal(df):
    """
    Multi-confluence 1-minute signal:
    trend (EMA), ADX/DI, MACD, RSI, stochastic, Supertrend,
    candle pattern and breakout/structure. Strict scoring aims to
    filter weak setups; it cannot guarantee a particular win rate.
    """
    d = add_indicators(df)
    if len(d) < 210:
        raise RuntimeError("Need at least 210 candles for EMA200")
    x, prev = d.iloc[-1], d.iloc[-2]

    required = [
        "ema9", "ema21", "ema50", "ema200", "rsi", "macd",
        "macd_signal", "macd_hist", "stoch_k", "stoch_d",
        "bb_upper", "bb_lower", "atr", "adx", "plus_di", "minus_di",
        "resistance", "support", "supertrend_dir"
    ]
    if not np.isfinite(np.array([x[k] for k in required], dtype=float)).all():
        raise RuntimeError("Indicator data is incomplete")

    call = 0
    put = 0
    reasons_call, reasons_put = [], []

    # Main trend alignment (stronger weight)
    if x.ema9 > x.ema21 and x.close > x.ema50:
        call += 2
        reasons_call.append("EMA trend")
    elif x.ema9 < x.ema21 and x.close < x.ema50:
        put += 2
        reasons_put.append("EMA trend")

    # Long trend filter
    if x.ema50 > x.ema200:
        call += 1
        reasons_call.append("long trend")
    elif x.ema50 < x.ema200:
        put += 1
        reasons_put.append("long trend")

    # ADX / directional movement confirmation
    if x.adx >= 20:
        if x.plus_di > x.minus_di:
            call += 1
            reasons_call.append("ADX/DI")
        elif x.minus_di > x.plus_di:
            put += 1
            reasons_put.append("ADX/DI")

    # MACD momentum
    if x.macd > x.macd_signal and x.macd_hist >= prev.macd_hist:
        call += 1
        reasons_call.append("MACD")
    elif x.macd < x.macd_signal and x.macd_hist <= prev.macd_hist:
        put += 1
        reasons_put.append("MACD")

    # RSI: avoid chasing extreme readings
    if 52 <= x.rsi <= 68:
        call += 1
        reasons_call.append("RSI")
    elif 32 <= x.rsi <= 48:
        put += 1
        reasons_put.append("RSI")

    # Stochastic cross/confirmation
    if x.stoch_k > x.stoch_d and x.stoch_k < 80:
        call += 1
        reasons_call.append("Stochastic")
    elif x.stoch_k < x.stoch_d and x.stoch_k > 20:
        put += 1
        reasons_put.append("Stochastic")

    # Supertrend confirmation
    if x.supertrend_dir == 1:
        call += 1
        reasons_call.append("Supertrend")
    elif x.supertrend_dir == -1:
        put += 1
        reasons_put.append("Supertrend")

    # Price action confirmation
    if bool(x.bull_engulf or x.bull_pin):
        call += 1
        reasons_call.append("bullish candle")
    if bool(x.bear_engulf or x.bear_pin):
        put += 1
        reasons_put.append("bearish candle")

    # Breakout confirmation. Use previous rolling level to avoid current-bar leakage.
    if x.close > x.resistance:
        call += 1
        reasons_call.append("breakout")
    elif x.close < x.support:
        put += 1
        reasons_put.append("breakdown")

    # Avoid abnormal candle spikes and very low ADX / flat markets.
    candle_range = float(x["high"] - x["low"])
    if not np.isfinite(x.atr) or x.atr <= 0:
        raise RuntimeError("ATR is invalid")
    if candle_range > 2.2 * x.atr:
        return None, max(call, put), "Skipped: abnormal volatility spike"
    if x.adx < 12:
        return None, max(call, put), "Skipped: market trend too weak"

    # Require a useful score and at least 2-point directional separation.
    # This intentionally filters weak setups instead of forcing random directions.
    if call >= 6 and call >= put + 2:
        return "CALL", call, "EMA + ADX + MACD + RSI/Stochastic + Supertrend + Price Action"
    if put >= 6 and put >= call + 2:
        return "PUT", put, "EMA + ADX + MACD + RSI/Stochastic + Supertrend + Price Action"

    return None, max(call, put), f"Filtered weak setup (CALL {call}, PUT {put})"

def make_chart(df, pair, direction):
    d = df.tail(60)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["datetime"], d["close"], label="Close", linewidth=1.4)
    enriched = add_indicators(d.copy())
    ax.plot(enriched["datetime"], enriched["ema9"], label="EMA 9", linewidth=1)
    ax.plot(enriched["datetime"], enriched["ema21"], label="EMA 21", linewidth=1)
    ax.set_title(f"{pair} | 1-minute | {direction}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = "market_chart.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def send_signal(pair, df, direction, score, strategy, state):
    """Send one text signal and return its scheduled trade details."""
    now = datetime.now(TZ)
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    closing = entry + timedelta(minutes=1)
    key = f"{pair}|{entry.strftime('%Y-%m-%d %H:%M')}"
    if key in state["sent_keys"]:
        logging.info("Duplicate signal suppressed: %s", key)
        return None

    message = (
        f"Non Mtg Signal\n"
        f"PAIR : {pair}\n"
        f"ENTRY TIME : {entry.strftime('%H:%M')}\n"
        f"CLOSING TIME : {closing.strftime('%H:%M')}\n"
        f"DIRECTION : {direction}\n\n"
        f"SIGNAL BY - {SIGNAL_BY}"
    )
    send_text(message)
    state["sent_keys"].append(key)
    state["sent_keys"] = state["sent_keys"][-500:]
    save_state(state)
    logging.info("Signal sent: %s %s score=%s strategy=%s", pair, direction, score, strategy)

    return {
        "pair": pair,
        "entry_time": entry,
        "closing_time": closing,
        "direction": direction,
    }


def wait_for_result(trade, state, max_wait_seconds=300):
    """
    Wait for the scheduled entry-minute candle to finish and become available.
    Evaluate that candle's open vs close for a consistent 1-minute result.
    """
    entry_time = trade["entry_time"]
    deadline = time.time() + max_wait_seconds
    last_error = None

    # Wait until the scheduled one-minute candle should have closed.
    wait_seconds = max(0, (trade["closing_time"] - datetime.now(TZ)).total_seconds()) + 8
    if wait_seconds > 0:
        time.sleep(min(wait_seconds, max_wait_seconds))

    while time.time() < deadline:
        try:
            df = fetch_candles(trade["pair"])
            times = pd.to_datetime(df["datetime"], errors="coerce")
            entry_naive = pd.Timestamp(entry_time.replace(tzinfo=None))
            rows = df[times == entry_naive]
            if not rows.empty:
                candle = rows.iloc[0]
                open_price = float(candle["open"])
                close_price = float(candle["close"])

                if close_price > open_price:
                    candle_direction = "CALL"
                elif close_price < open_price:
                    candle_direction = "PUT"
                else:
                    candle_direction = "DOJI"

                if candle_direction == "DOJI":
                    # Flat candle is counted as LOSS for conservative, deterministic accounting.
                    result = "LOSS"
                    state["losses"] += 1
                elif candle_direction == trade["direction"]:
                    result = "WIN"
                    state["wins"] += 1
                else:
                    result = "LOSS"
                    state["losses"] += 1

                total = state["wins"] + state["losses"]
                win_rate = (state["wins"] / total * 100.0) if total else 0.0
                message = (
                    f"PAIR : {trade['pair']}\n"
                    f"ENTRY TIME : {entry_time.strftime('%H:%M')}\n"
                    f"RESULT : {result}\n"
                    f"WIN : {state['wins']}\n"
                    f"LOSS : {state['losses']}\n"
                    f"WIN RATE : {win_rate:.1f}%\n\n"
                    f"SIGNAL BY {SIGNAL_BY}"
                )
                send_text(message)
                save_state(state)
                logging.info("Result sent: %s %s open=%s close=%s", trade["pair"], result, open_price, close_price)
                return True

            last_error = "Scheduled entry candle not available yet"
        except Exception as exc:
            last_error = str(exc)
            logging.warning("Waiting for result candle (%s): %s", trade["pair"], exc)
        time.sleep(10)

    diagnostic = (
        f"ALAMIN SIGNAL BOT - RESULT UNAVAILABLE\n"
        f"PAIR : {trade['pair']}\n"
        f"ENTRY TIME : {entry_time.strftime('%H:%M')}\n"
        f"Could not verify result from Twelve Data.\n"
        f"Reason : {(last_error or 'candle unavailable')[:180]}"
    )
    send_text(diagnostic)
    return False

def main():
    if not API_KEY:
        raise SystemExit("TWELVE_DATA_API_KEY missing in environment / GitHub Secrets")
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing in environment / GitHub Secrets")
    if not CHAT_ID:
        raise SystemExit("TELEGRAM_CHAT_ID missing in environment / GitHub Secrets")

    state = load_state()
    candidates = []
    errors = []

    # Analyze every configured pair, but only send ONE best signal per run.
    for pair in PAIRS:
        try:
            df = fetch_candles(pair)
            direction, score, strategy = get_signal(df)
            if direction:
                candidates.append({
                    "pair": pair,
                    "df": df,
                    "direction": direction,
                    "score": score,
                    "strategy": strategy,
                })
                logging.info("Analysis %s: %s score=%s strategy=%s", pair, direction, score, strategy)
            else:
                errors.append(f"{pair}: no signal ({strategy})")
        except Exception as exc:
            logging.exception("Pair failed (%s): %s", pair, exc)
            errors.append(f"{pair}: {str(exc)[:160]}")
        time.sleep(1)

    if not candidates:
        try:
            send_text(
                "ALAMIN SIGNAL BOT - NO SIGNAL\n"
                "No setup passed the stronger analysis filters this run.\n"
                + ("\n".join(errors[:4]) if errors else "Please check market data.")
            )
        except Exception:
            logging.exception("Could not send no-signal diagnostic")
        save_state(state)
        return

    # Highest score wins. Stable order breaks ties.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    trade = send_signal(
        best["pair"], best["df"], best["direction"],
        best["score"], best["strategy"], state
    )
    if trade is None:
        return

    # Keep this workflow run alive until the trade's result is available.
    wait_for_result(trade, state)
    save_state(state)


if __name__ == "__main__":
    main()
