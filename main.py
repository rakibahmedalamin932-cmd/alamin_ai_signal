
import os
import json
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
TZ = ZoneInfo("Asia/Dhaka")
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PAIRS = [
    p.strip().upper()
    for p in os.getenv(
        "PAIRS", "EUR/JPY,EUR/USD,GBP/USD,USD/JPY"
    ).split(",")
    if p.strip()
]

SIGNAL_BY = "RAKIB AHMED ALAMIN"
STATE_FILE = os.getenv("STATE_FILE", "signal_state.json")

CANDLE_COUNT = 250
MIN_SCORE = 5  # Original setting preserved
REQUEST_TIMEOUT = 20
RESULT_DELAY_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ---------- STATE ----------
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
    state.setdefault("result_keys", [])

    state["wins"] = max(0, int(state["wins"]))
    state["losses"] = max(0, int(state["losses"]))

    for key in ("pending", "sent_keys", "result_keys"):
        if not isinstance(state[key], list):
            state[key] = []

    return state


def save_state(state):
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, STATE_FILE)


# ---------- TELEGRAM ----------
def telegram(method, **kwargs):
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    response = requests.post(
        url, data=kwargs, timeout=REQUEST_TIMEOUT
    )

    try:
        data = response.json()
    except ValueError:
        data = {"description": response.text[:500]}

    if not response.ok or not data.get("ok"):
        raise RuntimeError(
            f"Telegram API error: HTTP {response.status_code}: {data}"
        )

    return data.get("result")


def send_text(text):
    return telegram("sendMessage", chat_id=CHAT_ID, text=text)


# ---------- MARKET DATA ----------
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

    response = requests.get(
        url, params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(
            f"Twelve Data error for {pair}: "
            f"{data.get('message', data)}"
        )

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(
            subset=["datetime", "open", "high", "low", "close"]
        )
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise RuntimeError(
            f"Not enough candles for {pair}: {len(df)}"
        )

    return df


# ---------- INDICATORS ----------
def add_indicators(df):
    d = df.copy()
    close, high, low = d["close"], d["high"], d["low"]

    # EMA
    for period in (9, 21, 50, 200):
        d[f"ema{period}"] = close.ewm(
            span=period, adjust=False
        ).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD(12,26,9)
    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    d["macd"] = macd
    d["macd_signal"] = macd.ewm(
        span=9, adjust=False
    ).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Stochastic (14,3)
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    d["stoch_k"] = (
        100 * (close - lowest)
        / (highest - lowest).replace(0, np.nan)
    )
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # Bollinger Bands
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()

    d["bb_mid"] = mid
    d["bb_upper"] = mid + 2 * std
    d["bb_lower"] = mid - 2 * std

    # ATR / ADX
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["atr"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    plus_di = (
        100 * plus_dm.ewm(alpha=1/14, adjust=False).mean()
        / d["atr"].replace(0, np.nan)
    )
    minus_di = (
        100 * minus_dm.ewm(alpha=1/14, adjust=False).mean()
        / d["atr"].replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    d["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    d["plus_di"] = plus_di
    d["minus_di"] = minus_di

    # Original Supertrend direction logic preserved
    hl2 = (high + low) / 2
    tr10 = tr.ewm(alpha=1/10, adjust=False).mean()
    upper = hl2 + 3 * tr10
    lower = hl2 - 3 * tr10

    st_dir = np.ones(len(d), dtype=int)

    for i in range(1, len(d)):
        if close.iloc[i] > upper.iloc[i - 1]:
            st_dir[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i - 1]

    d["supertrend_dir"] = st_dir

    # Support / resistance
    d["resistance"] = high.shift(1).rolling(20).max()
    d["support"] = low.shift(1).rolling(20).min()

    # Candle patterns
    d["body"] = (close - d["open"]).abs()
    d["range"] = (high - low).replace(0, np.nan)

    d["bull_engulf"] = (
        (close > d["open"])
        & (d["open"] < close.shift(1))
        & (close > d["open"].shift(1))
        & (close.shift(1) < d["open"].shift(1))
    )

    d["bear_engulf"] = (
        (close < d["open"])
        & (d["open"] > close.shift(1))
        & (close < d["open"].shift(1))
        & (close.shift(1) > d["open"].shift(1))
    )

    upper_wick = (
        high - pd.concat([close, d["open"]], axis=1).max(axis=1)
    )
    lower_wick = (
        pd.concat([close, d["open"]], axis=1).min(axis=1) - low
    )

    d["bull_pin"] = (
        (lower_wick > 2 * d["body"])
        & (lower_wick > upper_wick)
    )
    d["bear_pin"] = (
        (upper_wick > 2 * d["body"])
        & (upper_wick > lower_wick)
    )

    return d


# ---------- SIGNAL STRATEGY ----------
# Original score rules and thresholds preserved.
def get_signal(df):
    d = add_indicators(df)

    if len(d) < 210:
        raise RuntimeError("Need at least 210 candles for EMA200")

    x, prev = d.iloc[-1], d.iloc[-2]

    required = [
        "ema9", "ema21", "ema50", "ema200", "rsi",
        "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "bb_upper", "bb_lower",
        "atr", "adx", "plus_di", "minus_di",
        "resistance", "support", "supertrend_dir",
    ]

    if not np.isfinite(
        np.array([x[k] for k in required], dtype=float)
    ).all():
        raise RuntimeError("Indicator data is incomplete")

    call = 0
    put = 0

    if x.ema9 > x.ema21 and x.close > x.ema50:
        call += 2
    elif x.ema9 < x.ema21 and x.close < x.ema50:
        put += 2

    if x.ema50 > x.ema200:
        call += 1
    elif x.ema50 < x.ema200:
        put += 1

    if x.adx >= 20:
        if x.plus_di > x.minus_di:
            call += 1
        elif x.minus_di > x.plus_di:
            put += 1

    if x.macd > x.macd_signal and x.macd_hist >= prev.macd_hist:
        call += 1
    elif x.macd < x.macd_signal and x.macd_hist <= prev.macd_hist:
        put += 1

    if 52 <= x.rsi <= 68:
        call += 1
    elif 32 <= x.rsi <= 48:
        put += 1

    if x.stoch_k > x.stoch_d and x.stoch_k < 80:
        call += 1
    elif x.stoch_k < x.stoch_d and x.stoch_k > 20:
        put += 1

    if x.supertrend_dir == 1:
        call += 1
    elif x.supertrend_dir == -1:
        put += 1

    if bool(x.bull_engulf or x.bull_pin):
        call += 1
    if bool(x.bear_engulf or x.bear_pin):
        put += 1

    if x.close > x.resistance:
        call += 1
    elif x.close < x.support:
        put += 1

    candle_range = float(x.high - x.low)

    if not np.isfinite(x.atr) or x.atr <= 0:
        raise RuntimeError("ATR is invalid")

    if candle_range > 2.2 * x.atr:
        return None, max(call, put), "Skipped: volatility spike"

    if x.adx < 12:
        return None, max(call, put), "Skipped: weak trend"

    if call >= 6 and call >= put + 2:
        return (
            "CALL", call,
            "EMA + ADX + MACD + RSI/Stochastic + Supertrend + PA"
        )

    if put >= 6 and put >= call + 2:
        return (
            "PUT", put,
            "EMA + ADX + MACD + RSI/Stochastic + Supertrend + PA"
        )

    return None, max(call, put), f"Filtered weak setup (CALL {call}, PUT {put})"


# ---------- SIGNAL MESSAGE ----------
def send_signal(pair, direction, score, strategy, state):
    now = datetime.now(TZ)
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    closing = entry + timedelta(minutes=1)

    key = f"{pair}|{entry.strftime('%Y-%m-%d %H:%M')}"

    if key in state["sent_keys"]:
        logging.info("Duplicate signal suppressed: %s", key)
        return None

    message = (
        "Non Mtg Signal\n"
        f"PAIR : {pair}\n"
        f"ENTRY TIME : {entry.strftime('%H:%M')}\n"
        f"CLOSING TIME : {closing.strftime('%H:%M')}\n"
        f"DIRECTION : {direction}\n\n"
        f"SIGNAL BY - {SIGNAL_BY}"
    )

    send_text(message)

    state["sent_keys"].append(key)
    state["sent_keys"] = state["sent_keys"][-1000:]

    trade = {
        "pair": pair,
        "entry_time": entry.isoformat(),
        "closing_time": closing.isoformat(),
        "direction": direction,
        "result_key": key,
    }

    # Store pending trade before leaving this function
    state["pending"].append(trade)
    state["pending"] = state["pending"][-100:]
    save_state(state)

    logging.info(
        "Signal sent: %s %s score=%s strategy=%s",
        pair, direction, score, strategy
    )

    return trade


# ---------- RESULT CHECK ----------
def wait_for_result(trade, state, max_wait_seconds=300):
    entry_time = datetime.fromisoformat(trade["entry_time"])
    closing_time = datetime.fromisoformat(trade["closing_time"])
    pair = trade["pair"]
    direction = trade["direction"]
    result_key = trade["result_key"]

    if result_key in state["result_keys"]:
        logging.info("Result already recorded: %s", result_key)
        return True

    # Never evaluate before the scheduled candle should be closed
    wait_seconds = (
        closing_time - datetime.now(TZ)
    ).total_seconds() + RESULT_DELAY_SECONDS

    if wait_seconds > 0:
        time.sleep(min(wait_seconds, max_wait_seconds))

    deadline = time.time() + max_wait_seconds
    last_error = "Scheduled candle not available"

    while time.time() < deadline:
        try:
            # If the candle is still not expected to be closed, wait.
            if datetime.now(TZ) < closing_time + timedelta(
                seconds=RESULT_DELAY_SECONDS
            ):
                time.sleep(3)
                continue

            df = fetch_candles(pair)

            # Twelve Data timestamps are interpreted in Asia/Dhaka.
            times = pd.to_datetime(df["datetime"], errors="coerce")
            entry_naive = pd.Timestamp(
                entry_time.replace(tzinfo=None)
            )
            rows = df.loc[times == entry_naive]

            if rows.empty:
                last_error = "Entry candle not available yet"
                time.sleep(10)
                continue

            candle = rows.iloc[-1]
            open_price = float(candle["open"])
            close_price = float(candle["close"])

            if not (
                np.isfinite(open_price)
                and np.isfinite(close_price)
                and open_price > 0
                and close_price > 0
            ):
                last_error = "Invalid candle prices"
                time.sleep(10)
                continue

            # Flat candle: no WIN/LOSS count change.
            if np.isclose(open_price, close_price, rtol=0, atol=0):
                result = "DRAW / VERIFY"
            else:
                candle_direction = (
                    "CALL" if close_price > open_price else "PUT"
                )
                result = (
                    "PROFIT"
                    if candle_direction == direction
                    else "LOSS"
                )

            # Count only directional outcomes
            if result == "PROFIT":
                state["wins"] += 1
            elif result == "LOSS":
                state["losses"] += 1

            # Record the result once
            state["result_keys"].append(result_key)
            state["result_keys"] = state["result_keys"][-1000:]

            state["pending"] = [
                p for p in state["pending"]
                if p.get("result_key") != result_key
            ]

            total = state["wins"] + state["losses"]
            win_rate = (
                state["wins"] / total * 100
                if total else 0.0
            )

            # Save statistics before sending Telegram message
            save_state(state)

            message = (
                f"PAIR : {pair}\n"
                f"ENTRY TIME : {entry_time.strftime('%H:%M')}\n"
                f"RESULT : {result}\n"
                f"WIN : {state['wins']}\n"
                f"LOSS : {state['losses']}\n"
                f"WIN RATE : {win_rate:.1f}%\n\n"
                f"SIGNAL BY {SIGNAL_BY}"
            )

            send_text(message)

            logging.info(
                "Result: %s %s direction=%s open=%s close=%s",
                pair, result, direction, open_price, close_price
            )
            return True

        except Exception as exc:
            last_error = str(exc)
            logging.warning(
                "Result check failed for %s: %s", pair, exc
            )
            time.sleep(10)

    # Unverified results are NOT counted as losses
    send_text(
        "ALAMIN SIGNAL BOT - RESULT UNVERIFIED\n"
        f"PAIR : {pair}\n"
        f"ENTRY TIME : {entry_time.strftime('%H:%M')}\n"
        f"REASON : {last_error[:180]}\n"
        "WIN/LOSS কাউন্ট পরিবর্তন করা হয়নি।"
    )
    return False


# ---------- MAIN ----------
def main():
    if not API_KEY:
        raise SystemExit("TWELVE_DATA_API_KEY missing")
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")
    if not CHAT_ID:
        raise SystemExit("TELEGRAM_CHAT_ID missing")

    state = load_state()
    candidates = []
    errors = []

    for pair in PAIRS:
        try:
            df = fetch_candles(pair)

            # Preserve original signal strategy
            direction, score, strategy = get_signal(df)

            if direction:
                candidates.append({
                    "pair": pair,
                    "direction": direction,
                    "score": score,
                    "strategy": strategy,
                })
                logging.info(
                    "Analysis %s: %s score=%s strategy=%s",
                    pair, direction, score, strategy
                )
            else:
                errors.append(f"{pair}: {strategy}")

        except Exception as exc:
            logging.exception("Pair failed (%s)", pair)
            errors.append(f"{pair}: {str(exc)[:160]}")

        time.sleep(1)

    if not candidates:
        send_text(
            "ALAMIN SIGNAL BOT - NO SIGNAL\n"
            "No setup passed the existing filters.\n"
            + ("\n".join(errors[:4]) if errors else "")
        )
        save_state(state)
        return

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    trade = send_signal(
        best["pair"],
        best["direction"],
        best["score"],
        best["strategy"],
        state,
    )

    if trade is not None:
        wait_for_result(trade, state)

    save_state(state)


if __name__ == "__main__":
    main()
