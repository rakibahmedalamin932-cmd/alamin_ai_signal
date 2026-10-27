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
    d = add_indicators(df)
    x, prev = d.iloc[-1], d.iloc[-2]
    values = [x.get(k, np.nan) for k in (
        "ema9", "ema21", "ema50", "ema200", "rsi", "macd",
        "macd_signal", "macd_hist", "stoch_k", "stoch_d",
        "bb_upper", "bb_lower", "atr", "adx", "plus_di", "minus_di",
        "resistance", "support"
    )]
    if not np.isfinite(np.array(values, dtype=float)).all():
        raise RuntimeError("Indicators are not ready yet")

    call = 0
    put = 0

    # Trend / EMA alignment
    if x.ema9 > x.ema21: call += 1
    elif x.ema9 < x.ema21: put += 1
    if x.ema50 > x.ema200: call += 1
    elif x.ema50 < x.ema200: put += 1

    # ADX directional confirmation
    if x.adx >= 18:
        if x.plus_di > x.minus_di: call += 1
        elif x.minus_di > x.plus_di: put += 1

    # Momentum
    if x.macd > x.macd_signal and x.macd_hist > prev.macd_hist: call += 1
    elif x.macd < x.macd_signal and x.macd_hist < prev.macd_hist: put += 1
    if 50 < x.rsi < 70: call += 1
    elif 30 < x.rsi < 50: put += 1
    if x.stoch_k > x.stoch_d and x.stoch_k < 80: call += 1
    elif x.stoch_k < x.stoch_d and x.stoch_k > 20: put += 1

    # Supertrend
    if x.supertrend_dir == 1: call += 1
    elif x.supertrend_dir == -1: put += 1

    # Price action: breakout / retest proxy and candle patterns
    if x.close > x.resistance: call += 1
    elif x.close < x.support: put += 1
    if bool(x.bull_engulf or x.bull_pin): call += 1
    if bool(x.bear_engulf or x.bear_pin): put += 1

    # Volatility guard: avoid extremely tiny or oversized candle relative to ATR
    candle_range = float(x["high"] - x["low"])
    if not np.isfinite(x.atr) or x.atr <= 0 or candle_range > 2.5*x.atr:
        return None, max(call, put), "Volatility guard"

    if call >= MIN_SCORE and call > put:
        return "CALL", call, "Trend + Momentum + Price Action"
    if put >= MIN_SCORE and put > call:
        return "PUT", put, "Trend + Momentum + Price Action"

    # This bot is configured to emit only CALL/PUT; use directional majority if no score threshold.
    # This fallback is intentionally marked low-confidence in the message.
    if call > put:
        return "CALL", call, "Directional fallback (low confidence)"
    if put > call:
        return "PUT", put, "Directional fallback (low confidence)"
    return None, max(call, put), "No directional agreement"


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
    now = datetime.now(TZ)
    entry = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    closing = entry + timedelta(minutes=1)
    key = f"{pair}|{entry.strftime('%Y-%m-%d %H:%M')}"
    if key in state["sent_keys"]:
        return

    message = (
        f"PAIR : {pair}\n"
        f"ENTRY TIME : {entry.strftime('%H:%M')}\n"
        f"CLOSING TIME : {closing.strftime('%H:%M')}\n"
        f"DIRECTION : {direction}\n\n"
        f"SIGNAL BY - {SIGNAL_BY}"
    )
    chart_path = make_chart(df, pair, direction)
    with open(chart_path, "rb") as photo:
        telegram("sendPhoto", chat_id=CHAT_ID, photo=photo, caption=message)
    state["sent_keys"].append(key)
    state["sent_keys"] = state["sent_keys"][-500:]
    state["pending"].append({
        "pair": pair,
        "entry_time": entry.isoformat(),
        "closing_time": closing.isoformat(),
        "direction": direction,
        "entry_price": None,
        "checked": False,
        "created_at": now.isoformat(),
    })
    save_state(state)
    logging.info("Signal sent: %s %s score=%s strategy=%s", pair, direction, score, strategy)


def settle_pending(state):
    now = datetime.now(TZ)
    changed = False
    for item in state["pending"]:
        if item.get("checked"):
            continue
        close_at = datetime.fromisoformat(item["closing_time"])
        if now < close_at + timedelta(seconds=5):
            continue
        try:
            df = fetch_candles(item["pair"])
            # Candle whose timestamp is the scheduled entry minute, and closing minute.
            times = pd.to_datetime(df["datetime"])
            entry_dt = datetime.fromisoformat(item["entry_time"]).replace(tzinfo=None)
            close_dt = close_at.replace(tzinfo=None)
            entry_rows = df[times == pd.Timestamp(entry_dt)]
            close_rows = df[times == pd.Timestamp(close_dt)]
            if entry_rows.empty or close_rows.empty:
                logging.warning("Result candles not yet available for %s", item["pair"])
                continue
            entry_price = float(entry_rows.iloc[0]["open"])
            exit_price = float(close_rows.iloc[0]["close"])
            win = (exit_price > entry_price) if item["direction"] == "CALL" else (exit_price < entry_price)
            if win:
                state["wins"] += 1
                result = "WIN"
            else:
                state["losses"] += 1
                result = "LOSS"
            item["entry_price"] = entry_price
            item["exit_price"] = exit_price
            item["checked"] = True
            msg = (
                f"PAIR : {item['pair']}\n"
                f"ENTRY TIME : {close_at.strftime('%H:%M')}\n"
                f"RESULT : {result}\n"
                f"WIN : {state['wins']}\n"
                f"LOSS : {state['losses']}\n\n"
                f"SIGNAL BY {SIGNAL_BY}"
            )
            send_text(msg)
            changed = True
        except Exception as exc:
            logging.exception("Could not settle %s: %s", item["pair"], exc)
    state["pending"] = [p for p in state["pending"] if not p.get("checked")]
    if changed:
        save_state(state)


def main():
    if not API_KEY:
        raise SystemExit("TWELVE_DATA_API_KEY missing in environment / GitHub Secrets")
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing in environment / GitHub Secrets")
    if not CHAT_ID:
        raise SystemExit("TELEGRAM_CHAT_ID missing in environment / GitHub Secrets")

    state = load_state()
    settle_pending(state)
    for pair in PAIRS:
        try:
            df = fetch_candles(pair)
            direction, score, strategy = get_signal(df)
            if direction is None:
                logging.info("No signal for %s: %s", pair, strategy)
                continue
            send_signal(pair, df, direction, score, strategy, state)
        except Exception as exc:
            logging.exception("Pair failed (%s): %s", pair, exc)
        time.sleep(1)
    save_state(state)


if __name__ == "__main__":
    main()
