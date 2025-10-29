# students/aditya_kulkarni.py
# XRP — Risk & Regimes (Kraken) + Prediction (number only)
import os
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# Config
# ---------------------------
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = "xrp"
KRAKEN_PAIR = "XXRPZUSD"  # XRP/USD on Kraken (daily candles = interval=1440)

st.title("XRP — Risk & Regimes (Kraken) + Next-Day HIGH Prediction")
st.caption("Different from CG 7-day view: deeper window, risk analytics (volatility, ATR, drawdown, SMA regimes) from Kraken. Prediction shows only the number.")

# ---------------------------
# Sidebar & inputs
# ---------------------------
window_days = st.sidebar.selectbox("History window (days)", [90, 180, 365, 730], index=1)
st.sidebar.caption("Tip: increase the window to see longer regime shifts.")

# ---------------------------
# Data fetch (Kraken) with timezone-safe handling
# ---------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kraken_ohlc(pair: str, interval_mins: int = 1440) -> pd.DataFrame:
    """
    Returns a timezone-naive UTC DataFrame with columns:
    time, open, high, low, close, vwap, volume, count
    """
    url = "https://api.kraken.com/0/public/OHLC"
    try:
        r = requests.get(url, params={"pair": pair, "interval": interval_mins, "since": 0}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            # Kraken sometimes returns ["EAPI:Rate limit exceeded"] etc.
            raise RuntimeError(f"Kraken error: {data['error']}")
        rows = data["result"][pair]
        df = pd.DataFrame(
            rows,
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
        )
        # parse epoch seconds as UTC, then drop tz -> naive
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        for c in ["open", "high", "low", "close", "vwap", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        # Return empty DF with expected columns so the caller can handle gracefully
        cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
        empty = pd.DataFrame(columns=cols)
        empty.attrs["__error__"] = str(e)
        return empty

full = fetch_kraken_ohlc(KRAKEN_PAIR, interval_mins=1440)

# Display fetch error if any
fetch_err = full.attrs.get("__error__")
if fetch_err:
    st.error(f"Failed to load Kraken OHLC: {fetch_err}")
    st.stop()

if full.empty:
    st.warning("Kraken returned no rows. Try again in a minute (rate limiting) or widen your window.")
    st.stop()

# Build a timezone-naive UTC cutoff and filter
cutoff = (pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None)
          - pd.Timedelta(days=window_days + 2))
df = full[full["time"] >= cutoff].copy()

if df.empty:
    st.info("No rows after filtering by window. Try increasing the window or refreshing the page.")
    st.stop()

# ---------------------------
# Metrics & helpers
# ---------------------------
def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # 30-day rolling annualised volatility
    df["vol30_annual"] = df["ret"].rolling(30).std() * np.sqrt(365)

    # ATR-14 (Average True Range)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr14_pct"] = (df["atr14"] / df["close"]) * 100.0

    # Drawdown from rolling max close
    df["drawdown"] = (df["close"] / df["close"].cummax() - 1.0) * 100.0

    # SMA regimes
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["regime"] = np.where(df["sma20"] > df["sma50"], "BULL", "BEAR")
    return df

def top_moves(df: pd.DataFrame, n=10):
    d = df.dropna(subset=["ret"]).copy()
    d["pct"] = d["ret"] * 100
    best = d.nlargest(n, "pct")[["time", "open", "high", "low", "close", "pct"]]
    worst = d.nsmallest(n, "pct")[["time", "open", "high", "low", "close", "pct"]]
    return best, worst

df = add_metrics(df)

# ---------------------------
# Charts
# ---------------------------
st.subheader(f"Daily Candles + Volume — Kraken (last {window_days} days)")
price = df[["time", "open", "high", "low", "close"]].rename(columns={"time": "Date"})
volume = df[["time", "volume"]].rename(columns={"time": "Date"})

base = alt.Chart(price).encode(x="Date:T")
rule = base.mark_rule().encode(y="low:Q", y2="high:Q")
bar = base.mark_bar().encode(
    y="open:Q",
    y2="close:Q",
    color=alt.condition("datum.open <= datum.close", alt.value("#16a34a"), alt.value("#dc2626")),
)
candles = (rule + bar).properties(height=280)

vol_chart = (
    alt.Chart(volume)
    .mark_bar()
    .encode(x="Date:T", y=alt.Y("volume:Q", title="Volume"))
    .properties(height=80)
)

st.altair_chart((candles & vol_chart).resolve_scale(x="shared"), use_container_width=True)

# Quick KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Last Close (USD)", f"{df['close'].iloc[-1]:,.6f}")
c2.metric("Curr. Drawdown", f"{df['drawdown'].iloc[-1]:.2f}%")
c3.metric("Regime (20/50 SMA)", df["regime"].iloc[-1])

# Risk panels
st.subheader("Risk Metrics")
left, right = st.columns(2)
with left:
    st.caption("30-day Annualised Volatility")
    st.line_chart(df.set_index("time")[["vol30_annual"]])
with right:
    st.caption("ATR-14 (% of price)")
    st.line_chart(df.set_index("time")[["atr14_pct"]])

st.caption("Max drawdown (since window start)")
st.area_chart(df.set_index("time")[["drawdown"]])

best, worst = top_moves(df, n=10)
st.subheader("Top 10 Moves")
st.write("⬆️ Largest Up Days")
st.dataframe(best.rename(columns={"pct": "return_%"}), use_container_width=True)
st.write("⬇️ Largest Down Days")
st.dataframe(worst.rename(columns={"pct": "return_%"}), use_container_width=True)

# ---------------------------
# Prediction (number only)
# ---------------------------
st.subheader("Predicted Next-Day HIGH (t+1)")
colA, colB = st.columns(2)

with colA:
    if st.button("Predict (use model's embedded last_row_features)"):
        try:
            r = requests.post(f"{API_BASE}/predict_value/{TOKEN}", json={}, timeout=30)
            if not r.ok:
                r = requests.post(f"{API_BASE}/predict/{TOKEN}", json={}, timeout=30)
            if r.ok:
                st.metric("Predicted t+1 HIGH (USD)", f"{r.json()['prediction']:.6f}")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

with colB:
    # Optional: derive features from last Kraken row to override
    with st.popover("Advanced: derive features from last Kraken row"):
        last = df.iloc[-1]
        # Your model's features:
        # ["close","volume","marketCap","high_ret_1","log_marketCap","log_volume"]
        # We don't have market cap from Kraken; keep 0.0 or fetch CoinGecko if desired.
        features = {
            "close": float(last["close"]),
            "volume": float(last["volume"]),
            "marketCap": float(0.0),  # optional: replace via CoinGecko latest market cap
            "high_ret_1": float(last["high"] / df["high"].iloc[-2] - 1.0) if len(df) > 1 else 0.0,
            "log_marketCap": float(np.log1p(0.0)),
            "log_volume": float(np.log1p(max(last["volume"], 0.0))),
        }
        st.json(features)
        if st.button("Predict with above features"):
            try:
                r = requests.post(f"{API_BASE}/predict_value/{TOKEN}", json={"features": features}, timeout=30)
                if not r.ok:
                    r = requests.post(f"{API_BASE}/predict/{TOKEN}", json={"features": features}, timeout=30)
                if r.ok:
                    st.metric("Predicted t+1 HIGH (USD)", f"{r.json()['prediction']:.6f}")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
