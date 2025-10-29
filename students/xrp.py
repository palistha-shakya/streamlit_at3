# students/aditya_kulkarni.py
import os
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

import altair as alt  # for nicer charts

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = "xrp"
KRAKEN_PAIR = "XXRPZUSD"  # XRP/USD on Kraken (daily interval=1440)

st.title("XRP — Risk & Regimes (Kraken) + Next-Day HIGH Prediction")
st.caption("Source: Kraken OHLC (public). Metrics: volatility, ATR, drawdown, SMA regimes. Prediction from FastAPI.")

# ---------------------------
# Sidebar controls
# ---------------------------
window_days = st.sidebar.selectbox("History window (days)", [90, 180, 365, 730], index=1)
st.sidebar.caption("Different from CoinGecko 7-day chart: deeper window + risk metrics from Kraken.")

# ---------------------------
# Data fetch
# ---------------------------
@st.cache_data(ttl=300)
def fetch_kraken_ohlc(pair: str, interval_mins: int = 1440) -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/OHLC"
    r = requests.get(url, params={"pair": pair, "interval": interval_mins, "since": 0}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    rows = data["result"][pair]
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    for c in ["open","high","low","close","vwap","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df

try:
    full = fetch_kraken_ohlc(KRAKEN_PAIR, interval_mins=1440)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days+2)
    df = full[full["time"] >= cutoff].copy()
except Exception as e:
    st.error(f"Failed to load Kraken OHLC: {e}")
    df = pd.DataFrame()

# ---------------------------
# Metrics & helpers
# ---------------------------
def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # 30d rolling annualised volatility
    vol = df["ret"].rolling(30).std() * np.sqrt(365)
    df["vol30_annual"] = vol

    # ATR-14
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr14_pct"] = (df["atr14"] / df["close"]) * 100.0

    # Drawdown
    cummax = df["close"].cummax()
    df["drawdown"] = (df["close"]/cummax - 1.0) * 100.0

    # SMA regimes
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["regime"] = np.where(df["sma20"] > df["sma50"], "BULL", "BEAR")
    return df

def top_moves(df: pd.DataFrame, n=10):
    d = df.dropna(subset=["ret"]).copy()
    d["pct"] = d["ret"]*100
    best = d.nlargest(n, "pct")[["time","open","high","low","close","pct"]]
    worst = d.nsmallest(n, "pct")[["time","open","high","low","close","pct"]]
    return best, worst

if df.empty:
    st.info("No data loaded yet.")
    st.stop()

df = add_metrics(df)

# ---------------------------
# Charts
# ---------------------------
st.subheader(f"Daily Candles + Volume (last {window_days} days) — Kraken")
price = df[["time","open","high","low","close"]].rename(columns={"time":"Date"})
volume = df[["time","volume"]].rename(columns={"time":"Date"})

# Candlestick with Altair
base = alt.Chart(price).encode(x="Date:T")
rule = base.mark_rule().encode(y="low:Q", y2="high:Q")
bar = base.mark_bar().encode(y="open:Q", y2="close:Q",
                             color=alt.condition("datum.open <= datum.close",
                                                 alt.value("#16a34a"),  # green
                                                 alt.value("#dc2626"))) # red
candles = (rule + bar).properties(height=280)

vol_chart = alt.Chart(volume).mark_bar().encode(
    x="Date:T", y=alt.Y("volume:Q", title="Volume")
).properties(height=80)

st.altair_chart((candles & vol_chart).resolve_scale(x="shared"), use_container_width=True)

# Risk panels
c1, c2, c3 = st.columns(3)
c1.metric("Last Close (USD)", f"{df['close'].iloc[-1]:,.6f}")
c2.metric("Curr. Drawdown", f"{df['drawdown'].iloc[-1]:.2f}%")
reg = df["regime"].iloc[-1]
c3.metric("Regime (20/50 SMA)", reg)

st.subheader("Risk Metrics")
risk_left, risk_right = st.columns(2)

with risk_left:
    st.caption("30-day Annualised Volatility")
    st.line_chart(df.set_index("time")[["vol30_annual"]])

with risk_right:
    st.caption("ATR-14 (% of price)")
    st.line_chart(df.set_index("time")[["atr14_pct"]])

st.caption("Max Drawdown (since window start)")
st.area_chart(df.set_index("time")[["drawdown"]])

best, worst = top_moves(df, n=10)
st.subheader("Top 10 Moves")
st.write("⬆️ Largest Up Days")
st.dataframe(best.rename(columns={"pct":"return_%"}), use_container_width=True)
st.write("⬇️ Largest Down Days")
st.dataframe(worst.rename(columns={"pct":"return_%"}), use_container_width=True)

# ---------------------------
# Prediction (number only)
# ---------------------------
st.subheader("Predicted Next-Day HIGH (t+1)")
colA, colB = st.columns([1,1])

with colA:
    if st.button("Predict from model (use embedded last_row_features)"):
        r = requests.post(f"{API_BASE}/predict_value/{TOKEN}", json={})
        if not r.ok:
            # fallback to full endpoint
            r = requests.post(f"{API_BASE}/predict/{TOKEN}", json={})
        if r.ok:
            st.metric("Predicted t+1 HIGH (USD)", f"{r.json()['prediction']:.6f}")
        else:
            st.error(f"API error {r.status_code}: {r.text}")

with colB:
    # Optional custom override: infer features from last row of Kraken data
    with st.popover("Advanced: predict using features derived from last row"):
        # Build features expected by your model
        last = df.iloc[-1]
        # Derive your model's six features:
        # close, volume, marketCap (unknown from Kraken; set NaN/0 or ask CoinGecko),
        # high_ret_1, log_marketCap, log_volume
        # Here we fill marketCap with 0 (or you can fetch CoinGecko mcap if desired)
        features = {
            "close": float(last["close"]),
            "volume": float(last["volume"]),
            "marketCap": float(0.0),  # Optionally fetch CG market cap to be exact
            "high_ret_1": float(last["high"]/df["high"].iloc[-2] - 1.0) if len(df) > 1 else 0.0,
            "log_marketCap": float(np.log1p(0.0)),
            "log_volume": float(np.log1p(max(last["volume"], 0.0))),
        }
        st.json(features)
        if st.button("Predict with above features"):
            r = requests.post(f"{API_BASE}/predict_value/{TOKEN}", json={"features": features})
            if not r.ok:
                r = requests.post(f"{API_BASE}/predict/{TOKEN}", json={"features": features})
            if r.ok:
                st.metric("Predicted t+1 HIGH (USD)", f"{r.json()['prediction']:.6f}")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
