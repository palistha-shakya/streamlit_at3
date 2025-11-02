# students/bitcoin.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import math


API_BASE = st.secrets.get("API_BASE", "https://three6120-25sp-group27-25402328-at3-api.onrender.com")


FEATURES = [
    "close_pct", "high_pct", "volume_pct", "high_low_range_pct",
    "range_roll3_mean", "range_roll7_std",
    "volume_roll3_mean", "volume_roll7_mean",
    "close_roll3_mean", "close_roll7_mean",
    "volume_log1p",
    "close_pct_lag1", "close_pct_lag3", "close_pct_lag7",
    "high_pct_lag1", "high_pct_lag3", "high_pct_lag7",
    "range_zscore_7", "range_zscore_14",
    "atr_pct", "atr_pct_roll7",
    "dow_sin", "dow_cos",
]

ID_MAP = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "ripple": "ripple",
    "solana": "solana",
}

def fetch_coingecko_ohlc(token_id: str, days="90"):
    """Returns DataFrame: date, open, high, low, close (daily)."""
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
        df = df.drop(columns=["ts"]).sort_values("date").reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch CoinGecko OHLC data: {e}")
        return pd.DataFrame()


def fetch_coingecko_volume(token_id: str, days="max"):
    """Returns DataFrame: date, volume, marketCap (daily)."""
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    vols = js.get("total_volumes", [])
    caps = js.get("market_caps", [])
    df_v = pd.DataFrame(vols, columns=["ts", "volume"])
    df_v["date"] = pd.to_datetime(df_v["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
    df_v = df_v.drop(columns=["ts"])

    df_c = pd.DataFrame(caps, columns=["ts", "marketCap"])
    df_c["date"] = pd.to_datetime(df_c["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
    df_c = df_c.drop(columns=["ts"])
    out = pd.merge_asof(
        df_v.sort_values("date"), df_c.sort_values("date"),
        on="date", direction="nearest", tolerance=pd.Timedelta("1D")
    )
    return out.sort_values("date").reset_index(drop=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given daily OHLCV df, compute model FEATURES on last row."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # pct changes
    df["close_pct"] = df["close"].pct_change()
    df["high_pct"]  = df["high"].pct_change()
    df["volume_pct"] = df["volume"].pct_change()

    # ranges & rolling stats
    df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan)

    df["range_roll3_mean"] = df["high_low_range_pct"].rolling(3).mean()
    df["range_roll7_std"]  = df["high_low_range_pct"].rolling(7).std()

    df["volume_roll3_mean"] = df["volume"].rolling(3).mean()
    df["volume_roll7_mean"] = df["volume"].rolling(7).mean()

    df["close_roll3_mean"] = df["close"].rolling(3).mean()
    df["close_roll7_mean"] = df["close"].rolling(7).mean()

    df["volume_log1p"] = np.log1p(df["volume"].clip(lower=0))

    # lags
    for k in [1, 3, 7]:
        df[f"close_pct_lag{k}"] = df["close_pct"].shift(k)
        df[f"high_pct_lag{k}"]  = df["high_pct"].shift(k)

    # zscores
    df["range_zscore_7"]  = (df["high_low_range_pct"] - df["high_low_range_pct"].rolling(7).mean()) / (df["high_low_range_pct"].rolling(7).std() + 1e-12)
    df["range_zscore_14"] = (df["high_low_range_pct"] - df["high_low_range_pct"].rolling(14).mean()) / (df["high_low_range_pct"].rolling(14).std() + 1e-12)

    # ATR%
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr / df["close"].shift(1).replace(0, np.nan)
    df["atr_pct_roll7"] = df["atr_pct"].rolling(7).mean()

    # Day-of-week encoding
    dow = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)


    return df

def call_predict_api(features: dict) -> float:
    url = f"{API_BASE}/predict/bitcoin"
    r = requests.get(url, params=features, timeout=20)
    r.raise_for_status()
    js = r.json()
    return float(js["prediction_next_day_high_usd"])

def render(token: str):
    st.header("Bitcoin (Student: Zhikang)")

    if token != "bitcoin":
        st.info("This tab is for Bitcoin. Change token to Bitcoin to enable prediction.")
        return


    with st.spinner("Fetching OHLC and volume from CoinGecko..."):
        ohlc = fetch_coingecko_ohlc(ID_MAP[token], days="max")
        vol  = fetch_coingecko_volume(ID_MAP[token], days="max")
        df = ohlc.merge(vol, on="date", how="inner")

    st.subheader("Historical price (Close) and Volume")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("date")["close"], height=260)
    with c2:
        st.area_chart(df.set_index("date")["volume"], height=260)


    df_feat = build_features(df)
    last = df_feat.iloc[-1].copy()


    feats = {k: float(last[k]) for k in FEATURES}

    st.caption("Latest feature snapshot used for prediction:")
    st.dataframe(pd.DataFrame([feats]), use_container_width=True)


    try:
        with st.spinner("Calling API for next-day HIGH prediction..."):
            pred = call_predict_api(feats)
        st.success(f"Predicted next-day HIGH (USD): {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
