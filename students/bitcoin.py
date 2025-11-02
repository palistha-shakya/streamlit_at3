# students/bitcoin.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os

# =========================
# Config & constants
# =========================
API_BASE = st.secrets.get(
    "API_BASE",
    "https://three6120-25sp-group27-25402328-at3-api.onrender.com"
)

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
    "ripple": "ripple",   # XRP
    "solana": "solana",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
KRAKEN_BASE    = "https://api.kraken.com"

# =========================
# Lightweight local file cache (optional)
# =========================
def _save_tmp(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception:
        pass

def _load_tmp(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

# =========================
# HTTP helpers (with headers/key/backoff)
# =========================
def _rate_limited_get(url, params=None, timeout=30, retries=5, backoff=1.8):
    """GET with exponential backoff; respects Retry-After for 429. Adds UA and optional CoinGecko key."""
    params = params or {}
    last_err = None
    cg_key = st.secrets.get("COINGECKO_API_KEY", None)
    headers = {
        "Accept": "application/json",
        "User-Agent": "streamlit-at3-group27/1.0 (+https://appat3-group-27.streamlit.app)"
    }
    if cg_key:
        headers["x-cg-pro-api-key"] = cg_key

    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or int(backoff ** (i + 1))
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                r.raise_for_status()
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise last_err

def _kraken_get(url, params=None, timeout=30, retries=4, backoff=1.8):
    params = params or {}
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            if js.get("error"):
                raise RuntimeError(js["error"])
            return js["result"]
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise last_err

# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=3600)
def fetch_coingecko_ohlc(coin_id: str, days="365") -> pd.DataFrame:
    """CoinGecko OHLC (no volume). Default 365d to avoid rate limit; 'max' possible but slower."""
    cache_path = f"/tmp/{coin_id}_ohlc_{days}.parquet"
    cached = _load_tmp(cache_path)
    if cached is not None and len(cached) > 0:
        return cached

    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    r = _rate_limited_get(url, params=params, timeout=40, retries=6, backoff=2.0)
    data = r.json() or []

    if not data:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        _save_tmp(df, cache_path)
        return df

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    df["date"] = (
        pd.to_datetime(df["ts"], unit="ms", utc=True)
          .dt.tz_convert("UTC")
          .dt.normalize()
    )
    df = df.drop(columns=["ts"]).drop_duplicates(subset=["date"])
    df = df[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)
    _save_tmp(df, cache_path)
    return df

@st.cache_data(ttl=3600)
def fetch_coingecko_volume(coin_id: str, days="365") -> pd.DataFrame:
    """CoinGecko market_chart -> total_volumes + market_caps (daily)."""
    cache_path = f"/tmp/{coin_id}_volume_{days}.parquet"
    cached = _load_tmp(cache_path)
    if cached is not None and len(cached) > 0:
        return cached

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = _rate_limited_get(url, params=params, timeout=40, retries=6, backoff=2.0)
    js = r.json() or {}

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
    ).sort_values("date").reset_index(drop=True)

    _save_tmp(out, cache_path)
    return out

@st.cache_data(ttl=3600)
def fetch_kraken_ohlc_btc(days: int = 365) -> pd.DataFrame:
    """
    Kraken fallback for BTC (USD, daily interval).
    pair: XXBTZUSD, interval=1440 (daily). Includes volume.
    """
    now = int(time.time())
    since = now - days * 86400
    url = f"{KRAKEN_BASE}/0/public/OHLC"
    params = {"pair": "XXBTZUSD", "interval": 1440, "since": since}
    res = _kraken_get(url, params=params)

    series = None
    for k, v in res.items():
        if isinstance(v, list):
            series = v
            break
    if not series:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    cols = ["time","open","high","low","close","vwap","volume","count"]
    df = pd.DataFrame(series, columns=cols)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("UTC").dt.normalize()
    df = df[["date","open","high","low","close","volume"]].astype(
        {"open":float,"high":float,"low":float,"close":float,"volume":float}
    )
    return df.sort_values("date").reset_index(drop=True)

# =========================
# Feature engineering
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    df["close_pct"]  = df["close"].pct_change()
    df["high_pct"]   = df["high"].pct_change()
    df["volume_pct"] = df["volume"].pct_change()

    df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan)

    df["range_roll3_mean"] = df["high_low_range_pct"].rolling(3).mean()
    df["range_roll7_std"]  = df["high_low_range_pct"].rolling(7).std()

    df["volume_roll3_mean"] = df["volume"].rolling(3).mean()
    df["volume_roll7_mean"] = df["volume"].rolling(7).mean()

    df["close_roll3_mean"] = df["close"].rolling(3).mean()
    df["close_roll7_mean"] = df["close"].rolling(7).mean()

    df["volume_log1p"] = np.log1p(df["volume"].clip(lower=0))

    for k in [1, 3, 7]:
        df[f"close_pct_lag{k}"] = df["close_pct"].shift(k)
        df[f"high_pct_lag{k}"]  = df["high_pct"].shift(k)

    df["range_zscore_7"]  = (df["high_low_range_pct"] - df["high_low_range_pct"].rolling(7).mean()) / (df["high_low_range_pct"].rolling(7).std() + 1e-12)
    df["range_zscore_14"] = (df["high_low_range_pct"] - df["high_low_range_pct"].rolling(14).mean()) / (df["high_low_range_pct"].rolling(14).std() + 1e-12)

    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr / df["close"].shift(1).replace(0, np.nan)
    df["atr_pct_roll7"] = df["atr_pct"].rolling(7).mean()

    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df

# =========================
# API call to your FastAPI
# =========================
def call_predict_api(features: dict) -> float:
    r = requests.get(f"{API_BASE}/predict/bitcoin", params=features, timeout=25)
    r.raise_for_status()
    js = r.json()
    return float(js["prediction_next_day_high_usd"])

# =========================
# UI entry
# =========================
def render(token: str):
    st.header("Bitcoin (Student: Zhikang)")

    if token != "bitcoin":
        st.info("This tab is for Bitcoin. Change token to Bitcoin to enable prediction.")
        return


    rng = st.selectbox(
        "History range",
        ["90 days", "180 days", "365 days", "5 years", "max (slow)"],
        index=1
    )
    days_map = {
        "90 days": "90",
        "180 days": "180",
        "365 days": "365",
        "5 years": "1825",
        "max (slow)": "max",
    }
    days = days_map[rng]


    with st.spinner(f"Fetching OHLC & volume ({days})..."):
        try:
            ohlc = fetch_coingecko_ohlc(ID_MAP[token], days=days)
            vol  = fetch_coingecko_volume(ID_MAP[token], days=days)
            df = ohlc.merge(vol, on="date", how="inner")
            if len(df) < 10:
                raise RuntimeError("Too few rows after merge, triggering fallback.")
        except Exception as e:
            st.info(f"CoinGecko not available ({str(e)[:80]}). Falling back to Kraken…")
            fallback_days = 365 if (not days.isdigit()) else int(days)
            df = fetch_kraken_ohlc_btc(days=fallback_days)


    st.subheader("Historical price (Close) and Volume")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("date")["close"], height=260)
    with c2:
        st.area_chart(df.set_index("date")["volume"], height=260)


    df_feat = build_features(df)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
    if df_feat.empty:
        st.error("Not enough data to build features. Try a longer history range.")
        return
    last = df_feat.iloc[-1].copy()

    feats = {
        k: float(np.nan_to_num(last.get(k, np.nan), nan=0.0, posinf=0.0, neginf=0.0))
        for k in FEATURES
    }

    st.caption("Latest feature snapshot used for prediction:")
    st.dataframe(pd.DataFrame([feats]), use_container_width=True)


    try:
        with st.spinner("Calling API for next-day HIGH prediction..."):
            pred = call_predict_api(feats)
        st.success(f"Predicted next-day HIGH (USD): {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


    with st.expander("Debug • Connectivity & cache", expanded=False):
        st.write("Resolved API_BASE:", API_BASE)
        st.write("Rows:", len(df), "| Date range:", df["date"].min() if len(df) else None, "→", df["date"].max() if len(df) else None)
        try:
            hr = requests.get(f"{API_BASE}/health/", timeout=10)
            st.write("/health status:", hr.status_code, hr.text[:200])
        except Exception as e:
            st.error(f"/health failed: {e}")

