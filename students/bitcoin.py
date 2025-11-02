# students/bitcoin.py
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    "ripple": "ripple",  # XRP
    "solana": "solana",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
KRAKEN_BASE    = "https://api.kraken.com"

# =========================
# Lightweight local CSV cache (no pyarrow dependency)
# =========================
def _save_tmp(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass

def _load_tmp(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            # typical date column for OHLC/volume caches
            if path.endswith(".csv"):
                return pd.read_csv(path, parse_dates=["date"])
            return pd.read_csv(path)
    except Exception:
        return None
    return None

# =========================
# HTTP session with retries
# =========================
@st.cache_resource
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=(429, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "streamlit-at3-group27/1.0 (+https://appat3-group-27.streamlit.app)"})
    return s

# =========================
# HTTP helpers (UA, optional CG key, backoff)
# =========================
def _rate_limited_get(url, params=None, timeout=30, retries=5, backoff=1.8):
    """GET with exponential backoff; respects Retry-After for 429. Adds UA and optional CoinGecko key."""
    params = params or {}
    last_err = None
    cg_key = st.secrets.get("COINGECKO_API_KEY", None)
    headers = {"Accept": "application/json"}
    if cg_key:
        headers["x-cg-pro-api-key"] = cg_key

    session = _http_session()
    for i in range(retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or int(backoff ** (i + 1))
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                r.raise_for_status()  # trigger fallback
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise last_err

def _kraken_get(url, params=None, timeout=30, retries=4, backoff=1.8):
    params = params or {}
    last_err = None
    session = _http_session()
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
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
@st.cache_data(ttl=900)
def fetch_cg_market_chart_fast(coin_id: str, days: str = "7") -> pd.DataFrame:
    """
    Fast path: single /market_chart call for close/volume/marketCap.
    Approximate OHLC via rolling window to avoid /ohlc rate limits.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = _rate_limited_get(url, params=params, timeout=25, retries=4, backoff=1.8)
    js = r.json() or {}

    prices = js.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts", "close"])
    if df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","marketCap"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
    df.drop(columns=["ts"], inplace=True)

    vols = pd.DataFrame(js.get("total_volumes", []), columns=["ts","volume"])
    caps = pd.DataFrame(js.get("market_caps", []),     columns=["ts","marketCap"])
    for d in (vols, caps):
        if not d.empty:
            d["date"] = pd.to_datetime(d["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
            d.drop(columns=["ts"], inplace=True, errors="ignore")

    if not vols.empty:
        df = df.merge(vols[["date","volume"]], on="date", how="left")
    else:
        df["volume"] = np.nan
    if not caps.empty:
        df = df.merge(caps[["date","marketCap"]], on="date", how="left")
    else:
        df["marketCap"] = np.nan

    df.sort_values("date", inplace=True)
    df["open"] = df["close"].shift(1).bfill()
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"]  = df["close"].rolling(3, min_periods=1).min()

    return df[["date","open","high","low","close","volume","marketCap"]]

@st.cache_data(ttl=3600)
def fetch_coingecko_ohlc(coin_id: str, days="365") -> pd.DataFrame:
    """CoinGecko OHLC (no volume). Default 365 to reduce rate limits; 'max' allowed."""
    cache_path = f"/tmp/{coin_id}_ohlc_{days}.csv"
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
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
    df.drop(columns=["ts"], inplace=True)
    df.drop_duplicates(subset=["date"], inplace=True)
    df = df[["date","open","high","low","close"]].sort_values("date").reset_index(drop=True)
    _save_tmp(df, cache_path)
    return df

@st.cache_data(ttl=3600)
def fetch_coingecko_volume(coin_id: str, days="365") -> pd.DataFrame:
    """CoinGecko market_chart -> total_volumes + market_caps (daily)."""
    cache_path = f"/tmp/{coin_id}_volume_{days}.csv"
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
    df_v.drop(columns=["ts"], inplace=True)

    df_c = pd.DataFrame(caps, columns=["ts", "marketCap"])
    df_c["date"] = pd.to_datetime(df_c["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
    df_c.drop(columns=["ts"], inplace=True)

    out = pd.merge_asof(
        df_v.sort_values("date"),
        df_c.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta("1D")
    ).sort_values("date").reset_index(drop=True)

    _save_tmp(out, cache_path)
    return out

@st.cache_data(ttl=3600)
def fetch_kraken_ohlc_btc(days: int | str = 365) -> pd.DataFrame:
    """
    Kraken fallback for BTC (USD, daily interval).
    Accepts days as int or str; 'max' or invalid string -> 1825 (~5y).
    Includes volume.
    """
    if isinstance(days, str):
        if days.isdigit():
            days = int(days)
        else:
            days = 1825
    days = max(int(days), 30)  # minimal 30 days

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
# API warm-up & prediction
# =========================
def _warm_up_api(base: str):
    """Ping Render app to wake cold start; tolerate failures."""
    session = _http_session()
    try:
        for path in ["", "/", "/health", "/health/"]:
            try:
                session.get(f"{base}{path}", timeout=8)
            except Exception:
                pass
    except Exception:
        pass

def _compact_features(feats: dict, ndigits: int = 6) -> dict:
    out = {}
    for k, v in feats.items():
        try:
            out[k] = float(round(float(v), ndigits))
        except Exception:
            out[k] = 0.0
    return out

def call_predict_api(features: dict) -> float:
    session = _http_session()
    params = _compact_features(features, ndigits=6)

    # Warm up (Render cold start)
    _warm_up_api(API_BASE)

    # Try GET first (assignment spec)
    try:
        r = session.get(f"{API_BASE}/predict/bitcoin", params=params, timeout=60)
        if r.ok:
            js = r.json()
            return float(
                js.get("prediction_next_day_high_usd")
                or js.get("prediction")
                or js.get("predicted_high")
                or js.get("predicted_next_day_high")
            )
    except Exception:
        pass

    # Fallback: POST
    r = session.post(f"{API_BASE}/predict/bitcoin", json=params, timeout=60)
    if r.ok:
        js = r.json()
        return float(
            js.get("prediction_next_day_high_usd")
            or js.get("prediction")
            or js.get("predicted_high")
            or js.get("predicted_next_day_high")
        )
    raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")

# =========================
# UI entry
# =========================
def render(token: str):
    st.header("Bitcoin (Student: Zhikang)")

    if token != "bitcoin":
        st.info("This tab is for Bitcoin. Change token to Bitcoin to enable prediction.")
        return

    # Data mode
    mode = st.radio(
        "Data mode",
        ["Fast (market_chart, approximate OHLC)", "Full (OHLC + fallback)"],
        index=0,
        help="Fast: single API call, low rate-limit risk. Full: true OHLC with Kraken fallback."
    )

    # History window
    if mode.startswith("Fast"):
        rng = st.selectbox("History range", ["7 days", "14 days", "30 days", "90 days"], index=2)
        days_map = {"7 days":"7","14 days":"14","30 days":"30","90 days":"90"}
    else:
        rng = st.selectbox("History range", ["90 days", "180 days", "365 days", "5 years", "max (slow)"], index=1)
        days_map = {"90 days":"90","180 days":"180","365 days":"365","5 years":"1825","max (slow)":"max"}
    days = days_map[rng]

    # Fetch data
    if mode.startswith("Fast"):
        with st.spinner(f"Fetching market_chart (fast) {days}…"):
            try:
                df = fetch_cg_market_chart_fast(ID_MAP[token], days=days)
                if df.empty or len(df) < 5:
                    st.warning("Too few rows from market_chart. Try a longer window.")
                    st.stop()
            except Exception as e:
                st.error(f"Fast mode failed: {e}")
                st.stop()
    else:
        with st.spinner(f"Fetching OHLC & volume ({days})..."):
            try:
                ohlc = fetch_coingecko_ohlc(ID_MAP[token], days=days)
                vol  = fetch_coingecko_volume(ID_MAP[token], days=days)
                df = ohlc.merge(vol, on="date", how="inner")
                if len(df) < 10:
                    raise RuntimeError("Too few rows after merge, triggering fallback.")
            except Exception as e:
                st.info(f"CoinGecko not available ({str(e)[:80]}). Falling back to Kraken…")
                fallback_days = int(days) if isinstance(days, str) and days.isdigit() else (int(days) if isinstance(days, int) else 1825)
                df = fetch_kraken_ohlc_btc(days=fallback_days)

    # Charts
    st.subheader("Historical price (Close) and Volume")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("date")["close"], height=260)
    with c2:
        st.area_chart(df.set_index("date")["volume"], height=260)

    # Features + cleaning
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

    # Prediction (button-triggered)
    st.subheader("Predict Next-Day HIGH (t+1)")
    if st.button("Predict"):
        try:
            with st.spinner("Calling API for next-day HIGH prediction…"):
                pred = call_predict_api(feats)
            st.success(f"Predicted next-day HIGH (USD): {pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Debug
    with st.expander("Debug • Connectivity & cache", expanded=False):
        st.write("Resolved API_BASE:", API_BASE)
        st.write("Rows:", len(df), "| Date range:", (df["date"].min() if len(df) else None), "→", (df["date"].max() if len(df) else None))
        try:
            hr = _http_session().get(f"{API_BASE}/health/", timeout=10)
            st.write("/health status:", hr.status_code, hr.text[:200])
        except Exception as e:
            st.error(f"/health failed: {e}")
