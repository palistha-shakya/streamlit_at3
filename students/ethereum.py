import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np

# Your deployed FastAPI endpoint
FASTAPI_URL = "https://ethereum-currency-25548520-latest.onrender.com"


def run_eth_tab():
    st.title("Ethereum Dashboard – Real-Time Data & Prediction")
    st.write(
        "Displays live Ethereum data from CoinGecko and predicts the next-day HIGH price using the deployed ML model."
    )

    # SECTION 1: Fetch Data + Feature Engineering
    st.header("Live Ethereum Market Data")

    market_chart_url = (
        "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        "?vs_currency=usd&days=7"
    )

    try:
        market_data = requests.get(market_chart_url).json()

        if not market_data or "prices" not in market_data:
            st.warning("No recent market data available. Try again later.")
            return

        # Create base DataFrame
        df = pd.DataFrame(market_data["prices"], columns=["time", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        # Approximate OHLC values
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(3, min_periods=1).max()
        df["low"] = df["close"].rolling(3, min_periods=1).min()
        df.bfill(inplace=True)

        # Feature Engineering
        # Daily Return
        df["daily_return"] = df["close"].pct_change()
        df["daily_return"] = df["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 7-Day Moving Average
        df["ma_7"] = df["close"].rolling(window=7, min_periods=1).mean()

        # 7-Day Rolling Volatility
        df["volatility_7"] = df["daily_return"].rolling(window=7, min_periods=1).std().fillna(0)

        # Latest market snapshot
        latest = df.iloc[-1]
        volume = market_data["total_volumes"][-1][1]
        market_cap = market_data["market_caps"][-1][1]

        # Prepare features for API
        features = {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "ma_7": float(latest["ma_7"]),
            "volatility_7": float(latest["volatility_7"]),
            "daily_return": float(latest["daily_return"]),
            "volume": float(volume),
            "marketCap": float(market_cap),
        }

        # Display metrics
        cols = st.columns(5)
        cols[0].metric("Open", f"${features['open']:,.2f}")
        cols[1].metric("High", f"${features['high']:,.2f}")
        cols[2].metric("Low", f"${features['low']:,.2f}")
        cols[3].metric("Volume (M)", f"{features['volume']/1e6:,.2f}")
        cols[4].metric("Market Cap (B)", f"{features['marketCap']/1e9:,.2f}")

        # Display chart
        st.subheader("Ethereum Price Trend (7 Days)")
        fig = px.line(df, x="time", y="close", title="Ethereum Close Price (7 Days)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(), use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data from CoinGecko: {e}")
        return

    # SECTION 2: Automatic Prediction
    st.header("Predict Next-Day HIGH (Automatic)")
    st.write("Uses the latest engineered features to call the FastAPI model.")

    if st.button("Predict Automatically"):
        try:
            res = requests.get(f"{FASTAPI_URL}/predict/ethereum", params=features)
            if res.status_code == 200:
                data = res.json()
                predicted = (
                    data.get("predicted_next_day_high")
                    or data.get("predicted_high")
                    or data.get("predicted_high_price")
                )
                if predicted is not None:
                    st.success(f"Predicted Next-Day HIGH: **${predicted:,.2f} USD**")
                else:
                    st.warning("No prediction key found in API response.")
            else:
                st.error(f"API error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

    # SECTION 3: Manual Prediction
    st.header("Predict Next-Day HIGH (Manual Input)")
    st.write("Enter your own feature values to test the model manually.")

    with st.form("manual_form"):
        open_p = st.number_input("Open", value=features["open"])
        high_p = st.number_input("High", value=features["high"])
        low_p = st.number_input("Low", value=features["low"])
        close_p = st.number_input("Close", value=features["close"])
        ma7_p = st.number_input("MA_7 (7-day Moving Avg)", value=features["ma_7"])
        vol7_p = st.number_input("Volatility_7 (7-day Rolling)", value=features["volatility_7"])
        ret_p = st.number_input("Daily Return", value=features["daily_return"])
        vol_p = st.number_input("Volume", value=features["volume"])
        mcap_p = st.number_input("Market Cap", value=features["marketCap"])
        submitted = st.form_submit_button("Predict Manually")

        if submitted:
            params = {
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "ma_7": ma7_p,
                "volatility_7": vol7_p,
                "daily_return": ret_p,
                "volume": vol_p,
                "marketCap": mcap_p,
            }
            try:
                res = requests.get(f"{FASTAPI_URL}/predict/ethereum", params=params)
                if res.status_code == 200:
                    data = res.json()
                    predicted = (
                        data.get("predicted_next_day_high")
                        or data.get("predicted_high")
                        or data.get("predicted_high_price")
                    )
                    if predicted is not None:
                        st.success(f"Predicted Next-Day HIGH: **${predicted:,.2f} USD**")
                    else:
                        st.warning("No prediction returned from API.")
                else:
                    st.error(f"API error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"API request failed: {e}")
