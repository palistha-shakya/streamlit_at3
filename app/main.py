import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from students.ethereum import run_eth_tab
from students.bitcoin import render as run_btc_tab
from students.xrp import run_xrp_tab

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.title("Crypto Forecast Dashboard")
st.write("Access real-time data and next-day price predictions for selected cryptocurrency.")

tab = st.selectbox("Select Cryptocurrency", ["Ethereum", "Bitcoin", "XRP"])

if tab == "Ethereum":
    run_eth_tab()
elif tab == "Bitcoin":
    run_btc_tab(token="bitcoin")
elif tab == "XRP":
    run_xrp_tab()
