import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from students.ethereum import run_eth_tab
from students.bitcoin import render as run_btc_tab

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.title("Crypto Forecast Dashboard")
st.write("Access real-time data and next-day price predictions for Ethereum.")

# single tab for Ethereum
tab1, tab2 = st.tabs(["Bitcoin", "Ethereum"])
with tab1:
    run_btc_tab(token="bitcoin")

with tab2:
    run_eth_tab()
