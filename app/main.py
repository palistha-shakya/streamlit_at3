import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from students.ethereum import run_eth_tab
from students.xrp import run_xrp_tab

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")


st.title("Crypto Forecast Dashboard")
st.write("Access real-time data and next-day price predictions for Ethereum.")

# single tab for Ethereum
tab_eth, tab_xrp = st.tabs(["Ethereum", "XRP"])
with tab_eth:
    run_eth_tab()

with tab_xrp:
    run_xrp_tab()
