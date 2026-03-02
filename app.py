import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Alpha Quant v8.5 - Full Execution", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .insider-box { background: linear-gradient(90deg, #1e2130 0%, #0d1117 100%); padding: 15px; border-left: 5px solid #ffd700; border-radius: 8px; margin-bottom: 20px; }
    .veredicto-banner { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 10px; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO ---
def calculate_hurst(series):
    if len(series) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def get_alpha_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t
