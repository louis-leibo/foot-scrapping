# === main.py ===
import streamlit as st

# Set page config
st.set_page_config(page_title="⚽ Market Value Explorer", layout="wide")

import pandas as pd
from joblib import load


# === Load data ===
@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

# === Load model ===
@st.cache_resource
def load_model():
    return load("player_value_predictor_final_xgb.joblib")

model, cat_cols, num_cols = load_model()

# === App Title ===
st.title("⚽ Football Player Market Value Platform")

# === Tabs ===
from tabs import analysis, predict, compare, transfer, league_conversion, timeseries

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Analysis",
    "🧠 Predict Market Value",
    "🔄 Player Comparison",
    "💰 Transfer Recommendations",
    "⚖️ League Conversion",
    "📈 Time Series Analysis"
])

with tab1:
    analysis.show(df)

with tab2:
    predict.show(df, model)

with tab3:
    compare.show(df)

with tab4:
    transfer.show(df, model)

with tab5:
    league_conversion.show(df)

with tab6:
    timeseries.show(df)
