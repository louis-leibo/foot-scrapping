import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

@st.cache_data
def prepare_time_series_data(df, entity_type='player', entity_name=None):
    if entity_type == 'player' and entity_name:
        filtered_df = df[df['player'] == entity_name]
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
    elif entity_type == 'team' and entity_name:
        filtered_df = df[df['team'] == entity_name]
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
    elif entity_type == 'league' and entity_name:
        filtered_df = df[df['league'] == entity_name]
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
    else:
        grouped = df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()

    grouped.columns = ['ds', 'y']
    grouped['ds'] = pd.to_datetime(grouped['ds'], format='%Y')
    return grouped

@st.cache_data
def train_and_forecast(time_series_df, forecast_periods=5):
    model = Prophet(
        yearly_seasonality=True,
        growth='linear',
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    model.fit(time_series_df)
    future = model.make_future_dataframe(periods=forecast_periods, freq='Y')
    forecast = model.predict(future)
    return model, forecast

def show_time_series_tab():
    st.header("\U0001F4C8 Market Value Time Series Analysis")
    st.write("Forecast future market values based on historical trends.")

    entity_type = st.radio("Select Analysis Type", 
                          options=["Player", "Team", "League", "Overall Market"],
                          horizontal=True)

    if entity_type == "Player":
        entity_name = st.selectbox("Select Player", sorted(df["player"].dropna().unique()))
    elif entity_type == "Team":
        entity_name = st.selectbox("Select Team", sorted(df["team"].dropna().unique()))
    elif entity_type == "League":
        entity_name = st.selectbox("Select League", sorted(df["league"].dropna().unique()))
    else:
        entity_name = None

    forecast_years = st.slider("Forecast Years", min_value=1, max_value=10, value=5)

    if st.button("Generate Forecast"):
        with st.spinner("Analyzing historical data and generating forecast..."):
            try:
                entity_type_map = {
                    "Player": "player",
                    "Team": "team",
                    "League": "league",
                    "Overall Market": "overall"
                }
                time_series_data = prepare_time_series_data(df, entity_type_map[entity_type], entity_name)

                if len(time_series_data) < 3:
                    st.warning("\u26A0\uFE0F Not enough historical data for reliable forecasting. Need at least 3 seasons.")
                    return

                model, forecast = train_and_forecast(time_series_data, forecast_periods=forecast_years)

                historical = time_series_data.rename(columns={"y": "Market Value (€)", "ds": "Season"})
                historical["Season"] = pd.to_datetime(historical["Season"]).dt.year
                historical["Source"] = "Historical"

                forecast_data = forecast[["ds", "yhat"]].rename(columns={"ds": "Season", "yhat": "Market Value (€)"})
                forecast_data["Season"] = pd.to_datetime(forecast_data["Season"]).dt.year
                forecast_data["Source"] = "Forecast"

                combined = pd.concat([historical, forecast_data], ignore_index=True)

                fig = go.Figure()
                title = {
                    "Player": f"Market Value Forecast for {entity_name}",
                    "Team": f"Average Player Market Value Forecast for {entity_name}",
                    "League": f"Average Player Market Value Forecast for {entity_name}",
                    "Overall Market": "Overall Market Value Forecast"
                }[entity_type]

                fig.add_trace(go.Scatter(
                    x=historical["Season"],
                    y=historical["Market Value (€)"],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_data["Season"],
                    y=forecast_data["Market Value (€)"],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', dash='dash')
                ))

                fig.update_layout(
                    title=title,
                    xaxis_title="Season",
                    yaxis_title="Market Value (€)",
                    legend_title="Data Source",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Forecast Data")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Season', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_display['Season'] = forecast_display['Season'].dt.year
                for col in ['Forecast', 'Lower Bound', 'Upper Bound']:
                    forecast_display[col] = forecast_display[col].apply(lambda x: f"€{x:,.0f}")
                st.dataframe(forecast_display, use_container_width=True)

                st.subheader("Trend Components")
                st.write("Decomposition of the forecast into trend, yearly seasonality, and other components.")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Error generating forecast: {e}")