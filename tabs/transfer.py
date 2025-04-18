import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load

# === Load Data and Model ===
@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

@st.cache_resource

def load_model():
    return load("player_value_predictor_final_xgb.joblib")

model, cat_cols, num_cols = load_model()

@st.cache_data
def find_undervalued_players(df, _predict_func, n_recommendations=10):
    latest_data = df.sort_values('season').groupby('player').last().reset_index()

    prediction_data = latest_data.drop(columns=["player", "born", "Valeur marchande (euros)"], errors="ignore")

    try:
        prediction_data['value_per_goal'] = prediction_data['Valeur marchande (euros)'] / (prediction_data['Performance Gls'] + 1)
        prediction_data['minutes_played_ratio'] = prediction_data['Playing Time Min'] / (90 * 38)
        prediction_data['goals_per_90'] = prediction_data['Performance Gls'] / (prediction_data['Playing Time 90s'] + 0.001)
        prediction_data['assists_per_90'] = prediction_data['Performance Ast'] / (prediction_data['Playing Time 90s'] + 0.001)

        predictions = _predict_func(prediction_data)
        latest_data['predicted_value'] = predictions
        latest_data['value_difference'] = latest_data['predicted_value'] - latest_data['Valeur marchande (euros)']
        latest_data['value_difference_percent'] = (latest_data['value_difference'] / latest_data['Valeur marchande (euros)']) * 100

        undervalued = latest_data[latest_data['value_difference'] > 0].copy()
        undervalued.sort_values('value_difference_percent', ascending=False, inplace=True)
        undervalued = undervalued[undervalued['Playing Time 90s'] > 5]
        top_recommendations = undervalued.head(n_recommendations)

        return top_recommendations[['player', 'team', 'league', 'pos', 'age', 
                                   'Valeur marchande (euros)', 'predicted_value', 
                                   'value_difference', 'value_difference_percent']]
    except Exception as e:
        st.error(f"Error finding undervalued players: {e}")
        return pd.DataFrame()

def show_transfer_tab():
    st.header("💰 Transfer Recommendation Engine")
    st.write("Find undervalued players based on the difference between predicted and actual market values.")

    if "model" not in st.session_state:
        st.session_state["model"] = model
        st.session_state["cat_cols"] = cat_cols
        st.session_state["num_cols"] = num_cols

    if "model" not in st.session_state:
        st.warning("⚠️ Please train or load a model in the 'Predict Market Value' tab first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        min_age = st.number_input("Minimum Age", min_value=16, max_value=40, value=18)
    with col2:
        max_age = st.number_input("Maximum Age", min_value=16, max_value=40, value=30)
    with col3:
        position_filter = st.multiselect("Position", 
                                         options=sorted(df["pos"].unique()), 
                                         default=sorted(df["pos"].unique()))

    league_filter = st.multiselect("League", 
                                  options=sorted(df["league"].unique()), 
                                  default=sorted(df["league"].unique()))

    if st.button("Find Undervalued Players"):
        with st.spinner("Analyzing the market..."):
            filtered_df = df[(df["age"] >= min_age) & 
                             (df["age"] <= max_age) & 
                             (df["pos"].isin(position_filter)) &
                             (df["league"].isin(league_filter))]

            model = st.session_state["model"]

            def predict_values(data):
                return model.predict(data)

            recommendations = find_undervalued_players(filtered_df, predict_values)

            if not recommendations.empty:
                st.subheader("🌟 Top Undervalued Players")

                display_df = recommendations.copy()
                display_df['Current Value (€)'] = display_df['Valeur marchande (euros)'].apply(lambda x: f"{x:,.0f}")
                display_df['Predicted Value (€)'] = display_df['predicted_value'].apply(lambda x: f"{x:,.0f}")
                display_df['Value Difference (€)'] = display_df['value_difference'].apply(lambda x: f"{x:+,.0f}")
                display_df['Value Difference (%)'] = display_df['value_difference_percent'].apply(lambda x: f"{x:+.1f}%")

                st.dataframe(
                    display_df[['player', 'team', 'league', 'pos', 'age', 
                              'Current Value (€)', 'Predicted Value (€)', 
                              'Value Difference (€)', 'Value Difference (%)']],
                    use_container_width=True
                )

                fig = px.bar(
                    recommendations.head(10),
                    x='player',
                    y='value_difference_percent',
                    color='league',
                    hover_data=['pos', 'age', 'team'],
                    title='Top 10 Undervalued Players (% Difference)',
                    labels={
                        'player': 'Player',
                        'value_difference_percent': 'Undervalued by (%)',
                        'league': 'League'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No undervalued players found with the current criteria.")