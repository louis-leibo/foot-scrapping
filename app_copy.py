import streamlit as st

st.set_page_config(page_title="⚽ Market Value Explorer", layout="wide")


import pandas as pd
import plotly.express as px
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly

# === Load Data and Model ===
@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

@st.cache_resource
def load_model():
    return load("player_value_predictor_final_xgb.joblib")

model, cat_cols, num_cols = load_model()

# === Setup ===
# st.set_page_config(page_title="⚽ Market Value Explorer", layout="wide")
st.title("⚽ Football Player Market Value Platform (data from 2020-2021 season to 2023-2024 season)")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analysis", 
        "🧠 Predict Market Value", 
        "🔄 Player Comparison",
        "💰 Transfer Recommendations",
        "⚖️ League Conversion",
        "📈 Time Series Analysis"
    ])

# ---------------------
# 📊 ANALYSIS TAB
# ---------------------

# === New visualization functions ===
@st.cache_data
def plot_correlation_heatmap(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Drop columns with all NaN values
    numeric_df = numeric_df.dropna(axis=1, how='all')
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Filter relevant features (related to performance and market value)
    relevant_cols = [col for col in corr.columns if any(x in col.lower() for x in 
                    ['performance', 'expected', 'gls', 'ast', 'valeur', 'xg', 'age'])]
    
    # Handle if we have too many columns
    if len(relevant_cols) > 15:
        # Select the most important features based on correlation with market value
        market_value_corr = abs(corr['Valeur marchande (euros)']).sort_values(ascending=False)
        relevant_cols = market_value_corr.index[:15].tolist()
    
    filtered_corr = corr.loc[relevant_cols, relevant_cols]
    
    # Create the heatmap
    fig = px.imshow(
        filtered_corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Between Key Performance Metrics",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    return fig

@st.cache_data
def plot_market_value_vs_time_by_position(df):
    pos_time_value = (
        df.groupby(['season', 'pos'])['Valeur marchande (euros)']
        .mean()
        .reset_index()
    )
    
    fig = px.line(
        pos_time_value,
        x="season",
        y="Valeur marchande (euros)",
        color="pos",
        markers=True,
        title="Average Market Value Trend by Position",
        labels={"season": "Season", "Valeur marchande (euros)": "Avg Market Value (€)"}
    )
    return fig

@st.cache_data
def plot_age_vs_market_value_by_league(df):
    fig = px.scatter(
        df,
        x="age",
        y="Valeur marchande (euros)",
        color="league",
        size="Playing Time 90s",
        hover_name="player",
        facet_col="league",
        facet_col_wrap=2,
        opacity=0.7,
        title="Age vs Market Value by League (Size = Playing Time)",
        labels={"age": "Age", "Valeur marchande (euros)": "Market Value (€)"}
    )
    return fig

@st.cache_data
def plot_playing_time_vs_market_value(df):
    fig = px.scatter(
        df,
        x="Playing Time 90s",
        y="Valeur marchande (euros)",
        color="pos",
        size="age",
        hover_name="player",
        opacity=0.7,
        trendline="ols",
        title="Playing Time vs Market Value (Size = Age)",
        labels={"Playing Time 90s": "Games Played (90min)", "Valeur marchande (euros)": "Market Value (€)"}
    )
    return fig

@st.cache_data
def plot_performance_radar_charts(player_data):
    if player_data.empty:
        return None
    
    # Define performance metrics to include in radar chart
    metrics = [
        'Expected xG', 'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
        'Expected xAG', 'Performance G+A', 'Performance CrdY'
    ]
    
    # Filter metrics that exist in the data
    available_metrics = [m for m in metrics if m in player_data.columns]
    
    if not available_metrics:
        return None
    
    # Create radar chart for each season
    seasons = sorted(player_data['season'].unique())
    
    # Create figure
    fig = go.Figure()
    
    for season in seasons:
        season_data = player_data[player_data['season'] == season]
        
        if not season_data.empty:
            values = season_data[available_metrics].values.flatten().tolist()
            # Make sure we close the polygon
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill='toself',
                name=f'Season {season}'
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True)
        ),
        title=f"Performance Radar Chart by Season"
    )
    return fig

@st.cache_data
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        # Get feature importances from the model
        importances = model.feature_importances_
        
        # Create a DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        # Create bar chart
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 20 Feature Importances for Market Value Prediction',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature'}
        )
        return fig
    else:
        return None

@st.cache_data
def plot_prediction_vs_actual(y_true, y_pred):
    # Create DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Create scatter plot
    fig = px.scatter(
        results_df,
        x='Actual',
        y='Predicted',
        labels={'Actual': 'Actual Market Value (€)', 'Predicted': 'Predicted Market Value (€)'},
        title='Prediction vs Actual Market Values'
    )
    
    # Add diagonal line (perfect predictions)
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    return fig

@st.cache_data
def plot_player_performance_vs_market_value(player_data):
    if player_data.empty or len(player_data) < 2:
        return None
    
    # Select key performance metrics
    performance_metrics = ['Performance Gls', 'Performance Ast', 'Expected xG', 
                           'Expected xAG', 'Performance G+A']
    
    # Only keep metrics that exist in the data
    metrics = [m for m in performance_metrics if m in player_data.columns]
    
    if not metrics:
        return None
    
    # Create a figure with subplots
    fig = go.Figure()
    
    # Add market value line
    fig.add_trace(
        go.Scatter(
            x=player_data['season'],
            y=player_data['Valeur marchande (euros)'],
            mode='lines+markers',
            name='Market Value (€)',
            line=dict(color='black', width=3),
            yaxis='y2'
        )
    )
    
    # Add performance metrics
    colors = px.colors.qualitative.Plotly
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=player_data['season'],
                y=player_data[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i % len(colors)])
            )
        )
    
    # Update layout with second y-axis
    fig.update_layout(
        title=f"Performance Metrics vs Market Value Over Time",
        xaxis=dict(title='Season'),
        yaxis=dict(title='Performance Metrics', side='left'),
        yaxis2=dict(
            title='Market Value (€)',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    return fig

# 1. TRANSFER RECOMMENDATION ENGINE
# ==========================================
@st.cache_data
def find_undervalued_players(df, _predict_func, n_recommendations=10):
    """
    Find undervalued players by comparing their predicted market value
    with their actual market value.
    
    Args:
        df: DataFrame with player data
        predict_func: Function to predict player market value
        n_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame of recommended players
    """
    # Get latest data for each player to avoid duplicates
    latest_data = df.sort_values('season').groupby('player').last().reset_index()
    
    # Prepare prediction data
    prediction_data = latest_data.drop(columns=["player", "born", "Valeur marchande (euros)"], errors="ignore")
    
    # Make predictions
    try:
        # Add feature engineering for prediction (matching what's done in prediction tab)
        prediction_data['value_per_goal'] = prediction_data['Valeur marchande (euros)'] / (prediction_data['Performance Gls'] + 1)
        prediction_data['minutes_played_ratio'] = prediction_data['Playing Time Min'] / (90 * 38)
        prediction_data['goals_per_90'] = prediction_data['Performance Gls'] / (prediction_data['Playing Time 90s'] + 0.001)
        prediction_data['assists_per_90'] = prediction_data['Performance Ast'] / (prediction_data['Playing Time 90s'] + 0.001)
        
        # Predict values
        predictions = _predict_func(prediction_data)
        
        # Add predictions to original data
        latest_data['predicted_value'] = predictions
        
        # Calculate value difference and percentage
        latest_data['value_difference'] = latest_data['predicted_value'] - latest_data['Valeur marchande (euros)']
        latest_data['value_difference_percent'] = (latest_data['value_difference'] / latest_data['Valeur marchande (euros)']) * 100
        
        # Filter for positive differences (undervalued)
        undervalued = latest_data[latest_data['value_difference'] > 0].copy()
        
        # Sort by percentage difference (best deals first)
        undervalued.sort_values('value_difference_percent', ascending=False, inplace=True)
        
        # Filter to players with significant playing time
        undervalued = undervalued[undervalued['Playing Time 90s'] > 5]
        
        # Limit to top recommendations
        top_recommendations = undervalued.head(n_recommendations)
        
        return top_recommendations[['player', 'team', 'league', 'pos', 'age', 
                                   'Valeur marchande (euros)', 'predicted_value', 
                                   'value_difference', 'value_difference_percent']]
    except Exception as e:
        st.error(f"Error finding undervalued players: {e}")
        return pd.DataFrame()
    

def show_transfer_recommendations_tab():
    """
    Display the transfer recommendations tab in the Streamlit app
    """
    st.header("💰 Transfer Recommendation Engine")
    st.write("Find undervalued players based on the difference between predicted and actual market values.")
    
    # Check if model is available
    if "model" not in st.session_state:
        st.warning("⚠️ Please train or load a model in the 'Predict Market Value' tab first.")
        return
    
    # Get filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_age = st.number_input("Minimum Age", min_value=16, max_value=40, value=18)
    
    with col2:
        max_age = st.number_input("Maximum Age", min_value=16, max_value=40, value=30)
    
    with col3:
        position_filter = st.multiselect("Position", 
                                         options=sorted(df["pos"].unique()), 
                                         default=sorted(df["pos"].unique()))
    
    # Get league filter
    league_filter = st.multiselect("League", 
                                  options=sorted(df["league"].unique()), 
                                  default=sorted(df["league"].unique()))
    
    # Get recommended players
    if st.button("Find Undervalued Players"):
        with st.spinner("Analyzing the market..."):
            # Filter data based on user input
            filtered_df = df[(df["age"] >= min_age) & 
                             (df["age"] <= max_age) & 
                             (df["pos"].isin(position_filter)) &
                             (df["league"].isin(league_filter))]
            
            # Get model predictions
            model = st.session_state["model"]
            
            # Create prediction function
            def predict_values(data):
                return model.predict(data)
            
            # Find undervalued players
            recommendations = find_undervalued_players(filtered_df, predict_values)
            
            if not recommendations.empty:
                # Display results
                st.subheader("🌟 Top Undervalued Players")
                
                # Format the dataframe for display
                display_df = recommendations.copy()
                display_df['Current Value (€)'] = display_df['Valeur marchande (euros)'].apply(lambda x: f"{x:,.0f}")
                display_df['Predicted Value (€)'] = display_df['predicted_value'].apply(lambda x: f"{x:,.0f}")
                display_df['Value Difference (€)'] = display_df['value_difference'].apply(lambda x: f"{x:+,.0f}")
                display_df['Value Difference (%)'] = display_df['value_difference_percent'].apply(lambda x: f"{x:+.1f}%")
                
                # Show the dataframe
                st.dataframe(
                    display_df[['player', 'team', 'league', 'pos', 'age', 
                              'Current Value (€)', 'Predicted Value (€)', 
                              'Value Difference (€)', 'Value Difference (%)']],
                    use_container_width=True
                )
                
                # Visualization
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

# ==========================================
# 2. LEAGUE-TO-LEAGUE VALUE CONVERSION
# ==========================================
@st.cache_data
def calculate_league_value_coefficients(df):
    """
    Calculate the relative market value coefficients between leagues.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        DataFrame with league coefficients
    """
    # Group by league and calculate the average market value
    league_avg_values = df.groupby('league')['Valeur marchande (euros)'].mean().reset_index()
    
    # Calculate the global average market value
    global_avg = league_avg_values['Valeur marchande (euros)'].mean()
    
    # Calculate coefficient (how many times higher/lower than average)
    league_avg_values['coefficient'] = league_avg_values['Valeur marchande (euros)'] / global_avg
    
    # Calculate average age by league (for context)
    league_avg_age = df.groupby('league')['age'].mean().reset_index()
    
    # Merge the two dataframes
    result = pd.merge(league_avg_values, league_avg_age, on='league')
    
    return result


def show_league_conversion_tab():
    """
    Display the league-to-league value conversion tab in the Streamlit app
    """
    st.header("🔄 League-to-League Value Conversion")
    st.write("Estimate how a player's market value might change when moving between leagues.")
    
    # Calculate league coefficients
    league_coefficients = calculate_league_value_coefficients(df)
    
    # Display inputs
    col1, col2 = st.columns(2)
    
    with col1:
        source_league = st.selectbox("Source League", sorted(df["league"].unique()), key="source_league")
        current_value = st.number_input("Current Market Value (€)", min_value=0, value=10000000)
    
    with col2:
        target_league = st.selectbox("Target League", sorted(df["league"].unique()), key="target_league")
    
    # Get coefficients
    source_coef = league_coefficients[league_coefficients['league'] == source_league]['coefficient'].values[0]
    target_coef = league_coefficients[league_coefficients['league'] == target_league]['coefficient'].values[0]
    
    # Calculate new value
    conversion_ratio = target_coef / source_coef
    new_value = current_value * conversion_ratio
    
    # Display results
    st.subheader("Estimated Market Value After Transfer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Value", f"€{current_value:,.0f}")
    
    with col2:
        st.metric("Conversion Factor", f"{conversion_ratio:.2f}x")
    
    with col3:
        change = new_value - current_value
        change_percent = (change / current_value) * 100
        st.metric("Estimated New Value", f"€{new_value:,.0f}", f"{change:+,.0f} ({change_percent:+.1f}%)")
    
    # Show league coefficients
    st.subheader("League Value Coefficients")
    st.write("These coefficients represent the relative market value of players in each league compared to the global average.")
    
    # Sort by coefficient
    sorted_coefficients = league_coefficients.sort_values('coefficient', ascending=False)
    
    # Prepare display dataframe
    display_coefs = sorted_coefficients.copy()
    display_coefs['Average Value (€)'] = display_coefs['Valeur marchande (euros)'].apply(lambda x: f"{x:,.0f}")
    display_coefs['Coefficient'] = display_coefs['coefficient'].apply(lambda x: f"{x:.2f}x")
    display_coefs['Average Age'] = display_coefs['age'].apply(lambda x: f"{x:.1f}")
    
    # Display table
    st.dataframe(
        display_coefs[['league', 'Average Value (€)', 'Coefficient', 'Average Age']],
        use_container_width=True
    )
    
    # Visualization of league coefficients
    fig = px.bar(
        sorted_coefficients,
        x='league',
        y='coefficient',
        color='coefficient',
        hover_data=['Valeur marchande (euros)', 'age'],
        title='League Value Coefficients',
        labels={
            'league': 'League',
            'coefficient': 'Value Coefficient',
        },
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. TIME SERIES ANALYSIS
# ==========================================
@st.cache_data
def prepare_time_series_data(df, entity_type='player', entity_name=None):
    """
    Prepare time series data for forecasting.
    
    Args:
        df: DataFrame with player/team/league data
        entity_type: 'player', 'team', or 'league'
        entity_name: The specific entity to analyze
        
    Returns:
        DataFrame ready for time series forecasting
    """
    if entity_type == 'player' and entity_name:
        # Filter for specific player
        filtered_df = df[df['player'] == entity_name]
        
        # Group by season (in case there are multiple rows per season)
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
        
    elif entity_type == 'team' and entity_name:
        # Filter for specific team
        filtered_df = df[df['team'] == entity_name]
        
        # Group by season
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
        
    elif entity_type == 'league' and entity_name:
        # Filter for specific league
        filtered_df = df[df['league'] == entity_name]
        
        # Group by season
        grouped = filtered_df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
        
    else:
        # Default: all data, grouped by season
        grouped = df.groupby('season')['Valeur marchande (euros)'].mean().reset_index()
    
    # Rename columns for Prophet
    grouped.columns = ['ds', 'y']
    
    # Convert season to datetime (assuming season is the end year, e.g., 2020 means 2019-2020)
    grouped['ds'] = pd.to_datetime(grouped['ds'], format='%Y')
    
    return grouped

@st.cache_data
def train_and_forecast(time_series_df, forecast_periods=5):
    """
    Train a Prophet model and generate forecasts.
    
    Args:
        time_series_df: DataFrame with 'ds' and 'y' columns
        forecast_periods: Number of periods to forecast
        
    Returns:
        Prophet model and forecast DataFrame
    """
    # Initialize model
    model = Prophet(
        yearly_seasonality=True,
        growth='linear',
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    
    # Fit model
    model.fit(time_series_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq='Y')
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast


def show_time_series_tab():
    """
    Display the time series analysis tab in the Streamlit app
    """
    st.header("📈 Market Value Time Series Analysis")
    st.write("Forecast future market values based on historical trends.")
    
    # Entity selection
    entity_type = st.radio("Select Analysis Type", 
                          options=["Player", "Team", "League", "Overall Market"],
                          horizontal=True)
    
    if entity_type == "Player":
        entity_name = st.selectbox("Select Player", sorted(df["player"].dropna().unique()))
    elif entity_type == "Team":
        entity_name = st.selectbox("Select Team", sorted(df["team"].dropna().unique()))
    elif entity_type == "League":
        entity_name = st.selectbox("Select League", sorted(df["league"].dropna().unique()))
    else:  # Overall Market
        entity_name = None
    
    # Number of years to forecast
    forecast_years = st.slider("Forecast Years", min_value=1, max_value=10, value=5)
    
    # Run forecast
    if st.button("Generate Forecast"):
        with st.spinner("Analyzing historical data and generating forecast..."):
            try:
                # Map entity type to internal values
                entity_type_map = {
                    "Player": "player",
                    "Team": "team",
                    "League": "league",
                    "Overall Market": "overall"
                }
                
                # Prepare time series data
                time_series_data = prepare_time_series_data(
                    df, 
                    entity_type=entity_type_map[entity_type], 
                    entity_name=entity_name
                )
                
                if len(time_series_data) < 3:
                    st.warning("⚠️ Not enough historical data for reliable forecasting. Need at least 3 seasons.")
                    return
                
                # Train model and generate forecast
                model, forecast = train_and_forecast(time_series_data, forecast_periods=forecast_years)
                
                # Display forecast chart
                # fig = plot_plotly(model, forecast, xlabel='Season', ylabel='Market Value (€)')
                # Get historical data
                historical = time_series_data.rename(columns={"y": "Market Value (€)", "ds": "Season"})
                historical["Season"] = pd.to_datetime(historical["Season"]).dt.year
                historical["Source"] = "Historical"

                # Prepare forecast data
                forecast_data = forecast[["ds", "yhat"]].rename(columns={"ds": "Season", "yhat": "Market Value (€)"})
                forecast_data["Season"] = pd.to_datetime(forecast_data["Season"]).dt.year
                forecast_data["Source"] = "Forecast"

                # Combine
                combined = pd.concat([historical, forecast_data], ignore_index=True)
                
                fig = go.Figure()

                # Update title based on entity type
                if entity_type == "Player":
                    title = f"Market Value Forecast for {entity_name}"
                elif entity_type == "Team":
                    title = f"Average Player Market Value Forecast for {entity_name}"
                elif entity_type == "League":
                    title = f"Average Player Market Value Forecast for {entity_name}"
                else:
                    title = "Overall Market Value Forecast"

                # Add Historical
                fig.add_trace(go.Scatter(
                    x=historical["Season"],
                    y=historical["Market Value (€)"],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue'),
                    showlegend=True
                ))

                # Add Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_data["Season"],
                    y=forecast_data["Market Value (€)"],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', dash='dash'),
                    showlegend=True
                ))

                # Layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Season",
                    yaxis_title="Market Value (€)",
                    legend_title="Data Source",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

                
                # fig.update_layout(title=title)
                
                # st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data
                st.subheader("Forecast Data")
                
                # Format forecast data for display
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Season', 'Forecast', 'Lower Bound', 'Upper Bound']
                
                # Convert datetime to year only
                forecast_display['Season'] = forecast_display['Season'].dt.year
                
                # Format as currency
                for col in ['Forecast', 'Lower Bound', 'Upper Bound']:
                    forecast_display[col] = forecast_display[col].apply(lambda x: f"€{x:,.0f}")
                
                # Show the table
                st.dataframe(forecast_display, use_container_width=True)
                
                # Show trend components
                st.subheader("Trend Components")
                st.write("Decomposition of the forecast into trend, yearly seasonality, and other components.")
                
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")


######### TABS ORGANISATION ############
with tab1:
    # === Sidebar Filters ===
    st.sidebar.header("📊 Filters")
    selected_league = st.sidebar.multiselect("Select League(s):", df["league"].unique(), default=df["league"].unique())
    selected_season = st.sidebar.multiselect("Select Season(s):", sorted(df["season"].unique()), default=sorted(df["season"].unique()))

    filtered_df = df[df["league"].isin(selected_league) & df["season"].isin(selected_season)]

    # === Summary Stats ===
    st.subheader("📋 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Players", f"{filtered_df['player'].nunique():,}")
    col2.metric("Teams", f"{filtered_df['team'].nunique():,}")
    col3.metric("Total Market Value (€)", f"{filtered_df['Valeur marchande (euros)'].sum():,.0f}")

    # === Top 10 Most Valuable Players ===
    st.subheader("💰 Top 10 Most Valuable Players")
    top_players = (
        filtered_df[["player", "team", "league", "season", "Valeur marchande (euros)"]]
        .dropna()
        .sort_values("Valeur marchande (euros)", ascending=False)
        .drop_duplicates("player")
        .head(10)
    )
    st.dataframe(top_players.reset_index(drop=True))

    # === Market Value Distribution ===
    st.subheader("📈 Market Value Distribution")
    fig1 = px.histogram(
        filtered_df,
        x="Valeur marchande (euros)",
        nbins=50,
        title="Distribution of Market Values (€)",
        labels={"Valeur marchande (euros)": "Market Value (€)"},
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Average Market Value by Position ===
    st.subheader("📊 Average Market Value by Position")
    pos_value = (
        filtered_df.groupby("pos")["Valeur marchande (euros)"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig2 = px.bar(
        pos_value,
        x="pos",
        y="Valeur marchande (euros)",
        title="Average Market Value by Position",
        labels={"pos": "Position", "Valeur marchande (euros)": "Avg Market Value (€)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # === Market Value by League and Season ===
    st.subheader("🌍 Market Value by League and Season")
    grouped = filtered_df.groupby(["season", "league"])["Valeur marchande (euros)"].mean().reset_index()
    fig3 = px.line(
        grouped,
        x="season",
        y="Valeur marchande (euros)",
        color="league",
        markers=True,
        title="Avg Market Value per Player by League Over Seasons",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # === Top Teams by Total Market Value ===
    st.subheader("🏆 Top Teams by Average Seasonal Market Value")

    # Step 1: total value per team per season
    team_season_value = (
        filtered_df.groupby(["team", "season"])["Valeur marchande (euros)"]
        .sum()
        .reset_index()
    )

    # Step 2: average that over selected seasons
    team_avg_value = (
        team_season_value.groupby("team")["Valeur marchande (euros)"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    # Step 3: plot
    fig4 = px.bar(
        team_avg_value,
        x="team",
        y="Valeur marchande (euros)",
        title="Top 15 Teams by Average Total Market Value (per Season)",
        labels={"team": "Team", "Valeur marchande (euros)": "Avg Market Value (€)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    # === Age vs Market Value ===
    st.subheader("👶 Age vs. Market Value")
    fig5 = px.box(
        filtered_df,
        x="age",
        y="Valeur marchande (euros)",
        title="Player Age vs Market Value",
        labels={"age": "Age", "Valeur marchande (euros)": "Market Value (€)"},
    )
    st.plotly_chart(fig5, use_container_width=True)

    # === Player Distribution by Position ===
    st.subheader("🎯 Player Distribution by Position")
    pos_counts = filtered_df["pos"].value_counts().reset_index()
    pos_counts.columns = ["pos", "count"]
    fig6 = px.pie(
        pos_counts,
        names="pos",
        values="count",
        title="Position Distribution",
    )

    st.plotly_chart(fig6, use_container_width=True)

    # Show change in total market value by team between 2020 and 2024 : top rising teams : 

    st.subheader("📈 Top Rising Teams (2020 vs 2024)")

    team_year_value = (
        df[df["season"].isin([2020, 2023])]
        .groupby(["team", "season"])["Valeur marchande (euros)"]
        .sum()
        .unstack()
        .dropna()
        .assign(Change=lambda x: x[2023] - x[2020])
        .sort_values("Change", ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        team_year_value,
        x="team",
        y="Change",
        title="Top 10 Teams with Largest Increase in Total Market Value (2020 → 2024)",
        labels={"Change": "€ Change"},
    )
    st.plotly_chart(fig, use_container_width=True)


    # number of players by nationality per league : 
    st.subheader("🌍 National Representation by League")

    nationality_league = (
        filtered_df.groupby(["league", "nation"])["player"].nunique().reset_index(name="num_players")
    )

    fig = px.sunburst(
        nationality_league,
        path=["league", "nation"],
        values="num_players",
        title="Player Nationalities Across Leagues"
    )
    st.plotly_chart(fig, use_container_width=True)

    # corr scatterplots : xG vs market value : 
    st.subheader("📊 Expected Goals vs. Market Value")

    fig = px.scatter(
        filtered_df,
        x="Expected xG",
        y="Valeur marchande (euros)",
        hover_name="player",
        color="pos",
        trendline="ols",
        title="Expected Goals vs. Market Value",
        labels={"Expected xG": "xG", "Valeur marchande (euros)": "Market Value (€)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    #top scoreres per season : 
    top_scorers = filtered_df.sort_values("Performance Gls", ascending=False).dropna(subset=["Performance Gls"])
    top_scorers = top_scorers.drop_duplicates("player").head(10)

    fig = px.bar(
        top_scorers,
        x="player",
        y="Performance Gls",
        color="league",
        hover_data=["team", "season"],
        title="Top 10 Goal Scorers"
    )
    st.plotly_chart(fig, use_container_width=True)

        
    # Add this after the "📈 Market Value Distribution" section
    st.subheader("🔄 Correlation Between Key Metrics")
    correlation_fig = plot_correlation_heatmap(filtered_df)
    st.plotly_chart(correlation_fig, use_container_width=True)

    # Add this after "📊 Average Market Value by Position" section
    st.subheader("📊 Market Value Trends by Position")
    pos_trends_fig = plot_market_value_vs_time_by_position(filtered_df)
    st.plotly_chart(pos_trends_fig, use_container_width=True)

    # Add this after "👶 Age vs. Market Value" section
    st.subheader("⚽ Playing Time vs Market Value")
    playing_time_fig = plot_playing_time_vs_market_value(filtered_df)
    st.plotly_chart(playing_time_fig, use_container_width=True)

    # Add this after the last visualization in Analysis tab
    st.subheader("🌍 Age vs Market Value by League")
    age_league_fig = plot_age_vs_market_value_by_league(filtered_df)
    st.plotly_chart(age_league_fig, use_container_width=True)



# ---------------------
# 🧠 PREDICTION TAB
# ---------------------

# === UPDATES TO PREDICTION TAB ===
# Replace the prediction section with this enhanced version

with tab2:
    st.subheader("🧠 Predict a Player's Market Value for Next Season")

    # Create two columns for the prediction interface
    col1, col2 = st.columns([1, 2])

    with col1:
        # === Optional: Model Training ===
        st.markdown("### ⚙️ Train / Retrain Model")

        # Create model training form
        with st.form("model_training_form"):
            st.markdown("#### Model Configuration")
            n_estimators = st.slider("Number of Trees", 50, 500, 150, 50)
            max_depth = st.slider("Maximum Tree Depth", 5, 30, 15, 5)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 5, 1)
            train_button = st.form_submit_button("Train Model")
            
            if train_button:
                with st.spinner("Training model..."):
                    # Drop missing target
                    df_model = df[df["Valeur marchande (euros)"].notna()].copy()
                    
                    # Feature engineering
                    df_model['value_per_goal'] = df_model['Valeur marchande (euros)'] / (df_model['Performance Gls'] + 1)
                    df_model['minutes_played_ratio'] = df_model['Playing Time Min'] / (90 * 38)  # minutes / (90min * max games)
                    df_model['goals_per_90'] = df_model['Performance Gls'] / (df_model['Playing Time 90s'] + 0.001)
                    df_model['assists_per_90'] = df_model['Performance Ast'] / (df_model['Playing Time 90s'] + 0.001)
                    
                    # Prepare X and y
                    X = df_model.drop(columns=["Valeur marchande (euros)", "player", "born"], errors="ignore")
                    y = df_model["Valeur marchande (euros)"]
                    
                    # Identify column types
                    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
                    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
                    
                    # Preprocessing
                    preprocessor = ColumnTransformer([
                        ("num", SimpleImputer(strategy="median"), num_cols),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ]), cat_cols)
                    ], remainder='drop')
                    
                    # Full pipeline with improved Random Forest
                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("regressor", RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42))
                    ])
                    
                    # Train/test split and fit
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    # Save metrics
                    st.session_state["mae"] = mae
                    st.session_state["rmse"] = rmse
                    st.session_state["r2"] = r2
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    
                    # Save feature names (post one-hot encoding)
                    # Get feature names from preprocessing step
                    st.session_state["feature_names"] = num_cols + cat_cols
                    
                    # Save the model in session state
                    st.session_state["model"] = pipeline
                    st.session_state["cat_cols"] = cat_cols
                    st.session_state["num_cols"] = num_cols
                    
                    # Save to disk as well
                    dump((pipeline, cat_cols, num_cols), "player_value_predictor_final_xgb.joblib")
                    
                    st.success("✅ Model trained successfully!")
        
        # If model exists in session state, show metrics
        if "model" in st.session_state and "mae" in st.session_state:
            st.markdown("### 📊 Model Performance")
            st.metric("Mean Absolute Error (€)", f"{st.session_state['mae']:,.0f}")
            st.metric("RMSE (€)", f"{st.session_state['rmse']:,.0f}")
            st.metric("R² Score", f"{st.session_state['r2']:.3f}")

    with col2:
        # Load model if available
        if "model" not in st.session_state:
            try:
                from pathlib import Path
                if Path("player_value_predictor_final_xgb.joblib").exists():
                    loaded_model, cat_cols, num_cols = load("player_value_predictor_final_xgb.joblib")
                    st.session_state["model"] = loaded_model
                    st.session_state["cat_cols"] = cat_cols
                    st.session_state["num_cols"] = num_cols
                    st.info("✅ Loaded model from saved file.")
                else:
                    st.warning("⚠️ No trained model found. Please train one first.")
            except Exception as e:
                st.error(f"❌ Error loading saved model: {e}")
        
        # If model exists, show player prediction interface
        if "model" in st.session_state:
            model = st.session_state["model"]
            
            st.markdown("### 🔮 Player Value Prediction")
            
            # Player selection 
            player_list = df["player"].dropna().unique()
            selected_player = st.selectbox("Choose a player", sorted(player_list))
            
            # Get player data
            player_data = df[df["player"] == selected_player].sort_values("season")
            
            if not player_data.empty:
                # Performance vs market value visualization
                st.markdown("### 📈 Performance & Market Value History")
                player_perf_fig = plot_player_performance_vs_market_value(player_data)
                if player_perf_fig:
                    st.plotly_chart(player_perf_fig, use_container_width=True)
                    
                # Radar chart
                st.markdown("### 🎯 Player Performance Radar")
                radar_fig = plot_performance_radar_charts(player_data)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                # Get latest row to predict from
                latest_row = player_data.iloc[-1:].copy()
                
                # Add engineered features used during training
                latest_row["mv_lag1"] = player_data["Valeur marchande (euros)"].shift(1).iloc[-1]
                latest_row["rolling_mv_2"] = player_data["Valeur marchande (euros)"].rolling(2).mean().iloc[-1]
                latest_row["rolling_mv_3"] = player_data["Valeur marchande (euros)"].rolling(3).mean().iloc[-1]
                latest_row["gls_per_90"] = latest_row["Performance Gls"] / (latest_row["Playing Time 90s"] + 1e-3)
                latest_row["market_age_ratio"] = latest_row["Valeur marchande (euros)"] / (latest_row["age"] + 1e-3)

                # Calculate performance trends (year-over-year changes)
                if len(player_data) > 1:
                    previous_season = player_data.iloc[-2:-1]
                    
                    # Calculate YoY changes
                    for col in ['Performance Gls', 'Performance Ast', 'Expected xG', 'Playing Time 90s']:
                        if col in latest_row.columns and col in previous_season.columns:
                            latest_row[f'{col}_trend'] = latest_row[col].values[0] - previous_season[col].values[0]
                
                # Add feature engineering for prediction
                latest_row['value_per_goal'] = latest_row['Valeur marchande (euros)'] / (latest_row['Performance Gls'] + 1)
                latest_row['minutes_played_ratio'] = latest_row['Playing Time Min'] / (90 * 38)
                latest_row['goals_per_90'] = latest_row['Performance Gls'] / (latest_row['Playing Time 90s'] + 0.001)
                latest_row['assists_per_90'] = latest_row['Performance Ast'] / (latest_row['Playing Time 90s'] + 0.001)
                
                # Prepare for prediction
                prediction_data = latest_row.drop(columns=["player", "born", "Valeur marchande (euros)"], 
                                                  errors="ignore")
                
                # Predict
                try:
                    prediction = model.predict(prediction_data)[0]
                    
                    # Calculate projected change
                    current_value = latest_row["Valeur marchande (euros)"].values[0]
                    value_change = prediction - current_value
                    change_percent = (value_change / current_value) * 100 if current_value > 0 else 0
                    
                    # Display prediction
                    st.markdown("### 🤖 Predicted Market Value (Next Season)")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Value", f"€{current_value:,.0f}")
                    col2.metric("Predicted Value", f"€{prediction:,.0f}", 
                                f"{value_change:+,.0f} ({change_percent:+.1f}%)")
                    
                    age = latest_row["age"].values[0] + 1  # Increment age for next season
                    col3.metric("Player Age (Next Season)", f"{age:.0f}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
            else:
                st.warning("⚠️ No data available for selected player.")
                
        # Show feature importance plot if model and feature names are available
        if "model" in st.session_state and "feature_names" in st.session_state:
            st.markdown("### 🔍 Feature Importance")
            
            # Extract regressor from pipeline to get feature importances
            regressor = st.session_state["model"].named_steps['regressor']
            feature_names = st.session_state["feature_names"]
            
            # Generate and display feature importance plot
            importance_fig = plot_feature_importance(regressor, feature_names)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
        # Show prediction vs actual if available
        if "y_test" in st.session_state and "y_pred" in st.session_state:
            st.markdown("### 📊 Model Accuracy")
            pred_vs_actual_fig = plot_prediction_vs_actual(
                st.session_state["y_test"], 
                st.session_state["y_pred"]
            )
            st.plotly_chart(pred_vs_actual_fig, use_container_width=True)

# === PLAYER COMPARISON TAB ===
with tab3:
    st.subheader("🔄 Compare Players")

    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select First Player", sorted(df["player"].dropna().unique()), key="player1")
        
    with col2:
        player2 = st.selectbox("Select Second Player", sorted(df["player"].dropna().unique()), key="player2")
    
    # Get player data
    player1_data = df[df["player"] == player1].sort_values("season")
    player2_data = df[df["player"] == player2].sort_values("season")
    
    if not player1_data.empty and not player2_data.empty:
        # Create tabs for different comparison views
        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Market Value", "Performance", "Radar Charts"])
        
        with comp_tab1:
            # Market value comparison
            fig = go.Figure()
            
            # Add player 1 line
            fig.add_trace(
                go.Scatter(
                    x=player1_data['season'],
                    y=player1_data['Valeur marchande (euros)'],
                    mode='lines+markers',
                    name=player1,
                    line=dict(width=3)
                )
            )
            
            # Add player 2 line
            fig.add_trace(
                go.Scatter(
                    x=player2_data['season'],
                    y=player2_data['Valeur marchande (euros)'],
                    mode='lines+markers',
                    name=player2,
                    line=dict(width=3, dash='dash')
                )
            )
            
            fig.update_layout(
                title=f"Market Value Comparison: {player1} vs {player2}",
                xaxis=dict(title='Season'),
                yaxis=dict(title='Market Value (€)'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with comp_tab2:
            # Performance metrics comparison
            metrics = ['Performance Gls', 'Performance Ast', 'Expected xG', 
                      'Expected xAG', 'Performance G+A', 'Playing Time 90s']
            
            # Filter available metrics
            avail_metrics = [m for m in metrics if m in player1_data.columns and m in player2_data.columns]
            
            selected_metric = st.selectbox("Select Metric", avail_metrics)
            
            if selected_metric:
                fig = go.Figure()
                
                # Add player 1 line
                fig.add_trace(
                    go.Scatter(
                        x=player1_data['season'],
                        y=player1_data[selected_metric],
                        mode='lines+markers',
                        name=player1,
                        line=dict(width=3)
                    )
                )
                
                # Add player 2 line
                fig.add_trace(
                    go.Scatter(
                        x=player2_data['season'],
                        y=player2_data[selected_metric],
                        mode='lines+markers',
                        name=player2,
                        line=dict(width=3, dash='dash')
                    )
                )
                
                fig.update_layout(
                    title=f"{selected_metric} Comparison: {player1} vs {player2}",
                    xaxis=dict(title='Season'),
                    yaxis=dict(title=selected_metric),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        with comp_tab3:
            # Radar chart comparison
            # Get latest season data for each player
            player1_latest = player1_data.iloc[-1:] if not player1_data.empty else None
            player2_latest = player2_data.iloc[-1:] if not player2_data.empty else None
            
            if player1_latest is not None and player2_latest is not None:
                # Define performance metrics for radar chart
                radar_metrics = [
                    'Expected xG', 'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
                    'Expected xAG', 'Performance G+A', 'Performance CrdY'
                ]
                
                # Filter available metrics
                avail_radar_metrics = [m for m in radar_metrics 
                                      if m in player1_latest.columns and m in player2_latest.columns]
                
                if avail_radar_metrics:
                    # Create radar chart
                    fig = go.Figure()
                    
                    # Player 1 trace
                    p1_values = player1_latest[avail_radar_metrics].values.flatten().tolist()
                    p1_values.append(p1_values[0])  # Close the polygon
                    
                    fig.add_trace(go.Scatterpolar(
                        r=p1_values,
                        theta=avail_radar_metrics + [avail_radar_metrics[0]],
                        fill='toself',
                        name=player1
                    ))
                    
                    # Player 2 trace
                    p2_values = player2_latest[avail_radar_metrics].values.flatten().tolist()
                    p2_values.append(p2_values[0])  # Close the polygon
                    
                    fig.add_trace(go.Scatterpolar(
                        r=p2_values,
                        theta=avail_radar_metrics + [avail_radar_metrics[0]],
                        fill='toself',
                        name=player2
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True)
                        ),
                        title=f"Performance Comparison: {player1} vs {player2} (Latest Season)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a table comparing key stats
                    st.subheader("🔢 Key Stats Comparison")
                    
                    # Combine and display as a table
                    compare_metrics = avail_radar_metrics + ['age', 'Valeur marchande (euros)', 
                                                           'Playing Time MP', 'Playing Time Min']
                    
                    # Filter available metrics
                    avail_compare_metrics = [m for m in compare_metrics 
                                            if m in player1_latest.columns and m in player2_latest.columns]
                    
                    # Create comparison dataframe
                    compare_df = pd.DataFrame({
                        'Metric': avail_compare_metrics,
                        player1: player1_latest[avail_compare_metrics].values.flatten(),
                        player2: player2_latest[avail_compare_metrics].values.flatten(),
                        'Difference': player1_latest[avail_compare_metrics].values.flatten() - 
                                     player2_latest[avail_compare_metrics].values.flatten()
                    })
                    
                    st.dataframe(compare_df, use_container_width=True)
    else:
        st.warning("⚠️ Data missing for one or both players.")


with tab4:
    show_transfer_recommendations_tab()
    
with tab5:
    show_league_conversion_tab()
 

with tab6:
    show_time_series_tab()