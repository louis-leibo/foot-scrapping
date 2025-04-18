import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

@st.cache_resource
def load_model():
    return load("player_value_predictor_final_xgb.joblib")

def plot_player_performance_vs_market_value(player_data):
    if player_data.empty or len(player_data) < 2:
        return None

    import plotly.express as px

    # Select key performance metrics
    performance_metrics = ['Performance Gls', 'Performance Ast', 'Expected xG', 
                           'Expected xAG', 'Performance G+A']

    metrics = [m for m in performance_metrics if m in player_data.columns]

    if not metrics:
        return None

    fig = go.Figure()

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

def plot_performance_radar_charts(player_data):
    if player_data.empty:
        return None

    metrics = [
        'Expected xG', 'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
        'Expected xAG', 'Performance G+A', 'Performance CrdY'
    ]

    available_metrics = [m for m in metrics if m in player_data.columns]

    if not available_metrics:
        return None

    seasons = sorted(player_data['season'].unique())
    fig = go.Figure()

    for season in seasons:
        season_data = player_data[player_data['season'] == season]

        if not season_data.empty:
            values = season_data[available_metrics].values.flatten().tolist()
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

def plot_prediction_vs_actual(y_true, y_pred):
    import plotly.express as px
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })

    fig = px.scatter(
        results_df,
        x='Actual',
        y='Predicted',
        labels={'Actual': 'Actual Market Value (€)', 'Predicted': 'Predicted Market Value (€)'},
        title='Prediction vs Actual Market Values'
    )

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

def plot_feature_importance(model, feature_names):
    import plotly.express as px
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)

        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 20 Feature Importances for Market Value Prediction',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature'}
        )
        return fig
    return None

# ======================== MAIN PREDICTION TAB ==============================
def show_prediction_tab():
    st.subheader("🧠 Predict a Player's Market Value for Next Season")
    df = load_data()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Train / Retrain Model")

        with st.form("model_training_form"):
            st.markdown("#### Model Configuration")
            n_estimators = st.slider("Number of Trees", 50, 500, 150, 50)
            max_depth = st.slider("Maximum Tree Depth", 5, 30, 15, 5)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 5, 1)
            train_button = st.form_submit_button("Train Model")

            if train_button:
                with st.spinner("Training model..."):
                    df_model = df[df["Valeur marchande (euros)"].notna()].copy()

                    df_model['value_per_goal'] = df_model['Valeur marchande (euros)'] / (df_model['Performance Gls'] + 1)
                    df_model['minutes_played_ratio'] = df_model['Playing Time Min'] / (90 * 38)
                    df_model['goals_per_90'] = df_model['Performance Gls'] / (df_model['Playing Time 90s'] + 0.001)
                    df_model['assists_per_90'] = df_model['Performance Ast'] / (df_model['Playing Time 90s'] + 0.001)

                    X = df_model.drop(columns=["Valeur marchande (euros)", "player", "born"], errors="ignore")
                    y = df_model["Valeur marchande (euros)"]

                    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
                    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

                    preprocessor = ColumnTransformer([
                        ("num", SimpleImputer(strategy="median"), num_cols),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ]), cat_cols)
                    ], remainder='drop')

                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("regressor", RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42))
                    ])

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipeline.fit(X_train, y_train)

                    y_pred = pipeline.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    st.session_state.update({
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "feature_names": num_cols + cat_cols,
                        "model": pipeline,
                        "cat_cols": cat_cols,
                        "num_cols": num_cols
                    })

                    dump((pipeline, cat_cols, num_cols), "player_value_predictor_final_xgb.joblib")
                    st.success("✅ Model trained successfully!")

        if "model" in st.session_state and "mae" in st.session_state:
            st.markdown("### 📊 Model Performance")
            st.metric("Mean Absolute Error (€)", f"{st.session_state['mae']:,.0f}")
            st.metric("RMSE (€)", f"{st.session_state['rmse']:,.0f}")
            st.metric("R² Score", f"{st.session_state['r2']:.3f}")

    with col2:
        if "model" not in st.session_state:
            try:
                model, cat_cols, num_cols = load_model()
                st.session_state.update({
                    "model": model,
                    "cat_cols": cat_cols,
                    "num_cols": num_cols
                })
                st.info("✅ Loaded model from saved file.")
            except Exception as e:
                st.error(f"❌ Error loading saved model: {e}")

        if "model" in st.session_state:
            model = st.session_state["model"]

            st.markdown("### 🔮 Player Value Prediction")
            player_list = df["player"].dropna().unique()
            selected_player = st.selectbox("Choose a player", sorted(player_list))

            player_data = df[df["player"] == selected_player].sort_values("season")

            if not player_data.empty:
                st.markdown("### 📈 Performance & Market Value History")
                fig1 = plot_player_performance_vs_market_value(player_data)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)

                st.markdown("### 🎯 Player Performance Radar")
                radar_fig = plot_performance_radar_charts(player_data)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)

                latest_row = player_data.iloc[-1:].copy()

                latest_row["mv_lag1"] = player_data["Valeur marchande (euros)"].shift(1).iloc[-1]
                latest_row["rolling_mv_2"] = player_data["Valeur marchande (euros)"].rolling(2).mean().iloc[-1]
                latest_row["rolling_mv_3"] = player_data["Valeur marchande (euros)"].rolling(3).mean().iloc[-1]
                latest_row["gls_per_90"] = latest_row["Performance Gls"] / (latest_row["Playing Time 90s"] + 1e-3)
                latest_row["market_age_ratio"] = latest_row["Valeur marchande (euros)"] / (latest_row["age"] + 1e-3)

                if len(player_data) > 1:
                    previous_season = player_data.iloc[-2:-1]
                    for col in ['Performance Gls', 'Performance Ast', 'Expected xG', 'Playing Time 90s']:
                        if col in latest_row.columns and col in previous_season.columns:
                            latest_row[f'{col}_trend'] = latest_row[col].values[0] - previous_season[col].values[0]

                latest_row['value_per_goal'] = latest_row['Valeur marchande (euros)'] / (latest_row['Performance Gls'] + 1)
                latest_row['minutes_played_ratio'] = latest_row['Playing Time Min'] / (90 * 38)
                latest_row['goals_per_90'] = latest_row['Performance Gls'] / (latest_row['Playing Time 90s'] + 0.001)
                latest_row['assists_per_90'] = latest_row['Performance Ast'] / (latest_row['Playing Time 90s'] + 0.001)

                prediction_data = latest_row.drop(columns=["player", "born", "Valeur marchande (euros)"], errors="ignore")

                try:
                    prediction = model.predict(prediction_data)[0]
                    current_value = latest_row["Valeur marchande (euros)"].values[0]
                    value_change = prediction - current_value
                    change_percent = (value_change / current_value) * 100 if current_value > 0 else 0

                    st.markdown("### 🤖 Predicted Market Value (Next Season)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Value", f"€{current_value:,.0f}")
                    col2.metric("Predicted Value", f"€{prediction:,.0f}", f"{value_change:+,.0f} ({change_percent:+.1f}%)")
                    col3.metric("Player Age (Next Season)", f"{latest_row['age'].values[0] + 1:.0f}")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

            else:
                st.warning("⚠️ No data available for selected player.")

        if "model" in st.session_state and "feature_names" in st.session_state:
            st.markdown("### 🔍 Feature Importance")
            regressor = st.session_state["model"].named_steps['regressor']
            feature_names = st.session_state["feature_names"]
            importance_fig = plot_feature_importance(regressor, feature_names)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)

        if "y_test" in st.session_state and "y_pred" in st.session_state:
            st.markdown("### 📊 Model Accuracy")
            pred_vs_actual_fig = plot_prediction_vs_actual(
                st.session_state["y_test"], 
                st.session_state["y_pred"]
            )
            st.plotly_chart(pred_vs_actual_fig, use_container_width=True)
