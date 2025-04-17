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


# === Load Data and Model ===
@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

@st.cache_resource
def load_model():
    return load("player_value_predictor.joblib")

model, cat_cols, num_cols = load_model()

# === Setup ===
# st.set_page_config(page_title="⚽ Market Value Explorer", layout="wide")
st.title("⚽ Football Player Market Value Platform")

# === Tabs ===
tab1, tab2 = st.tabs(["📊 Analysis", "🧠 Predict Market Value"])

# ---------------------
# 📊 ANALYSIS TAB
# ---------------------
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


# ---------------------
# 🧠 PREDICTION TAB
# ---------------------
with tab2:
    st.subheader("🧠 Predict a Player's Market Value for Next Season")

    # === Optional: Model Training ===
    st.markdown("### ⚙️ Train / Retrain Model")

    if st.button("Train Model"):
        with st.spinner("Training model..."):

            # Drop missing target
            df_model = df[df["Valeur marchande (euros)"].notna()].copy()

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
            ])

            # Full pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
            ])

            # Train/test split and fit
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            pipeline.fit(X_train, y_train)

            # Save the model in memory (not to disk)
            model = pipeline
            st.session_state["model"] = pipeline
            st.session_state["cat_cols"] = cat_cols
            st.session_state["num_cols"] = num_cols

            st.success("✅ Model trained successfully!")

    # === Prediction UI ===
    from pathlib import Path

    # Try to load model from session OR file
    if "model" not in st.session_state:
        try:
            if Path("player_value_predictor.joblib").exists():
                loaded_model, cat_cols, num_cols = load("player_value_predictor.joblib")
                st.session_state["model"] = loaded_model
                st.session_state["cat_cols"] = cat_cols
                st.session_state["num_cols"] = num_cols
                st.info("✅ Loaded model from saved file.")
            else:
                st.warning("⚠️ No trained model found. Please train one first.")
        except Exception as e:
            st.error(f"❌ Error loading saved model: {e}")

    if "model" in st.session_state:
        model = st.session_state["model"]

        player_list = df["player"].dropna().unique()
        selected_player = st.selectbox("Choose a player", sorted(player_list))

        player_data = df[df["player"] == selected_player].sort_values("season")

        if not player_data.empty:
            st.markdown("### 📈 Market Value History")
            fig3 = px.line(
                player_data,
                x="season",
                y="Valeur marchande (euros)",
                title=f"{selected_player} - Market Value Over Time",
                markers=True
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### 🤖 Predicted Market Value (Next Season)")

            # Get latest row to predict from
            latest_row = player_data.iloc[-1:].drop(columns=["player", "born", "Valeur marchande (euros)"], errors="ignore")

            # Predict
            try:
                prediction = model.predict(latest_row)[0]
                st.success(f"📊 Predicted value for next season: **€{prediction:,.0f}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

        else:
            st.warning("⚠️ No data available for selected player.")
