import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# === Load Data ===
@st.cache_data

def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

st.title("📊 Football Market Value Analysis")

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
team_season_value = (
    filtered_df.groupby(["team", "season"])["Valeur marchande (euros)"]
    .sum()
    .reset_index()
)
team_avg_value = (
    team_season_value.groupby("team")["Valeur marchande (euros)"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
fig4 = px.bar(
    team_avg_value,
    x="team",
    y="Valeur marchande (euros)",
    title="Top 15 Teams by Avg Total Market Value (per Season)",
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

# === Top Rising Teams (2020 vs 2024) ===
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
fig7 = px.bar(
    team_year_value,
    x="team",
    y="Change",
    title="Top 10 Teams with Largest Increase in Total Market Value (2020 → 2024)",
    labels={"Change": "€ Change"},
)
st.plotly_chart(fig7, use_container_width=True)

# === National Representation by League ===
st.subheader("🌍 National Representation by League")
nationality_league = (
    filtered_df.groupby(["league", "nation"])["player"].nunique().reset_index(name="num_players")
)
fig8 = px.sunburst(
    nationality_league,
    path=["league", "nation"],
    values="num_players",
    title="Player Nationalities Across Leagues"
)
st.plotly_chart(fig8, use_container_width=True)

# === xG vs Market Value ===
st.subheader("📊 Expected Goals vs. Market Value")
fig9 = px.scatter(
    filtered_df,
    x="Expected xG",
    y="Valeur marchande (euros)",
    hover_name="player",
    color="pos",
    trendline="ols",
    title="Expected Goals vs. Market Value",
    labels={"Expected xG": "xG", "Valeur marchande (euros)": "Market Value (€)"}
)
st.plotly_chart(fig9, use_container_width=True)

# === Top Goal Scorers ===
top_scorers = filtered_df.sort_values("Performance Gls", ascending=False).dropna(subset=["Performance Gls"])
top_scorers = top_scorers.drop_duplicates("player").head(10)
fig10 = px.bar(
    top_scorers,
    x="player",
    y="Performance Gls",
    color="league",
    hover_data=["team", "season"],
    title="Top 10 Goal Scorers"
)
st.plotly_chart(fig10, use_container_width=True)

# === Correlation Heatmap ===
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1, how='all')
    corr = numeric_df.corr()
    relevant_cols = [col for col in corr.columns if any(x in col.lower() for x in 
                    ['performance', 'expected', 'gls', 'ast', 'valeur', 'xg', 'age'])]
    if len(relevant_cols) > 15:
        market_value_corr = abs(corr['Valeur marchande (euros)']).sort_values(ascending=False)
        relevant_cols = market_value_corr.index[:15].tolist()
    filtered_corr = corr.loc[relevant_cols, relevant_cols]
    fig = px.imshow(
        filtered_corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Between Key Performance Metrics",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    return fig

st.subheader("🔄 Correlation Between Key Metrics")
st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)

# === Market Value Trends by Position ===
st.subheader("📊 Market Value Trends by Position")
pos_time_value = (
    df.groupby(['season', 'pos'])['Valeur marchande (euros)'].mean().reset_index()
)
fig11 = px.line(
    pos_time_value,
    x="season",
    y="Valeur marchande (euros)",
    color="pos",
    markers=True,
    title="Average Market Value Trend by Position",
    labels={"season": "Season", "Valeur marchande (euros)": "Avg Market Value (€)"}
)
st.plotly_chart(fig11, use_container_width=True)

# === Playing Time vs Market Value ===
st.subheader("⚽ Playing Time vs Market Value")
fig12 = px.scatter(
    filtered_df,
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
st.plotly_chart(fig12, use_container_width=True)

# === Age vs Market Value by League ===
st.subheader("🌍 Age vs Market Value by League")
fig13 = px.scatter(
    filtered_df,
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
st.plotly_chart(fig13, use_container_width=True)
