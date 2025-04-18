import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

@st.cache_data
def calculate_league_value_coefficients(df):
    """
    Calculate the relative market value coefficients between leagues.
    """
    league_avg_values = df.groupby('league')['Valeur marchande (euros)'].mean().reset_index()
    global_avg = league_avg_values['Valeur marchande (euros)'].mean()
    league_avg_values['coefficient'] = league_avg_values['Valeur marchande (euros)'] / global_avg
    league_avg_age = df.groupby('league')['age'].mean().reset_index()
    result = pd.merge(league_avg_values, league_avg_age, on='league')
    return result

def show_league_conversion_tab():
    st.header("🔄 League-to-League Value Conversion")
    st.write("Estimate how a player's market value might change when moving between leagues.")

    league_coefficients = calculate_league_value_coefficients(df)

    col1, col2 = st.columns(2)
    with col1:
        source_league = st.selectbox("Source League", sorted(df["league"].unique()), key="source_league")
        current_value = st.number_input("Current Market Value (€)", min_value=0, value=10000000)

    with col2:
        target_league = st.selectbox("Target League", sorted(df["league"].unique()), key="target_league")

    source_coef = league_coefficients[league_coefficients['league'] == source_league]['coefficient'].values[0]
    target_coef = league_coefficients[league_coefficients['league'] == target_league]['coefficient'].values[0]

    conversion_ratio = target_coef / source_coef
    new_value = current_value * conversion_ratio

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

    st.subheader("League Value Coefficients")
    st.write("These coefficients represent the relative market value of players in each league compared to the global average.")

    sorted_coefficients = league_coefficients.sort_values('coefficient', ascending=False)
    display_coefs = sorted_coefficients.copy()
    display_coefs['Average Value (€)'] = display_coefs['Valeur marchande (euros)'].apply(lambda x: f"€{x:,.0f}")
    display_coefs['Coefficient'] = display_coefs['coefficient'].apply(lambda x: f"{x:.2f}x")
    display_coefs['Average Age'] = display_coefs['age'].apply(lambda x: f"{x:.1f}")

    st.dataframe(
        display_coefs[['league', 'Average Value (€)', 'Coefficient', 'Average Age']],
        use_container_width=True
    )

    fig = px.bar(
        sorted_coefficients,
        x='league',
        y='coefficient',
        color='coefficient',
        hover_data=['Valeur marchande (euros)', 'age'],
        title='League Value Coefficients',
        labels={
            'league': 'League',
            'coefficient': 'Value Coefficient'
        },
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
