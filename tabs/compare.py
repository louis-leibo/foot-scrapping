import streamlit as st
import pandas as pd
import plotly.graph_objects as go

@st.cache_data
def load_data():
    return pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

df = load_data()

def show_comparison_tab():
    st.subheader("🔄 Compare Players")

    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox("Select First Player", sorted(df["player"].dropna().unique()), key="player1")

    with col2:
        player2 = st.selectbox("Select Second Player", sorted(df["player"].dropna().unique()), key="player2")

    player1_data = df[df["player"] == player1].sort_values("season")
    player2_data = df[df["player"] == player2].sort_values("season")

    if not player1_data.empty and not player2_data.empty:
        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Market Value", "Performance", "Radar Charts"])

        with comp_tab1:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=player1_data['season'],
                y=player1_data['Valeur marchande (euros)'],
                mode='lines+markers',
                name=player1,
                line=dict(width=3)
            ))

            fig.add_trace(go.Scatter(
                x=player2_data['season'],
                y=player2_data['Valeur marchande (euros)'],
                mode='lines+markers',
                name=player2,
                line=dict(width=3, dash='dash')
            ))

            fig.update_layout(
                title=f"Market Value Comparison: {player1} vs {player2}",
                xaxis=dict(title='Season'),
                yaxis=dict(title='Market Value (€)'),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with comp_tab2:
            metrics = ['Performance Gls', 'Performance Ast', 'Expected xG', 'Expected xAG', 'Performance G+A', 'Playing Time 90s']
            avail_metrics = [m for m in metrics if m in player1_data.columns and m in player2_data.columns]

            selected_metric = st.selectbox("Select Metric", avail_metrics)

            if selected_metric:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=player1_data['season'],
                    y=player1_data[selected_metric],
                    mode='lines+markers',
                    name=player1,
                    line=dict(width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=player2_data['season'],
                    y=player2_data[selected_metric],
                    mode='lines+markers',
                    name=player2,
                    line=dict(width=3, dash='dash')
                ))

                fig.update_layout(
                    title=f"{selected_metric} Comparison: {player1} vs {player2}",
                    xaxis=dict(title='Season'),
                    yaxis=dict(title=selected_metric),
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

        with comp_tab3:
            player1_latest = player1_data.iloc[-1:] if not player1_data.empty else None
            player2_latest = player2_data.iloc[-1:] if not player2_data.empty else None

            if player1_latest is not None and player2_latest is not None:
                radar_metrics = [
                    'Expected xG', 'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
                    'Expected xAG', 'Performance G+A', 'Performance CrdY'
                ]
                avail_radar_metrics = [m for m in radar_metrics if m in player1_latest.columns and m in player2_latest.columns]

                if avail_radar_metrics:
                    fig = go.Figure()

                    p1_values = player1_latest[avail_radar_metrics].values.flatten().tolist()
                    p1_values.append(p1_values[0])

                    fig.add_trace(go.Scatterpolar(
                        r=p1_values,
                        theta=avail_radar_metrics + [avail_radar_metrics[0]],
                        fill='toself',
                        name=player1
                    ))

                    p2_values = player2_latest[avail_radar_metrics].values.flatten().tolist()
                    p2_values.append(p2_values[0])

                    fig.add_trace(go.Scatterpolar(
                        r=p2_values,
                        theta=avail_radar_metrics + [avail_radar_metrics[0]],
                        fill='toself',
                        name=player2
                    ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        title=f"Performance Comparison: {player1} vs {player2} (Latest Season)"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("🔢 Key Stats Comparison")
                    compare_metrics = avail_radar_metrics + ['age', 'Valeur marchande (euros)', 'Playing Time MP', 'Playing Time Min']
                    avail_compare_metrics = [m for m in compare_metrics if m in player1_latest.columns and m in player2_latest.columns]

                    compare_df = pd.DataFrame({
                        'Metric': avail_compare_metrics,
                        player1: player1_latest[avail_compare_metrics].values.flatten(),
                        player2: player2_latest[avail_compare_metrics].values.flatten(),
                        'Difference': player1_latest[avail_compare_metrics].values.flatten() - player2_latest[avail_compare_metrics].values.flatten()
                    })

                    st.dataframe(compare_df, use_container_width=True)
    else:
        st.warning("⚠️ Data missing for one or both players.")
