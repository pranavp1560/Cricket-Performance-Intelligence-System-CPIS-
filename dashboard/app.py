import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import optimizer and ML paths for integrations
from src.optimization.xi_optimizer import PlayingXIOptimizer

# Page configuration
st.set_page_config(
    page_title="Cricket Performance Intelligence System (CPIS)",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Navy, Teal, Slate layout)
st.markdown("""
<style>
    /* CSS for premium look */
    .stApp {
        background-color: #0B132B;
        color: #F8FAFC;
    }
    .css-1d391kg {
        background-color: #1C2541;
    }
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background: rgba(28, 37, 65, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #38BDF8;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #06B6D4;
    }
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 5px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #06B6D4 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
    }
    .stButton>button:hover {
        background-color: #0891B2 !important;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to connect to SQLite
def get_db_connection():
    db_path = "data/processed/cricket_intelligence.db"
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}. Please run cleaning and analytics pipelines first.")
        st.stop()
    return sqlite3.connect(db_path)

# Load database cache
@st.cache_data
def load_overall_metrics():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM player_overall_metrics ORDER BY cpis_rating DESC", conn)
    conn.close()
    return df

@st.cache_data
def load_venue_stats():
    conn = get_db_connection()
    df_stats = pd.read_sql("SELECT * FROM player_venue_stats", conn)
    df_classes = pd.read_sql("SELECT * FROM venue_classification", conn)
    df_insights = pd.read_sql("SELECT * FROM player_venue_insights", conn)
    conn.close()
    return df_stats, df_classes, df_insights

@st.cache_data
def load_predictions():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM player_predictions", conn)
    conn.close()
    return df

# Main logic
def main():
    st.sidebar.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-cricket-sports-and-games-flatart-icons-flat-flatarticons.png", width=80)
    st.sidebar.title("CPIS Navigation")
    
    page = st.sidebar.selectbox(
        "Select Analytics Dashboard",
        ["Executive Overview", "Player Profiler & Comparison", "Venue Intelligence", "Playing XI Optimizer", "Predictions & ML Insights", "Reports & Exports"]
    )
    
    # Load all data
    rankings_df = load_overall_metrics()
    venue_stats_df, venue_classes_df, venue_insights_df = load_venue_stats()
    predictions_df = load_predictions()
    
    # ------------------ 1. EXECUTIVE OVERVIEW ------------------
    if page == "Executive Overview":
        st.title("🏆 Cricket Performance Intelligence System")
        st.subheader("Industry-grade player evaluation, ranking, and squad optimization platform")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">122</div><div class="metric-label">Players Evaluated</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">300</div><div class="metric-label">Matches Tracked</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">7</div><div class="metric-label">Venues Analysed</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">2021-2025</div><div class="metric-label">Historical Range</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### Top 10 Player Rankings (Composite CPIS Rating)")
            display_cols = ["rank", "player", "role", "matches", "match_impact_score", "consistency_index", "pressure_performance_index", "cpis_rating"]
            top_10_display = rankings_df[display_cols].head(10).copy()
            top_10_display.columns = ["Rank", "Player", "Role", "Matches", "Match Impact (MIS)", "Consistency Index (CI)", "Pressure Index (PPI)", "CPIS Rating"]
            
            # Highlight with a nice table
            st.dataframe(top_10_display.style.background_gradient(cmap="Blues", subset=["CPIS Rating"]), use_container_width=True)
            
        with col_right:
            st.markdown("### CPIS Rating Distribution")
            fig_hist = px.histogram(
                rankings_df, x="cpis_rating", 
                nbins=15, 
                labels={"cpis_rating": "CPIS Player Rating"},
                color_discrete_sequence=["#06B6D4"],
                template="plotly_dark"
            )
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.write("")
        
        st.markdown("### Player Clustering: Match Impact vs Consistency")
        fig_scatter = px.scatter(
            rankings_df, x="match_impact_score", y="consistency_index",
            color="role", size="cpis_rating", hover_name="player",
            labels={"match_impact_score": "Match Impact Score (MIS)", "consistency_index": "Consistency Index (CI)"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------ 2. PLAYER PROFILER & COMPARISON ------------------
    elif page == "Player Profiler & Comparison":
        st.title("👤 Player Profiler & Comparison Engine")
        
        tabs = st.tabs(["Player Profiler", "Player Comparison"])
        
        with tabs[0]:
            st.subheader("Individual Performance Profile")
            selected_player = st.selectbox("Search and Select Player", rankings_df["player"].unique())
            
            p_data = rankings_df[rankings_df["player"] == selected_player].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Basic Bio Card
                st.markdown(f"""
                <div style="background:rgba(28,37,65,0.8); border:1px solid #38BDF8; border-radius:12px; padding:25px; margin-top:20px;">
                    <h2 style="margin-top:0; color:#38BDF8;">{selected_player}</h2>
                    <p style="font-size:16px; color:#06B6D4; font-weight:bold;">Role: {p_data['role']}</p>
                    <p style="font-size:14px; margin-bottom:5px;"><b>Overall Rank:</b> #{int(p_data['rank'])}</p>
                    <p style="font-size:14px; margin-bottom:5px;"><b>Matches Played:</b> {int(p_data['matches'])}</p>
                    <p style="font-size:14px; margin-bottom:5px;"><b>Runs Scored:</b> {int(p_data['runs'])} (Avg: {p_data['batting_average']:.1f})</p>
                    <p style="font-size:14px; margin-bottom:5px;"><b>Strike Rate:</b> {p_data['strike_rate']:.1f}</p>
                    <p style="font-size:14px; margin-bottom:5px;"><b>Wickets Taken:</b> {int(p_data['wickets'])} (Econ: {p_data['economy']:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show venue recommendation
                p_insights = venue_insights_df[venue_insights_df["player"] == selected_player]
                if len(p_insights) > 0:
                    st.write("")
                    st.info(f"💡 **Venue Analyst Tip:** {p_insights.iloc[0]['recommendation']}")
            
            with col2:
                # Radar Chart for CPIS metric dimensions
                categories = ["Match Impact (MIS)", "Consistency Index (CI)", "Pressure Index (PPI)", "Batting Raw", "Bowling Raw"]
                values = [
                    p_data["match_impact_score"],
                    p_data["consistency_index"],
                    p_data["pressure_performance_index"],
                    p_data["batting_raw_rating"],
                    p_data["bowling_raw_rating"]
                ]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    fillcolor='rgba(6, 182, 212, 0.3)',
                    line=dict(color='#06B6D4', width=2),
                    name=selected_player
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100]),
                        gridshape='circular'
                    ),
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
        with tabs[1]:
            st.subheader("Head-to-Head Player Comparison")
            compare_players = st.multiselect("Select Players to Compare (Max 3)", rankings_df["player"].unique(), default=rankings_df["player"].head(2).tolist())
            
            if len(compare_players) > 0:
                fig_compare = go.Figure()
                colors = ["#38BDF8", "#F43F5E", "#10B981"]
                
                for idx, player in enumerate(compare_players):
                    p_data = rankings_df[rankings_df["player"] == player].iloc[0]
                    values = [
                        p_data["match_impact_score"],
                        p_data["consistency_index"],
                        p_data["pressure_performance_index"],
                        p_data["batting_raw_rating"],
                        p_data["bowling_raw_rating"]
                    ]
                    fig_compare.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        name=player
                    ))
                    
                fig_compare.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Comparison table
                compare_data = rankings_df[rankings_df["player"].isin(compare_players)][["player", "role", "matches", "cpis_rating", "match_impact_score", "consistency_index", "pressure_performance_index"]]
                compare_data.columns = ["Player", "Role", "Matches", "CPIS Rating", "MIS", "Consistency", "Pressure"]
                st.table(compare_data.set_index("Player"))

    # ------------------ 3. VENUE INTELLIGENCE ------------------
    elif page == "Venue Intelligence":
        st.title("🏟️ Venue Intelligence Engine")
        
        selected_venue = st.selectbox("Select Venue to Analyze", venue_classes_df["venue"].unique())
        
        v_class = venue_classes_df[venue_classes_df["venue"] == selected_venue].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{v_class["classification"]}</div><div class="metric-label">Venue Character</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{v_class["run_rate"]:.2f}</div><div class="metric-label">Average Run Rate (RPO)</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{v_class["avg_first_innings_score"]:.1f}</div><div class="metric-label">Average 1st Inn Score</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"### Best Batters at {selected_venue}")
            # Filter venue stats for this venue
            v_stats = venue_stats_df[venue_stats_df["venue"] == selected_venue].copy()
            v_stats_bat = v_stats[v_stats["bat_matches"] > 0].sort_values(by="bat_runs", ascending=False).head(5)
            
            st.dataframe(v_stats_bat[["player", "bat_matches", "bat_runs", "bat_average", "bat_strike_rate"]].rename(columns={
                "player": "Batsman", "bat_matches": "Matches", "bat_runs": "Runs", "bat_average": "Average", "bat_strike_rate": "Strike Rate"
            }).style.background_gradient(cmap="Blues", subset=["Runs"]), use_container_width=True)
            
        with col_right:
            st.markdown(f"### Best Bowlers at {selected_venue}")
            v_stats_bowl = v_stats[v_stats["bowl_matches"] > 0].sort_values(by="bowl_wickets", ascending=False).head(5)
            
            st.dataframe(v_stats_bowl[["player", "bowl_matches", "bowl_wickets", "bowl_economy", "bowl_average"]].rename(columns={
                "player": "Bowler", "bowl_matches": "Matches", "bowl_wickets": "Wickets", "bowl_economy": "Economy", "bowl_average": "Bowling Avg"
            }).style.background_gradient(cmap="Greens", subset=["Wickets"]), use_container_width=True)

    # ------------------ 4. PLAYING XI OPTIMIZER ------------------
    elif page == "Playing XI Optimizer":
        st.title("🏏 Mixed-Integer Linear Programming playing XI Optimizer")
        st.markdown("### Maximize total team CPIS Player Rating under positional constraints")
        
        col_ctrl, col_pitch = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("### Squad Constraints")
            # Create user adjusters for LP Constraints
            min_openers = st.slider("Min Openers", 1, 3, 2)
            max_openers = st.slider("Max Openers", 2, 4, 2)
            min_middle = st.slider("Min Middle Order/Finishers", 2, 6, 3)
            max_middle = st.slider("Max Middle Order/Finishers", 4, 8, 5)
            min_all_rounders = st.slider("Min All-Rounders", 0, 4, 1)
            max_all_rounders = st.slider("Max All-Rounders", 1, 5, 3)
            min_bowlers = st.slider("Min Bowlers", 2, 6, 3)
            max_bowlers = st.slider("Max Bowlers", 4, 8, 5)
            
            # Run optimizer button
            if st.button("Generate Optimal Playing XI"):
                with st.spinner("Running LP Solver..."):
                    optimizer = PlayingXIOptimizer()
                    constraints = {
                        'min_openers': min_openers, 'max_openers': max_openers,
                        'min_middle_order': min_middle, 'max_middle_order': max_middle,
                        'min_all_rounders': min_all_rounders, 'max_all_rounders': max_all_rounders,
                        'min_bowlers': min_bowlers, 'max_bowlers': max_bowlers,
                        'min_wks': 1, 'max_wks': 2
                    }
                    
                    squad, capt, vc = optimizer.optimize_xi(rankings_df, constraints)
                    
                    # Store in session state for rendering
                    st.session_state["opt_squad"] = squad
                    st.session_state["opt_capt"] = capt
                    st.session_state["opt_vc"] = vc
                    st.success("Optimal XI Found!")
                    
        with col_pitch:
            st.markdown("### Optimal Team Composition")
            if "opt_squad" in st.session_state:
                squad = st.session_state["opt_squad"]
                capt = st.session_state["opt_capt"]
                vc = st.session_state["opt_vc"]
                
                # Format squad display
                squad_disp = squad[["player", "role", "cpis_rating", "consistency_index", "match_impact_score"]].copy()
                squad_disp.columns = ["Player", "Role", "CPIS Rating", "Consistency (CI)", "Match Impact (MIS)"]
                
                # Add badges
                def add_badge(player_name):
                    if player_name == capt:
                        return f"👑 {player_name} (Captain)"
                    elif player_name == vc:
                        return f"⭐ {player_name} (Vice-Captain)"
                    return player_name
                    
                squad_disp["Player"] = squad_disp["Player"].apply(add_badge)
                
                st.dataframe(squad_disp.style.background_gradient(cmap="Blues", subset=["CPIS Rating"]), use_container_width=True)
                st.info(f"👑 **Captain Recommendation:** {capt} (Selected for maximum overall CPIS rating).")
                st.info(f"⭐ **Vice-Captain Recommendation:** {vc} (Selected for maximum consistency index amongst squad players).")
            else:
                st.write("Click 'Generate Optimal Playing XI' to run the linear programming solver.")

    # ------------------ 5. PREDICTIONS & ML INSIGHTS ------------------
    elif page == "Predictions & ML Insights":
        st.title("🔮 Machine Learning Predictive Forecasts")
        st.subheader("Predicting 2026 performance shifts using Random Forest, XGBoost, and LightGBM models")
        
        # Display model performance metrics
        # Load evaluation metrics if pkl file exists
        metrics_path = "models/evaluation_metrics.pkl"
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                metrics_dict = pickle.load(f)
                
            st.markdown("### Model Evaluation Metrics (Test Split)")
            eval_rows = []
            for target, models in metrics_dict.items():
                for model, vals in models.items():
                    eval_rows.append({
                        "Target Variable": target.replace("_next", ""),
                        "Model": model,
                        "R2 Score": f"{vals['R2']:.3f}",
                        "MAE": f"{vals['MAE']:.2f}",
                        "RMSE": f"{vals['RMSE']:.2f}"
                    })
            st.table(pd.DataFrame(eval_rows))
            
        st.write("")
        st.markdown("### Player Forecast Explorer")
        st.write("Search a player to see their predicted stats for the 2026 season:")
        
        target_player = st.selectbox("Select Player to Forecast", predictions_df["player"].unique())
        
        p_pred = predictions_df[predictions_df["player"] == target_player].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rating card
            diff = p_pred["rating_predicted_2026"] - p_pred["rating_actual_2025"]
            diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
            st.metric(
                label="CPIS Rating Prediction",
                value=f"{p_pred['rating_predicted_2026']:.2f}",
                delta=f"{diff_str} change vs 2025"
            )
            
        with col2:
            # Runs card
            diff_runs = int(p_pred["runs_predicted_2026"] - p_pred["runs_actual_2025"])
            diff_runs_str = f"+{diff_runs}" if diff_runs >= 0 else f"{diff_runs}"
            st.metric(
                label="Predicted Runs",
                value=int(p_pred["runs_predicted_2026"]),
                delta=f"{diff_runs_str} runs change vs 2025"
            )
            
        with col3:
            # Wickets card
            diff_wicks = int(p_pred["wickets_predicted_2026"] - p_pred["wickets_actual_2025"])
            diff_wicks_str = f"+{diff_wicks}" if diff_wicks >= 0 else f"{diff_wicks}"
            st.metric(
                label="Predicted Wickets",
                value=int(p_pred["wickets_predicted_2026"]),
                delta=f"{diff_wicks_str} wickets change vs 2025"
            )

    # ------------------ 6. REPORTS & EXPORTS ------------------
    elif page == "Reports & Exports":
        st.title("📁 Reports & Analytical Exports")
        st.subheader("Download complete performance intelligence digests and datasets")
        
        st.write("")
        
        # Download widgets for generated report artifacts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background:rgba(28,37,65,0.7); border:1px solid #06B6D4; border-radius:12px; padding:25px; text-align:center;">
                <h3 style="color:#38BDF8; margin-top:0;">PDF Intelligence Report</h3>
                <p style="font-size:14px; color:#94A3B8;">Contains visual ranking sheets, venue recommendations, executive summary, and graphs.</p>
            </div>
            """, unsafe_allow_html=True)
            pdf_path = "reports/player_intelligence_report.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f.read(),
                        file_name="player_intelligence_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("PDF Report not found on disk. Please compile it first.")
                
        with col2:
            st.markdown("""
            <div style="background:rgba(28,37,65,0.7); border:1px solid #10B981; border-radius:12px; padding:25px; text-align:center;">
                <h3 style="color:#10B981; margin-top:0;">Excel Raw Analytics Workbook</h3>
                <p style="font-size:14px; color:#94A3B8;">Includes multi-sheet exports: Player overall stats, venue stats, and predictive shifts.</p>
            </div>
            """, unsafe_allow_html=True)
            xlsx_path = "reports/player_intelligence_report.xlsx"
            if os.path.exists(xlsx_path):
                with open(xlsx_path, "rb") as f:
                    st.download_button(
                        label="Download Excel Workbook",
                        data=f.read(),
                        file_name="player_intelligence_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Excel Workbook not found on disk. Please compile it first.")

if __name__ == "__main__":
    import pickle
    main()
