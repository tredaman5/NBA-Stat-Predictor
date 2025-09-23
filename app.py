import streamlit as st
import joblib
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# ---------------------------
# Load trained model
# ---------------------------
model = joblib.load("nba_model.pkl")

# ---------------------------
# Helper Functions
# ---------------------------

def get_player_id(name):
    """Search for a player ID by name using nba_api."""
    try:
        all_players = players.get_players()
        match = next((p for p in all_players if name.lower() in p["full_name"].lower()), None)
        if match:
            return match["id"], match["full_name"]
        else:
            return None, None
    except Exception as e:
        st.error(f"❌ Error searching for player: {e}")
        return None, None


def get_recent_stats(player_id, season="2023-24", last_n_games=5):
    """Fetch recent game logs for a player."""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]

        if df.empty:
            return None

        # Keep only the most recent N games
        df = df.head(last_n_games)

        return df
    except Exception as e:
        st.error(f"❌ Error fetching recent game logs: {e}")
        return None


def prepare_features(df):
    """Prepare input features [AST, REB] for the model."""
    avg_ast = df["AST"].mean()
    avg_reb = df["REB"].mean()
    return np.array([[avg_ast, avg_reb]])


# ---------------------------
# Streamlit App UI (Improved)
# ---------------------------

# App config
st.set_page_config(page_title="🏀 NBA Stat Predictor", page_icon="🏀", layout="wide")

# Header
st.markdown(
    """
    <div style="text-align:center; padding:15px; background-color:#0b1e2d; border-radius:10px;">
        <h1 style="color:#ffffff;">🏀 NBA Stat Predictor</h1>
        <p style="color:#cccccc;">Powered by <b>nba_api</b> | Predict player performance based on recent stats</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Player search input
st.sidebar.header("🔎 Search Player")
player_name = st.sidebar.text_input("Enter Player Name (e.g., Stephen Curry, LeBron James):")

# Prediction button
if st.sidebar.button("Predict"):
    if not player_name.strip():
        st.sidebar.error("⚠️ Please enter a player name.")
    else:
        pid, full_name = get_player_id(player_name)
        if pid is None:
            st.sidebar.error("❌ Player not found. Try a different name.")
        else:
            st.success(f"🔎 Found player: {full_name} (ID: {pid})")

            stats_df = get_recent_stats(pid)
            if stats_df is None:
                st.error("❌ Could not fetch recent games for this player.")
            else:
                # Player card
                st.markdown(
                    f"""
                    <div style="padding:20px; border-radius:15px; background-color:#f8f9fa; box-shadow:0px 4px 12px rgba(0,0,0,0.2); margin-bottom:20px;">
                        <h2 style="color:#0b1e2d;">📋 {full_name}</h2>
                        <p style="color:#333;">Recent performance stats used for prediction</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show recent stats
                st.write("📊 Recent Games:")
                st.dataframe(stats_df[["GAME_DATE", "PTS", "AST", "REB"]])

                # Make prediction
                features = prepare_features(stats_df)
                prediction = model.predict(features)

                # Display prediction in a metric card
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 Predicted Points", f"{prediction[0]:.2f}")
                col2.metric("📈 Avg Assists", f"{stats_df['AST'].mean():.1f}")
                col3.metric("🏀 Avg Rebounds", f"{stats_df['REB'].mean():.1f}")
