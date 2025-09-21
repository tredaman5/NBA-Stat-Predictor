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
# Streamlit App UI
# ---------------------------

st.title("🏀 NBA Stat Predictor (Powered by nba_api)")
st.write("Enter an NBA player's name to predict their next game's points using AST + REB averages.")

player_name = st.text_input("Enter Player Name (e.g., Stephen Curry, LeBron James):")

if st.button("Predict"):
    if not player_name.strip():
        st.error("⚠️ Please enter a player name.")
    else:
        pid, full_name = get_player_id(player_name)
        if pid is None:
            st.error("❌ Player not found. Try a different name.")
        else:
            st.info(f"🔎 Found player: {full_name} (ID: {pid})")

            stats_df = get_recent_stats(pid)
            if stats_df is None:
                st.error("❌ Could not fetch recent games for this player.")
            else:
                st.write("📊 Recent Games (used for prediction):")
                st.dataframe(stats_df[["GAME_DATE", "PTS", "AST", "REB"]])

                features = prepare_features(stats_df)
                prediction = model.predict(features)

                st.success(f"🎯 Predicted Points for {full_name}: **{prediction[0]:.2f}**")
