import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# --- Step 1: Get data from API ---
players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Kevin Durant", "Luka Doncic"]
stats_data = []

def get_player_id(name):
    url = f"https://www.balldontlie.io/api/v1/players?search={name}"
    r = requests.get(url).json()
    if r["data"]:
        return r["data"][0]["id"]
    return None

def get_recent_games(player_id, num_games=20):
    url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page={num_games}"
    r = requests.get(url).json()
    return r["data"]

# Fetch data for each player
for player in players:
    pid = get_player_id(player)
    if pid:
        games = get_recent_games(pid)
        for g in games:
            stats_data.append({
                "name": player,
                "points": g["pts"],
                "assists": g["ast"],
                "rebounds": g["reb"]
            })

# Convert to DataFrame
df = pd.DataFrame(stats_data)
print(df.head())

# --- Step 2: Train a simple model ---
X = df[["assists", "rebounds"]]
y = df["points"]

model = LinearRegression()
model.fit(X, y)

# --- Step 3: Save the model ---
joblib.dump(model, "nba_model.pkl")

print("✅ Model trained and saved as nba_model.pkl")
