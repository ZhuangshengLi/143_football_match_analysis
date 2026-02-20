import sqlite3
import pandas as pd
import numpy as np

# load database
con = sqlite3.connect("database.sqlite")

matches = pd.read_sql_query("SELECT * FROM Match;", con)
player_att = pd.read_sql_query("SELECT * FROM Player_Attributes;", con)

con.close()

# select attributes
selected_attributes = [
    "player_api_id",
    "date",
    "overall_rating",
    "finishing",
    "shot_power",
    "dribbling",
    "short_passing",
    "long_passing",
    "vision",
    "acceleration",
    "sprint_speed",
    "stamina",
    "strength",
    "marking",
    "standing_tackle",
    "sliding_tackle"
]

player_att = player_att[selected_attributes]

player_att = player_att.sort_values("date")
player_att = player_att.groupby("player_api_id").last().reset_index()
player_att = player_att.drop(columns=["date"])
# print(player_att.isna().sum())
player_att = player_att.fillna(0) # clean player data

match_columns = [
    "match_api_id",
    "home_team_goal",
    "away_team_goal"
]

for i in range(1, 12):
    match_columns.append(f"home_player_{i}")
    match_columns.append(f"away_player_{i}")
matches = matches[match_columns]
# print(matches.isna().sum())
matches = matches.dropna() # clean matches data

matches["home_win"] = (matches["home_team_goal"] > matches["away_team_goal"]).astype(int)

def get_team_matrix(player_ids):
    team_data = []

    for pid in player_ids:
        if pd.isna(pid):
            team_data.append(np.zeros(len(selected_attributes) - 2))
        else:
            player_row = player_att[player_att["player_api_id"] == pid]
            if len(player_row) == 0:
                team_data.append(np.zeros(len(selected_attributes) - 2))
            else:
                team_data.append(player_row.iloc[0, 1:].values)

    return np.array(team_data)

X = []
y = []

for _, row in matches.iterrows():

    home_players = [row[f"home_player_{i}"] for i in range(1, 12)]
    away_players = [row[f"away_player_{i}"] for i in range(1, 12)]

    home_matrix = get_team_matrix(home_players)
    away_matrix = get_team_matrix(away_players)

    combined = np.vstack((home_matrix, away_matrix))

    X.append(combined)
    y.append(row["home_win"])

X = np.array(X)
y = np.array(y)

# print("Input shape:", X.shape)   # (25979, 22, 14)
# print("Labels shape:", y.shape)

np.savez("processed_dataset.npz", X=X, y=y)