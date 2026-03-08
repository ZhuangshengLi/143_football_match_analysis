"""L1/L2/L3 feature engineering for European Soccer DB.

Feature levels:
  L1 - Player static ability (overall, attack, defense, passing, pace, physical, gk)
  L2 - Team tactical style (buildUpPlay, chanceCreation, defence)
  L3 - Dynamic state (last 5 win rate, goal diff, Elo, rest days)
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Player_Attributes columns for L1 aggregation
PLAYER_ATTACK = ["finishing", "shot_power", "dribbling", "long_shots", "positioning", "penalties"]
PLAYER_DEFENSE = ["marking", "standing_tackle", "sliding_tackle", "interceptions"]
PLAYER_PASSING = ["short_passing", "long_passing", "vision", "ball_control"]
PLAYER_PACE = ["acceleration", "sprint_speed"]
PLAYER_PHYSICAL = ["stamina", "strength", "agility"]
PLAYER_GK = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
PLAYER_SHOOTING = ["finishing", "shot_power", "long_shots"]

# Team_Attributes columns for L2
TEAM_STYLE_COLS = [
    "buildUpPlaySpeed", "buildUpPlayPassing",
    "chanceCreationPassing", "chanceCreationShooting",
    "defencePressure", "defenceAggression",
]


def _safe_mean(arr: np.ndarray) -> float:
    """Mean of finite values; nan if empty. Used for team-level aggregation."""
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def _get_player_attr_at_date(player_att: pd.DataFrame, player_ids: list, match_date: pd.Timestamp) -> pd.DataFrame:
    """Lookup player attributes at match_date. Uses most recent record with date <= match_date (no leakage)."""
    cols = ["overall_rating"] + PLAYER_ATTACK + PLAYER_DEFENSE + PLAYER_PASSING + PLAYER_PACE + PLAYER_PHYSICAL + PLAYER_GK + PLAYER_SHOOTING
    cols = [c for c in cols if c in player_att.columns]
    out = []
    for pid in player_ids:
        if pd.isna(pid):
            out.append({c: np.nan for c in cols})
            continue
        sub = player_att[(player_att["player_api_id"] == pid) & (player_att["date"] <= match_date)]
        if len(sub) == 0:
            out.append({c: np.nan for c in cols})
            continue
        row = sub.sort_values("date").iloc[-1]  # most recent before match
        out.append({c: row.get(c, np.nan) for c in cols})
    return pd.DataFrame(out)


def _team_index(players_df: pd.DataFrame, cols: list) -> float:
    """Mean of available attribute columns for a team's 11 players."""
    avail = [c for c in cols if c in players_df.columns]
    if not avail:
        return np.nan
    return players_df[avail].values.mean()


def _build_L1_features(home_players: pd.DataFrame, away_players: pd.DataFrame) -> dict:
    """Aggregate player attributes: overall, attack, defense, passing, pace, gk, cross-terms."""
    def idx(df, cols):
        return _team_index(df, cols)

    h_overall = _safe_mean(home_players["overall_rating"]) if "overall_rating" in home_players.columns else np.nan
    a_overall = _safe_mean(away_players["overall_rating"]) if "overall_rating" in away_players.columns else np.nan

    h_attack = idx(home_players, PLAYER_ATTACK)
    a_attack = idx(away_players, PLAYER_ATTACK)
    h_defense = idx(home_players, PLAYER_DEFENSE)
    a_defense = idx(away_players, PLAYER_DEFENSE)
    h_passing = idx(home_players, PLAYER_PASSING)
    a_passing = idx(away_players, PLAYER_PASSING)
    h_pace = idx(home_players, PLAYER_PACE)
    a_pace = idx(away_players, PLAYER_PACE)
    h_physical = idx(home_players, PLAYER_PHYSICAL)
    a_physical = idx(away_players, PLAYER_PHYSICAL)
    h_gk = idx(home_players, PLAYER_GK)
    a_gk = idx(away_players, PLAYER_GK)
    h_shooting = idx(home_players, PLAYER_SHOOTING)
    a_shooting = idx(away_players, PLAYER_SHOOTING)

    return {
        "L1_home_overall_mean": h_overall,
        "L1_away_overall_mean": a_overall,
        "L1_home_attack_index": h_attack,
        "L1_away_attack_index": a_attack,
        "L1_home_defense_index": h_defense,
        "L1_away_defense_index": a_defense,
        "L1_home_passing_index": h_passing,
        "L1_away_passing_index": a_passing,
        "L1_home_pace_index": h_pace,
        "L1_away_pace_index": a_pace,
        "L1_home_physical_index": h_physical,
        "L1_away_physical_index": a_physical,
        "L1_home_gk_index": h_gk,
        "L1_away_gk_index": a_gk,
        "L1_diff_overall": h_overall - a_overall if np.isfinite(h_overall) and np.isfinite(a_overall) else np.nan,
        "L1_diff_attack": h_attack - a_attack if np.isfinite(h_attack) and np.isfinite(a_attack) else np.nan,
        "L1_attack_home_vs_defense_away": h_attack - a_defense if np.isfinite(h_attack) and np.isfinite(a_defense) else np.nan,
        "L1_gk_home_vs_shooting_away": h_gk - a_shooting if np.isfinite(h_gk) and np.isfinite(a_shooting) else np.nan,
    }


def _get_team_attr_at_date(team_att: pd.DataFrame, team_id: int, match_date: pd.Timestamp) -> pd.Series:
    """Lookup team tactical style at match_date. Most recent record with date <= match_date."""
    sub = team_att[(team_att["team_api_id"] == team_id) & (team_att["date"] <= match_date)]
    if len(sub) == 0:
        return pd.Series({c: np.nan for c in TEAM_STYLE_COLS if c in team_att.columns})
    row = sub.sort_values("date").iloc[-1]
    return row[[c for c in TEAM_STYLE_COLS if c in team_att.columns]]


def _build_L2_features(home_style: pd.Series, away_style: pd.Series) -> dict:
    """Team tactical style: buildUpPlay, chanceCreation, defence, diffs."""
    out = {}
    for c in TEAM_STYLE_COLS:
        if c in home_style.index and c in away_style.index:
            h, a = home_style.get(c, np.nan), away_style.get(c, np.nan)
            out[f"L2_home_{c}"] = h
            out[f"L2_away_{c}"] = a
            if np.isfinite(h) and np.isfinite(a):
                out[f"L2_diff_{c}"] = h - a

    h_pass = home_style.get("chanceCreationPassing", np.nan)
    a_press = away_style.get("defencePressure", np.nan)
    if np.isfinite(h_pass) and np.isfinite(a_press):
        out["L2_home_creation_vs_away_pressure"] = h_pass - a_press
    return out


def _compute_rolling_and_elo(matches_sorted: pd.DataFrame) -> dict:
    """Compute L3 features per match. Uses only past matches (date < row date) to avoid leakage."""
    K = 20  # Elo K-factor
    elo = defaultdict(lambda: 1500.0)
    team_history = defaultdict(list)  # (date, goal_diff, win, is_home) per team

    features = {}
    for _, row in tqdm(matches_sorted.iterrows(), total=len(matches_sorted), desc="L3 rolling"):
        date = pd.to_datetime(row["date"])
        match_id = int(row["match_api_id"])
        hid, aid = int(row["home_team_api_id"]), int(row["away_team_api_id"])
        hg, ag = int(row["home_team_goal"]), int(row["away_team_goal"])
        goal_diff = hg - ag
        home_win = 1 if hg > ag else 0

        def last5(team_id):
            """Win rate, avg goal diff, home win rate over last 5 games (before this match)."""
            hist = [x for x in team_history[team_id] if x[0] < date][-5:]
            if not hist:
                return np.nan, np.nan, np.nan
            wins = sum(x[2] for x in hist)
            gd = sum(x[1] for x in hist)
            home_wins = sum(x[2] for x in hist if x[3])
            home_games = sum(1 for x in hist if x[3])
            home_win_rate = home_wins / home_games if home_games > 0 else np.nan
            return wins / len(hist), gd / len(hist), home_win_rate

        h_win_rate, h_gd, h_home_win = last5(hid)
        a_win_rate, a_gd, a_home_win = last5(aid)

        def rest_days(team_id):
            """Days since team's last match (before this match)."""
            hist = [x[0] for x in team_history[team_id] if x[0] < date]
            if not hist:
                return np.nan
            return (date - hist[-1]).days

        h_rest = rest_days(hid)
        a_rest = rest_days(aid)

        # Elo update: expected prob from ratings, actual 1/0.5/0 for win/draw/loss
        eh, ea = elo[hid], elo[aid]
        qh, qa = 10 ** (eh / 400), 10 ** (ea / 400)
        exp_h = qh / (qh + qa)
        actual = 1.0 if home_win else (0.5 if hg == ag else 0.0)
        elo[hid] = eh + K * (actual - exp_h)
        elo[aid] = ea + K * ((1 - actual) - (1 - exp_h))

        features[match_id] = {
            "L3_home_win_rate_5": h_win_rate,
            "L3_away_win_rate_5": a_win_rate,
            "L3_home_gd_5": h_gd,
            "L3_away_gd_5": a_gd,
            "L3_home_home_win_rate_5": h_home_win,
            "L3_away_home_win_rate_5": a_home_win,
            "L3_home_rest_days": h_rest,
            "L3_away_rest_days": a_rest,
            "L3_home_elo": eh,
            "L3_away_elo": ea,
            "L3_elo_diff": eh - ea,
        }

        away_win = 1 if ag > hg else 0
        team_history[hid].append((date, goal_diff, home_win, True))
        team_history[aid].append((date, -goal_diff, away_win, False))

    return features


def _implied_prob(odds: float) -> float:
    if pd.isna(odds) or odds <= 0:
        return np.nan
    return 1.0 / odds


def build_match_dataset(database_path: str | Path, output_path: str | Path, min_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Load DB, build L1/L2/L3 per match, save .npz or .parquet."""
    database_path = Path(database_path)
    output_path = Path(output_path)
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    con = sqlite3.connect(str(database_path))
    matches = pd.read_sql_query("SELECT * FROM Match", con)
    player_att = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
    team_att = pd.read_sql_query("SELECT * FROM Team_Attributes", con)
    con.close()

    player_att["date"] = pd.to_datetime(player_att["date"])
    team_att["date"] = pd.to_datetime(team_att["date"])
    matches["date"] = pd.to_datetime(matches["date"])

    # Require starting 11 for both sides; filter by min_date for valid L2
    req = [f"home_player_{i}" for i in range(1, 12)] + [f"away_player_{i}" for i in range(1, 12)]
    matches = matches.dropna(subset=[c for c in req if c in matches.columns])
    if min_date is not None:
        min_date = pd.to_datetime(min_date)
        matches = matches[matches["date"] >= min_date].reset_index(drop=True)
    matches = matches.sort_values("date").reset_index(drop=True)
    L3_map = _compute_rolling_and_elo(matches)  # precompute all L3 (rolling/Elo) in one pass

    rows = []
    for i, row in tqdm(matches.iterrows(), total=len(matches), desc="Building features"):
        date = row["date"]
        hid, aid = row["home_team_api_id"], row["away_team_api_id"]
        home_players = [row[f"home_player_{j}"] for j in range(1, 12)]
        away_players = [row[f"away_player_{j}"] for j in range(1, 12)]

        hp = _get_player_attr_at_date(player_att, home_players, date)
        ap = _get_player_attr_at_date(player_att, away_players, date)
        L1 = _build_L1_features(hp, ap)
        home_style = _get_team_attr_at_date(team_att, hid, date)
        away_style = _get_team_attr_at_date(team_att, aid, date)
        L2 = _build_L2_features(home_style, away_style)
        match_id = int(row["match_api_id"])
        L3 = L3_map.get(match_id, {})

        rec = {
            "match_api_id": row["match_api_id"],
            "date": date,
            "home_team_api_id": hid,
            "away_team_api_id": aid,
            "home_win": 1 if row["home_team_goal"] > row["away_team_goal"] else 0,
            "B365H": row.get("B365H", np.nan),
            "B365D": row.get("B365D", np.nan),
            "B365A": row.get("B365A", np.nan),
            **L1,
            **L2,
            **L3,
        }
        rows.append(rec)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".npz"):
        # Save X, y, feature_names, plus metadata for reproducibility
        meta_cols = ["match_api_id", "date", "home_team_api_id", "away_team_api_id"]
        feature_cols = [c for c in df.columns if c not in meta_cols and c != "home_win"]
        X = df[feature_cols].values.astype(np.float64)
        y = df["home_win"].values.astype(np.int32)
        np.savez(
            output_path,
            X=X,
            y=y,
            feature_names=np.array(feature_cols, dtype=object),
            date=df["date"].values,
            match_api_id=df["match_api_id"].values,
            home_team_api_id=df["home_team_api_id"].values,
            away_team_api_id=df["away_team_api_id"].values,
            allow_pickle=True,
        )
    else:
        df.to_parquet(output_path, index=False)
    return df


def get_feature_groups(df: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return L1, L2, L3, odds column lists for ablation. If df given, filter to existing cols."""
    L1 = [
        "L1_home_overall_mean", "L1_away_overall_mean",
        "L1_home_attack_index", "L1_away_attack_index",
        "L1_home_defense_index", "L1_away_defense_index",
        "L1_home_passing_index", "L1_away_passing_index",
        "L1_home_pace_index", "L1_away_pace_index",
        "L1_home_physical_index", "L1_away_physical_index",
        "L1_home_gk_index", "L1_away_gk_index",
        "L1_diff_overall", "L1_diff_attack",
        "L1_attack_home_vs_defense_away", "L1_gk_home_vs_shooting_away",
    ]
    L2 = [
        "L2_home_buildUpPlaySpeed", "L2_away_buildUpPlaySpeed",
        "L2_home_buildUpPlayPassing", "L2_away_buildUpPlayPassing",
        "L2_home_chanceCreationPassing", "L2_away_chanceCreationPassing",
        "L2_home_chanceCreationShooting", "L2_away_chanceCreationShooting",
        "L2_home_defencePressure", "L2_away_defencePressure",
        "L2_home_defenceAggression", "L2_away_defenceAggression",
        "L2_diff_buildUpPlaySpeed", "L2_diff_buildUpPlayPassing",
        "L2_diff_chanceCreationPassing", "L2_diff_chanceCreationShooting",
        "L2_diff_defencePressure", "L2_diff_defenceAggression",
        "L2_home_creation_vs_away_pressure",
    ]
    L3 = [
        "L3_home_win_rate_5", "L3_away_win_rate_5",
        "L3_home_gd_5", "L3_away_gd_5",
        "L3_home_home_win_rate_5", "L3_away_home_win_rate_5",
        "L3_home_rest_days", "L3_away_rest_days",
        "L3_home_elo", "L3_away_elo", "L3_elo_diff",
    ]
    odds = ["B365H", "B365D", "B365A"]
    out = {"L1": L1, "L2": L2, "L3": L3, "odds": odds}
    if df is not None:
        out = {k: [c for c in v if c in df.columns] for k, v in out.items()}
    return out


def load_match_dataset(path: str | Path) -> pd.DataFrame:
    """Load .npz (our format) or .parquet. Returns DataFrame with feature cols + home_win + metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if str(path).endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "feature_names" not in data:
            raise ValueError(
                "This .npz is in old format. Re-run: python main.py process"
            )
        feature_names = list(data["feature_names"])
        df = pd.DataFrame(data["X"], columns=feature_names)
        df["home_win"] = data["y"]
        df["date"] = pd.to_datetime(data["date"])
        df["match_api_id"] = data["match_api_id"]
        df["home_team_api_id"] = data["home_team_api_id"]
        df["away_team_api_id"] = data["away_team_api_id"]
        return df
    return pd.read_parquet(path)
