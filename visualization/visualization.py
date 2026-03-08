"""Plots: data overview, match/league stats, feature impact, non-linear (SVD/t-SNE/KPCA)."""
from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "inputs" / "raw" / "database.sqlite"
DATA_PATH = ROOT / "outputs" / "clean" / "processed_dataset.npz"

TOP_LEAGUES = [
    "England Premier League",
    "France Ligue 1",
    "Germany 1. Bundesliga",
    "Italy Serie A",
    "Spain LIGA BBVA",
]


def load_match_league_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT Match.id, League.name AS league_name, season,
               home_team_goal, away_team_goal
        FROM Match JOIN League ON League.id = Match.league_id;
    """
    df = pd.read_sql(query, conn)
    df["total_goals"] = df["home_team_goal"] + df["away_team_goal"]
    conn.close()
    return df


def load_player_attributes(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    df = pd.read_sql(
        "SELECT finishing, shot_power, overall_rating, acceleration FROM Player_Attributes",
        conn,
    )
    conn.close()
    return df


def load_buildup_winrate_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    matches = pd.read_sql(
        "SELECT home_team_api_id, away_team_api_id, home_team_goal, away_team_goal FROM Match",
        conn,
    )

    def get_winner(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return row["home_team_api_id"]
        if row["away_team_goal"] > row["home_team_goal"]:
            return row["away_team_api_id"]
        return None

    matches["winner"] = matches.apply(get_winner, axis=1)
    home_cnt = matches.groupby("home_team_api_id").size()
    away_cnt = matches.groupby("away_team_api_id").size()
    total_matches = home_cnt.add(away_cnt, fill_value=0)
    wins = matches.dropna(subset=["winner"]).groupby("winner").size()
    win_rate = (wins / total_matches.reindex(wins.index).fillna(0)).fillna(0)

    team_attr = pd.read_sql("SELECT * FROM Team_Attributes", conn)
    team_attr_avg = team_attr.groupby("team_api_id").mean(numeric_only=True).reset_index()
    analysis_df = team_attr_avg.merge(
        win_rate.rename("win_rate").reset_index(),
        left_on="team_api_id",
        right_on="winner",
        how="inner",
    )
    conn.close()
    return analysis_df


def _extract_possession(xml_data) -> tuple[Optional[float], Optional[float]]:
    try:
        if not xml_data:
            return None, None
        tree = ET.fromstring(xml_data)
        vals = tree.findall(".//value")
        if len(vals) >= 2:
            h = float(vals[-2].text) if vals[-2].text else None
            a = float(vals[-1].text) if vals[-1].text else None
            return h, a
    except Exception:
        pass
    return None, None


def load_possession_win_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT home_team_goal, away_team_goal, possession
        FROM Match WHERE possession IS NOT NULL
    """
    df_raw = pd.read_sql(query, conn)
    conn.close()

    rows = []
    for _, row in df_raw.iterrows():
        h_pos, a_pos = _extract_possession(row["possession"])
        if h_pos is not None:
            if row["home_team_goal"] > row["away_team_goal"]:
                win_score = 1.0
            elif row["home_team_goal"] < row["away_team_goal"]:
                win_score = 0.0
            else:
                win_score = 0.5
            rows.append({"possession": h_pos, "win_score": win_score})
    return pd.DataFrame(rows)


def _count_xml_occurrences(xml_string: str, tag_name: str = "value") -> int:
    try:
        if not xml_string:
            return 0
        tree = ET.fromstring(xml_string)
        return len(tree.findall(f".//{tag_name}"))
    except Exception:
        return 0


def load_aggression_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT home_team_goal, away_team_goal, goal, foulcommit, card
        FROM Match WHERE foulcommit IS NOT NULL
    """
    df_raw = pd.read_sql(query, conn)
    conn.close()

    stats = []
    for _, row in df_raw.iterrows():
        fouls = _count_xml_occurrences(row["foulcommit"], "value")
        cards = _count_xml_occurrences(row["card"], "value")
        goals = _count_xml_occurrences(row["goal"], "value") if row["goal"] else 0
        stats.append({"fouls": fouls, "cards": cards, "goals": goals})
    return pd.DataFrame(stats)


def load_aggression_eda_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT home_team_goal, away_team_goal, foulcommit, card
        FROM Match WHERE foulcommit IS NOT NULL
    """
    df_raw = pd.read_sql(query, conn)
    conn.close()

    rows = []
    for _, row in df_raw.iterrows():
        h_goal, a_goal = row["home_team_goal"], row["away_team_goal"]
        outcome = 3 if h_goal > a_goal else (1 if h_goal == a_goal else 0)
        rows.append({
            "fouls": _count_xml_occurrences(row["foulcommit"], "value"),
            "cards": _count_xml_occurrences(row["card"], "value"),
            "total_goals": h_goal + a_goal,
            "match_outcome": outcome,
        })
    return pd.DataFrame(rows)


def load_league_aggression_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT l.name AS league_name, m.home_team_goal, m.away_team_goal, m.foulcommit, m.card
        FROM Match m JOIN League l ON l.id = m.league_id
        WHERE m.foulcommit IS NOT NULL
        AND l.name IN ('England Premier League', 'France Ligue 1', 'Germany 1. Bundesliga',
                       'Italy Serie A', 'Spain LIGA BBVA')
    """
    df_raw = pd.read_sql(query, conn)
    conn.close()

    rows = []
    for _, row in df_raw.iterrows():
        h_goal, a_goal = row["home_team_goal"], row["away_team_goal"]
        outcome = 3 if h_goal > a_goal else (1 if h_goal == a_goal else 0)
        rows.append({
            "league": row["league_name"],
            "fouls": _count_xml_occurrences(row["foulcommit"], "value"),
            "cards": _count_xml_occurrences(row["card"], "value"),
            "goals": h_goal + a_goal,
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def load_epl_aggression_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT m.home_team_goal, m.away_team_goal, m.foulcommit, m.card
        FROM Match m JOIN League l ON l.id = m.league_id
        WHERE m.foulcommit IS NOT NULL AND l.name = 'England Premier League'
    """
    df_raw = pd.read_sql(query, conn)
    conn.close()

    rows = []
    for _, row in df_raw.iterrows():
        h_goal, a_goal = row["home_team_goal"], row["away_team_goal"]
        outcome = 3 if h_goal > a_goal else (1 if h_goal == a_goal else 0)
        rows.append({
            "fouls": _count_xml_occurrences(row["foulcommit"], "value"),
            "cards": _count_xml_occurrences(row["card"], "value"),
            "total_goals": h_goal + a_goal,
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def _check_result(row) -> str:
    if row["home_team_goal"] > row["away_team_goal"]:
        return "Home Win"
    if row["away_team_goal"] > row["home_team_goal"]:
        return "Away Win"
    return "Draw"


def load_outcome_counts(db_path: Optional[Path] = None) -> pd.Series:
    conn = sqlite3.connect(db_path or DB_PATH)
    df = pd.read_sql("SELECT home_team_goal, away_team_goal FROM Match", conn)
    conn.close()
    df["result"] = df.apply(_check_result, axis=1)
    return df["result"].value_counts()


def load_league_outcome_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT League.name AS league_name, home_team_goal, away_team_goal
        FROM Match JOIN League ON League.id = Match.league_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["result"] = df.apply(_check_result, axis=1)
    return df


def load_aging_curve_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT p.birthday, pa.date AS rating_date, pa.overall_rating, pa.potential
        FROM Player p
        JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["birthday"] = pd.to_datetime(df["birthday"])
    df["rating_date"] = pd.to_datetime(df["rating_date"])
    df["age"] = df["rating_date"].dt.year - df["birthday"].dt.year
    df = df[(df["age"] >= 15) & (df["age"] <= 45)].dropna(subset=["overall_rating"])
    return df


def load_top_players(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT p.player_name, MAX(pa.overall_rating) AS max_rating
        FROM Player p
        JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
        GROUP BY p.player_name
        ORDER BY max_rating DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_improved_players(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT p.player_name,
               (MAX(pa.overall_rating) - MIN(pa.overall_rating)) AS improvement
        FROM Player p
        JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
        GROUP BY p.player_name
        HAVING COUNT(pa.id) > 5
        ORDER BY improvement DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_top_teams_home_wins(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    query = """
        SELECT Team.team_long_name, COUNT(Match.id) AS total_wins
        FROM Team
        JOIN Match ON Team.team_api_id = Match.home_team_api_id
        WHERE Match.home_team_goal > Match.away_team_goal
        GROUP BY Team.team_long_name
        ORDER BY total_wins DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def _get_formation(row) -> str:
    ys = [row.get(f"home_player_Y{i}") for i in range(2, 12)]
    ys = [y for y in ys if pd.notna(y)]
    if len(ys) < 11:
        return "Unknown"
    ys_sorted = sorted(ys)
    lines = [0, 0, 0]
    for y in ys_sorted:
        if y < 33:
            lines[0] += 1
        elif y < 66:
            lines[1] += 1
        else:
            lines[2] += 1
    return f"{lines[0]}-{lines[1]}-{lines[2]}"


def load_formation_buildup_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or DB_PATH)
    y_cols = ", ".join([f"home_player_Y{i}" for i in range(2, 12)])
    query = f"SELECT id, home_team_api_id, {y_cols} FROM Match"
    df_match = pd.read_sql(query, conn)
    df_match["formation"] = df_match.apply(_get_formation, axis=1)

    team_attr = pd.read_sql(
        "SELECT team_api_id, buildUpPlaySpeed AS rating FROM Team_Attributes",
        conn,
    )
    team_avg = team_attr.groupby("team_api_id")["rating"].mean().reset_index()
    df_plot = df_match.merge(
        team_avg,
        left_on="home_team_api_id",
        right_on="team_api_id",
        how="inner",
    )[["formation", "rating"]]
    top5 = df_plot["formation"].value_counts().head(5).index.tolist()
    df_plot = df_plot[df_plot["formation"].isin(top5)]
    conn.close()
    return df_plot


def load_data_overview(db_path: Optional[Path] = None) -> dict:
    conn = sqlite3.connect(db_path or DB_PATH)
    league = pd.read_sql_query("SELECT id, name, country_id FROM League;", conn)
    country = pd.read_sql_query("SELECT id, name FROM Country;", conn)
    match = pd.read_sql_query("""
        SELECT id, country_id, league_id, season, date,
               home_team_api_id, away_team_api_id,
               home_team_goal, away_team_goal
        FROM Match;
    """, conn)
    player = pd.read_sql_query("SELECT id, player_api_id, player_name, birthday FROM Player;", conn)
    player_attr = pd.read_sql_query("""
        SELECT player_api_id, date, overall_rating
        FROM Player_Attributes;
    """, conn)
    conn.close()

    match["date"] = pd.to_datetime(match["date"])
    player["birthday"] = pd.to_datetime(player["birthday"])
    player_attr["date"] = pd.to_datetime(player_attr["date"])

    match_league = match.merge(league, left_on="league_id", right_on="id", suffixes=("", "_league"))
    match_league = match_league.merge(country, left_on="country_id", right_on="id", suffixes=("", "_country"))

    league_counts = match_league.groupby("name")["id"].count().sort_values(ascending=False)
    season_counts = match.dropna(subset=["season"]).groupby("season")["id"].count().sort_index()
    match["year"] = match["date"].dt.year
    year_counts = match.dropna(subset=["year"]).groupby("year")["id"].count().sort_index()

    ref_date = match["date"].max()
    player_age = player.copy()
    player_age["age_years"] = (ref_date - player_age["birthday"]).dt.days / 365.25
    player_age = player_age[(player_age["age_years"] >= 10) & (player_age["age_years"] <= 60)]

    rating = player_attr.dropna(subset=["overall_rating"])
    rating = rating[(rating["overall_rating"] >= 1) & (rating["overall_rating"] <= 100)]

    return {
        "league_counts": league_counts,
        "season_counts": season_counts,
        "year_counts": year_counts,
        "player_age": player_age,
        "rating": rating,
    }


def load_feature_impact_df(data_path: Optional[Path] = None) -> pd.DataFrame:
    path = data_path or DATA_PATH
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    if "feature_names" in data.files:
        fn = data["feature_names"]
        feature_names = [
            f.decode("utf-8") if isinstance(f, (bytes, np.bytes_)) else str(f)
            for f in fn
        ]
    else:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y.astype(int)
    return df


def plot_total_goals_by_league(df: pd.DataFrame, ax=None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAS_SNS:
        sns.barplot(
            data=df, x="total_goals", y="league_name",
            estimator=sum, errorbar=None, palette="viridis", ax=ax,
        )
    else:
        agg = df.groupby("league_name")["total_goals"].sum().sort_values()
        agg.plot(kind="barh", ax=ax, color="steelblue")
    plt.title("Total Goals Scored by League (All Seasons)")
    plt.xlabel("Total Goals")
    plt.ylabel("League")
    plt.tight_layout()
    plt.show()


def plot_goals_over_seasons_by_league(
    df: pd.DataFrame,
    top_leagues: Optional[list] = None,
    ax=None,
) -> None:
    top_leagues = top_leagues or TOP_LEAGUES
    df_top = df[df["league_name"].isin(top_leagues)]
    agg = df_top.groupby(["season", "league_name"])["total_goals"].mean().reset_index()
    plt.figure(figsize=(12, 6))
    if HAS_SNS:
        sns.lineplot(data=agg, x="season", y="total_goals", hue="league_name", marker="o")
    else:
        for league in agg["league_name"].unique():
            sub = agg[agg["league_name"] == league]
            plt.plot(sub["season"], sub["total_goals"], marker="o", label=league)
        plt.legend()
    plt.title("Average Goals per Match Over Seasons")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_home_vs_away_goals(df: pd.DataFrame, ax=None) -> None:
    goals_summary = df[["home_team_goal", "away_team_goal"]].mean()
    plt.figure(figsize=(6, 6))
    goals_summary.plot(kind="bar", color=["blue", "red"])
    plt.title("Average Goals: Home vs Away")
    plt.ylabel("Average Goals")
    plt.xticks([0, 1], ["Home Goals", "Away Goals"], rotation=0)
    plt.tight_layout()
    plt.show()


def plot_finishing_vs_overall_rating(
    df: pd.DataFrame,
    sample_size: int = 500,
    ax=None,
) -> None:
    sample = df.sample(min(sample_size, len(df))) if len(df) > sample_size else df
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.regplot(
            data=sample, x="finishing", y="overall_rating",
            scatter_kws={"alpha": 0.3}, line_kws={"color": "red"},
        )
    else:
        plt.scatter(sample["finishing"], sample["overall_rating"], alpha=0.3)
        z = np.polyfit(sample["finishing"].dropna(), sample["overall_rating"].dropna(), 1)
        p = np.poly1d(z)
        x = np.linspace(sample["finishing"].min(), sample["finishing"].max(), 100)
        plt.plot(x, p(x), "r-")
    plt.title("Relationship: Finishing Attribute vs. Overall Player Rating")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_buildup_speed_vs_win_rate(analysis_df: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.regplot(
            data=analysis_df, x="buildUpPlaySpeed", y="win_rate",
            scatter_kws={"alpha": 0.4}, line_kws={"color": "green"},
        )
    else:
        plt.scatter(analysis_df["buildUpPlaySpeed"], analysis_df["win_rate"], alpha=0.4)
    plt.title("Team Build Up Play Speed vs. Win Rate")
    plt.xlabel("Build Up Play Speed (Attribute Score)")
    plt.ylabel("Win Rate (0.0 to 1.0)")
    plt.tight_layout()
    plt.show()


def plot_possession_vs_win_score(df_final: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(12, 7))
    if HAS_SNS:
        sns.regplot(
            data=df_final, x="possession", y="win_score",
            scatter_kws={"alpha": 0.5, "color": "blue"},
            line_kws={"color": "red", "linewidth": 3},
            x_bins=20,
        )
    else:
        plt.scatter(df_final["possession"], df_final["win_score"], alpha=0.5)
    plt.title("Does Possession Lead to Winning?", fontsize=16)
    plt.xlabel("Possession Percentage (%)", fontsize=12)
    plt.ylabel("Win Probability (0 = Loss, 0.5 = Draw, 1 = Win)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fouls_cards_goals_correlation(df_aggression: pd.DataFrame, ax=None) -> None:
    correlation = df_aggression.corr()
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    else:
        plt.imshow(correlation, aspect="auto", cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
        plt.yticks(range(len(correlation)), correlation.columns)
    plt.title("Correlation: Fouls, Cards, and Goals")
    plt.tight_layout()
    plt.show()


def plot_cards_distribution(df_aggression: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.histplot(df_aggression["cards"], bins=15, kde=True, color="gold")
    else:
        plt.hist(df_aggression["cards"], bins=15, color="gold", edgecolor="black", alpha=0.7)
    plt.title("Distribution of Yellow/Red Cards per Match")
    plt.xlabel("Number of Cards")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_match_outcome_donut(outcome_counts: pd.Series, ax=None) -> None:
    colors = ["#4CAF50", "#FFC107", "#F44336"]
    plt.figure(figsize=(8, 8))
    plt.pie(
        outcome_counts,
        labels=outcome_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[: len(outcome_counts)],
        pctdistance=0.85,
        explode=(0.05, 0, 0)[: len(outcome_counts)],
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("European Soccer: Home vs. Away Win Rates (All Seasons)", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_league_outcome_distribution(df_leagues: pd.DataFrame, ax=None) -> None:
    league_analysis = pd.crosstab(
        df_leagues["league_name"], df_leagues["result"], normalize="index"
    ) * 100
    colors = ["#4CAF50", "#FFC107", "#F44336"]
    cols = [c for c in ["Home Win", "Draw", "Away Win"] if c in league_analysis.columns]
    league_analysis = league_analysis[cols]
    plt.figure(figsize=(12, 8))
    league_analysis.plot(
        kind="barh", stacked=True, color=colors[: len(cols)], ax=plt.gca()
    )
    plt.title("Win/Loss/Draw Distribution by League")
    plt.xlabel("Percentage (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_aging_curve(df_age: pd.DataFrame, peak_age: int = 27, ax=None) -> None:
    agg = df_age.groupby("age")["overall_rating"].mean().reset_index()
    plt.figure(figsize=(12, 6))
    if HAS_SNS:
        sns.lineplot(
            data=agg, x="age", y="overall_rating",
            marker="o", color="royalblue", errorbar=None,
        )
    else:
        plt.plot(agg["age"], agg["overall_rating"], "o-", color="royalblue")
    plt.axvline(x=peak_age, color="red", linestyle="--", label=f"Average Peak (Age {peak_age})")
    plt.title("The Soccer Aging Curve: Age vs. Overall Rating", fontsize=16)
    plt.xlabel("Player Age", fontsize=12)
    plt.ylabel("Average Overall Rating", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rating_vs_potential_by_age(df_age: pd.DataFrame, ax=None) -> None:
    agg = df_age.groupby("age")[["overall_rating", "potential"]].mean().reset_index()
    plt.figure(figsize=(12, 6))
    if HAS_SNS:
        sns.lineplot(data=agg, x="age", y="overall_rating", label="Current Rating", color="blue")
        sns.lineplot(data=agg, x="age", y="potential", label="Potential Rating", color="orange")
    else:
        plt.plot(agg["age"], agg["overall_rating"], "o-", label="Current Rating", color="blue")
        plt.plot(agg["age"], agg["potential"], "o-", label="Potential Rating", color="orange")
    plt.title("Current Rating vs. Potential by Age")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_rated_players(top_players: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.barplot(data=top_players, x="max_rating", y="player_name", palette="bright")
    else:
        plt.barh(top_players["player_name"], top_players["max_rating"], color="steelblue")
    plt.xlim(85, 96)
    plt.title("Top 10 Highest Rated Players (All Time)", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_most_improved_players(improved_players: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.barplot(data=improved_players, x="improvement", y="player_name", palette="rocket")
    else:
        plt.barh(improved_players["player_name"], improved_players["improvement"], color="coral")
    plt.title("Top 10 Most Improved Players")
    plt.xlabel("Rating Increase (Points)")
    plt.tight_layout()
    plt.show()


def plot_top_teams_by_home_wins(top_teams: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(10, 6))
    if HAS_SNS:
        sns.barplot(data=top_teams, x="total_wins", y="team_long_name", palette="viridis")
    else:
        plt.barh(top_teams["team_long_name"], top_teams["total_wins"], color="steelblue")
    plt.title("Top 10 Teams by Total Home Wins")
    plt.tight_layout()
    plt.show()


def plot_formation_vs_buildup_speed(df_plot: pd.DataFrame, ax=None) -> None:
    plt.figure(figsize=(12, 6))
    if HAS_SNS:
        sns.boxplot(data=df_plot, x="formation", y="rating", palette="Set2")
    else:
        df_plot.boxplot(column="rating", by="formation", ax=plt.gca())
        plt.suptitle("")
    plt.title("Team Build-Up Speed Rating by Formation Type")
    plt.xlabel("Formation (Def-Mid-Fwd)")
    plt.ylabel("Average Team Build-Up Speed")
    plt.tight_layout()
    plt.show()


AGGRESSION_EDA_FEATURES = ["fouls", "cards", "total_goals", "match_outcome"]
LEAGUE_AGGRESSION_FEATURES = ["fouls", "cards", "goals", "outcome"]


def plot_aggression_outcome_pairplot(
    df_eda: pd.DataFrame,
    features: Optional[list] = None,
    ax=None,
) -> None:
    features = features or AGGRESSION_EDA_FEATURES
    cols = [c for c in features if c in df_eda.columns]
    if not cols:
        return
    if HAS_SNS:
        g = sns.pairplot(
            df_eda[cols], hue="match_outcome", palette="viridis", diag_kind="kde",
            height=2.5,
        )
        g.fig.suptitle(
            "EDA: Non-linear Relationships between Aggression and Outcome", y=1.02
        )
    else:
        plt.figure(figsize=(12, 10))
        pd.plotting.scatter_matrix(df_eda[cols], alpha=0.5, figsize=(12, 10))
        plt.suptitle(
            "EDA: Non-linear Relationships between Aggression and Outcome", y=1.02
        )
    plt.tight_layout()
    plt.show()


def plot_svd_explained_variance(
    X: np.ndarray,
    n_components: Optional[int] = None,
    ax=None,
) -> None:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    U, Sigma, VT = np.linalg.svd(x_scaled, full_matrices=False)
    explained_variance = (Sigma ** 2) / np.sum(Sigma ** 2)
    if n_components:
        explained_variance = explained_variance[:n_components]
    plt.figure(figsize=(8, 4))
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
        alpha=0.7,
        color="teal",
    )
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Components (SVD)")
    plt.title("SVD: Information Density per Component")
    plt.tight_layout()
    plt.show()


def plot_svd_2d_projection(
    X: np.ndarray,
    y: np.ndarray,
    ax=None,
) -> None:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    U, Sigma, VT = np.linalg.svd(x_scaled, full_matrices=False)
    projection = x_scaled @ VT[:2].T
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        projection[:, 0], projection[:, 1],
        c=y, cmap="coolwarm", alpha=0.6, s=10,
    )
    plt.colorbar(scatter, label="Match Outcome")
    plt.xlabel("First Singular Vector")
    plt.ylabel("Second Singular Vector")
    plt.title("SVD Projection: Mapping to 2D Space")
    plt.tight_layout()
    plt.show()


def plot_svd_explained_variance_df(
    df_eda: pd.DataFrame,
    features: Optional[list] = None,
    ax=None,
) -> None:
    features = features or AGGRESSION_EDA_FEATURES
    cols = [c for c in features if c in df_eda.columns]
    X = df_eda[cols].values
    plot_svd_explained_variance(X, ax=ax)


def plot_svd_2d_projection_df(
    df_eda: pd.DataFrame,
    features: Optional[list] = None,
    outcome_col: str = "match_outcome",
    ax=None,
) -> None:
    features = features or AGGRESSION_EDA_FEATURES
    cols = [c for c in features if c in df_eda.columns]
    X = df_eda[cols].values
    y = df_eda[outcome_col].values if outcome_col in df_eda.columns else np.zeros(len(df_eda))
    plot_svd_2d_projection(X, y, ax=ax)


def plot_league_correlation_comparison(
    df_leagues: pd.DataFrame,
    features: Optional[list] = None,
    ax=None,
) -> None:
    features = features or LEAGUE_AGGRESSION_FEATURES
    cols = [c for c in features if c in df_leagues.columns]
    leagues = df_leagues["league"].unique()
    n = min(5, len(leagues))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for i, league in enumerate(leagues[:n]):
        sub = df_leagues[df_leagues["league"] == league][cols]
        corr = sub.corr()
        if HAS_SNS:
            sns.heatmap(
                corr, annot=True, cmap="RdBu_r", center=0,
                ax=axes[i], cbar=(i == n - 1), fmt=".2f",
            )
        else:
            im = axes[i].imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            if i == n - 1:
                plt.colorbar(im, ax=axes[i])
            axes[i].set_xticks(range(len(cols)))
            axes[i].set_xticklabels(cols, rotation=45, ha="right")
            axes[i].set_yticks(range(len(cols)))
            axes[i].set_yticklabels(cols)
        axes[i].set_title(league.replace(" ", "\n"))
    plt.suptitle(
        "Comparison of Correlation: Fouls, Cards, Goals, and Outcome across Top 5 Leagues",
        y=1.05, fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_tsne_by_league(
    df_leagues: pd.DataFrame,
    features: Optional[list] = None,
    sample_per_league: int = 800,
    perplexity: int = 30,
    random_state: int = 42,
    ax=None,
) -> None:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    features = features or LEAGUE_AGGRESSION_FEATURES
    cols = [c for c in features if c in df_leagues.columns]
    leagues = df_leagues["league"].unique()
    n = min(6, len(leagues))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    for i, league in enumerate(leagues[:n]):
        sub = df_leagues[df_leagues["league"] == league]
        sub = sub.sample(min(sample_per_league, len(sub)), random_state=random_state)
        X = sub[cols].values
        X_scaled = StandardScaler().fit_transform(X)
        tsne = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=random_state, init="pca", learning_rate="auto",
        )
        X_emb = tsne.fit_transform(X_scaled)
        ax_i = axes_flat[i]
        scatter = ax_i.scatter(
            X_emb[:, 0], X_emb[:, 1],
            c=sub["outcome"].values, cmap="coolwarm", alpha=0.6, s=15,
        )
        ax_i.set_title(f"t-SNE: {league}")
        if i == 4:
            plt.colorbar(scatter, ax=ax_i, label="Outcome (3:Win, 1:Draw, 0:Loss)")
    for j in range(i + 1, 6):
        axes_flat[j].set_visible(False)
    plt.suptitle(
        "Non-linear Analysis (t-SNE) of Discipline & Success across Leagues",
        y=1.02, fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_tsne_vs_kpca_epl(
    df_epl: pd.DataFrame,
    features: Optional[list] = None,
    kpca_gamma: float = 15.0,
    kpca_alpha: float = 0.1,
    perplexity: int = 30,
    random_state: int = 42,
    ax=None,
) -> None:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler

    feat_cols = features or ["fouls", "cards", "total_goals", "outcome"]
    cols = [c for c in feat_cols if c in df_epl.columns]
    X = df_epl[cols].values
    X_scaled = StandardScaler().fit_transform(X)
    y = df_epl["outcome"].values

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=kpca_gamma, alpha=kpca_alpha)
    X_kpca = kpca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.6)
    ax1.set_title("EPL: Non-linear Analysis using t-SNE")
    ax1.set_xlabel("t-SNE component 1")
    ax1.set_ylabel("t-SNE component 2")

    scatter2 = ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap="coolwarm", alpha=0.6)
    ax2.set_title("EPL: Non-linear Analysis using Kernel PCA (RBF)")
    ax2.set_xlabel("KPCA component 1")
    ax2.set_ylabel("KPCA component 2")

    plt.colorbar(scatter2, ax=[ax1, ax2], label="Outcome (3:Win, 1:Draw, 0:Loss)")
    plt.show()


def plot_feature_correlation_with_outcome(
    df: pd.DataFrame,
    top_k: int = 25,
    ax=None,
) -> None:
    corr_to_y = (
        df.drop(columns=["y"])
        .corrwith(df["y"])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    top_corr = corr_to_y.head(top_k)
    plt.figure(figsize=(10, 6))
    top_corr.sort_values().plot(kind="barh")
    plt.title(f"Top {top_k} Features Correlated with Home Win (Pearson corr with y)")
    plt.xlabel("Correlation with y (0=not win, 1=win)")
    plt.tight_layout()
    plt.show()


def plot_feature_distribution_by_label(
    df: pd.DataFrame,
    feature_names: Optional[list] = None,
    top_k: int = 6,
    use_violin: bool = True,
    ax=None,
) -> None:
    corr_to_y = df.drop(columns=["y"]).corrwith(df["y"]).abs()
    strong = corr_to_y[corr_to_y >= 0.20]
    names = feature_names or list(strong.head(top_k).index)
    if not names:
        names = list(corr_to_y.head(top_k).index)
    for col in names:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        if HAS_SNS and use_violin:
            sns.violinplot(data=df, x="y", y=col, inner="quartile")
            plt.title(f"Distribution by Label (y) — {col}")
        else:
            df.boxplot(column=col, by="y")
            plt.title(f"Boxplot by Label (y) — {col}")
            plt.suptitle("")
        plt.xlabel("y (0=not win, 1=win)")
        plt.tight_layout()
        plt.show()


def plot_feature_feature_correlation_heatmap(
    df: pd.DataFrame,
    top_k: int = 15,
    ax=None,
) -> None:
    corr_to_y = (
        df.drop(columns=["y"])
        .corrwith(df["y"])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    subset_names = list(corr_to_y.head(top_k).index)
    corr_mat = df[subset_names].corr()
    plt.figure(figsize=(10, 8))
    if HAS_SNS:
        sns.heatmap(corr_mat, annot=False, cmap="coolwarm", center=0)
    else:
        plt.imshow(corr_mat, aspect="auto", cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(subset_names)), subset_names, rotation=90)
        plt.yticks(range(len(subset_names)), subset_names)
    plt.title("Feature-Feature Correlation Heatmap (subset)")
    plt.tight_layout()
    plt.show()


def plot_matches_per_league(league_counts: pd.Series, ax=None) -> None:
    plt.figure()
    league_counts.plot(kind="bar")
    plt.title("Number of Matches per League")
    plt.xlabel("League")
    plt.ylabel("Match Count")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.show()


def plot_matches_per_season(season_counts: pd.Series, ax=None) -> None:
    plt.figure()
    season_counts.plot(kind="line", marker="o")
    plt.title("Number of Matches per Season")
    plt.xlabel("Season")
    plt.ylabel("Match Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_matches_per_year(year_counts: pd.Series, ax=None) -> None:
    plt.figure()
    year_counts.plot(kind="line", marker="o")
    plt.title("Number of Matches per Year")
    plt.xlabel("Year")
    plt.ylabel("Match Count")
    plt.tight_layout()
    plt.show()


def plot_player_age_distribution(player_age: pd.DataFrame, bins: int = 30, ax=None) -> None:
    plt.figure()
    plt.hist(player_age["age_years"], bins=bins)
    plt.title("Player Age Distribution (at reference date)")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_overall_rating_distribution(rating: pd.DataFrame, bins: int = 30, ax=None) -> None:
    plt.figure()
    plt.hist(rating["overall_rating"], bins=bins)
    plt.title("Overall Rating Distribution (Player_Attributes)")
    plt.xlabel("Overall Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def run_all_visualizations(db_path: Optional[Path] = None, data_path: Optional[Path] = None,
                          skip_db_plots: bool = False, skip_feature_impact: bool = False) -> None:
    db = db_path or DB_PATH
    data = data_path or DATA_PATH

    if not skip_db_plots and db.exists():
        df = load_match_league_df(db)
        plot_total_goals_by_league(df)
        plot_goals_over_seasons_by_league(df)
        plot_home_vs_away_goals(df)

        df_attr = load_player_attributes(db)
        plot_finishing_vs_overall_rating(df_attr)

        analysis_df = load_buildup_winrate_df(db)
        plot_buildup_speed_vs_win_rate(analysis_df)

        df_poss = load_possession_win_df(db)
        plot_possession_vs_win_score(df_poss)

        df_agg = load_aggression_df(db)
        plot_fouls_cards_goals_correlation(df_agg)
        plot_cards_distribution(df_agg)

        # ECE143_Non-Linear_Visualization
        df_eda = load_aggression_eda_df(db)
        if len(df_eda) > 0:
            plot_aggression_outcome_pairplot(df_eda)
            plot_svd_explained_variance_df(df_eda)
            plot_svd_2d_projection_df(df_eda)
        df_leagues = load_league_aggression_df(db)
        if len(df_leagues) > 0:
            plot_league_correlation_comparison(df_leagues)
            plot_tsne_by_league(df_leagues)
        df_epl = load_epl_aggression_df(db)
        if len(df_epl) > 0:
            plot_tsne_vs_kpca_epl(df_epl)

        outcome = load_outcome_counts(db)
        plot_match_outcome_donut(outcome)

        df_league = load_league_outcome_df(db)
        plot_league_outcome_distribution(df_league)

        df_age = load_aging_curve_df(db)
        plot_aging_curve(df_age)
        plot_rating_vs_potential_by_age(df_age)

        top_players = load_top_players(db)
        plot_top_rated_players(top_players)

        improved = load_improved_players(db)
        plot_most_improved_players(improved)

        top_teams = load_top_teams_home_wins(db)
        plot_top_teams_by_home_wins(top_teams)

        df_form = load_formation_buildup_df(db)
        plot_formation_vs_buildup_speed(df_form)

        overview = load_data_overview(db)
        plot_matches_per_league(overview["league_counts"])
        plot_matches_per_season(overview["season_counts"])
        plot_matches_per_year(overview["year_counts"])
        plot_player_age_distribution(overview["player_age"])
        plot_overall_rating_distribution(overview["rating"])

    if not skip_feature_impact and data.exists():
        df_feat = load_feature_impact_df(data)
        plot_feature_correlation_with_outcome(df_feat)
        plot_feature_distribution_by_label(df_feat)
        plot_feature_feature_correlation_heatmap(df_feat)


if __name__ == "__main__":
    run_all_visualizations()
