import pandas as pd
import nflreadpy as nfl
import joblib
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Globals (populated on startup)
# --------------------------------------------------
pipeline      = None
X_test        = None
y_test        = None
feature_names = None
stats_df      = None
schedules_df  = None

MODEL_PATH = "nfl_predictor.pkl"

# --------------------------------------------------
# Training logic (mirrors your original script)
# --------------------------------------------------
def train_model():
    global pipeline, X_test, y_test, feature_names, stats_df, schedules_df

    print("Loading NFL data...")
    seasons = [2021, 2022, 2023]
    team_stats = nfl.load_team_stats(seasons).to_pandas()
    schedules  = nfl.load_schedules(seasons).to_pandas()
    schedules_df = schedules

    # Prepare schedules
    games = schedules[[
        'season', 'week', 'home_team', 'away_team',
        'home_score', 'away_score', 'spread_line',
        'home_moneyline', 'away_moneyline'
    ]].dropna(subset=['home_score', 'away_score'])
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

    # Prepare team stats
    stats = team_stats[[
        'season', 'week', 'team', 'passing_yards', 'rushing_yards',
        'attempts', 'carries', 'sacks_suffered', 'penalties',
        'passing_interceptions', 'rushing_fumbles_lost',
        'receiving_fumbles_lost', 'sack_fumbles_lost'
    ]].copy()
    stats = stats.sort_values(['team', 'season', 'week'])

    stats['turnovers'] = (
        stats['passing_interceptions']
        + stats['rushing_fumbles_lost']
        + stats['receiving_fumbles_lost']
        + stats['sack_fumbles_lost']
    )
    stats['yards_per_play'] = (
        (stats['passing_yards'] + stats['rushing_yards']) /
        (stats['attempts'] + stats['carries'])
    )

    rolling_cols = [
        'passing_yards', 'rushing_yards', 'attempts', 'carries',
        'sacks_suffered', 'penalties', 'turnovers', 'yards_per_play'
    ]
    for col in rolling_cols:
        stats[f'{col}_roll5'] = (
            stats.groupby('team')[col]
                 .rolling(5, min_periods=1)
                 .mean()
                 .reset_index(level=0, drop=True)
        )

    # Points rolling average
    points = schedules[['season','week','home_team','away_team','home_score','away_score']].dropna()
    home_pts = points[['season','week','home_team','home_score']].rename(columns={'home_team':'team','home_score':'points'})
    away_pts = points[['season','week','away_team','away_score']].rename(columns={'away_team':'team','away_score':'points'})
    points_long = pd.concat([home_pts, away_pts]).sort_values(['team','season','week'])
    points_long['points_roll5'] = (
        points_long.groupby('team')['points']
                   .rolling(5, min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
    )
    stats = stats.merge(
        points_long[['season','week','team','points_roll5']],
        on=['season','week','team'], how='left'
    )
    stats_df = stats

    # Merge home/away stats
    home = games.merge(stats, left_on=['season','week','home_team'], right_on=['season','week','team'], how='left')
    away = games.merge(stats, left_on=['season','week','away_team'], right_on=['season','week','team'], how='left')

    df = pd.DataFrame({
        'home_win':              games['home_win'].values,
        'vegas_spread':          games['spread_line'].values,
        'moneyline_diff':        (games['away_moneyline'] - games['home_moneyline']).values,
        'pass_yards_diff':       (home['passing_yards_roll5'] - away['passing_yards_roll5']).values,
        'rush_yards_diff':       (home['rushing_yards_roll5'] - away['rushing_yards_roll5']).values,
        'yards_per_play_diff':   (home['yards_per_play_roll5'] - away['yards_per_play_roll5']).values,
        'total_plays_diff':      ((home['attempts_roll5'] + home['carries_roll5']) - (away['attempts_roll5'] + away['carries_roll5'])).values,
        'sacks_suffered_diff':   (home['sacks_suffered_roll5'] - away['sacks_suffered_roll5']).values,
        'penalties_diff':        (home['penalties_roll5'] - away['penalties_roll5']).values,
        'turnovers_diff':        (home['turnovers_roll5'] - away['turnovers_roll5']).values,
        'points_per_game_diff':  (home['points_roll5'] - away['points_roll5']).values,
    }).dropna(subset=['vegas_spread', 'moneyline_diff'])

    X = df.drop(columns='home_win')
    y = df['home_win']
    feature_names = list(X.columns)

    X_train, X_test_local, y_train, y_test_local = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, C=0.5, penalty='l2'))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    pipeline = model
    X_test   = X_test_local
    y_test   = y_test_local

# --------------------------------------------------
# Load or train on startup
# --------------------------------------------------
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    # Still need to rebuild stats_df even if model is cached
    train_model()
else:
    train_model()

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.route("/api/metrics")
def metrics():
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return jsonify({
        "accuracy":      round(float(accuracy_score(y_test, y_pred)), 3),
        "roc_auc":       round(float(roc_auc_score(y_test, y_prob)), 3),
        "train_samples": len(y_test) * 4,
        "test_samples":  len(y_test),
        "seasons":       "2021–2023",
        "confusion_matrix": { "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn) }
    })


@app.route("/api/coefficients")
def coefficients():
    coefs = pipeline.named_steps['model'].coef_[0]
    result = [
        {
            "feature": f.replace("_", " ").replace("diff", "Diff").title(),
            "coef":    round(float(c), 3),
            "abs":     round(abs(float(c)), 3)
        }
        for f, c in zip(feature_names, coefs)
    ]
    result.sort(key=lambda x: x["abs"], reverse=True)
    return jsonify(result)


@app.route("/api/teams")
def teams():
    team_list = sorted(stats_df['team'].unique().tolist())
    return jsonify(team_list)


@app.route("/api/team-stats")
def team_stats():
    """Return rolling-5 stats for all teams for radar chart comparisons."""
    team = request.args.get("team")
    season = int(request.args.get("season", 2023))

    stat_cols = [
        'passing_yards_roll5', 'rushing_yards_roll5',
        'yards_per_play_roll5', 'sacks_suffered_roll5',
        'penalties_roll5', 'turnovers_roll5', 'points_roll5'
    ]

    mask = (stats_df['team'] == team) & (stats_df['season'] == season)
    row = stats_df[mask].sort_values('week').iloc[-1]

    return jsonify({
        "team":            team,
        "passing_yards":   round(float(row['passing_yards_roll5']), 1),
        "rushing_yards":   round(float(row['rushing_yards_roll5']), 1),
        "yards_per_play":  round(float(row['yards_per_play_roll5']), 2),
        "sacks_suffered":  round(float(row['sacks_suffered_roll5']), 1),
        "penalties":       round(float(row['penalties_roll5']), 1),
        "turnovers":       round(float(row['turnovers_roll5']), 2),
        "points":          round(float(row['points_roll5']), 1),
    })


@app.route("/api/home-win-rate")
def home_win_rate():
    """Return home win rate by week across all seasons."""
    if schedules_df is None:
        return jsonify([])
    df = schedules_df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    by_week = df.groupby('week')['home_win'].mean().reset_index()
    by_week.columns = ['week', 'rate']
    return jsonify([
        {"week": int(r['week']), "rate": round(float(r['rate']), 3)}
        for _, r in by_week.iterrows()
    ])


@app.route("/api/predict")
def predict():
    home     = request.args.get("home")
    away     = request.args.get("away")
    week     = int(request.args.get("week", 18))
    season   = int(request.args.get("season", 2023))
    spread   = float(request.args.get("spread", -3.0))
    home_ml  = float(request.args.get("home_ml", -150))
    away_ml  = float(request.args.get("away_ml", 130))

    if not home or not away:
        return jsonify({"error": "home and away params required"}), 400
    if home == away:
        return jsonify({"error": "home and away teams must differ"}), 400

    def get_row(team):
        mask = (
            (stats_df['team'] == team) &
            (stats_df['season'] == season) &
            (stats_df['week'] < week)
        )
        filtered = stats_df[mask].sort_values('week')
        if filtered.empty:
            raise ValueError(f"No stats found for {team} before week {week} in {season}")
        return filtered.iloc[-1]

    try:
        h = get_row(home)
        a = get_row(away)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    game = pd.DataFrame([{
        "vegas_spread":         spread,
        "moneyline_diff":       away_ml - home_ml,
        "pass_yards_diff":      h["passing_yards_roll5"]  - a["passing_yards_roll5"],
        "rush_yards_diff":      h["rushing_yards_roll5"]  - a["rushing_yards_roll5"],
        "yards_per_play_diff":  h["yards_per_play_roll5"] - a["yards_per_play_roll5"],
        "total_plays_diff":    (h["attempts_roll5"] + h["carries_roll5"]) -
                               (a["attempts_roll5"] + a["carries_roll5"]),
        "sacks_suffered_diff":  h["sacks_suffered_roll5"] - a["sacks_suffered_roll5"],
        "penalties_diff":       h["penalties_roll5"]      - a["penalties_roll5"],
        "turnovers_diff":       h["turnovers_roll5"]      - a["turnovers_roll5"],
        "points_per_game_diff": h["points_roll5"]         - a["points_roll5"],
    }])

    prob = pipeline.predict_proba(game)[0][1]
    return jsonify({
        "home_team":      home,
        "away_team":      away,
        "home_win_prob":  round(float(prob), 3),
        "away_win_prob":  round(1 - float(prob), 3),
        "home_stats":     {
            "points":       round(float(h["points_roll5"]), 1),
            "pass_yards":   round(float(h["passing_yards_roll5"]), 1),
            "rush_yards":   round(float(h["rushing_yards_roll5"]), 1),
            "yards_per_play": round(float(h["yards_per_play_roll5"]), 2),
            "turnovers":    round(float(h["turnovers_roll5"]), 2),
        },
        "away_stats":     {
            "points":       round(float(a["points_roll5"]), 1),
            "pass_yards":   round(float(a["passing_yards_roll5"]), 1),
            "rush_yards":   round(float(a["rushing_yards_roll5"]), 1),
            "yards_per_play": round(float(a["yards_per_play_roll5"]), 2),
            "turnovers":    round(float(a["turnovers_roll5"]), 2),
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
