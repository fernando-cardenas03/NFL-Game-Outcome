import pandas as pd
import nflreadpy as nfl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
print("Loading NFL team stats and schedules...")

seasons = [2021, 2022, 2023]

team_stats = nfl.load_team_stats(seasons).to_pandas()
schedules = nfl.load_schedules(seasons).to_pandas()


# --------------------------------------------------
# 2. Prepare schedules (home win label)
# --------------------------------------------------
games = schedules[[
    'season',
    'week',
    'home_team',
    'away_team',
    'home_score',
    'away_score',
    'spread_line',
    'home_moneyline',
    'away_moneyline'
]].dropna(subset=['home_score', 'away_score'])

games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# --------------------------------------------------
# 3. Prepare team stats
# --------------------------------------------------
stats = team_stats[[
    'season',
    'week',
    'team',
    'passing_yards',
    'rushing_yards',
    'attempts',
    'carries',
    'sacks_suffered',
    'penalties',
    'passing_interceptions',
    'rushing_fumbles_lost',
    'receiving_fumbles_lost',
    'sack_fumbles_lost'
]]

stats = stats.sort_values(['team', 'season', 'week'])

# --------------------------------------------------
# 4. Turnovers
# --------------------------------------------------
stats['turnovers'] = (
    stats['passing_interceptions']
    + stats['rushing_fumbles_lost']
    + stats['receiving_fumbles_lost']
    + stats['sack_fumbles_lost']
)

# --------------------------------------------------
# 5. Yards per play
# --------------------------------------------------
stats['yards_per_play'] = (
    (stats['passing_yards'] + stats['rushing_yards']) /
    (stats['attempts'] + stats['carries'])
)

# --------------------------------------------------
# 6. Rolling 5-game averages
# --------------------------------------------------
rolling_cols = [
    'passing_yards',
    'rushing_yards',
    'attempts',
    'carries',
    'sacks_suffered',
    'penalties',
    'turnovers',
    'yards_per_play'
]

for col in rolling_cols:
    stats[f'{col}_roll5'] = (
        stats.groupby('team')[col]
             .rolling(5, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
    )

# --------------------------------------------------
# 7. Points per game (rolling)
# --------------------------------------------------
points = schedules[[
    'season', 'week',
    'home_team', 'away_team',
    'home_score', 'away_score'
]].dropna()

home_points = points[['season', 'week', 'home_team', 'home_score']] \
    .rename(columns={'home_team': 'team', 'home_score': 'points'})

away_points = points[['season', 'week', 'away_team', 'away_score']] \
    .rename(columns={'away_team': 'team', 'away_score': 'points'})

points_long = pd.concat([home_points, away_points])
points_long = points_long.sort_values(['team', 'season', 'week'])

points_long['points_roll5'] = (
    points_long.groupby('team')['points']
               .rolling(5, min_periods=1)
               .mean()
               .reset_index(level=0, drop=True)
)

stats = stats.merge(
    points_long[['season', 'week', 'team', 'points_roll5']],
    on=['season', 'week', 'team'],
    how='left'
)

# --------------------------------------------------
# 8. Merge home & away stats
# --------------------------------------------------
home = games.merge(
    stats,
    left_on=['season', 'week', 'home_team'],
    right_on=['season', 'week', 'team'],
    how='left'
)

away = games.merge(
    stats,
    left_on=['season', 'week', 'away_team'],
    right_on=['season', 'week', 'team'],
    how='left'
)

# --------------------------------------------------
# 9. Feature differences
# --------------------------------------------------
df = pd.DataFrame({
    'home_win': games['home_win'],
    'vegas_spread': games['spread_line'],
    
    'moneyline_diff': 
        games['away_moneyline'] - games['home_moneyline'],

    'pass_yards_diff':
        home['passing_yards_roll5'] - away['passing_yards_roll5'],

    'rush_yards_diff':
        home['rushing_yards_roll5'] - away['rushing_yards_roll5'],

    'yards_per_play_diff':
        home['yards_per_play_roll5'] - away['yards_per_play_roll5'],

    'total_plays_diff':
        (home['attempts_roll5'] + home['carries_roll5']) -
        (away['attempts_roll5'] + away['carries_roll5']),

    'sacks_suffered_diff':
        home['sacks_suffered_roll5'] - away['sacks_suffered_roll5'],

    'penalties_diff':
        home['penalties_roll5'] - away['penalties_roll5'],

    'turnovers_diff':
        home['turnovers_roll5'] - away['turnovers_roll5'],

    'points_per_game_diff':
        home['points_roll5'] - away['points_roll5']
})

df = df.dropna(subset=[
    'vegas_spread',
    'moneyline_diff'
])

# --------------------------------------------------
# 10. Train-test split
# --------------------------------------------------
X = df.drop(columns='home_win')
y = df['home_win']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 11. Logistic Regression pipeline
# --------------------------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        C=0.5,
        penalty='l2'
    ))
])

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 12. Evaluation
# --------------------------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

# --------------------------------------------------
# 13. Feature coefficients
# --------------------------------------------------
coefs = pipeline.named_steps['model'].coef_[0]
features = X.columns

print("\nFeature Coefficients:")
for f, c in zip(features, coefs):
    print(f"{f}: {round(c, 3)}")

# --------------------------------------------------
# 14. Example prediction
# --------------------------------------------------
example = X_test.iloc[[0]]
prob = pipeline.predict_proba(example)[0][1]

print("\nExample Prediction:")
print("Predicted probability home team wins:", round(prob, 3))