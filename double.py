"""
Two-Stage Model for NCAA Final Four Analytics Challenge

Predicts tournament seeds (0 = did not make tournament, 1-68 = seed) using:
  Stage 1: Classification — which teams make the tournament (at-large selection).
  Stage 2: Regression — seed assignment (1-68) for the predicted field.

Evaluation metric: RMSE on predicted seeds.
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATS_FILES = {
    2021: 'data/NCAA_Statistics_2020-2021.csv',
    2022: 'data/NCAA_Statistics_2021-2022.csv',
    2023: 'data/NCAA_Statistics_2022-2023.csv',
    2024: 'data/NCAA_Statistics_2023-2024.csv',
    2025: 'data/NCAA_Statistics_2024-2025.csv',
}
RANKINGS_FILE = 'data/ncaa_rankings_2021_2025_fixed.csv'
SUBMISSION_PATH = '2025_submission_two_stage.csv'

FEATS_STAGE1 = [
    'NETRank', 'PrevNET', 'NETSOS', 'Win_Pct', 'Quadrant1_W', 'Q1_Win_Pct',
    'Quality_Wins', 'Bad_Losses', 'Conf_ID',
]
FEATS_STAGE2 = FEATS_STAGE1 + ['Is_AQ']

RANKINGS_NAME_MAPPING = {
    'USC': 'Southern California',
    'Eastern Washington': 'Eastern Wash.',
    'Western Kentucky': 'Western Ky.',
    'Northern Kentucky': 'Northern Ky.',
    'Appalachian St.': 'App State',
    'Texas A&M–Corpus Christi': 'A&M-Corpus Christi',
    'Charleston': 'Col. of Charleston',
    'Southeast Missouri St.': 'Southeast Mo. St.',
    'Florida Atlantic': 'Fla. Atlantic',
    'Fairleigh Dickinson': 'FDU',
    'Grambling St.': 'Grambling',
    'UNC Wilmington': 'UNCW',
    'SIU Edwardsville': 'SIUE',
    "Saint Mary's": "Saint Mary's (CA)",
    "St. John's": "St. John's (NY)",
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load season stats and rankings from CSV files.
    Adds Season column and concatenates stats across years.
    Returns (stats_df, rankings_df).
    """
    stats_dfs = []
    for year, filepath in STATS_FILES.items():
        df = pd.read_csv(filepath)
        df['Season'] = year
        stats_dfs.append(df)
    stats = pd.concat(stats_dfs, ignore_index=True)

    rankings = pd.read_csv(RANKINGS_FILE)
    return stats, rankings


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------

def _parse_wl(value):
    """
    Parse W-L style columns (e.g. '31-3' or Excel date bug '8-Sep').
    Returns (wins, losses) as integers.
    """
    x = str(value)
    months = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    }
    if '-' in x:
        parts = x.split('-')
        for m, num in months.items():
            if parts[0] == m:
                return num, int(parts[1])  # e.g. Sep-8 -> 9, 8
            if parts[1] == m:
                return int(parts[0]), num   # e.g. 8-Sep -> 8, 9
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
    return 0, 0


def _normalize_rankings_name(name):
    """Normalize school names from rankings file to match stats team names."""
    name = str(name).strip()
    name = name.replace(' State', ' St.')
    if name == 'NC St.':
        return 'NC State'
    return RANKINGS_NAME_MAPPING.get(name, name)


def clean_data(stats, rankings):
    """
    Standardize column names, clean team names, merge stats with rankings,
    and create target variables.
    Returns merged DataFrame with targets and cleaned columns.
    """
    # Standardize quadrant column names (Q1 -> Quadrant1, etc.)
    stats = stats.rename(columns={
        'Q1': 'Quadrant1', 'Q2': 'Quadrant2',
        'Q3': 'Quadrant3', 'Q4': 'Quadrant4',
    })

    # Clean team names: remove (seed numbers) and (AQ)
    stats['Team_Clean'] = stats['Team'].astype(str).apply(
        lambda x: re.sub(r'\s*\(\d+\)|\s*\(AQ\)', '', x, flags=re.IGNORECASE).strip()
    )

    # Normalize ranking school names for merge
    rankings = rankings.copy()
    rankings['School_Clean'] = rankings['School'].apply(_normalize_rankings_name)

    # Merge on clean team/school name and season
    df = pd.merge(
        stats, rankings,
        left_on=['Team_Clean', 'Season'],
        right_on=['School_Clean', 'Season'],
        how='left',
    )

    # Target variables
    df['Overall Seed'] = df['Overall Seed'].fillna(0)
    df['Made_Tournament'] = (df['Overall Seed'] > 0).astype(int)
    df['Is_AQ'] = (df['Berth Type'] == 'Automatic').astype(int)

    # Targeted renames for consistency (avoid breaking 'Overall Seed')
    df = df.rename(columns={
        'Conf.Record': 'ConfRecord',
        'Non-ConferenceRecord': 'NonConfRecord',
        'NET Rank': 'NETRank',
    })

    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df):
    """
    Parse W-L columns, compute derived features, encode conference.
    Modifies df in place and returns it.
    """
    wl_cols = ['WL', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']

    for col in wl_cols:
        if col in df.columns:
            parsed = df[col].apply(_parse_wl)
            df[f'{col}_W'] = parsed.str[0]
            df[f'{col}_L'] = parsed.str[1]
        

    df['Win_Pct'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
    df['Q1_Win_Pct'] = (
        df['Quadrant1_W'] / (df['Quadrant1_W'] + df['Quadrant1_L'] + 1e-6)
    )
    df['Quality_Wins'] = df['Quadrant1_W'] + df['Quadrant2_W']
    df['Bad_Losses'] = df['Quadrant3_L'] + df['Quadrant4_L']

    le = LabelEncoder()
    df['Conf_ID'] = le.fit_transform(df['Conference'].astype(str))

    # Fill missing feature values with 0
    df[FEATS_STAGE2] = df[FEATS_STAGE2].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Stage 1: Tournament Selection
# ---------------------------------------------------------------------------

def train_selection_model(train_data):
    """
    Train GradientBoostingClassifier for at-large selection.
    Trains only on non-AQ teams (Is_AQ == 0).
    Returns fitted classifier.
    """
    pool = train_data[train_data['Is_AQ'] == 0]
    clf = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        random_state=42,
    )
    clf.fit(pool[FEATS_STAGE1], pool['Made_Tournament'])
    return clf


def predict_field(clf, test_data):
    """
    Using the Stage 1 classifier, select at-large teams and combine with AQs
    to form the predicted field of 68.
    Returns DataFrame of 68 teams (predicted field).
    """
    known_aqs = test_data[test_data['Is_AQ'] == 1]
    n_aqs = len(known_aqs)
    n_spots_remaining = 68 - n_aqs

    non_aq = test_data[test_data['Is_AQ'] == 0].copy()
    non_aq['Selection_Prob'] = clf.predict_proba(non_aq[FEATS_STAGE1])[:, 1]

    predicted_at_larges = non_aq.nlargest(n_spots_remaining, 'Selection_Prob')
    predicted_field = pd.concat([known_aqs, predicted_at_larges], ignore_index=True)

    return predicted_field


# ---------------------------------------------------------------------------
# Stage 2: Tournament Seeding
# ---------------------------------------------------------------------------

def train_seeding_model(train_data):
    """
    Train GradientBoostingRegressor for seeding.
    Trains only on teams that made the tournament (Made_Tournament == 1).
    Returns fitted regressor.
    """
    pool = train_data[train_data['Made_Tournament'] == 1]
    reg = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        random_state=42,
    )
    reg.fit(pool[FEATS_STAGE2], pool['Overall Seed'])
    return reg


def assign_seeds(reg, predicted_field):
    """
    Predict Seed_Score for the predicted field and convert to seeds 1-68.
    Modifies predicted_field in place; adds Predicted_Seed column.
    Returns predicted_field sorted by seed.
    """
    predicted_field = predicted_field.copy()
    predicted_field['Seed_Score'] = reg.predict(predicted_field[FEATS_STAGE2])
    predicted_field = predicted_field.sort_values('Seed_Score').reset_index(drop=True)
    predicted_field['Predicted_Seed'] = range(1, 69)
    return predicted_field


# ---------------------------------------------------------------------------
# Submission & Validation
# ---------------------------------------------------------------------------

def generate_submission(test_data, predicted_field, output_path=SUBMISSION_PATH):
    """
    Build submission: RecordID = Season-TeamNameWithoutSpaces, Overall Seed.
    Teams in predicted field get seeds 1-68; all others get 0.
    Saves CSV and returns the submission DataFrame.
    """
    merge_df = test_data[['Season', 'Team']].merge(
        predicted_field[['Team', 'Predicted_Seed']],
        on='Team',
        how='left',
    )
    merge_df['Predicted_Seed'] = merge_df['Predicted_Seed'].fillna(0).astype(int)
    merge_df['RecordID'] = (
        merge_df['Season'].astype(str) + '-' + merge_df['Team'].str.replace(' ', '')
    )
    submission = merge_df[['RecordID', 'Predicted_Seed']].rename(
        columns={'Predicted_Seed': 'Overall Seed'}
    )
    submission.to_csv(output_path, index=False)
    return submission


def print_feature_importance(clf, reg, stage1_feats=FEATS_STAGE1, stage2_feats=FEATS_STAGE2):
    """Print feature importance for Stage 1 and Stage 2 models."""
    imp1 = pd.DataFrame({
        'Feature': stage1_feats,
        'Importance': clf.feature_importances_,
    }).sort_values('Importance', ascending=False)
    imp1['Importance (%)'] = 100 * imp1['Importance'] / imp1['Importance'].sum()
    print("\n--- Stage 1 Feature Importance (Tournament Selection) ---")
    print(imp1.to_string(index=False))

    imp2 = pd.DataFrame({
        'Feature': stage2_feats,
        'Importance': reg.feature_importances_,
    }).sort_values('Importance', ascending=False)
    imp2['Importance (%)'] = 100 * imp2['Importance'] / imp2['Importance'].sum()
    print("\n--- Stage 2 Feature Importance (Seeding) ---")
    print(imp2.to_string(index=False))


def evaluate_2025(test_data, predicted_field):
    """
    Compute and print validation metrics for 2025:
    - Number of predicted tournament teams
    - Selection accuracy (at-large)
    - MAE and RMSE on seeds for teams that made the tournament
    """
    actual_field = test_data[test_data['Made_Tournament'] == 1]
    n_predicted = len(predicted_field)
    print(f"\nNumber of predicted tournament teams: {n_predicted}")

    # Selection: compare predicted at-large vs actual at-large
    actual_at_large = test_data[
        (test_data['Made_Tournament'] == 1) & (test_data['Is_AQ'] == 0)
    ]
    predicted_at_large_teams = set(
        predicted_field[predicted_field['Is_AQ'] == 0]['Team']
    )
    actual_at_large_teams = set(actual_at_large['Team'])
    correct_at_large = len(predicted_at_large_teams & actual_at_large_teams)
    n_at_large = len(actual_at_large_teams)
    selection_accuracy = correct_at_large / n_at_large if n_at_large else 0.0
    print(f"Selection accuracy (at-large): {correct_at_large}/{n_at_large} = {selection_accuracy:.1%}")

    # Seeding: MAE and RMSE on teams that made the tournament
    eval_df = predicted_field.merge(
        actual_field[['Team', 'Overall Seed']],
        on='Team',
        how='inner',
        suffixes=('', '_Actual'),
    )
    if len(eval_df) == 0:
        print("No overlapping teams for seeding evaluation.")
        return
    eval_df['Overall Seed_Actual'] = eval_df['Overall Seed_Actual'].astype(float)
    mae = mean_absolute_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
    rmse = root_mean_squared_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
    print(f"MAE (seeding): {mae:.2f} seed positions")
    print(f"RMSE (seeding): {rmse:.2f} seed positions")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full two-stage pipeline: load -> clean -> features -> train -> predict -> submit."""
    print("Loading data...")
    stats, rankings = load_data()

    print("Cleaning and merging...")
    df = clean_data(stats, rankings)
    df = engineer_features(df)

    train_data = df[df['Season'] < 2025].copy()
    test_data = df[df['Season'] == 2025].copy()

    # Stage 1: Tournament selection
    print("\nTraining Stage 1: Tournament Selection (Classification)...")
    clf = train_selection_model(train_data)
    predicted_field = predict_field(clf, test_data)

    # Stage 2: Seeding
    print("Training Stage 2: Tournament Seeding (Regression)...")
    reg = train_seeding_model(train_data)
    predicted_field = assign_seeds(reg, predicted_field)

    # Output
    print_feature_importance(clf, reg)
    generate_submission(test_data, predicted_field)
    print(f"\nSubmission saved to {SUBMISSION_PATH}")

    # Validation metrics for 2025
    print("\n" + "=" * 60)
    print("2025 VALIDATION METRICS")
    print("=" * 60)
    evaluate_2025(test_data, predicted_field)
    print("=" * 60)


if __name__ == '__main__':
    main()
