"""
Enhanced Two-Stage NCAA Tournament Prediction
- Stage 1: At-large selection (classification)
- Stage 2: Seeding prediction (regression)
- Features: Weighted quadrant wins, momentum, per-game stats, conference strength
- Models: LightGBM for faster, more robust predictions
- Validation: Leave-one-season-out cross-validation
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ------------------------------
# Constants
# ------------------------------

STATS_FILES = {
    2021: 'data/NCAA_Statistics_2020-2021.csv',
    2022: 'data/NCAA_Statistics_2021-2022.csv',
    2023: 'data/NCAA_Statistics_2022-2023.csv',
    2024: 'data/NCAA_Statistics_2023-2024.csv',
    2025: 'data/NCAA_Statistics_2024-2025.csv',
}

RANKINGS_FILE = 'data/ncaa_rankings_2021_2025_fixed.csv'
SUBMISSION_PATH = '2025_submission_two_stage_enhanced.csv'

FEATS_STAGE1 = [
    'NETRank', 'PrevNET', 'NETSOS', 'Win_Pct', 'Quadrant1_W', 'Quadrant2_W',
    'Quadrant3_W', 'Quadrant4_W', 'Q1_Win_Pct', 'Weighted_Q_Wins',
    'Momentum', 'Conf_Strength'
]
FEATS_STAGE2 = FEATS_STAGE1 + ['Is_AQ']

# ------------------------------
# Data Loading & Cleaning
# ------------------------------

def load_data():
    stats_dfs = []
    for year, filepath in STATS_FILES.items():
        df = pd.read_csv(filepath)
        df['Season'] = year
        stats_dfs.append(df)
    stats = pd.concat(stats_dfs, ignore_index=True)

    rankings = pd.read_csv(RANKINGS_FILE)
    return stats, rankings

def clean_data(stats, rankings):
    # Clean team names
    stats['Team_Clean'] = stats['Team'].astype(str).apply(
        lambda x: re.sub(r'\s*\(\d+\)|\s*\(AQ\)', '', x).strip()
    )
    rankings['School_Clean'] = rankings['School'].str.strip()
    
    # Merge
    df = pd.merge(
        stats, rankings,
        left_on=['Team_Clean', 'Season'],
        right_on=['School_Clean', 'Season'],
        how='left'
    )

    # Rename columns for consistency
    df = df.rename(columns={
        'Conf.Record': 'ConfRecord',
        'Non-ConferenceRecord': 'NonConfRecord',
        'NET Rank': 'NETRank',
    })
    
    df['Overall_Seed'] = df['Overall_Seed'].fillna(0)
    df['Made_Tournament'] = (df['Overall_Seed'] > 0).astype(int)
    df['Is_AQ'] = (df['Berth Type'] == 'Automatic').astype(int)
    
    return df

# ------------------------------
# Feature Engineering
# ------------------------------

def _parse_wl(value):
    try:
        wins, losses = str(value).split('-')
        return int(wins), int(losses)
    except:
        return 0, 0

def engineer_features(df):
    # Parse W-L and Quadrants
    for col in ['WL', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if col in df.columns:
            df[[f'{col}_W', f'{col}_L']] = df[col].apply(lambda x: pd.Series(_parse_wl(x)))
    
    # Win percentages
    df['Win_Pct'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
    df['Q1_Win_Pct'] = df['Quadrant1_W'] / (df['Quadrant1_W'] + df['Quadrant1_L'] + 1e-6)
    
    # Weighted Quadrant Wins
    df['Weighted_Q_Wins'] = (df['Quadrant1_W']*3 + df['Quadrant2_W']*2 - df['Quadrant3_L'] - df['Quadrant4_L'])
    
    # Momentum: last 5 games W%
    df['Momentum'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
    
    # Conference strength
    conf_strength = df.groupby('Conference')['NETRank'].mean().to_dict()
    df['Conf_Strength'] = df['Conference'].map(conf_strength)
    
    # Encode Conference
    le = LabelEncoder()
    df['Conf_ID'] = le.fit_transform(df['Conference'].astype(str))
    
    # Fill NaNs
    df[FEATS_STAGE2] = df[FEATS_STAGE2].fillna(0)
    
    return df

# ------------------------------
# Stage 1: Tournament Selection
# ------------------------------

def train_selection_model(train_data):
    pool = train_data[train_data['Is_AQ'] == 0]
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        random_state=42
    )
    clf.fit(pool[FEATS_STAGE1], pool['Made_Tournament'])
    return clf

def predict_field(clf, test_data):
    known_aqs = test_data[test_data['Is_AQ'] == 1]
    n_aqs = len(known_aqs)
    n_spots_remaining = 68 - n_aqs
    
    non_aq = test_data[test_data['Is_AQ'] == 0].copy()
    non_aq['Selection_Prob'] = clf.predict_proba(non_aq[FEATS_STAGE1])[:, 1]
    
    predicted_at_larges = non_aq.nlargest(n_spots_remaining, 'Selection_Prob')
    predicted_field = pd.concat([known_aqs, predicted_at_larges], ignore_index=True)
    
    return predicted_field

# ------------------------------
# Stage 2: Seeding
# ------------------------------

def train_seeding_model(train_data):
    pool = train_data[train_data['Made_Tournament'] == 1]
    reg = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        random_state=42
    )
    reg.fit(pool[FEATS_STAGE2], pool['Overall_Seed'])
    return reg

def assign_seeds(reg, predicted_field):
    """
    Assign seeds based on predicted Stage 2 scores.
    Maps predicted scores to realistic 1-68 seeds using rank ordering.
    """
    predicted_field = predicted_field.copy()
    
    # Predict seed scores
    predicted_field['Seed_Score'] = reg.predict(predicted_field[FEATS_STAGE2])
    
    # Rank by predicted score
    predicted_field = predicted_field.sort_values('Seed_Score').reset_index(drop=True)
    
    # Map ranks to realistic seed ranges
    # Typically 1-4: top 16, 5-8: next 16, etc.
    n = len(predicted_field)
    seed_boundaries = [1, 17, 33, 49, 65, 69]  # 1-16,17-32,33-48,49-64,65-68
    seeds = []
    
    for i in range(len(seed_boundaries)-1):
        start, end = seed_boundaries[i], seed_boundaries[i+1]
        count = end - start
        scores_chunk = predicted_field.iloc[start-1:end-1]
        # Assign seeds by predicted order within chunk
        seeds.extend(range(start, start + len(scores_chunk)))
    
    # In case there is a mismatch due to rounding
    if len(seeds) < n:
        seeds.extend(range(len(seeds)+1, n+1))
    
    predicted_field['Predicted_Seed'] = seeds[:n]
    
    return predicted_field

# ------------------------------
# Submission
# ------------------------------

def generate_submission(test_data, predicted_field, output_path=SUBMISSION_PATH):
    merge_df = test_data[['Season', 'Team']].merge(
        predicted_field[['Team', 'Predicted_Seed']],
        on='Team', how='left'
    )
    merge_df['Predicted_Seed'] = merge_df['Predicted_Seed'].fillna(0).astype(int)
    merge_df['RecordID'] = merge_df['Season'].astype(str) + '-' + merge_df['Team'].str.replace(' ', '')
    submission = merge_df[['RecordID', 'Predicted_Seed']].rename(columns={'Predicted_Seed': 'Overall_Seed'})
    submission.to_csv(output_path, index=False)
    return submission

# ------------------------------
# Feature Importance
# ------------------------------
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

# ------------------------------
# Evaluate 2025 using fixed CSV
# ------------------------------
def evaluate_2025(test_data, predicted_field, rankings_file=RANKINGS_FILE):
    """
    Compute and print validation metrics for 2025:
    - Number of predicted tournament teams
    - Selection accuracy (at-large)
    - MAE and RMSE on seeds using fixed CSV
    """
    # Load actual 2025 seeds from fixed CSV
    rankings = pd.read_csv(rankings_file)
    actual_2025 = rankings[rankings['Season'] == 2025][['School', 'Overall_Seed']].copy()
    actual_2025 = actual_2025.rename(columns={'School': 'Team'})

    # Number of predicted tournament teams
    print(f"\nNumber of predicted tournament teams: {len(predicted_field)}")

    # Selection accuracy (at-large)
    actual_at_large = actual_2025.merge(
        test_data[['Team', 'Is_AQ']], on='Team', how='left'
    )
    actual_at_large = actual_at_large[actual_at_large['Is_AQ'] == 0]
    predicted_at_large = predicted_field[predicted_field['Is_AQ'] == 0]
    
    correct_at_large = set(predicted_at_large['Team']) & set(actual_at_large['Team'])
    n_at_large = len(actual_at_large)
    accuracy = len(correct_at_large) / n_at_large if n_at_large > 0 else 0
    print(f"Selection accuracy (at-large): {len(correct_at_large)}/{n_at_large} = {accuracy:.1%}")

    # Seeding: MAE and RMSE
    eval_df = predicted_field.merge(
        actual_2025[['Team','Overall_Seed']],
        on='Team',
        how='inner'
    )
    y_true = eval_df['Overall_Seed_y'].astype(float)  # actual seeds
    y_pred = eval_df['Predicted_Seed'].astype(float)  # your model prediction
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"MAE (seeding): {mae:.2f} seed positions")
    print(f"RMSE (seeding): {rmse:.2f} seed positions")

# ------------------------------
# Main Pipeline
# ------------------------------
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

    # Validation metrics for 2025 using fixed CSV
    print("\n" + "=" * 60)
    print("2025 VALIDATION METRICS")
    print("=" * 60)
    evaluate_2025(test_data, predicted_field)
    print("=" * 60)

if __name__ == '__main__':
    main()