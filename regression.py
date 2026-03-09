import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# File paths
files = {
    2021: '../new_data/NCAA Statistics 2020-2021 (1).csv',
    2022: '../new_data/NCAA Statistics2021-2022 (1).csv',
    2023: '../new_data/NCAA Statistics2022-2023 (1).csv',
    2024: '../new_data/NCAA Statistics2023-2024 (1).csv',
    2025: '../new_data/NCAA Statistics2024-2025 (1).csv'
}

stats_dfs = []
for year, filepath in files.items():
    df = pd.read_csv(filepath)
    df['Season'] = year
    
    # Standardize column names (2025 uses 'Q1', older years use 'Quadrant1')
    df = df.rename(columns={'Q1': 'Quadrant1', 'Q2': 'Quadrant2', 
                            'Q3': 'Quadrant3', 'Q4': 'Quadrant4'})
    
    # Clean Stats team names (Remove seeds/AQ tags)
    df['Team_Clean'] = df['Team'].astype(str).apply(
        lambda x: re.sub(r'\s*\(\d+\)|\s*\(AQ\)', '', x, flags=re.IGNORECASE).strip()
    )
    stats_dfs.append(df)

stats = pd.concat(stats_dfs, ignore_index=True)
rankings = pd.read_csv('../new_data/ncaa_rankings_2021_2025_fixed.csv')

# 100% Perfect Normalization for Rankings
def normalize_rankings_name(name):
    name = str(name).strip()
    name = name.replace(' State', ' St.')
    if name == 'NC St.': return 'NC State'
    
    mapping = {
        'USC': 'Southern California', 'Eastern Washington': 'Eastern Wash.',
        'Western Kentucky': 'Western Ky.', 'Northern Kentucky': 'Northern Ky.',
        'Appalachian St.': 'App State', 'Texas A&M–Corpus Christi': 'A&M-Corpus Christi',
        'Charleston': 'Col. of Charleston', 'Southeast Missouri St.': 'Southeast Mo. St.',
        'Florida Atlantic': 'Fla. Atlantic', 'Fairleigh Dickinson': 'FDU',
        'Grambling St.': 'Grambling', 'UNC Wilmington': 'UNCW',
        'SIU Edwardsville': 'SIUE', "Saint Mary's": "Saint Mary's (CA)",
        "St. John's": "St. John's (NY)"
    }
    return mapping.get(name, name)

rankings['School_Clean'] = rankings['School'].apply(normalize_rankings_name)

# Merge Datasets
df = pd.merge(stats, rankings, left_on=['Team_Clean', 'Season'], right_on=['School_Clean', 'Season'], how='left')

# Setup Target Variables
df['Overall Seed'] = df['Overall Seed'].fillna(0)
df['Made_Tournament'] = (df['Overall Seed'] > 0).astype(int)
df['Is_AQ'] = (df['Berth Type'] == 'Automatic').astype(int)

# Fix the Excel Date bug (e.g., "8-Sep") and parse standard "W-L"
def parse_wl(x):
    x = str(x)
    months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
              'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    if '-' in x:
        p = x.split('-')
        for m, num in months.items():
            if p[0] == m: return num, int(p[1]) # e.g. Sep-8 -> 9-8
            if p[1] == m: return int(p[0]), num # e.g. 8-Sep -> 8-9
        try: return int(p[0]), int(p[1])
        except: pass
    return 0, 0

# Apply parsing
wl_cols = ['WL', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']

# THIS LINE REMOVES ALL SPACES FROM COLUMN NAMES
df.columns = [c.replace(' ', '').replace('.', '') for c in df.columns]

for c in wl_cols:
    if c in df.columns:
        df[[f'{c}_W', f'{c}_L']] = df[c].apply(lambda x: pd.Series(parse_wl(x)))

# Calculate critical percentages and aggregates
df['Win_Pct'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
df['Q1_Win_Pct'] = df['Quadrant1_W'] / (df['Quadrant1_W'] + df['Quadrant1_L'] + 1e-6)
df['Quality_Wins'] = df['Quadrant1_W'] + df['Quadrant2_W']
df['Bad_Losses'] = df['Quadrant3_L'] + df['Quadrant4_L']

le = LabelEncoder()
df['Conf_ID'] = le.fit_transform(df['Conference'].astype(str))

# Define Features
feats_stage1 = ['NETRank', 'PrevNET', 'NETSOS', 'Win_Pct', 'Quadrant1_W', 'Q1_Win_Pct', 'Quality_Wins', 'Bad_Losses', 'Conf_ID']
feats_stage2 = feats_stage1 + ['Is_AQ'] # Stage 2 gets the AQ flag
df[feats_stage2] = df[feats_stage2].fillna(0)

train_data = df[df['Season'] < 2025]
test_data = df[df['Season'] == 2025].copy()

print("Training Stage 1: At-Large Selection...")

# Train ONLY on non-AQs (Is_AQ == 0)
train_pool_s1 = train_data[train_data['Is_AQ'] == 0]
clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
clf.fit(train_pool_s1[feats_stage1], train_pool_s1['Made_Tournament'])

# Predict At-Large probabilities for 2025 non-AQs
test_pool_s1 = test_data[test_data['Is_AQ'] == 0].copy()
test_pool_s1['Selection_Prob'] = clf.predict_proba(test_pool_s1[feats_stage1])[:, 1]

# Count known AQs and calculate remaining spots
known_aqs_2025 = test_data[test_data['Is_AQ'] == 1]
n_spots_remaining = 68 - len(known_aqs_2025)

# Select the Top N teams based on probability
predicted_at_larges = test_pool_s1.nlargest(n_spots_remaining, 'Selection_Prob')

# Combine AQs and Predicted At-Larges to form the Field of 68
predicted_field = pd.concat([known_aqs_2025, predicted_at_larges])

s1_importance = pd.DataFrame({
    'Feature': feats_stage1,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
s1_importance['Importance (%)'] = 100 * s1_importance['Importance'] / s1_importance['Importance'].sum()


print("\n--- Stage 1 Feature Importance (At-Large Selection) ---")
print(s1_importance.to_string(index=False))

print("Training Stage 2: Seeding Model...")

# Train ONLY on teams that actually made the tournament
train_pool_s2 = train_data[train_data['Made_Tournament'] == 1]
reg = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)

# ---> FIXED: Changed 'Overall Seed' to 'OverallSeed' <---
reg.fit(train_pool_s2[feats_stage2], train_pool_s2['OverallSeed'])

# Predict Seeds for our 68-team predicted field
predicted_field['Seed_Score'] = reg.predict(predicted_field[feats_stage2])

# Rank 1 to 68 based on predicted score
predicted_field = predicted_field.sort_values('Seed_Score')
predicted_field['Predicted_Seed'] = range(1, 69)


s2_importance = pd.DataFrame({
    'Feature': feats_stage2,
    'Importance': reg.feature_importances_
}).sort_values(by='Importance', ascending=False)
s2_importance['Importance (%)'] = 100 * s2_importance['Importance'] / s2_importance['Importance'].sum()

print("\n--- Stage 2 Feature Importance (Seeding Model) ---")
print(s2_importance.to_string(index=False))

# Measure Selection Accuracy
actual_field = test_data[test_data['Made_Tournament'] == 1]
correct_selections = set(predicted_field['Team']).intersection(set(actual_field['Team']))

print("\n" + "="*70)
print("2025 VALIDATION RESULTS")
print("="*70)
print(f"Correctly Selected Teams: {len(correct_selections)} / 68")

# Get actual at-large teams from 2025
actual_at_large = test_data[(test_data['Made_Tournament'] == 1) & (test_data['Is_AQ'] == 0)]
predicted_at_large_teams = set(predicted_at_larges['Team'])
actual_at_large_teams = set(actual_at_large['Team'])

# False Negatives: Teams that got at-large bids but we didn't predict
missed_at_large = actual_at_large_teams - predicted_at_large_teams

# False Positives: Teams we predicted for at-large but didn't actually make it
incorrect_at_large = predicted_at_large_teams - actual_at_large_teams

print(f"\n--- AT-LARGE BID ANALYSIS ---")
print(f"Actual At-Large Teams: {len(actual_at_large_teams)}")
print(f"Predicted At-Large Teams: {len(predicted_at_large_teams)}")
print(f"Correctly Predicted At-Large: {len(predicted_at_large_teams & actual_at_large_teams)}")

if missed_at_large:
    print(f"\n❌ MISSED AT-LARGE TEAMS ({len(missed_at_large)}):")
    print("-" * 70)
    missed_df = actual_at_large[actual_at_large['Team'].isin(missed_at_large)][
        ['Team', 'OverallSeed', 'NETRank', 'Quadrant1_W', 'Quality_Wins', 'Win_Pct']
    ].sort_values('OverallSeed')
    
    for idx, row in missed_df.iterrows():
        prob = 0
        if row['Team'] in test_pool_s1['Team'].values:
            prob = test_pool_s1[test_pool_s1['Team'] == row['Team']]['Selection_Prob'].values[0]
        
        print(f"  {row['Team']:<30} Seed: {row['OverallSeed']:>2.0f} | "
              f"NET: {row['NETRank']:>3.0f} | Q1: {row['Quadrant1_W']:>2.0f} | "
              f"Quality: {row['Quality_Wins']:>2.0f} | WinPct: {row['Win_Pct']:.3f} | "
              f"Prob: {prob:.3f}")
else:
    print("\n✓ No at-large teams were missed!")

if incorrect_at_large:
    print(f"\n❌ INCORRECTLY SELECTED (False Positives) ({len(incorrect_at_large)}):")
    print("-" * 70)
    incorrect_df = predicted_at_larges[predicted_at_larges['Team'].isin(incorrect_at_large)][
        ['Team', 'NETRank', 'Quadrant1_W', 'Quality_Wins', 'Win_Pct', 'Selection_Prob']
    ].sort_values('Selection_Prob', ascending=False)
    
    for idx, row in incorrect_df.iterrows():
        print(f"  {row['Team']:<30} NET: {row['NETRank']:>3.0f} | Q1: {row['Quadrant1_W']:>2.0f} | "
              f"Quality: {row['Quality_Wins']:>2.0f} | WinPct: {row['Win_Pct']:.3f} | "
              f"Prob: {row['Selection_Prob']:.3f}")


print(f"\n--- SEEDING ANALYSIS ---")

# Merge predictions with actuals for teams that made it
# Note: Column names have no spaces due to earlier processing
eval_df = predicted_field.merge(
    actual_field[['Team', 'OverallSeed', 'BerthType']], 
    on='Team', 
    how='inner',
    suffixes=('', '_Actual')
)

if len(eval_df) > 0:
    eval_df['Seed_Error'] = eval_df['Predicted_Seed'] - eval_df['OverallSeed_Actual']
    eval_df['Abs_Error'] = abs(eval_df['Seed_Error'])
    
    mae = mean_absolute_error(eval_df['OverallSeed_Actual'], eval_df['Predicted_Seed'])
    rmse = root_mean_squared_error(eval_df['OverallSeed_Actual'], eval_df['Predicted_Seed'])
    
    print(f"Teams evaluated for seeding: {len(eval_df)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} seed positions")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} seed positions")
    
    # Show biggest misses
    print(f"\n❌ BIGGEST SEEDING ERRORS (Top 15):")
    print("-" * 90)
    print(f"{'Team':<30} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Type':<8} {'NET':<6}")
    print("-" * 90)
    
    worst_predictions = eval_df.nlargest(15, 'Abs_Error')
    for idx, row in worst_predictions.iterrows():
        error_sign = "+" if row['Seed_Error'] > 0 else ""
        bid_type = "AQ" if row['BerthType'] == 'Automatic' else "AL"
        print(f"{row['Team']:<30} {row['Predicted_Seed']:<10} {row['OverallSeed_Actual']:<10.0f} "
              f"{error_sign}{row['Seed_Error']:<9.0f} {bid_type:<8} {row['NETRank']:<6.0f}")
    
    # Show perfect or near-perfect predictions
    print(f"\n✓ BEST PREDICTIONS (Within 2 seeds):")
    print("-" * 90)
    print(f"{'Team':<30} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Type':<8} {'NET':<6}")
    print("-" * 90)
    
    best_predictions = eval_df[eval_df['Abs_Error'] <= 2].sort_values('Abs_Error')
    for idx, row in best_predictions.head(15).iterrows():
        error_sign = "+" if row['Seed_Error'] > 0 else ""
        bid_type = "AQ" if row['BerthType'] == 'Automatic' else "AL"
        print(f"{row['Team']:<30} {row['Predicted_Seed']:<10} {row['OverallSeed_Actual']:<10.0f} "
              f"{error_sign}{row['Seed_Error']:<9.0f} {bid_type:<8} {row['NETRank']:<6.0f}")
    
    print(f"\nTeams within 2 seeds: {len(best_predictions)} / {len(eval_df)} "
          f"({100*len(best_predictions)/len(eval_df):.1f}%)")
    print(f"Teams within 5 seeds: {len(eval_df[eval_df['Abs_Error'] <= 5])} / {len(eval_df)} "
          f"({100*len(eval_df[eval_df['Abs_Error'] <= 5])/len(eval_df):.1f}%)")
    
    # Breakdown by bid type
    print(f"\n--- SEEDING ERROR BY BID TYPE ---")
    for bid_type in ['Automatic', 'At-Large']:
        subset = eval_df[eval_df['BerthType'] == bid_type]
        if len(subset) > 0:
            mae_type = mean_absolute_error(subset['OverallSeed_Actual'], subset['Predicted_Seed'])
            print(f"{bid_type:<12}: {len(subset):>2} teams | MAE: {mae_type:.2f} seeds")
else:
    print("⚠️ No teams matched between predictions and actual tournament!")

# Prepare Final Output
final_output = test_data[['Season', 'Team']].merge(
    predicted_field[['Team', 'Predicted_Seed']], 
    on='Team', 
    how='left'
)
final_output['Predicted_Seed'] = final_output['Predicted_Seed'].fillna(0).astype(int)
final_output['RecordID'] = (final_output['Season'].astype(str) + '-' + 
                             final_output['Team'].str.replace(' ', ''))

# Save submission
final_output[['RecordID', 'Predicted_Seed']].rename(
    columns={'Predicted_Seed': 'Overall Seed'}
).to_csv('2025_submission.csv', index=False)

print("\n" + "="*70)
print("✓ Predictions saved to 2025_submission.csv")
print("="*70)