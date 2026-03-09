import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid

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
    df = df.rename(columns={'Q1': 'Quadrant1', 'Q2': 'Quadrant2', 'Q3': 'Quadrant3', 'Q4': 'Quadrant4'})
    df['Team_Clean'] = df['Team'].astype(str).apply(lambda x: re.sub(r'\s*\(\d+\)|\s*\(AQ\)', '', x, flags=re.IGNORECASE).strip())
    stats_dfs.append(df)

stats = pd.concat(stats_dfs, ignore_index=True)
rankings = pd.read_csv('../new_data/ncaa_rankings_2021_2025_fixed.csv')

def normalize_rankings_name(name):
    name = str(name).strip().replace(' State', ' St.')
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
df = pd.merge(stats, rankings, left_on=['Team_Clean', 'Season'], right_on=['School_Clean', 'Season'], how='left')

df['Overall Seed'] = df['Overall Seed'].fillna(0)
df['Made_Tournament'] = (df['Overall Seed'] > 0).astype(int)
df['Is_AQ'] = (df['Berth Type'] == 'Automatic').astype(int)

def parse_wl(x):
    x = str(x)
    months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    if '-' in x:
        p = x.split('-')
        for m, num in months.items():
            if p[0] == m: return num, int(p[1])
            if p[1] == m: return int(p[0]), num
        try: return int(p[0]), int(p[1])
        except: pass
    return 0, 0

# TARGETED renaming to protect 'Overall Seed'
df = df.rename(columns={
    'Conf.Record': 'ConfRecord', 
    'Non-ConferenceRecord': 'NonConfRecord',
    'NET Rank': 'NETRank'
})

wl_cols = ['WL', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']

for c in wl_cols:
    if c in df.columns:
        df[[f'{c}_W', f'{c}_L']] = df[c].apply(lambda x: pd.Series(parse_wl(x)))

df['Win_Pct'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
df['Q1_Win_Pct'] = df['Quadrant1_W'] / (df['Quadrant1_W'] + df['Quadrant1_L'] + 1e-6)
df['Quality_Wins'] = df['Quadrant1_W'] + df['Quadrant2_W']
df['Bad_Losses'] = df['Quadrant3_L'] + df['Quadrant4_L']

le = LabelEncoder()
df['Conf_ID'] = le.fit_transform(df['Conference'].astype(str))

# Separate features for the two distinct models
feats_stage1 = ['NETRank', 'PrevNET', 'NETSOS', 'Win_Pct', 'Quadrant1_W', 'Q1_Win_Pct', 'Quality_Wins', 'Bad_Losses', 'Conf_ID']
feats_stage2 = feats_stage1 + ['Is_AQ']  
df[feats_stage2] = df[feats_stage2].fillna(0)

train_data = df[df['Season'] < 2025].copy()
test_data = df[df['Season'] == 2025].copy()

print("Tuning Stage 1: At-Large Selection Model (LOSO CV)...")
param_grid_clf = {
    'n_estimators': [50, 100, 150],  
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.1]
}

best_clf_score = -1
best_clf_params = None

for params in ParameterGrid(param_grid_clf):
    season_accuracies = []
    
    for val_season in [2021, 2022, 2023, 2024]:
        t_data = train_data[(train_data['Season'] != val_season) & (train_data['Is_AQ'] == 0)]
        v_data = train_data[(train_data['Season'] == val_season) & (train_data['Is_AQ'] == 0)].copy()
        
        clf = GradientBoostingClassifier(**params, random_state=42)
        clf.fit(t_data[feats_stage1], t_data['Made_Tournament'])
        v_data['Prob'] = clf.predict_proba(v_data[feats_stage1])[:, 1]
        
        actual_aqs = train_data[(train_data['Season'] == val_season) & (train_data['Is_AQ'] == 1)]
        spots = 68 - len(actual_aqs)
        
        preds = v_data.nlargest(spots, 'Prob')
        actual_at_large = v_data[v_data['Made_Tournament'] == 1]
        
        accuracy = len(set(preds['Team']).intersection(set(actual_at_large['Team']))) / spots
        season_accuracies.append(accuracy)
        
    avg_acc = np.mean(season_accuracies)
    if avg_acc > best_clf_score:
        best_clf_score = avg_acc
        best_clf_params = params

print(f" -> Best Selection Accuracy: {best_clf_score:.2%} with params: {best_clf_params}\n")

# Train Final Stage 1 Model
train_pool_s1 = train_data[train_data['Is_AQ'] == 0]
final_clf = GradientBoostingClassifier(**best_clf_params, random_state=42)
final_clf.fit(train_pool_s1[feats_stage1], train_pool_s1['Made_Tournament'])

# Predict 2025 Field
test_pool_s1 = test_data[test_data['Is_AQ'] == 0].copy()
test_pool_s1['Selection_Prob'] = final_clf.predict_proba(test_pool_s1[feats_stage1])[:, 1]
known_aqs_2025 = test_data[test_data['Is_AQ'] == 1]
predicted_at_larges = test_pool_s1.nlargest(68 - len(known_aqs_2025), 'Selection_Prob')
predicted_field_2025 = pd.concat([known_aqs_2025, predicted_at_larges])

print("Training Stage 2: Static Tournament Seeding Model...")

# Train ONLY on teams that actually made the tournament (2021-2024)
train_pool_s2 = train_data[train_data['Made_Tournament'] == 1]

# Using the static, un-tuned model from work.py that generalized better
final_reg = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
final_reg.fit(train_pool_s2[feats_stage2], train_pool_s2['Overall Seed'])

# Rank the 2025 Field
predicted_field_2025['Seed_Score'] = final_reg.predict(predicted_field_2025[feats_stage2])
predicted_field_2025 = predicted_field_2025.sort_values('Seed_Score')
predicted_field_2025['Predicted_Seed'] = range(1, 69)

final_output = test_data[['Season', 'Team']].merge(predicted_field_2025[['Team', 'Predicted_Seed']], on='Team', how='left')
final_output['Predicted_Seed'] = final_output['Predicted_Seed'].fillna(0).astype(int)

final_output['RecordID'] = final_output['Season'].astype(str) + '-' + final_output['Team'].str.replace(' ', '')
final_output[['RecordID', 'Predicted_Seed']].to_csv('2025_submission_hybrid.csv', index=False)

# Evaluate the 2025 Validation Data
actual_2025_field = test_data[test_data['Made_Tournament'] == 1]
eval_df = predicted_field_2025.merge(actual_2025_field[['Team', 'Overall Seed', 'Berth Type', 'NETRank']], on='Team', suffixes=('', '_Actual'))
eval_df = eval_df.dropna(subset=['Overall Seed_Actual']) 

final_2025_rmse = root_mean_squared_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
final_2025_mae = mean_absolute_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])

print("=== 2025 FINAL HOLD-OUT VALIDATION ===")
print(f"Final 2025 RMSE: {final_2025_rmse:.2f}")
print(f"Final 2025 MAE: {final_2025_mae:.2f} seeds off on average")
print("Saved complete file to '2025_submission_hybrid.csv'")

print("\n" + "="*70)
print("2025 VALIDATION RESULTS")
print("="*70)

actual_at_large = test_data[(test_data['Made_Tournament'] == 1) & (test_data['Is_AQ'] == 0)]
predicted_at_large_teams = set(predicted_at_larges['Team'])
actual_at_large_teams = set(actual_at_large['Team'])
missed_at_large = actual_at_large_teams - predicted_at_large_teams
incorrect_at_large = predicted_at_large_teams - actual_at_large_teams

print(f"\n--- AT-LARGE BID ANALYSIS ---")
print(f"Actual At-Large Teams: {len(actual_at_large_teams)}")
print(f"Predicted At-Large Teams: {len(predicted_at_large_teams)}")
print(f"Correctly Predicted At-Large: {len(predicted_at_large_teams & actual_at_large_teams)}")
print(f"Selection Accuracy: {len(predicted_at_large_teams & actual_at_large_teams) / len(actual_at_large_teams):.1%}")

if len(eval_df) > 0:
    eval_df['Seed_Error'] = eval_df['Predicted_Seed'] - eval_df['Overall Seed_Actual']
    eval_df['Abs_Error'] = abs(eval_df['Seed_Error'])
    
    print(f"\n--- SEEDING ANALYSIS ---")
    print(f"Mean error (signed): {eval_df['Seed_Error'].mean():.2f}")
    print(f"Median absolute error: {eval_df['Abs_Error'].median():.2f}")
    
    # Error distribution
    print(f"\n--- ERROR DISTRIBUTION ---")
    print(f"Within 2 seeds: {len(eval_df[eval_df['Abs_Error'] <= 2])}")
    print(f"Within 5 seeds: {len(eval_df[eval_df['Abs_Error'] <= 5])}")
    
    print(f"\n❌ BIGGEST SEEDING ERRORS (Top 10):")
    print("-" * 90)
    print(f"{'Team':<30} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Type':<8} {'NET':<6}")
    print("-" * 90)
    
    worst_predictions = eval_df.nlargest(10, 'Abs_Error')
    for idx, row in worst_predictions.iterrows():
        error_sign = "+" if row['Seed_Error'] > 0 else ""
        bid_type = "AQ" if row['Berth Type'] == 'Automatic' else "AL"
        print(f"{row['Team']:<30} {row['Predicted_Seed']:<10} {row['Overall Seed_Actual']:<10.0f} "
              f"{error_sign}{row['Seed_Error']:<9.0f} {bid_type:<8} {row['NETRank']:<6.0f}")
else:
    print("⚠️ No teams matched between predictions and actual tournament!")