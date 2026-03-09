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

# TARGETED renaming instead of blind replacement to protect 'Overall Seed'
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
    'n_estimators': [50, 100, 150],  # Equivalent to epochs
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.1]
}

best_clf_score = -1
best_clf_params = None

for params in ParameterGrid(param_grid_clf):
    season_accuracies = []
    
    for val_season in [2021, 2022, 2023, 2024]:
        # Train on remaining, test on holdout
        t_data = train_data[(train_data['Season'] != val_season) & (train_data['Is_AQ'] == 0)]
        v_data = train_data[(train_data['Season'] == val_season) & (train_data['Is_AQ'] == 0)].copy()
        
        clf = GradientBoostingClassifier(**params, random_state=42)
        clf.fit(t_data[feats_stage1], t_data['Made_Tournament'])
        v_data['Prob'] = clf.predict_proba(v_data[feats_stage1])[:, 1]
        
        # Calculate spots based on AQs for that year
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

print("Tuning Stage 2: Tournament Seeding Model (LOSO CV)...")
param_grid_reg = {
    'n_estimators': [50, 100, 150, 200],  # Different epoch bounds for regression
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1]
}

best_reg_mse = float('inf')
best_reg_params = None

for params in ParameterGrid(param_grid_reg):
    season_mses = []
    
    for val_season in [2021, 2022, 2023, 2024]:
        # Train only on actual tournament teams
        t_data = train_data[(train_data['Season'] != val_season) & (train_data['Made_Tournament'] == 1)]
        v_data = train_data[(train_data['Season'] == val_season) & (train_data['Made_Tournament'] == 1)].copy()
        
        reg = GradientBoostingRegressor(**params, random_state=42)
        reg.fit(t_data[feats_stage2], t_data['Overall Seed'])
        v_data['Predicted_Score'] = reg.predict(v_data[feats_stage2])
        
        # Rank 1-68
        v_data = v_data.sort_values('Predicted_Score')
        v_data['Predicted_Seed'] = range(1, len(v_data) + 1)
        
        # Calculate MSE for this holdout season
        mse = mean_squared_error(v_data['Overall Seed'], v_data['Predicted_Seed'])
        season_mses.append(mse)
        
    avg_mse = np.mean(season_mses)
    if avg_mse < best_reg_mse:
        best_reg_mse = avg_mse
        best_reg_params = params

print(f" -> Best Seeding MSE: {best_reg_mse:.2f} with params: {best_reg_params}\n")

print("Training final distinct models with optimal parameters on 2021-2024 data...")

# Stage 1 Final Model
train_pool_s1 = train_data[train_data['Is_AQ'] == 0]
final_clf = GradientBoostingClassifier(**best_clf_params, random_state=42)
final_clf.fit(train_pool_s1[feats_stage1], train_pool_s1['Made_Tournament'])

# Predict 2025 Field
test_pool_s1 = test_data[test_data['Is_AQ'] == 0].copy()
test_pool_s1['Selection_Prob'] = final_clf.predict_proba(test_pool_s1[feats_stage1])[:, 1]
known_aqs_2025 = test_data[test_data['Is_AQ'] == 1]
predicted_at_larges = test_pool_s1.nlargest(68 - len(known_aqs_2025), 'Selection_Prob')
predicted_field_2025 = pd.concat([known_aqs_2025, predicted_at_larges])

# Stage 2 Final Model
train_pool_s2 = train_data[train_data['Made_Tournament'] == 1]
final_reg = GradientBoostingRegressor(**best_reg_params, random_state=42)
final_reg.fit(train_pool_s2[feats_stage2], train_pool_s2['Overall Seed'])

# Rank the 2025 Field
predicted_field_2025['Seed_Score'] = final_reg.predict(predicted_field_2025[feats_stage2])
predicted_field_2025 = predicted_field_2025.sort_values('Seed_Score')
predicted_field_2025['Predicted_Seed'] = range(1, 69)

# Fill 0 for all non-tournament teams
final_output = test_data[['Season', 'Team']].merge(predicted_field_2025[['Team', 'Predicted_Seed']], on='Team', how='left')
final_output['Predicted_Seed'] = final_output['Predicted_Seed'].fillna(0).astype(int)

# Create submission
final_output['RecordID'] = final_output['Season'].astype(str) + '-' + final_output['Team'].str.replace(' ', '')
final_output[['RecordID', 'Predicted_Seed']].to_csv('2025_submission.csv', index=False)

# Evaluate the 2025 Validation Data
actual_2025_field = test_data[test_data['Made_Tournament'] == 1]
eval_df = predicted_field_2025.merge(actual_2025_field[['Team', 'Overall Seed']], on='Team', suffixes=('', '_Actual'))
eval_df = eval_df.dropna(subset=['Overall Seed_Actual']) # Only eval actual made teams

final_2025_rmse = root_mean_squared_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
final_2025_mae = mean_absolute_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])

print("=== 2025 FINAL HOLD-OUT VALIDATION ===")
print(f"Final 2025 RMSE: {final_2025_rmse:.2f}")
print(f"Final 2025 MAE: {final_2025_mae:.2f} seeds off on average")
print("Saved complete file to '2025_submission.csv'")


print("\n" + "="*70)
print("2025 VALIDATION RESULTS")
print("="*70)


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
print(f"Selection Accuracy: {len(predicted_at_large_teams & actual_at_large_teams) / len(actual_at_large_teams):.1%}")

if missed_at_large:
    print(f"\n❌ MISSED AT-LARGE TEAMS ({len(missed_at_large)}):")
    print("-" * 70)
    missed_df = actual_at_large[actual_at_large['Team'].isin(missed_at_large)][
        ['Team', 'Overall Seed', 'NETRank', 'Quadrant1_W', 'Quality_Wins', 'Win_Pct']
    ].sort_values('Overall Seed')
    
    for idx, row in missed_df.iterrows():
        prob = 0
        if row['Team'] in test_pool_s1['Team'].values:
            prob = test_pool_s1[test_pool_s1['Team'] == row['Team']]['Selection_Prob'].values[0]
        
        print(f"  {row['Team']:<30} Seed: {row['Overall Seed']:>2.0f} | "
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
else:
    print("\n✓ No incorrect at-large selections!")

print(f"\n--- SEEDING ANALYSIS ---")

# Merge predictions with actuals for teams that made it
eval_df = predicted_field_2025.merge(
    actual_2025_field[['Team', 'Overall Seed', 'Berth Type']], 
    on='Team', 
    how='inner',
    suffixes=('', '_Actual')
)

if len(eval_df) > 0:
    eval_df['Seed_Error'] = eval_df['Predicted_Seed'] - eval_df['Overall Seed_Actual']
    eval_df['Abs_Error'] = abs(eval_df['Seed_Error'])
    
    mae = mean_absolute_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
    rmse = root_mean_squared_error(eval_df['Overall Seed_Actual'], eval_df['Predicted_Seed'])
    
    # Normalized RMSE (as percentage of seed range)
    seed_range = eval_df['Overall Seed_Actual'].max() - eval_df['Overall Seed_Actual'].min()
    nrmse_range = (rmse / seed_range) * 100 if seed_range > 0 else 0
    
    # Alternative: Normalized by standard deviation
    seed_std = eval_df['Overall Seed_Actual'].std()
    nrmse_std = (rmse / seed_std) * 100 if seed_std > 0 else 0
    
    print(f"Teams evaluated for seeding: {len(eval_df)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} seed positions")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} seed positions")
    print(f"Normalized RMSE (% of range): {nrmse_range:.1f}%")
    print(f"Normalized RMSE (% of std dev): {nrmse_std:.1f}%")
    
    # Show biggest misses
    print(f"\n❌ BIGGEST SEEDING ERRORS (Top 15):")
    print("-" * 90)
    print(f"{'Team':<30} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Type':<8} {'NET':<6}")
    print("-" * 90)
    
    worst_predictions = eval_df.nlargest(15, 'Abs_Error')
    for idx, row in worst_predictions.iterrows():
        error_sign = "+" if row['Seed_Error'] > 0 else ""
        bid_type = "AQ" if row['Berth Type'] == 'Automatic' else "AL"
        print(f"{row['Team']:<30} {row['Predicted_Seed']:<10} {row['Overall Seed_Actual']:<10.0f} "
              f"{error_sign}{row['Seed_Error']:<9.0f} {bid_type:<8} {row['NETRank']:<6.0f}")
    
    # Show perfect or near-perfect predictions
    print(f"\n✓ BEST PREDICTIONS (Within 2 seeds):")
    print("-" * 90)
    print(f"{'Team':<30} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Type':<8} {'NET':<6}")
    print("-" * 90)
    
    best_predictions = eval_df[eval_df['Abs_Error'] <= 2].sort_values('Abs_Error')
    for idx, row in best_predictions.head(15).iterrows():
        error_sign = "+" if row['Seed_Error'] > 0 else ""
        bid_type = "AQ" if row['Berth Type'] == 'Automatic' else "AL"
        print(f"{row['Team']:<30} {row['Predicted_Seed']:<10} {row['Overall Seed_Actual']:<10.0f} "
              f"{error_sign}{row['Seed_Error']:<9.0f} {bid_type:<8} {row['NETRank']:<6.0f}")
    
    # Accuracy metrics
    within_2 = len(eval_df[eval_df['Abs_Error'] <= 2])
    within_5 = len(eval_df[eval_df['Abs_Error'] <= 5])
    within_10 = len(eval_df[eval_df['Abs_Error'] <= 10])
    
    print(f"\n--- ACCURACY BREAKDOWN ---")
    print(f"Perfect predictions (0 error): {len(eval_df[eval_df['Abs_Error'] == 0])} / {len(eval_df)} "
          f"({100*len(eval_df[eval_df['Abs_Error'] == 0])/len(eval_df):.1f}%)")
    print(f"Within 2 seeds: {within_2} / {len(eval_df)} "
          f"({100*within_2/len(eval_df):.1f}%)")
    print(f"Within 5 seeds: {within_5} / {len(eval_df)} "
          f"({100*within_5/len(eval_df):.1f}%)")
    print(f"Within 10 seeds: {within_10} / {len(eval_df)} "
          f"({100*within_10/len(eval_df):.1f}%)")
    
    # Breakdown by bid type
    print(f"\n--- SEEDING ERROR BY BID TYPE ---")
    for bid_type in ['Automatic', 'At-Large']:
        subset = eval_df[eval_df['Berth Type'] == bid_type]
        if len(subset) > 0:
            mae_type = mean_absolute_error(subset['Overall Seed_Actual'], subset['Predicted_Seed'])
            rmse_type = root_mean_squared_error(subset['Overall Seed_Actual'], subset['Predicted_Seed'])
            print(f"{bid_type:<12}: {len(subset):>2} teams | MAE: {mae_type:.2f} | RMSE: {rmse_type:.2f} seeds")
    
    # Error distribution
    print(f"\n--- ERROR DISTRIBUTION ---")
    print(f"Over-seeded (predicted higher seed #): {len(eval_df[eval_df['Seed_Error'] > 0])} teams")
    print(f"Under-seeded (predicted lower seed #): {len(eval_df[eval_df['Seed_Error'] < 0])} teams")
    print(f"Mean error (signed): {eval_df['Seed_Error'].mean():.2f}")
    print(f"Median absolute error: {eval_df['Abs_Error'].median():.2f}")
    print(f"Max error: {eval_df['Abs_Error'].max():.0f} seeds")
    
else:
    print("⚠️ No teams matched between predictions and actual tournament!")

print("\n" + "="*70)
print("✓ Analysis complete! Predictions saved to 2025_submission.csv")
print("="*70)