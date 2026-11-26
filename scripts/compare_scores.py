import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR

# 1. Fixed flip rules
FIXED_FLIP_RULES = {
    'ff': {'x': 1,  'z': 1}, 'ch': {'x': 1,  'z': -1},
    'cu': {'x': -1, 'z': -1}, 'st': {'x': -1, 'z': -1},
    'si': {'x': 1,  'z': -1}, 'fs': {'x': 1,  'z': -1},
    'sl': {'x': -1, 'z': -1}, 'fc': {'x': -1, 'z': 1},
}

def standardize_series(s):
    if s.std() == 0 or np.isnan(s.std()): return s - s.mean()
    return (s - s.mean()) / s.std()

def process_matchup_data(folder_path):
    folder = Path(folder_path)
    files = list(folder.glob('*.csv'))
    if not files: return pd.DataFrame()

    dfs = []
    for f in files:
        # Extract the pitch type from the file name (e.g. statcast_ff.csv -> ff)
        pt = f.stem.split('_')[-1].lower()
        df_raw = pd.read_csv(f)
        
        needed = ['player_id', 'player_name', 'year', 'pitches', 'api_break_x_arm', 'api_break_z_induced']
        df_sub = df_raw[[c for c in needed if c in df_raw.columns]].copy()
        
        # Rename columns per pitch type
        df_sub = df_sub.rename(columns={
            'pitches': f'{pt}_pitches',
            'api_break_x_arm': f'{pt}_x',
            'api_break_z_induced': f'{pt}_z'
        })
        dfs.append(df_sub)

    # Merge on player/year
    df_merged = dfs[0]
    for d in dfs[1:]:
        df_merged = pd.merge(df_merged, d, on=['player_id', 'player_name', 'year'], how='outer')

    # List of pitch types
    pitch_types = [c.split('_')[0] for c in df_merged.columns if c.endswith('_pitches')]
    
    # Total pitches and usage shares
    pitch_count_cols = [f'{pt}_pitches' for pt in pitch_types]
    df_merged[pitch_count_cols] = df_merged[pitch_count_cols].fillna(0)
    df_merged['total_pitches'] = df_merged[pitch_count_cols].sum(axis=1)

    # Compute H/Z scores (usage-weighted)
    df_merged['H_weighted_sum'] = 0.0
    df_merged['V_weighted_sum'] = 0.0

    for pt in pitch_types:
        if f'{pt}_x' not in df_merged.columns: continue
        
        # Standardize and apply flips
        h_std = standardize_series(df_merged[f'{pt}_x']) * FIXED_FLIP_RULES.get(pt, {'x':1})['x']
        v_std = standardize_series(df_merged[f'{pt}_z']) * FIXED_FLIP_RULES.get(pt, {'z':1})['z']
        usage = df_merged[f'{pt}_pitches'] / df_merged['total_pitches']
        
        df_merged['H_weighted_sum'] += h_std.fillna(0) * usage
        df_merged['V_weighted_sum'] += v_std.fillna(0) * usage

    # [Movement bias] (softsign applied)
    diff = df_merged['H_weighted_sum'] - df_merged['V_weighted_sum']
    df_merged['Movement_Bias'] = diff / (1 + np.abs(diff))
    
    return df_merged[['player_id', 'player_name', 'year', 'Movement_Bias', 'H_weighted_sum', 'V_weighted_sum']]

def find_polar_flip_pitchers(base_path, folder_a, folder_b):
    df_a = process_matchup_data(Path(base_path) / folder_a)
    df_b = process_matchup_data(Path(base_path) / folder_b)

    if df_a.empty or df_b.empty: return pd.DataFrame()

    merged = pd.merge(df_a, df_b, on=['player_id', 'player_name', 'year'], suffixes=(f'_{folder_a}', f'_{folder_b}'))

    # Condition for the bias sign to change (positive <-> negative)
    # Movement_Bias > 0 means H-dominant, < 0 means V-dominant
    bias_a = f'Movement_Bias_{folder_a}'
    bias_b = f'Movement_Bias_{folder_b}'
    
    flip_condition = (merged[bias_a] * merged[bias_b] < 0)
    
    result = merged[flip_condition].copy()
    
    # Size of the change
    result['Bias_Diff_Abs'] = np.abs(result[bias_a] - result[bias_b])
    return result.sort_values(by='Bias_Diff_Abs', ascending=False)

# Run
BASE_PATH = DATA_DIR


def main():
    # Example: RHP tendency vs RHB (r_vs_r) vs. vs LHB (r_vs_l)
    flip_pitchers = find_polar_flip_pitchers(BASE_PATH, 'r_vs_r', 'r_vs_l')

    print(f"--- Pitchers whose movement tendency (H/V) flips by matchup ---")
    if not flip_pitchers.empty:
        print(flip_pitchers[['player_name', 'year', 'Movement_Bias_r_vs_r', 'Movement_Bias_r_vs_l', 'Bias_Diff_Abs']].head(20))
        print(f"\n{len(flip_pitchers)} cases found")
    else:
        print("No pitchers with a flipped tendency.")


if __name__ == "__main__":
    main()
