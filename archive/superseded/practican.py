import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 1. Settings and data-loading logic
pd.set_option('display.max_columns', None)

def merge_csv_folder(folderpath, matchup_name):
    files = list(Path(folderpath).glob("*.csv"))
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        pitch = f.stem.split("_")[-1].lower()
        raw = pd.read_csv(f)
        needed = ["player_id", "player_name", "pitches", "arm_angle", "api_break_x_arm", "api_break_z_induced"]
        if "year" in raw.columns: needed.append("year")
        df = raw[[c for c in needed if c in raw.columns]].copy()
        
        rename_dict = {col: f"{pitch}_{col}" for col in df.columns if col not in ["player_id", "player_name", "year"]}
        df = df.rename(columns=rename_dict)
        dfs.append(df)
    
    df_merged = dfs[0]
    merge_on = ["player_id", "player_name"]
    if "year" in df_merged.columns: merge_on.append("year")
    for d in dfs[1:]: df_merged = df_merged.merge(d, on=merge_on, how="outer")
    
    df_merged["matchup"] = matchup_name
    pitch_cols = [c for c in df_merged.columns if c.endswith("_pitches")]
    df_merged[pitch_cols] = df_merged[pitch_cols].fillna(0)
    df_merged["total_pitches"] = df_merged[pitch_cols].sum(axis=1)
    return df_merged

def standardize_series(s):
    return (s - s.mean()) / s.std() if s.std() != 0 else s - s.mean()

def process_scores(df):
    df = df.copy()
    # Average arm angle
    angle_cols = [c for c in df.columns if "arm_angle" in c]
    df["avg_arm_angle"] = df[angle_cols].mean(axis=1)
    
    # Usage-weighted scores (H_score, Z_score)
    pitch_types = {c.split("_")[0] for c in df.columns if c.endswith("_api_break_x_arm")}
    H, V = np.zeros(len(df)), np.zeros(len(df))
    for pt in pitch_types:
        x_std = standardize_series(df[f"{pt}_api_break_x_arm"])
        z_std = standardize_series(df[f"{pt}_api_break_z_induced"])
        pct = np.where(df["total_pitches"] > 0, df[f"{pt}_pitches"] / df["total_pitches"], 0)
        H += pct * x_std.fillna(0).values
        V += pct * z_std.fillna(0).values
    df["H_score"], df["Z_score"] = H, V
    return df

# 2. Load all data
BASE_PATH = str(PROJECT_ROOT / "data")
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]

all_dfs = []
for f in FOLDERS:
    df_f = merge_csv_folder(Path(BASE_PATH) / f, f)
    if not df_f.empty: all_dfs.append(df_f)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = process_scores(full_df)

# 3. Extract ONLY IN 1 & ONLY IN 2
def find_exact_n_difference(df_full):
    pitch_types = {c.split("_")[0] for c in df_full.columns if c.endswith("_api_break_x_arm")}
    
    # --- Filtering for analysis 1 (Standardized Max Diff) ---
    idx1 = []
    for idx, row in df_full.iterrows():
        p_vals = []
        for pt in pitch_types:
            # Count pitch types that have pitch counts and movement data
            if row.get(f"{pt}_pitches", 0) > 0 and pd.notnull(row.get(f"{pt}_api_break_x_arm")):
                p_vals.append(row.get(f"{pt}_api_break_x_arm"))
        
        # Condition: two or more pitch types AND an arm angle
        if len(p_vals) >= 2 and pd.notnull(row['avg_arm_angle']):
            idx1.append(idx)
    
    # --- Filtering for analysis 2 (Integrated Score) ---
    # Condition: arm angle present AND H/Z scores present (same as the dropna in the original code)
    df2 = df_full.dropna(subset=['avg_arm_angle', 'H_score', 'Z_score']).copy()
    idx2 = list(df2.index)

    # Set comparison
    set1, set2 = set(idx1), set(idx2)
    only_in_1 = list(set1 - set2)
    only_in_2 = list(set2 - set1)

    # 4. Print results
    print(f"\n" + "="*80)
    print(f"🔍 Sample comparison results")
    print(f"Total rows loaded: {len(df_full)}")
    print(f"Analysis 1 sample size (Max Diff): {len(set1)}")
    print(f"Analysis 2 sample size (Score): {len(set2)}")
    print(f"Sample size difference: {abs(len(set1) - len(set2))}")
    print("="*80)

    if only_in_1:
        print(f"\n✅ [Only in Analysis 1] ({len(only_in_1)} rows)")
        print("These pitchers are in analysis 1 but not in analysis 2 (Score).")
        print(df_full.loc[only_in_1, ['year', 'matchup', 'player_name', 'avg_arm_angle', 'H_score', 'Z_score']].head(10))
    else:
        print("\n✅ No pitchers appear only in Analysis 1.")

    if only_in_2:
        print(f"\n✅ [Only in Analysis 2] ({len(only_in_2)} rows)")
        print("These pitchers are in analysis 2 but not in analysis 1 (Max Diff).")
        # Print extra information to check the number of pitch types
        temp_df = df_full.loc[only_in_2].copy()
        print(temp_df[['year', 'matchup', 'player_name', 'avg_arm_angle']].head(10))
    else:
        print("\n✅ No pitchers appear only in Analysis 2.")

    return only_in_1, only_in_2

# Run
o1, o2 = find_exact_n_difference(full_df)