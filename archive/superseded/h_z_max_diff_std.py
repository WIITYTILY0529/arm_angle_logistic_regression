import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde


pd.set_option('display.max_columns', None)

# [Settings and preprocessing logic kept as is]
FIXED_FLIP_RULES = {
    "ff": {"x": 1,  "z": 1},
    "ch": {"x": 1,  "z": -1},
    "cu": {"x": -1, "z": -1},
    "st": {"x": -1, "z": -1},
    "si": {"x": 1,  "z": -1},
    "fs": {"x": 1,  "z": -1},
    "sl": {"x": -1, "z": -1},
    "fc": {"x": -1, "z": 1},
}

def merge_csv_folder(folderpath):
    files = list(Path(folderpath).glob("*.csv"))
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        pitch = f.stem.split("_")[-1].lower()
        raw = pd.read_csv(f)
        needed = ["player_id", "player_name", "pitches", "arm_angle", "api_break_x_arm", "api_break_z_induced"]
        if "year" in raw.columns: needed.append("year")
        df = raw[[c for c in needed if c in raw.columns]].copy()
        df = df.rename(columns={"pitches": f"{pitch}_pitches", "arm_angle": f"{pitch}_arm_angle",
                                "api_break_x_arm": f"{pitch}_api_break_x_arm", "api_break_z_induced": f"{pitch}_api_break_z_induced"})
        dfs.append(df)
    df_merged = dfs[0]
    merge_on = ["player_id", "player_name"]
    if "year" in df_merged.columns: merge_on.append("year")
    for d in dfs[1:]: df_merged = df_merged.merge(d, on=merge_on, how="outer")
    pitch_cols = [c for c in df_merged.columns if c.endswith("_pitches")]
    df_merged[pitch_cols] = df_merged[pitch_cols].fillna(0)
    df_merged["total_pitches"] = df_merged[pitch_cols].sum(axis=1)
    for c in pitch_cols:
        pt = c.split('_')[0]
        df_merged[f"{pt}_pct"] = np.where(df_merged["total_pitches"] > 0, df_merged[c] / df_merged["total_pitches"], 0)
    return df_merged

def standardize_series(s):
    return (s - s.mean()) / s.std() if s.std() != 0 else s - s.mean()

def process_hv_pipeline(df):
    df = df.copy()
    angle_cols = [c for c in df.columns if "arm_angle" in c]
    df["avg_arm_angle"] = df[angle_cols].mean(axis=1)

    pitch_types = {c.split("_")[0] for c in df.columns if c.endswith("_api_break_x_arm")}
    for pt in pitch_types:
        x_col, z_col = f"{pt}_api_break_x_arm", f"{pt}_api_break_z_induced"
        if x_col in df.columns: df[f"{pt}_x_std"] = standardize_series(df[x_col])
        if z_col in df.columns: df[f"{pt}_z_std"] = standardize_series(df[z_col])
        if pt in FIXED_FLIP_RULES:
            if f"{pt}_x_std" in df.columns: df[f"{pt}_x_std"] *= FIXED_FLIP_RULES[pt]["x"]
            if f"{pt}_z_std" in df.columns: df[f"{pt}_z_std"] *= FIXED_FLIP_RULES[pt]["z"]
    usage_cols = [c for c in df.columns if c.endswith("_pct")]
    H, V = np.zeros(len(df)), np.zeros(len(df))
    for col in usage_cols:
        pt = col.split("_")[0]
        hx, vz = df.get(f"{pt}_x_std"), df.get(f"{pt}_z_std")
        if hx is None or vz is None: continue
        H += df[col].values * hx.fillna(0).values
        V += df[col].values * vz.fillna(0).values
    df["H_score"], df["Z_score"] = H, V
    df["H_std"], df["Z_std"] = standardize_series(df["H_score"]), standardize_series(df["Z_score"])
    # diff = df["H_std"] - df["Z_std"]
    diff = df['H_score'] - df["Z_score"]
    df["HZ_balance_final"] = diff / (1 + np.abs(diff))
    return df

BASE_PATH = str(PROJECT_ROOT / "data")
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]

all_dfs = []
for f in FOLDERS:
    df_f = merge_csv_folder(Path(BASE_PATH) / f)
    if not df_f.empty:
        df_f["matchup"] = f
        all_dfs.append(df_f)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = process_hv_pipeline(full_df)

df_analysis = full_df.dropna(subset=['avg_arm_angle', 'HZ_balance_final']).copy()

def analyze_pitch_separation_3d(df):
    """
    For each pitcher, compute the maximum x_std difference and the maximum z_std difference between pitch types
    """
    results = []
    # Collect pitch types that have an x_std column (e.g. ff_x_std, sl_x_std)
    pitch_types = {c.split("_")[0] for c in df.columns if c.endswith("_x_std")}
    
    for idx, row in df.iterrows():
        pitch_values = []
        for pt in pitch_types:
            x_val = row.get(f"{pt}_x_std")
            z_val = row.get(f"{pt}_z_std")
            pct_val = row.get(f"{pt}_pct")
            
            # Keep only pitch types with valid data
            if pd.notnull(x_val) and pd.notnull(z_val) and pct_val > 0:
                pitch_values.append({'pt': pt, 'x_std': x_val, 'z_std': z_val, 'pct': pct_val})
        
        # Only pitchers with two or more pitch types can be analyzed
        if len(pitch_values) >= 2:
            max_x_diff = 0
            max_z_diff = 0
            best_x_usage = 0
            best_z_usage = 0
            
            # Loop over every pitch-type pair to find the maximum difference
            for i in range(len(pitch_values)):
                for j in range(i + 1, len(pitch_values)):
                    # Horizontal (X) separation
                    x_diff = abs(pitch_values[i]['x_std'] - pitch_values[j]['x_std'])
                    if x_diff > max_x_diff:
                        max_x_diff = x_diff
                        best_x_usage = (pitch_values[i]['pct'] + pitch_values[j]['pct'])
                    
                    # Vertical (Z) separation
                    z_diff = abs(pitch_values[i]['z_std'] - pitch_values[j]['z_std'])
                    if z_diff > max_z_diff:
                        max_z_diff = z_diff
                        best_z_usage = (pitch_values[i]['pct'] + pitch_values[j]['pct'])
            
            results.append({
                'avg_arm_angle': row['avg_arm_angle'],
                'x_separation': max_x_diff * best_x_usage,
                'z_separation': max_z_diff * best_z_usage,
                'player_name': row['player_name']
            })
            
    return pd.DataFrame(results)

df_3d = analyze_pitch_separation_3d(df_analysis)

def run_standardized_diff_logistic_analysis(df_input):
    # 1. Compute Max Break Diff per pitcher
    results = []
    pitch_types = {c.split("_")[0] for c in df_input.columns if c.endswith("_api_break_x_arm")}
    
    for idx, row in df_input.iterrows():
        pitch_values = []
        for pt in pitch_types:
            x_val = row.get(f"{pt}_api_break_x_arm")
            z_val = row.get(f"{pt}_api_break_z_induced")
            if pd.notnull(x_val) and pd.notnull(z_val) and row.get(f"{pt}_pitches", 0) > 0:
                pitch_values.append({'x': x_val, 'z': z_val})
        
        if len(pitch_values) >= 2:
            max_h_diff = 0
            max_z_diff = 0
            for i in range(len(pitch_values)):
                for j in range(i + 1, len(pitch_values)):
                    max_h_diff = max(max_h_diff, abs(pitch_values[i]['x'] - pitch_values[j]['x']))
                    max_z_diff = max(max_z_diff, abs(pitch_values[i]['z'] - pitch_values[j]['z']))
            
            results.append({
                'avg_arm_angle': row['avg_arm_angle'],
                'max_h_break_diff': max_h_diff,
                'max_z_break_diff': max_z_diff
            })
    
    df_res = pd.DataFrame(results)

    # 2. Standardize across all pitchers (z-score)
    # So horizontal and vertical can be compared even if their units (inches, etc.) differ
    df_res['h_diff_std'] = (df_res['max_h_break_diff'] - df_res['max_h_break_diff'].mean()) / df_res['max_h_break_diff'].std()
    df_res['z_diff_std'] = (df_res['max_z_break_diff'] - df_res['max_z_break_diff'].mean()) / df_res['max_z_break_diff'].std()

    # 3. Labeling (1 if the standardized vertical difference exceeds the horizontal difference)
    df_res['binary_player_type'] = (df_res['z_diff_std'] > df_res['h_diff_std']).astype(int)

    # 4. Fit logistic regression
    df_final = df_res.dropna()
    X = df_final[['avg_arm_angle']].values
    y = df_final['binary_player_type'].values

    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)

    # Compute metrics
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    boundary = -intercept / slope
    acc = accuracy_score(y, model.predict(X))

    # 5. Visualization
    plt.figure(figsize=(12, 8))
    
    # KDE for density-based coloring
    x0, x1 = X[y == 0].flatten(), X[y == 1].flatten()
    colors = np.zeros(len(X))
    if len(x0) > 1: colors[y == 0] = gaussian_kde(x0)(x0)
    if len(x1) > 1: colors[y == 1] = gaussian_kde(x1)(x1)

    plt.scatter(X, y, c=colors, cmap='coolwarm', s=35, alpha=0.5, label='Pitchers')
    plt.axvline(boundary, color='green', linestyle='--', linewidth=3, label=f'Boundary: {boundary:.2f}°')
    
    X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    probs = model.predict_proba(X_test)[:, 1]
    plt.plot(X_test, probs, color='black', linewidth=2, alpha=0.6, label='Z-dominant Probability')

    plt.title(f"Logistic Regression: Arm Angle vs Standardized Separation Type\nAccuracy: {acc:.3f} | n: {len(df_final)}", fontsize=15)
    plt.xlabel("Average Arm Angle (degrees)")
    plt.ylabel("Probability (1.0 = Standardized Z-dominant)")
    plt.yticks([0, 1], ['H-diff Dominant (0)', 'Z-diff Dominant (1)'])
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')
    plt.show()

    print(f"Decision-boundary arm angle: {boundary:.4f} deg")
    print(f"Model accuracy: {acc:.4f}")

# Run
run_standardized_diff_logistic_analysis(df_analysis)