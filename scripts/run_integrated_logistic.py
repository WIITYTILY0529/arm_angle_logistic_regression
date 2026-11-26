import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR, FIGURES_DIR
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
    diff = df["H_std"] - df["Z_std"]
    df["HZ_balance_final"] = diff / (1 + np.abs(diff))
    return df

def run_integrated_logistic_analysis(df):
    # 1. Drop missing values from the combined data and label it
    # 1 if H_score <= Z_score (vertical-dominant), otherwise 0 (horizontal-dominant)
    analysis_df = df.dropna(subset=['avg_arm_angle', 'H_score', 'Z_score']).copy()
    analysis_df['binary_player_type'] = (analysis_df['H_score'] <= analysis_df['Z_score']).astype(int)

    X = analysis_df[['avg_arm_angle']].values
    y = analysis_df['binary_player_type'].values

    # 2. Fit logistic regression (balanced class weights)
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)

    # 3. Decision boundary and performance metrics
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    boundary = -intercept / slope
    acc = accuracy_score(y, model.predict(X))

    # 4. Visualization
    plt.figure(figsize=(12, 8))

    # KDE density-based coloring (shows how dense the data is)
    x0 = X[y == 0].flatten()
    x1 = X[y == 1].flatten()
    colors = np.zeros(len(X))
    if len(x0) > 1:
        kde0 = gaussian_kde(x0)
        colors[y == 0] = kde0(x0)
    if len(x1) > 1:
        kde1 = gaussian_kde(x1)
        colors[y == 1] = kde1(x1)

    # Scatter: horizontal (0) and vertical (1) pitchers
    plt.scatter(X, y, c=colors, cmap='coolwarm', s=35, alpha=0.5, label='Pitchers (All Matchups)')
    
    # Decision boundary (vertical dashed line)
    plt.axvline(boundary, color='green', linestyle='--', linewidth=3, 
                label=f'Integrated Boundary: {boundary:.2f}°')
    
    # Logistic sigmoid (probability) curve
    X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    probs = model.predict_proba(X_test)[:, 1]
    plt.plot(X_test, probs, color='black', linewidth=2, alpha=0.6, label='V-dom Probability')

    # Plot styling
    plt.title(f"Logistic Regression: Arm Angle vs Pitcher Type\nAccuracy: {acc:.3f} | n: {len(analysis_df)}", fontsize=15)
    plt.xlabel("Average Arm Angle (degrees)", fontsize=12)
    plt.ylabel("Probability of being V-dominant (1.0 = V-dominant)", fontsize=12)
    plt.yticks([0, 1], ['H-dominant (0)', 'V-dominant (1)'])
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')

    plt.savefig(FIGURES_DIR / 'logistic' / 'integrated_logi_reg.png')
    plt.show()

    print(f"Combined decision-boundary arm angle: {boundary:.4f} deg")
    print(f"Model accuracy: {acc:.4f}")

BASE_PATH = DATA_DIR
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]

all_dfs = []
for f in FOLDERS:
    df_f = merge_csv_folder(Path(BASE_PATH) / f)
    if not df_f.empty:
        df_f["matchup"] = f
        all_dfs.append(df_f)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = process_hv_pipeline(full_df)

# Run the combined analysis
run_integrated_logistic_analysis(full_df)
