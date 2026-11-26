from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

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

BASE_PATH = str(PROJECT_ROOT / 'data')
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]

all_dfs = []
for f in FOLDERS:
    df_f = merge_csv_folder(Path(BASE_PATH) / f)
    if not df_f.empty:
        df_f["matchup"] = f
        all_dfs.append(df_f)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = process_hv_pipeline(full_df)

# 1. Extract analysis data and drop missing values
df_reg = full_df.dropna(subset=['avg_arm_angle', 'HZ_balance_final']).copy()

X = df_reg[['avg_arm_angle']].values  # (n_samples, 1)
y = df_reg['HZ_balance_final'].values  # (n_samples,)

# 2. Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# 3. Predictions (for the regression line)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_range)

# 4. Model performance metrics
y_fitted = model.predict(X)
r2 = r2_score(y, y_fitted)
slope = model.coef_[0]
intercept = model.intercept_

# 5. Visualization
plt.figure(figsize=(12, 8))

# Scatter (actual data)
plt.scatter(X, y, s=15, alpha=0.3, color='gray', label='Pitcher Data')

# Regression line
plt.plot(X_range, y_pred, color='red', linewidth=3, label=f'Linear Fit: y = {slope:.4f}x + {intercept:.4f}')

# Reference line (H/Z balance point)
plt.axhline(0, color='blue', linestyle='--', alpha=0.5, label='HZ Balance (0)')

# Plot styling
plt.title(f"Linear Regression: Arm Angle vs HZ Balance\n(R² = {r2:.3f}, n = {len(df_reg)})", fontsize=15, fontweight='bold')
plt.xlabel("Average Arm Angle (degrees)", fontsize=12)
plt.ylabel("HZ Balance Score (H dominates > 0 > Z dominates)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

# Print results
print(f"--- Linear Regression Results ---")
print(f"Slope: {slope:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r2:.6f}")