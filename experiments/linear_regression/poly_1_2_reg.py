from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from pathlib import Path

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

# [Assumes data loading and process_hv_pipeline preprocessing are done]
df_reg = full_df.dropna(subset=['avg_arm_angle', 'HZ_balance_final']).copy()
X = df_reg[['avg_arm_angle']].values
y = df_reg['HZ_balance_final'].values

# Range for predictions
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Plot layout (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# ---------------------------------------------------------
# 1. Linear Regression (degree 1)
# ---------------------------------------------------------
model1 = LinearRegression()
model1.fit(X, y)
y_pred1 = model1.predict(X_range)
r2_1 = r2_score(y, model1.predict(X))

axes[0].scatter(X, y, s=15, alpha=0.2, color='gray')
axes[0].plot(X_range, y_pred1, color='red', linewidth=3, 
             label=f'Linear: R²={r2_1:.3f}\ny={model1.coef_[0]:.4f}x + {model1.intercept_:.4f}')
axes[0].set_title("1st Order Linear Regression", fontsize=14, fontweight='bold')

# ---------------------------------------------------------
# 2. Quadratic Regression (degree 2)
# ---------------------------------------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_range_poly = poly.transform(X_range)

model2 = LinearRegression()
model2.fit(X_poly, y)
y_pred2 = model2.predict(X_range_poly)
r2_2 = r2_score(y, model2.predict(X_poly))

# Extract coefficients (y = ax^2 + bx + c)
c, b, a = model2.intercept_, model2.coef_[1], model2.coef_[2]

axes[1].scatter(X, y, s=15, alpha=0.2, color='gray')
axes[1].plot(X_range, y_pred2, color='blue', linewidth=3, 
             label=f'Quadratic: R²={r2_2:.3f}\ny={a:.5f}x² + {b:.4f}x + {c:.2f}')
axes[1].set_title("2nd Order Polynomial Regression", fontsize=14, fontweight='bold')

# Shared settings
for ax in axes:
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Average Arm Angle (degrees)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

axes[0].set_ylabel("Movement Bias")

plt.suptitle(f"Arm Angle vs Movement Bias: Regression Comparison (n={len(df_reg)})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('1,2 reg.png')
plt.show()

# Print detailed values
print(f"[Linear]    R2: {r2_1:.6f}")
print(f"[Quadratic] R2: {r2_2:.6f}")