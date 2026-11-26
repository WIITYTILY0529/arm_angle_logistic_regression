import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR
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
    diff = df['H_score'] - df["Z_score"]
    df["HZ_balance_final"] = diff / (1 + np.abs(diff))
    return df

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

df_analysis = full_df.dropna(subset=['avg_arm_angle', 'HZ_balance_final']).copy()
X = df_analysis[['avg_arm_angle']].values
y = df_analysis['HZ_balance_final'].values

gam = LinearGAM(s(0, n_splines=25)).gridsearch(X, y, lam=np.logspace(-3, 2, 20), progress=False)
gam.summary()

# Get the p-value for the arm-angle term only
p_value = gam.statistics_['p_values'][0]
print(f"\nArm Angle Term p-value: {p_value:.10f}")

# R^2 extraction fixed (handles dict results)
r2_val = gam.statistics_.get('pseudo_r2', 0)
if isinstance(r2_val, dict):
    r2_val = list(r2_val.values())[0]

xx = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
yy = gam.predict(xx)
ci = gam.confidence_intervals(xx)
dydx = np.gradient(gam.partial_dependence(term=0, X=xx), xx.flatten())

plt.figure(figsize=(12, 8))
plt.scatter(X, y, s=15, alpha=0.2, color='gray', label='Pitchers (All Matchups)')
plt.plot(xx, yy, color='black', linewidth=3, label='GAM Trend')
plt.fill_between(xx.flatten(), ci[:,0], ci[:,1], alpha=0.15, color='gray')

bins = np.arange(np.floor(xx.min()/5)*5, np.ceil(xx.max()/5)*5+5, 5)
for i in range(len(bins)-1):
    m = (xx.flatten() >= bins[i]) & (xx.flatten() < bins[i+1])
    if m.sum() < 5: continue
    seg_x, seg_y = xx.flatten()[m], yy[m]
    slope = dydx[m].mean()
    c = 'blue' if slope > 0 else 'red'
    plt.plot(seg_x, seg_y, color=c, linewidth=5)
    plt.text((bins[i]+bins[i+1])/2, np.interp((bins[i]+bins[i+1])/2, xx.flatten(), yy), 
             f"{slope:+.3f}", color=c, fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round', fc='white', ec=c, alpha=0.8))

plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.ylim(-1, 1)

# Set the title using the fixed R^2 value
plt.title(f"Arm angle and Movement Bias GAM| n={len(df_analysis)} | R²={round(float(r2_val), 3)}", fontsize=14)

plt.xlabel("Average Arm Angle (degrees)")
plt.ylabel("Movement Bias")
plt.grid(alpha=0.3)
plt.legend()
# plt.savefig('softsign gam.png')
# plt.show()

# Check the basis of the design matrix
X_basis = gam._modelmat(X)

print("Basis matrix shape:", X_basis.shape)

# Knot positions are stored inside the term


# term summary
# print(term)

# Check the knots (recommended)
term = gam.terms[0]

print(term.info.keys())
