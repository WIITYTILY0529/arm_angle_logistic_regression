import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

FIXED_FLIP_RULES = {
    'ff': {'x': 1,  'z': 1},
    'ch': {'x': 1,  'z': -1},
    'cu': {'x': -1, 'z': -1},
    'st': {'x': -1, 'z': -1},
    'si': {'x': 1,  'z': -1},
    'fs': {'x': 1,  'z': -1},
    'sl': {'x': -1, 'z': -1},
    'fc': {'x': -1, 'z': 1},
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
    """Standardize against the full dataset"""
    return (s - s.mean()) / s.std() if s.std() != 0 else s - s.mean()

def process_global_h_score(df):
    """Global standardization over all data, then compute H_score"""
    df = df.copy()

    # 1. Average arm angle
    angle_cols = [c for c in df.columns if 'arm_angle' in c]
    df['avg_arm_angle'] = df[angle_cols].mean(axis=1)

    # 2. Global standardization and flip rules
    pitch_types = {c.split('_')[0] for c in df.columns if c.endswith('_api_break_x_arm')}
    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        if x_col in df.columns:
            # Standardize against all matchup data
            df[f'{pt}_x_std'] = standardize_series(df[x_col])
            
            # Apply flip rules
            if pt in FIXED_FLIP_RULES:
                df[f'{pt}_x_std'] *= FIXED_FLIP_RULES[pt]['x']

    # 3. Compute H_score (vectorized, usage-weighted)
    usage_cols = [c for c in df.columns if c.endswith('_pct')]
    h_sum = np.zeros(len(df))
    
    for col in usage_cols:
        pt = col.split('_')[0]
        x_std_col = f'{pt}_x_std'
        if x_std_col in df.columns:
            # usage share of this pitch type * standardized horizontal movement
            h_sum += df[col].values * df[x_std_col].fillna(0).values
            
    df['H_score'] = h_sum
    return df

# =============================================================================
# Run the GAM analysis and plot
# =============================================================================

BASE_PATH = DATA_DIR
FOLDERS = ['l_vs_l', 'l_vs_r', 'r_vs_l', 'r_vs_r']

# 1. Load data from all folders
all_dfs = []
for f in FOLDERS:
    tmp = merge_csv_folder(Path(BASE_PATH) / f)
    if not tmp.empty:
        tmp['matchup'] = f
        all_dfs.append(tmp)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = process_global_h_score(full_df)


df_analysis = full_df.dropna(subset=['avg_arm_angle', 'H_score']).copy()
X = df_analysis[['avg_arm_angle']].values
y = df_analysis['H_score'].values

gam = LinearGAM(s(0, n_splines=25)).gridsearch(X, y, lam=np.logspace(-3, 2, 20), progress=False)


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

bins = np.arange(np.floor(xx.min()/10)*10, np.ceil(xx.max()/10)*10+10, 10)
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
# plt.ylim(-1, 1)

# Set the title using the fixed R^2 value
plt.title(f"All Matchups Combined Analysis | n={len(df_analysis)} | R²={round(float(r2_val), 3)}", fontsize=14)

plt.xlabel("Average Arm Angle (degrees)")
plt.ylabel("H Score")
plt.grid(alpha=0.3)
plt.legend()
plt.show()









# # 3. Plot per matchup
# fig, axes = plt.subplots(2, 2, figsize=(20, 16))
# axes = axes.flatten()

# for ax, folder in zip(axes, FOLDERS):
#     # Filter the matchup and drop missing values
#     res_df = full_df[full_df['matchup'] == folder].dropna(subset=['avg_arm_angle', 'H_score'])
    
#     if len(res_df) < 20:
#         continue

#     X = res_df[['avg_arm_angle']].values
#     y = res_df['H_score'].values

#     # Fit the GAM
#     gam = LinearGAM(s(0, n_splines=20))
#     gam.gridsearch(X, y, lam=np.logspace(-3, 2, 20), progress=False)

#     # Predictions and confidence intervals
#     xx = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
#     yy = gam.predict(xx)
#     ci = gam.confidence_intervals(xx)

#     # Compute slopes (partial dependence)
#     pdp = gam.partial_dependence(term=0, X=xx)
#     dydx = np.gradient(pdp, xx.flatten())

#     # Visualization
#     ax.scatter(X, y, alpha=0.25, s=12, color='gray')
#     ax.plot(xx, yy, color='lightgray', linewidth=3)
#     ax.fill_between(xx.flatten(), ci[:, 0], ci[:, 1], color='gray', alpha=0.15)

#     # Plot slopes per interval
#     angle = xx.flatten()
#     bins = np.arange(np.floor(angle.min() / 5) * 5, np.ceil(angle.max() / 5) * 5 + 5, 5)

#     for i in range(len(bins) - 1):
#         mask = (angle >= bins[i]) & (angle < bins[i + 1])
#         if mask.sum() < 10: continue

#         slope = dydx[mask].mean()
#         color = 'blue' if slope > 0 else 'red'
#         ax.plot(angle[mask], yy[mask], color=color, linewidth=4)
        
#         # Add slope text
#         x_mid = (bins[i] + bins[i + 1]) / 2
#         y_mid = np.interp(x_mid, angle, yy)
#         ax.text(x_mid, y_mid, f"{slope:+.2f}", fontsize=10, ha='center', va='bottom', 
#                 color=color, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

#     ax.set_title(f"{folder.upper()} | n={len(res_df)} | EDF={gam.statistics_['edof']:.2f}", fontsize=14)
#     ax.set_xlabel('Avg Arm Angle')
#     ax.set_ylabel('H_score (Weighted Standardized X)')
#     ax.grid(alpha=0.3)

# # plt.tight_layout()
# plt.show()
