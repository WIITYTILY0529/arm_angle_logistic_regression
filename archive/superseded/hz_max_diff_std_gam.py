import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 1. Settings
pd.set_option('display.max_columns', None)

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
        df = df.rename(columns={
            "pitches": f"{pitch}_pitches", 
            "arm_angle": f"{pitch}_arm_angle",
            "api_break_x_arm": f"{pitch}_api_break_x_arm", 
            "api_break_z_induced": f"{pitch}_api_break_z_induced"
        })
        dfs.append(df)
    
    df_merged = dfs[0]
    merge_on = ["player_id", "player_name"]
    if "year" in df_merged.columns: merge_on.append("year")
    for d in dfs[1:]: df_merged = df_merged.merge(d, on=merge_on, how="outer")
    
    pitch_cols = [c for c in df_merged.columns if c.endswith("_pitches")]
    df_merged[pitch_cols] = df_merged[pitch_cols].fillna(0)
    df_merged["total_pitches"] = df_merged[pitch_cols].sum(axis=1)
    return df_merged

# 2. Load and preprocess data
BASE_PATH = str(PROJECT_ROOT / "data")
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]

all_dfs = []
for f in FOLDERS:
    df_f = merge_csv_folder(Path(BASE_PATH) / f)
    if not df_f.empty:
        df_f["matchup"] = f
        all_dfs.append(df_f)

full_df = pd.concat(all_dfs, ignore_index=True)

# 3. Build the target from Max Break Difference (logic from the first script)
def prepare_gam_data(df_input):
    results = []
    pitch_types = {c.split("_")[0] for c in df_input.columns if c.endswith("_api_break_x_arm")}
    
    for idx, row in df_input.iterrows():
        pitch_values = []
        for pt in pitch_types:
            x_val = row.get(f"{pt}_api_break_x_arm")
            z_val = row.get(f"{pt}_api_break_z_induced")
            # Average arm angle (mean of the angles present in the row)
            angle_cols = [c for c in df_input.columns if "arm_angle" in c]
            avg_angle = row[angle_cols].mean()
            
            if pd.notnull(x_val) and pd.notnull(z_val) and row.get(f"{pt}_pitches", 0) > 0:
                pitch_values.append({'x': x_val, 'z': z_val})
        
        # Only rows with two or more pitch types and an arm angle
        if len(pitch_values) >= 2 and pd.notnull(avg_angle):
            max_h_diff = 0
            max_z_diff = 0
            for i in range(len(pitch_values)):
                for j in range(i + 1, len(pitch_values)):
                    max_h_diff = max(max_h_diff, abs(pitch_values[i]['x'] - pitch_values[j]['x']))
                    max_z_diff = max(max_z_diff, abs(pitch_values[i]['z'] - pitch_values[j]['z']))
            
            results.append({
                'avg_arm_angle': avg_angle,
                'max_h_break_diff': max_h_diff,
                'max_z_break_diff': max_z_diff
            })
    
    df_res = pd.DataFrame(results)
    
    # Standardize (z-score)
    df_res['h_diff_std'] = (df_res['max_h_break_diff'] - df_res['max_h_break_diff'].mean()) / df_res['max_h_break_diff'].std()
    df_res['z_diff_std'] = (df_res['max_z_break_diff'] - df_res['max_z_break_diff'].mean()) / df_res['max_z_break_diff'].std()
    
    # [Key] Apply softsign to z_std - h_std
    # Positive: vertical separation dominates; negative: horizontal separation dominates
    diff = df_res['h_diff_std'] - df_res['z_diff_std']
    df_res['separation_bias_softsign'] = diff / (1 + np.abs(diff))
    
    return df_res

df_gam_ready = prepare_gam_data(full_df)

# 4. Run the GAM analysis
X = df_gam_ready[['avg_arm_angle']].values
y = df_gam_ready['separation_bias_softsign'].values

# Grid search for the best lambda
gam = LinearGAM(s(0, n_splines=25)).gridsearch(X, y, lam=np.logspace(-3, 2, 20), progress=False)

# Extract result metrics
p_value = gam.statistics_['p_values'][0]
r2_val = gam.statistics_.get('pseudo_r2', 0)
if isinstance(r2_val, dict):
    r2_val = list(r2_val.values())[0]

# 5. Visualization
xx = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
yy = gam.predict(xx)
ci = gam.confidence_intervals(xx)
dydx = np.gradient(gam.partial_dependence(term=0, X=xx), xx.flatten())

plt.figure(figsize=(12, 8))
plt.scatter(X, y, s=15, alpha=0.2, color='gray', label='Pitchers (Max Separation)')
plt.plot(xx, yy, color='black', linewidth=3, label='GAM Trend')
plt.fill_between(xx.flatten(), ci[:,0], ci[:,1], alpha=0.15, color='gray')

# Show slopes per interval (every 5 degrees)
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

plt.axhline(0, color='darkgreen', linestyle='--', alpha=0.5, label='Balance Line')
plt.ylim(-1.1, 1.1)

plt.title(f"Arm Angle vs Standardized Separation Bias (Softsign) | n={len(df_gam_ready)} | R²={float(r2_val):.3}", fontsize=14)
plt.xlabel("Average Arm Angle (degrees)")
plt.ylabel("Separation Bias (<- H-dominant | Z-dominant ->)")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('pngs/hzmaxdiffstdgam.png')
plt.show()

print(f"GAM analysis done. n={len(df_gam_ready)}, R2={r2_val:.4f}, p-value={p_value:.10f}")