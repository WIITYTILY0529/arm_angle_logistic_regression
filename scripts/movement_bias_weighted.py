import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# =========================
# 1. Flip rules and target pitch types
# =========================
FIXED_FLIP_RULES = {
    'ff': {'x': 1,  'z': 1}, 'ch': {'x': 1,  'z': -1},
    'cu': {'x': -1, 'z': -1}, 'st': {'x': -1, 'z': -1},
    'si': {'x': 1,  'z': -1}, 'fs': {'x': 1,  'z': -1},
    'sl': {'x': -1, 'z': -1}, 'fc': {'x': -1, 'z': 1},
}
TARGET_PITCHES = ['ff', 'ch', 'cu', 'fc', 'fs', 'si', 'sl', 'st']

def standardize_series(s):
    if s.std() == 0 or np.isnan(s.std()):
        return s - s.mean()
    return (s - s.mean()) / s.std()

# =========================
# 2. Load data and compute the usage-weighted bias
# =========================
def load_and_preprocess_weighted(base_path, pitch_type):
    dfs = []
    base_path = Path(base_path)
    for cond_dir in base_path.iterdir():
        if not cond_dir.is_dir(): continue
        for file in cond_dir.glob(f'*{pitch_type}.csv'):
            df = pd.read_csv(file)
            # Without the pitches column, usage cannot be computed, so check for it
            if 'pitches' not in df.columns: continue
            
            df['pitch_type'] = pitch_type
            dfs.append(df)
    
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 1) Usage of this pitch type relative to total pitches (assumes total_pitches is in the file)
    # If it is not in the file, use the logic from the merge_csv step
    if 'total_pitches' not in df.columns:
        # Total pitches must be defined for the dataset (here, the maximum within each file is used as an example)
        df['usage'] = df['pitches'] / df['pitches'].max() # temporary
    else:
        df['usage'] = df['pitches'] / df['total_pitches']

    # 2) Standardize movement
    df['h_std_raw'] = standardize_series(df['api_break_x_arm'])
    df['v_std_raw'] = standardize_series(df['api_break_z_induced'])
    
    # 3) Apply flip rules
    rule = FIXED_FLIP_RULES.get(pitch_type, {'x': 1, 'z': 1})
    df['h_std_flipped'] = df['h_std_raw'] * rule['x']
    df['v_std_flipped'] = df['v_std_raw'] * rule['z']
    
    # 4) [Key] Multiply by usage
    df['h_weighted'] = df['h_std_flipped'] * df['usage']
    df['v_weighted'] = df['v_std_flipped'] * df['usage']
    
    # 5) Compute the weighted movement bias
    diff = df['h_weighted'] - df['v_weighted']
    df['Movement_Bias'] = diff / (1 + np.abs(diff))
    
    return df

# =========================
# 3. Plot function (GAM)
# =========================
def plot_gam_sub(df, ax, pitch_name):
    df = df.dropna(subset=['arm_angle', 'Movement_Bias'])
    if len(df) < 15:
        ax.text(0.5, 0.5, f"Low Data: {pitch_name.upper()}", ha='center')
        return

    X = df[['arm_angle']].values
    y = df['Movement_Bias'].values

    # Fit the model
    gam = LinearGAM(s(0, n_splines=20)).gridsearch(X, y, progress=False)
    
    r2 = gam.statistics_.get('pseudo_r2', 0)
    if isinstance(r2, dict): r2 = list(r2.values())[0]

    xx = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    yy = gam.predict(xx)
    ci = gam.confidence_intervals(xx)
    dydx = np.gradient(gam.partial_dependence(term=0, X=xx), xx.flatten())

    ax.scatter(X, y, s=10, alpha=0.2, color='gray')
    ax.plot(xx, yy, color='black', linewidth=2)
    ax.fill_between(xx.flatten(), ci[:, 0], ci[:, 1], alpha=0.15, color='blue')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    # Show slopes every 10 degrees
    bins = np.arange(np.floor(X.min()/5)*5, np.ceil(X.max()/5)*5+5, 5)
    for i in range(len(bins)-1):
        mask = (xx.flatten() >= bins[i]) & (xx.flatten() < bins[i+1])
        if mask.sum() < 5: continue
        slope = dydx[mask].mean()
        color = 'blue' if slope > 0 else 'red'
        ax.plot(xx[mask], yy[mask], color=color, linewidth=4)
        
        x_mid = (bins[i] + bins[i+1]) / 2
        y_mid = np.interp(x_mid, xx.flatten(), yy)
        # ax.text(x_mid, y_mid + 0.05, f"{slope:+.2f}", color=color, 
        #         fontsize=9, fontweight='bold', ha='center',
        #         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.7))

    ax.set_title(f"{pitch_name.upper()} Bias | R²: {r2:.3f}", fontsize=11)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.2)

# =========================
# 4. Main
# =========================
BASE_PATH = DATA_DIR

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for i, pt in enumerate(TARGET_PITCHES):
    data = load_and_preprocess_weighted(BASE_PATH, pt)
    if not data.empty:
        plot_gam_sub(data, axes[i], pt)
    else:
        axes[i].text(0.5, 0.5, f"No Data: {pt.upper()}", ha='center')

plt.suptitle("Movement Bias GAM Analysis per Pitch type", fontsize=22)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("pitchtype gam(1).png")
plt.show()
