from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# ============================================================
# 1. Shared functions (kept as in the original code)
# ============================================================

def merge_csv(folderpath):
    folder_path = Path(folderpath)
    df = pd.DataFrame()

    files = list(folder_path.glob('*.csv'))
    if not files:
        print(f"Warning: No CSV files found in {folderpath}")
        return df

    for file_path in files:
        pitch_class = file_path.name[7:9].lower()
        df_raw = pd.read_csv(file_path)

        shared = ['player_id', 'player_name', 'total_pitches']
        metrics = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced']
        metrics = [m for m in metrics if m in df_raw.columns]

        df_adj = df_raw[shared + metrics]
        rename_map = {m: pitch_class + '_' + m for m in metrics}
        df_adj = df_adj.rename(columns=rename_map)

        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on=shared, how='outer')

    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)

    pitch_cols = [c for c in df.columns if 'pitches' in c and c not in ['total_pitches']]
    if pitch_cols:
        df['tot_pitches'] = df[pitch_cols].sum(axis=1)
        for col in df.columns:
            if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
                df[f'{col[:2]} %'] = df[col] / df['tot_pitches']

    return df


def avg_arm_angle(row):
    cols = [c for c in row.index if 'arm_angle' in c]
    vals = row[cols].dropna()
    row['avg_arm_angle'] = vals.mean() if len(vals) else np.nan
    return row


def standardize_series(s):
    std = s.std()
    mean = s.mean()
    if std == 0 or np.isnan(std):
        return s - mean
    return (s - mean) / std


def add_standardized_breaks_dynamic(df, folder_name):
    df = df.copy()
    folder_lower = folder_name.lower()

    if folder_lower.startswith('l_vs'):
        flip_rules = {
            'ff': {'x': 1,  'z': -1},
            'ch': {'x': 1,  'z': 1},
            'cu': {'x': -1, 'z': 1},
            'st': {'x': -1, 'z': 1},
            'si': {'x': 1,  'z': 1},
            'fs': {'x': 1,  'z': 1},
            'sl': {'x': -1, 'z': 1},
            'fc': {'x': -1, 'z': -1},
        }
    else:
        flip_rules = {
            'ff': {'x': 1,  'z': 1},
            'ch': {'x': 1,  'z': -1},
            'cu': {'x': -1, 'z': 1},
            'st': {'x': -1, 'z': 1},
            'si': {'x': 1,  'z': 1},
            'fs': {'x': 1,  'z': 1},
            'sl': {'x': -1, 'z': 1},
            'fc': {'x': -1, 'z': -1},
        }

    pitch_types = sorted({col[:2].lower() for col in df.columns if col.endswith('api_break_x_arm')})

    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        z_col = f'{pt}_api_break_z_induced'

        if x_col in df.columns: df[f'{pt}_x_std'] = standardize_series(df[x_col])
        if z_col in df.columns: df[f'{pt}_z_std'] = standardize_series(df[z_col])

        if pt in flip_rules:
            fx = flip_rules[pt]['x']
            fz = flip_rules[pt]['z']
            if f'{pt}_x_std' in df.columns: df[f'{pt}_x_std'] *= fx
            if f'{pt}_z_std' in df.columns: df[f'{pt}_z_std'] *= fz

    return df


def compute_pitcher_HV(df):
    H_list = []
    V_list = []
    usage_cols = [c for c in df.columns if c.endswith('%')]

    for idx, row in df.iterrows():
        H_total = 0.0
        V_total = 0.0
        for col in usage_cols:
            pitch = col[:2]
            u = row[col]
            h_col = f'{pitch}_x_std'
            v_col = f'{pitch}_z_std'

            if (h_col in df.columns) and (v_col in df.columns):
                h = row[h_col]
                v = row[v_col]
                if pd.isna(h) or pd.isna(v) or pd.isna(u): continue
                H_total += u * h
                V_total += u * v
        H_list.append(H_total)
        V_list.append(V_total)

    df['H_score'] = H_list
    df['V_score'] = V_list
    return df


# ============================================================
# 2. Compute HV_angle
# ============================================================

def add_HV_angle(df):
    df = df.copy()
    mask = ~(df[['H_score', 'V_score']].isna().any(axis=1))

    HV = df.loc[mask, ['H_score', 'V_score']].values
    H = HV[:, 0]
    V = HV[:, 1]

    angles = np.arctan2(V, H)

    out = np.full(df.shape[0], np.nan)
    out[mask] = angles
    df['HV_angle'] = out

    return df

# ============================================================
# 3. Run the analysis
# ============================================================

def run_analysis_angle(df):
    df_final = df.dropna(subset=['HV_angle', 'avg_arm_angle']).reset_index(drop=True)
    if len(df_final) < 10:
        return None

    X = df_final[['HV_angle']].values
    y = df_final['avg_arm_angle'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gam = LinearGAM(s(0, n_splines=20))
    try:
        lam_grid = np.logspace(-2, 2, 20)
        gam.gridsearch(X_train, y_train, lam=lam_grid, progress=False)
    except:
        gam.fit(X_train, y_train)

    y_pred = gam.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    xx = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    yy = gam.predict(xx)

    return {
        'X': X,
        'y': y,
        'xx': xx,
        'yy': yy,
        'r2': r2,
        'rmse': rmse,
        'count': len(df_final)
    }


# ============================================================
# 4. Main: Movement angle -> Arm Angle (2x2)
# ============================================================

BASE_PATH = str(PROJECT_ROOT / 'data')
FOLDERS = ['l_vs_l', 'l_vs_r', 'r_vs_l', 'r_vs_r']

fig, axes = plt.subplots(2, 2, figsize=(24, 16))
axes = axes.flatten()

for i, folder in enumerate(FOLDERS):
    ax = axes[i]
    full_path = Path(BASE_PATH) / folder

    df = merge_csv(full_path)
    if df.empty:
        ax.text(0.5, 0.5, f"No Data\n({folder})", ha='center', va='center')
        continue

    df = df.apply(avg_arm_angle, axis=1)
    df = add_standardized_breaks_dynamic(df, folder)
    df = compute_pitcher_HV(df)
    df = add_HV_angle(df)

    result = run_analysis_angle(df)

    if result:
        ax.scatter(result['X'], result['y'], alpha=0.3, color='gray', s=15, label='Data')
        ax.plot(result['xx'], result['yy'], color='red', linewidth=2.5, label='GAM fit')

        ax.text(
            0.02, 0.02,
            f"{folder.upper()} (n={result['count']})\nR²={result['r2']:.3f}, RMSE={result['rmse']:.3f}",
            transform=ax.transAxes,
            fontsize=28,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        ax.set_xlabel('Movement Angle (HV_angle)', fontsize=20)
        ax.set_ylabel('Avg Arm Angle', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=18)
    else:
        ax.text(0.5, 0.5, f"No Data or Error\n({folder})",
                ha='center', va='center', transform=ax.transAxes)

plt.suptitle('GAM Regression: Movement Angle → Arm Angle', fontsize=32, y=0.99)
plt.tight_layout()
# plt.savefig(f'{BASE_PATH}/pngs/movement_angle_to_arm_angle.png')
plt.show()