from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

def merge_csv(folderpath):
    folder_path = Path(folderpath)
    df = pd.DataFrame()

    for file_path in folder_path.glob('*.csv'):
        pitch_class = file_path.name[7:9].lower()
        df_raw = pd.read_csv(file_path)

        shared = ['player_id', 'player_name', 'total_pitches']
        metrics = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced']

        df_adj = df_raw[shared + metrics]
        rename_map = {m: pitch_class + '_' + m for m in metrics}
        df_adj = df_adj.rename(columns=rename_map)

        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on=shared, how='outer')

    # handle nan pitches
    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)

    # calculate total pitches
    pitch_cols = [c for c in df.columns if 'pitches' in c and c not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)

    # usage %
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

def add_standardized_breaks(df):
    df = df.copy()

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

    pitch_types = sorted({col[:2].lower()
                          for col in df.columns
                          if col.endswith('api_break_x_arm')})

    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        z_col = f'{pt}_api_break_z_induced'

        # standardize
        if x_col in df.columns:
            df[f'{pt}_x_std'] = standardize_series(df[x_col])
        if z_col in df.columns:
            df[f'{pt}_z_std'] = standardize_series(df[z_col])

        # flip
        if pt in flip_rules:
            fx = flip_rules[pt]['x']
            fz = flip_rules[pt]['z']

            if f'{pt}_x_std' in df.columns:
                df[f'{pt}_x_std'] *= fx
            if f'{pt}_z_std' in df.columns:
                df[f'{pt}_z_std'] *= fz

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
                if pd.isna(h) or pd.isna(v) or pd.isna(u):
                    continue

                H_total += u * h
                V_total += u * v

        H_list.append(H_total)
        V_list.append(V_total)

    df['H_score'] = H_list
    df['V_score'] = V_list
    return df

df = merge_csv('data/l_vs_r')
df = df.apply(avg_arm_angle, axis=1)
df = add_standardized_breaks(df)
df = compute_pitcher_HV(df)

df_plot = df.dropna(subset=['H_score','V_score','avg_arm_angle']).copy()

H = df_plot['H_score'].values.reshape(-1,1)
V = df_plot['V_score'].values.reshape(-1,1)
A = df_plot['avg_arm_angle'].values.reshape(-1,1)

X = np.hstack([H, V])

reg_plane = LinearRegression().fit(X, A)
A_pred = reg_plane.predict(X)

print("Regression coefficients (H, V):", reg_plane.coef_)
print("Intercept:", reg_plane.intercept_)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(H, V, A, s=25, alpha=0.5, color='gray', label='data')

H_range = np.linspace(H.min(), H.max(), 40)
V_range = np.linspace(V.min(), V.max(), 40)
H_grid, V_grid = np.meshgrid(H_range, V_range)
A_grid = (reg_plane.intercept_ +
          reg_plane.coef_[0][0] * H_grid +
          reg_plane.coef_[0][1] * V_grid)

ax.plot_surface(H_grid, V_grid, A_grid, alpha=0.3, color='red', linewidth=0)

ax.set_xlabel('H_score')
ax.set_ylabel('V_score')
ax.set_zlabel('avg_arm_angle')
ax.set_title('3D Scatter + Regression Plane')

plt.tight_layout()
plt.show()