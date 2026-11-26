from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

##########################################################
# Libraries
##########################################################
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

##########################################################
# 1. Merge CSVs
##########################################################
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

    # Handle missing pitches
    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)

    # Compute tot_pitches
    pitch_cols = [c for c in df.columns if 'pitches' in c and c not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)

    # usage %
    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[:2]} %'] = df[col] / df['tot_pitches']

    return df

##########################################################
# 2. Compute average arm angle
##########################################################
def avg_arm_angle(row):
    cols = [c for c in row.index if 'arm_angle' in c]
    vals = row[cols].dropna()
    row['avg_arm_angle'] = vals.mean() if len(vals) else np.nan
    return row

##########################################################
# 3. Standardize + flip
##########################################################
def standardize_series(s):
    std = s.std()
    mean = s.mean()
    if std == 0 or np.isnan(std):
        return s - mean
    return (s - mean) / std

def add_standardized_breaks(df):
    df = df.copy()

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

    pitch_types = sorted({col[:2].lower()
                          for col in df.columns
                          if col.endswith('api_break_x_arm')})

    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        z_col = f'{pt}_api_break_z_induced'

        # Standardize
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

##########################################################
# 4. Compute HV
##########################################################
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

def add_HV_angle(df):
    df = df.copy()
    # Compute only when both H_score and V_score exist
    mask = ~(df[['H_score', 'V_score']].isna().any(axis=1))
    
    HV = df.loc[mask, ['H_score', 'V_score']].values
    H = HV[:, 0]
    V = HV[:, 1]

    angles = np.arctan2(V, H)   # -pi ~ +pi

    out = np.full(df.shape[0], np.nan)
    out[mask] = angles
    df['HV_angle'] = out

    return df

##########################################################
# 5. PCA
##########################################################
def HV_to_1D_via_PCA(df):
    hv = df[['H_score', 'V_score']].copy()

    mask = ~(hv.isna().any(axis=1))
    hv_clean = hv[mask].values

    if hv_clean.shape[0] < 2:
        raise RuntimeError("Not enough valid H/V samples.")

    scaler = StandardScaler()
    hv_std = scaler.fit_transform(hv_clean)

    pca = PCA(n_components=1)
    hv_1d_temp = pca.fit_transform(hv_std)

    if pca.components_[0, 0] < 0:
        print("Note: flipping the PCA sign (for consistency).")
        pca.components_[0] *= -1     # flip the direction of the component (axis)
        hv_1d_temp *= -1

    hv_1d = hv_1d_temp.flatten()

    out = np.full(df.shape[0], np.nan)
    out[mask] = hv_1d
    df['HV_1D'] = out

    print("PCA components:", pca.components_)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    return df, scaler, pca

##########################################################
# 6. Run
##########################################################
df = merge_csv(str(PROJECT_ROOT / 'data/r_vs_l'))
df = df.apply(avg_arm_angle, axis=1)
df = add_standardized_breaks(df)
df = compute_pitcher_HV(df)
df = add_HV_angle(df)   
##########################################################
# Debug 1: list the standardized columns
##########################################################
print("x_std, z_std columns:")
print([c for c in df.columns if c.endswith('_x_std') or c.endswith('_z_std')])

##########################################################
# Debug 2: HV sanity check
##########################################################
print("HV head:")
print(df[['H_score','V_score']].head())
print("NaN counts in H_score/V_score:", df[['H_score','V_score']].isna().sum())

##########################################################
# PCA
##########################################################
df, scaler_HV, pca_HV = HV_to_1D_via_PCA(df)

##########################################################
# Debug 3: number of NaNs in the PCA input
##########################################################
print("HV rows with NaN:", df[['H_score','V_score']].isna().any(axis=1).sum())

df_final = df.dropna(subset=['HV_1D', 'avg_arm_angle']).reset_index(drop=True)

##########################################################
# 7. GAM regression
##########################################################
X = df_final[['avg_arm_angle']].values
y = df_final['HV_1D'].values
# X = df_final['HV_angle'].values.reshape(-1, 1)
# y = df_final['avg_arm_angle'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gam = LinearGAM(s(0, n_splines=20))
lam_grid = np.logspace(-2, 2, 20)
gam.gridsearch(X_train, y_train, lam=lam_grid)

y_pred = gam.predict(X_test)

##########################################################
# Debug 4: check the RSS components
##########################################################
rss_model = np.sum((y_test - y_pred)**2)
rss_null = np.sum((y_test - y_test.mean())**2)
print("rss_model:", rss_model)
print("rss_null:", rss_null)

pseudo_r2 = 1 - rss_model / rss_null if rss_null != 0 else np.nan

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Test R²: {r2:.4f}')
print(f'Pseudo R²: {pseudo_r2:.4f}')
print(f'RMSE: {rmse:.4f}')

##########################################################
# Visualization
##########################################################
xx = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
yy = gam.predict(xx)

plt.figure(figsize=(10, 6))
plt.scatter(X.flatten(), y, alpha=0.35, color='gray', s=20, label='data')
plt.plot(xx.flatten(), yy, color='red', linewidth=2.5, label='GAM fit')
plt.xlabel('avg_arm_angle')
plt.ylabel('HV_1D')
plt.title(f'GAM Regression (R²={r2:.3f}, RMSE={rmse:.3f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.savefig('outputs/figures/r_vs_r')
plt.show()