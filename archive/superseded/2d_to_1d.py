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

#h, v score calculation
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
                if pd.isna(h) or pd.isna(v) or pd.isna(u): #u = usage
                    continue

                H_total += u * h
                V_total += u * v

        H_list.append(H_total)
        V_list.append(V_total)

    df['H_score'] = H_list
    df['V_score'] = V_list
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
    hv_std = scaler.fit_transform(hv_clean) #numpy array

    pca = PCA(n_components=1)
    hv_1d = pca.fit_transform(hv_std).flatten() #hv_std => 2dim ->[h_score_std, v_score_std] -> after PCA, it becomes[N,1] -> flatten result [N, ] (scalar value)

    out = np.full(df.shape[0], np.nan) # length is same w/ # of original df rows, full w/ nan
    out[mask] = hv_1d
    df['HV_1D'] = out

    print("PCA components:", pca.components_)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    return df, scaler, pca

def compute_HV_3types(df):
    if 'H_score' not in df.columns or 'V_score' not in df.columns:
        raise RuntimeError("No H_score / V_score")

    H = df['H_score'].values
    V = df['V_score'].values

    H_center = np.array([1.0, 0.0])
    V_center = np.array([0.0, 1.0])

    HV = np.vstack([H, V]).T  # shape (n_samples, 2)

    dist_H = np.linalg.norm(HV - H_center, axis=1)
    dist_V = np.linalg.norm(HV - V_center, axis=1)

    logits = np.vstack([-dist_H, -dist_V]).T  # shape (n_samples, 2)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax_vals = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    df['HV_softmax_H'] = softmax_vals[:, 0]
    df['HV_softmax_V'] = softmax_vals[:, 1]


    df['HV_norm'] = np.sqrt(H**2 + V**2)

    hv = df[['H_score', 'V_score']].copy()
    mask = ~(hv.isna().any(axis=1))
    hv_clean = hv[mask].values

    if hv_clean.shape[0] >= 2:
        scaler = StandardScaler()
        hv_std = scaler.fit_transform(hv_clean)

        pca = PCA(n_components=1)
        hv_1d = pca.fit_transform(hv_std).flatten()

        out = np.full(df.shape[0], np.nan)
        out[mask] = hv_1d
        df['HV_PCA'] = out

        print("PCA components:", pca.components_)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
    else:
        df['HV_PCA'] = np.nan

    return df

df = merge_csv('data/l_vs_r')
df = df.apply(avg_arm_angle, axis=1)
df = add_standardized_breaks(df)
df = compute_pitcher_HV(df)
df = compute_HV_3types(df)

print(df[['player_id', 'player_name', 'HV_softmax_H', 'HV_softmax_V', 'HV_norm', 'HV_PCA']])