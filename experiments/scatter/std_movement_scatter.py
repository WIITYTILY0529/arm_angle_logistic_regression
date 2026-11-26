import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

def merge_csv(folderpath):
    folder_path = Path(folderpath)
    df = pd.DataFrame()

    for file_path in folder_path.glob('*.csv'):
        pitch_class = file_path.name[7:9] #ex ) sl, sl etc..
        df_raw = pd.read_csv(file_path)

        shared = ['player_id', 'player_name', 'total_pitches', 'year']
        metrics = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced']

        df_adj = df_raw[shared + metrics]
        rename_map = {m: pitch_class + '_' + m for m in metrics}
        df_adj = df_adj.rename(columns=rename_map)

        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on=shared, how='outer')

    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0) # treat pitch types not thrown as 0 pitches

    pitch_cols = [c for c in df.columns if 'pitches' in c and c not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)

    for col in pitch_cols:
        df[f'{col[:2]} %'] = df[col] / df['tot_pitches']

    return df

def standardize_series(s):
    std = s.std()
    mean = s.mean()
    if std == 0 or np.isnan(std):
        return s - mean
    return (s - mean) / std

def add_standardized_breaks(df):
    df = df.copy()

    pitch_types = sorted({col[:2] for col in df.columns if col.endswith('api_break_x_arm')})

    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        z_col = f'{pt}_api_break_z_induced'

        if x_col in df.columns:
            df[f'{pt}_x_std'] = standardize_series(df[x_col])
        if z_col in df.columns:
            df[f'{pt}_z_std'] = standardize_series(-1*df[z_col])

    return df

# def plot_pitch_scatte_with_name(df, pitch_type, player_name=None):
#     xcol = f'{pitch_type}_x_std'
#     zcol = f'{pitch_type}_z_std'

#     if xcol not in df.columns or zcol not in df.columns:
#         raise RuntimeError(f'No standardized values exist for pitch type {pitch_type}.')

#     if player_name is not None:
#         sub = df[df['player_name'].str.contains(player_name, case=False, na=False)]
#         if sub.empty:
#             raise RuntimeError(f'No pitcher with this name: {player_name}')
#     else:
#         sub = df

#     plt.figure(figsize=(6,6))
#     plt.scatter(sub[xcol], sub[zcol])
#     plt.xlabel(f'{pitch_type} x_std')
#     plt.ylabel(f'{pitch_type} z_std')
#     plt.title(f'{pitch_type} movement scatter (standardized)')
#     plt.grid(True)
#     plt.show()

def plot_pitch_scatter(df, pitch_type):
    xcol = f'{pitch_type}_x_std'
    zcol = f'{pitch_type}_z_std'

    if xcol not in df.columns or zcol not in df.columns:
        raise RuntimeError(f'No standardized values exist for pitch type {pitch_type}.')

    plt.figure(figsize=(6,6))
    plt.scatter(df[xcol], df[zcol])
    plt.xlabel(f'{pitch_type} x_std')
    plt.ylabel(f'{pitch_type} z_std')
    plt.title(f'{pitch_type} movement scatter (standardized)')
    plt.grid(True)
    plt.show()

df = merge_csv('data/l_vs_l')
df = add_standardized_breaks(df)
df = df[df['player_name'] == 'Fried, Max']
# df = df.sort_values(by = ['year'], ascending = False)
print(df[['player_name', 'year', 'ff_z_std', 'ff_x_std']])


plot_pitch_scatter(df, 'ff')
# plot_pitch_scatter(df, 'SL')

# plot_pitch_scatte_with_name(df, 'FF', player_name='Gerrit Cole')