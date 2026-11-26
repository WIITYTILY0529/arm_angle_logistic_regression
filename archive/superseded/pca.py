from pathlib import Path
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from pygam import LinearGAM, s
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

def merge_csv(folderpath):
    folder_path = Path(folderpath)
    df = pd.DataFrame()
    for file_path in folder_path.glob('*.csv'):
        pitch_class = file_path.name[7:9]
        df_raw = pd.read_csv(file_path)
        shared_cols = ['player_id', 'player_name', 'total_pitches']
        metric_cols = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced']
        df_adj = df_raw[shared_cols + metric_cols]
        sel_dict = {x: pitch_class + '_' + x for x in df_adj.columns if x in metric_cols}
        df_adj = df_adj.rename(columns=sel_dict)
        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on=shared_cols, how='outer')
    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)
    pitch_cols = [col for col in df.columns if 'pitches' in col and col not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)
    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[0:2]} %'] = df[col] / df['tot_pitches']
    return df

df = merge_csv('data/r_vs_l')
minmax = {col: (df[col].min(), df[col].max()) 
          for col in df.columns if 'api_break' in col}

def avg_arm_angle(row):
    arm_angle = [col for col in df.columns if 'arm_angle' in col]
    row['avg_arm_angle'] = row[arm_angle].mean()
    return row

df = df.apply(avg_arm_angle, axis=1)

def manipulate(row, df, minmax):
    usage = [col for col in df.columns if '%' in col]
    h_score = 0
    z_score = 0
    for col in usage:
        if row[col] > 0:
            pitch = col[0:2]
            x_break = row[f'{pitch}_api_break_x_arm']
            z_break = row[f'{pitch}_api_break_z_induced']
            x_min, x_max = minmax[f'{pitch}_api_break_x_arm']
            z_min, z_max = minmax[f'{pitch}_api_break_z_induced']
            x_norm = (x_break - x_min) / (x_max - x_min)
            z_norm = (z_break - z_min) / (z_max - z_min)
            h_score += abs(row[col] * x_norm)
            z_score += abs(row[col] * z_norm)
    row['h_score'] = h_score
    row['z_score'] = z_score
    return row

df = df.apply(lambda row: manipulate(row, df, minmax), axis=1)

break_cols = [col for col in df.columns 
              if ('api_break_x_arm' in col) or ('api_break_z_induced' in col)]

pca_input = df[break_cols].fillna(0).copy()

#Standardization before pca
scaler = StandardScaler()
pca_input_scaled = scaler.fit_transform(pca_input) #movement standardization

pca = PCA(n_components=1)
pca_fit = pca.fit(pca_input_scaled)

# PC2 score per player
pc_scores = pca_fit.transform(pca_input_scaled)
df['shape_axis_pc2'] = pc_scores[:,0]  # the second axis is the shape axis
print(pca.explained_variance_ratio_)
# Check the shape-axis distribution
print(df[['player_id', 'player_name', 'shape_axis_pc2']].head())