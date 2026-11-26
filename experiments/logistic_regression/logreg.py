from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


pd.set_option('display.max_columns', None)

def merge_csv(folderpath):
    folder_path = Path(folderpath)

    df = pd.DataFrame()

    for file_path in folder_path.glob('*.csv'):
        pitch_class = file_path.name[7:9]
        df_raw = pd.read_csv(file_path)

        shared_cols = ['player_id', 'player_name', 'total_pitches']
        metric_cols = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced','velocity']

        df_adj = df_raw[shared_cols + metric_cols]
        sel_dict = {x: pitch_class + '_' + x for x in df_adj.columns if x in metric_cols}
        df_adj = df_adj.rename(columns=sel_dict)

        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on = shared_cols, how = 'outer')

    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)

    pitch_cols = [col for col in df.columns if 'pitches' in col and col not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)
    
    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[0:2]} %'] = df[col] / df['tot_pitches']
    
    return df

def classification(row):
    usage = [col for col in df.columns if '%' in col]

    max1_pitch_percentage = row[usage].idxmax()
    
    temp_df = row[usage].copy()
    temp_df[max1_pitch_percentage] = float('-inf')
    max2_pitch_percentage = temp_df.idxmax()

    max1_pitch = max1_pitch_percentage[0:2]
    max2_pitch = max2_pitch_percentage[0:2]

    row['max1_pitch'] = max1_pitch
    row['max2_pitch'] = max2_pitch

    x_break_diff = abs(row[f'{max1_pitch}_api_break_x_arm'] - row[f'{max2_pitch}_api_break_x_arm'])
    z_break_diff = abs(row[f'{max1_pitch}_api_break_z_induced'] - row[f'{max2_pitch}_api_break_z_induced'])

    if x_break_diff > z_break_diff:
        row['player_type'] = 'x_diff_more'
    else:
        row['player_type'] = 'z_diff_more'

    row['max1_pitch'] = max1_pitch
    row['max2_pitch'] = max2_pitch

    row['max1_pitch_arm_angle'] = row[f'{max1_pitch}_arm_angle']
    row['max2_pitch_arm_angle'] = row[f'{max2_pitch}_arm_angle']

    row['max1_pitch_api_break_x_arm'] = row[f'{max1_pitch}_api_break_x_arm']
    row['max2_pitch_api_break_x_arm'] = row[f'{max2_pitch}_api_break_x_arm']

    row['max1_pitch_api_break_z_induced'] = row[f'{max1_pitch}_api_break_z_induced']
    row['max2_pitch_api_break_z_induced'] = row[f'{max2_pitch}_api_break_z_induced']

    row['player_type'] = 'x_diff_more' if x_break_diff > z_break_diff else 'z_diff_more'

    return row

df = merge_csv(str(PROJECT_ROOT / 'data/r_vs_r'))
df = df.apply(classification, axis = 1)

df['binary_player_type'] = 0
df['binary_player_type'] = df['player_type'].map({'z_diff_more' : 1, 'x_diff_more' : 0})

df['avg_arm_angle'] = df[['max1_pitch_arm_angle', 'max2_pitch_arm_angle']].mean(axis=1, skipna=True)

logistic_col = ['player_id', 'player_name', 'binary_player_type', 'max1_pitch', 'max2_pitch', 
                'max1_pitch_api_break_x_arm', 'max2_pitch_api_break_x_arm',
                'max1_pitch_api_break_z_induced', 'max2_pitch_api_break_z_induced',
                'max1_pitch_arm_angle', 'max2_pitch_arm_angle', 'avg_arm_angle']

df_cleaned = df.dropna(subset = ['avg_arm_angle'])
logistic_df = df_cleaned[logistic_col]

# Density-based colors
x = logistic_df['avg_arm_angle'].values
y = logistic_df['binary_player_type'].values

# Density of each point with a Gaussian KDE
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x, y, c=z, s=30, cmap='coolwarm', alpha=0.6)

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density', rotation=270, labelpad=20)

ax.set_xlabel('avg_arm_angle')
ax.set_ylabel('binary_player_type')
ax.set_title('Player Type Distribution by Arm Angle (density-based colors)')

plt.tight_layout()
plt.show()