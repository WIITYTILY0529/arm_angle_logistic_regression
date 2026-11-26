from pathlib import Path as _WorkspacePath
PROJECT_ROOT = _WorkspacePath(__file__).resolve().parents[2]

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

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

    # df['tot_pitches'] = sum(df[col] for col in df.columns if 'pitches' in col) - df['total_pitches']

    pitch_cols = [col for col in df.columns if 'pitches' in col and col not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)
    
    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[0:2]} %'] = df[col] / df['tot_pitches']
    
    return df
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

    # if x_break_diff > z_break_diff:
    #     row['player_type'] = 'x_diff_more'
    # else:
    #     row['player_type'] = 'z_diff_more'

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

df = merge_csv(str(PROJECT_ROOT / 'data/l_vs_r'))
df = df.apply(classification, axis = 1)
df = add_standardized_breaks(df)
df = compute_pitcher_HV(df)

df['binary_player_type'] = 0

df.loc[df['H_score'] < df['V_score'], 'binary_player_type'] = 1

# df['binary_player_type'] = df['player_type'].map({'z_diff_more' : 1, 'x_diff_more' : 0})
# if df['max1_pitch_arm_angle'] is None:
#     df['avg_arm_angle'] = df['max2_pitch_arm_angle']
# elif df['max2_pitch_arm_angle'] is None:
#     df['avg_arm_angle'] = df['max1_pitch_arm_angle']
# else:
#     df['avg_arm_angle'] = (df['max1_pitch_arm_angle'] + df['max2_pitch_arm_angle'] ) / 2

df['avg_arm_angle'] = df[['max1_pitch_arm_angle', 'max2_pitch_arm_angle']].mean(axis=1, skipna=True)

logistic_col = ['player_id', 'player_name', 'binary_player_type', 'max1_pitch', 'max2_pitch', 
                'max1_pitch_api_break_x_arm', 'max2_pitch_api_break_x_arm',
                'max1_pitch_api_break_z_induced', 'max2_pitch_api_break_z_induced',
                'max1_pitch_arm_angle', 'max2_pitch_arm_angle', 'avg_arm_angle']

df_cleaned = df.dropna(subset = ['avg_arm_angle'])
logistic_df = df[logistic_col]

logistic_df.plot.scatter(x = 'avg_arm_angle', y = 'binary_player_type')
plt.show()

print(logistic_df.loc[logistic_df['binary_player_type'] == 1].head())

X = logistic_df[['avg_arm_angle']]
y = logistic_df[['binary_player_type']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# coef = model.coef_[0][0]
# intercept = model.intercept_[0]

# boundary = -intercept / coef
# print(f'boundary = {boundary}')

# print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))