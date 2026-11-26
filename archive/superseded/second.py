import pandas as pd
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
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

    # df['tot_pitches'] = sum(df[col] for col in df.columns if 'pitches' in col) - df['total_pitches']

    pitch_cols = [col for col in df.columns if 'pitches' in col and col not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)
    
    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[0:2]} %'] = df[col] / df['tot_pitches']
    
    return df

def manipulate(row, df):
    usage = [col for col in df.columns if '%' in col]

    max1_pitch_percentage = row[usage].idxmax()
    
    temp_df = row[usage].copy()
    temp_df[max1_pitch_percentage] = float('-inf')
    max2_pitch_percentage = temp_df.idxmax()

    max1_pitch = max1_pitch_percentage[0:2]
    max2_pitch = max2_pitch_percentage[0:2]

    x_break_diff = abs(row[f'{max1_pitch}_api_break_x_arm'] - row[f'{max2_pitch}_api_break_x_arm'])
    z_break_diff = abs(row[f'{max1_pitch}_api_break_z_induced'] - row[f'{max2_pitch}_api_break_z_induced'])

    velo_diff = abs(row[f'{max1_pitch}_velocity'] - row[f'{max2_pitch}_velocity'])

    row['max1_pitch'] = max1_pitch
    row['max2_pitch'] = max2_pitch

    row['max1_pitch_arm_angle'] = row[f'{max1_pitch}_arm_angle']
    row['max2_pitch_arm_angle'] = row[f'{max2_pitch}_arm_angle']

    row['avg_arm_angle'] = row[['max1_pitch_arm_angle', 'max2_pitch_arm_angle']].mean(skipna=True)

    row['max1_pitch_api_break_x_arm'] = row[f'{max1_pitch}_api_break_x_arm']
    row['max2_pitch_api_break_x_arm'] = row[f'{max2_pitch}_api_break_x_arm']

    row['max1_pitch_api_break_z_induced'] = row[f'{max1_pitch}_api_break_z_induced']
    row['max2_pitch_api_break_z_induced'] = row[f'{max2_pitch}_api_break_z_induced']

    row['x_break_diff'] = abs(row['max1_pitch_api_break_x_arm'] - row['max2_pitch_api_break_x_arm'])
    row['z_break_diff'] = abs(row['max1_pitch_api_break_z_induced'] - row['max2_pitch_api_break_z_induced'])

    # row['max1_pitch_velocity'] = row[f'{max1_pitch}_velocity']
    # row['max2_pitch_velocity'] = row[f'{max2_pitch}_velocity']

    # row['velo_diff'] = velo_diff
    # row['velo_diff'] = abs(row['max1_pitch_velocity'] - row['max2_pitch_velocity'])

    # row['player_type'] = 'x_diff_more' if x_break_diff > z_break_diff else 'z_diff_more'
    return row

df_r_vs_r = merge_csv(str(PROJECT_ROOT / 'data/r_vs_r'))
df_r_vs_l = merge_csv(str(PROJECT_ROOT / 'data/r_vs_l'))
df_l_vs_r = merge_csv(str(PROJECT_ROOT / 'data/l_vs_r'))
df_l_vs_l = merge_csv(str(PROJECT_ROOT / 'data/l_vs_l'))

df1 = df_r_vs_r.apply(manipulate, axis = 1, df = df_r_vs_r)
df2 = df_r_vs_l.apply(manipulate, axis = 1, df = df_r_vs_l)
df3 = df_l_vs_r.apply(manipulate, axis = 1, df = df_l_vs_r)
df4 = df_l_vs_l.apply(manipulate, axis = 1, df = df_l_vs_l)

df1.dropna(subset = ['x_break_diff', 'z_break_diff', 'avg_arm_angle'], inplace = True)
df2.dropna(subset = ['x_break_diff', 'z_break_diff', 'avg_arm_angle'], inplace = True)
df3.dropna(subset = ['x_break_diff', 'z_break_diff', 'avg_arm_angle'], inplace = True)
df4.dropna(subset = ['x_break_diff', 'z_break_diff', 'avg_arm_angle'], inplace = True)

df_list = [df1, df2, df3, df4]

df = pd.concat(df_list, ignore_index= True)

# print(df.head())

def classification(row):
    x_diff = row['x_break_diff']
    z_diff = row['z_break_diff']
    # velo_diff = row['velo_diff']

    x_std = (x_diff - df['x_break_diff'].mean()) / df['x_break_diff'].std()
    z_std = (z_diff - df['z_break_diff'].mean()) / df['z_break_diff'].std()

    row['x_std'] = x_std
    row['z_std'] = z_std

    x_norm = (x_diff - df['x_break_diff'].min()) /(df['x_break_diff'].max() - df['x_break_diff'].min())
    z_norm = (z_diff - df['z_break_diff'].min()) /(df['z_break_diff'].max() - df['z_break_diff'].min())

    row['x_norm'] = x_norm
    row['z_norm'] = z_norm

    # v_norm = velo_diff / df['velo_diff'].std()
    # if x_break_diff > z_break_diff:
    #     row['player_type'] = 'x_diff_more'
    # else:
    #     row['player_type'] = 'z_diff_more'

    # if abs(x_norm - z_norm) < threshold:
    #     return 'movement_dominant'
    # elif abs(v_norm - max(x_norm, z_norm)) < threshold:
    #     return 'balanced'
    # elif x_norm > z_norm and x_norm > v_norm:
    #     return 'x_diff_dominant'
    # elif z_norm > x_norm and z_norm > v_norm:
    #     return 'z_diff_dominant'
    # else:
    #     return 'velo_diff_dominant'

    return row

df = df.apply(classification, axis=1)

# df['numericlal_player_type'] = df['player_type'].map({'movement_dominant' : 0, 'balanced' : 1, 'x_diff_dominant' : 2, 
#                                                       'z_diff_dominant' : 3, 'velo_diff_dominant' : 4})

def class_grouping(row):
    if row['x_std'] >= 0 and row['z_std'] >= 0:
        return 'both above 0'
    elif row['x_std'] >= 0 and row['z_std'] < 0:
        return 'x>=0'
    elif row['x_std'] < 0 and row['z_std'] >= 0:
        return 'z>=0'
    else:
        return 'both<0'

df['quadrant'] = df.apply(class_grouping, axis = 1)

def angle_group(angle):
    if angle < 0:
        return '<0'
    elif angle < 20:
        return '0-20'
    elif angle < 40:
        return '20-40'
    elif angle < 60:
        return '40-60'
    else:
        return '60+'

df['angle_group'] = df['avg_arm_angle'].apply(angle_group)

df['pitch_combo_unordered'] = df.apply(
    lambda row: '-'.join(sorted([row['max1_pitch'], row['max2_pitch']])),
    axis=1
)
# combo_angle_table = pd.crosstab(df['pitch_combo_unordered'], df['angle_group'])
# print(combo_angle_table)

# result = pd.crosstab(df['angle_group'], df['quadrant'])
# print(result)

# combo_class_table = pd.crosstab(df['pitch_combo_unordered'], df['quadrant'])
# print(combo_class_table)

df['color_map'] = ''

df.loc[(df['avg_arm_angle'] < 0), 'color_map'] = 'red'
df.loc[(df['avg_arm_angle'] >=0) & (df['avg_arm_angle'] < 20), 'color_map'] = 'green'
df.loc[(df['avg_arm_angle'] >=20) & (df['avg_arm_angle'] < 40), 'color_map'] = 'yellow'
df.loc[(df['avg_arm_angle'] >=40) & (df['avg_arm_angle'] < 60), 'color_map'] = 'blue'
# df.loc[(df['avg_arm_angle'] >=45) & (df['avg_arm_angle'] < 60), 'color_map'] = ''
df.loc[(df['avg_arm_angle'] >=60), 'color_map'] = 'black'

plt.scatter(df['x_std'], df['z_std'], alpha = 0.5, c = df['color_map'])
plt.xlabel('x_std')
plt.ylabel('z_std')
plt.axhline(0, color = 'black', linestyle = '--')
plt.axvline(0, color = 'black', linestyle = '--')
plt.title('x_std vs z_std')
plt.show()

# X = df[['avg_arm_angle']]
# y = df['numericlal_player_type']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

'''
fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['x_break_diff'] ,df['z_break_diff'], df['velo_diff'], color = 'r', alpha = 0.5)
print(df.head())
plt.show()

regression
X = df[['x_break_diff', 'z_break_diff']]
y = df['velo_diff']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X,y)

x1_range = np.linspace(df['x_break_diff'].min(), df['x_break_diff'].max(), 10)
x2_range = np.linspace(df['z_break_diff'].min(), df['z_break_diff'].max(), 10)

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

y_pred = model.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()]).reshape(x1_grid.shape)
ax.plot_surface(x1_grid, x2_grid, y_pred, alpha=0.5, color='blue', label='Regression plane')

plt.show()
'''