import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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

df = merge_csv(str(PROJECT_ROOT / 'data/r_vs_r'))

minmax = {col: (df[col].min(), df[col].max()) 
          for col in df.columns if 'api_break' in col}

def avg_arm_angle(row):
    arm_angle = [col for col in df.columns if 'arm_angle' in col]

    row['avg_arm_angle'] = row[arm_angle].mean()

    return row

df = df.apply(avg_arm_angle, axis = 1)

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

# print(df.head())

# plt.scatter(df['avg_arm_angle'], df['h_score'] / df['z_score'], alpha = 0.5)
# plt.show()

df_reg = df[['avg_arm_angle', 'h_score', 'z_score']].dropna()
df_reg = df_reg[df_reg['z_score'] != 0]  # avoid division by zero

X = df_reg[['avg_arm_angle']]
y = df_reg['h_score'] / df_reg['z_score']
# y = df_reg['h_score']
# y = df_reg['z_score']

# model = LinearRegression()
# model.fit(X, y)

# print("Intercept:", model.intercept_)
# print("Slope:", model.coef_[0])

# print("R²:", model.score(X, y))

# plt.scatter(X, y, alpha=0.5, label='data')
# plt.plot(X, model.predict(X), color='red', label='regression line')
# plt.xlabel('avg_arm_angle')
# plt.ylabel('h_score / z_score')
# plt.legend()
# plt.show()

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)
# model_poly = LinearRegression()
# model_poly.fit(X_poly, y)

# print("Quadratic polynomial R²:", model_poly.score(X_poly, y))

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(n_estimators=200, random_state=42)
# rf.fit(X, y)
# print('R²:', rf.score(X, y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit the model
# rf = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=None,
#     random_state=42,
#     n_jobs=-1
# )
# rf.fit(X_train, y_train)

# # Predict
# y_pred = rf.predict(X_test)

# # Evaluate performance
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print(f'R²: {r2:.4f}')
# print(f'MSE: {mse:.4f}')

# # Visualize
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test, y_test, alpha=0.4, label='Actual')
# plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
# plt.xlabel('Average Arm Angle')
# plt.ylabel('h_score / z_score')
# plt.title('Random Forest Regression: h/z ratio vs Arm Angle')
# plt.legend()
# plt.show()
print(df_reg[['avg_arm_angle', 'h_score', 'z_score']].corr())


plt.scatter(df_reg['avg_arm_angle'], y)
plt.xlabel('avg_arm_angle')
plt.ylabel('h_score / z_score')
plt.show()