# Required library: if pygam is missing, run pip install pygam
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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

df = merge_csv(str(PROJECT_ROOT / 'data/r_vs_l'))
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

df_reg = df[['avg_arm_angle', 'h_score', 'z_score']].dropna()
df_reg = df_reg[df_reg['z_score'] != 0]

# Define variables
df_reg['h_z_ratio'] = df_reg['h_score'] / (df_reg['h_score'] + df_reg['z_score'])

mask = (df_reg['avg_arm_angle'] > -50) & (df_reg['h_z_ratio'] < 2.5)
df_reg = df_reg[mask].copy()

# df_reg: DataFrame built earlier. Minimum columns: 'avg_arm_angle', 'h_score', 'z_score'
# y = h / (h+z)
df = df_reg.copy()
df = df[(df['h_score'] + df['z_score']) > 0]   # safeguard

X = df[['avg_arm_angle']].values
y = (df['h_score'] / (df['h_score'] + df['z_score'])).values

# train/test split (stratify is not needed for a ratio distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1) Basic GAM + grid search for lambda (regularization)
# s(0, n_splines=20): the number of splines can be adjusted to the data size
gam = LinearGAM(s(0, n_splines=20))

# gridsearch: automatically searches lambda (=lam) candidates
# lam is the smoothing penalty; use a wide log-scale range
lam_grid = np.logspace(-3, 3, 25)
gam.gridsearch(X_train, y_train, lam=lam_grid)
print(gam.summary()) 
# 2) Evaluation: predictions, R² (standard), pseudo-R² (1 - RSS/RSS_null), RMSE
y_pred_test = gam.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

rss_model = np.sum((y_test - y_pred_test)**2)
rss_null = np.sum((y_test - y_test.mean())**2)
pseudo_r2 = 1 - rss_model / rss_null

print("GAM lam chosen:", gam.lam)
print(f"Test R² (sklearn): {r2:.4f}")
print(f"Test pseudo-R² (1 - RSS/RSS_null): {pseudo_r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# 3) Visualization: fit curve + 95% CI (prediction intervals)
xx = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
yy_pred = gam.predict(xx)
# Use prediction_intervals from pygam
try:
    lower, upper = gam.prediction_intervals(xx, width=0.95)
except Exception:
    # For older pygam versions, build the CI with a simple bootstrap (included in the bootstrap below)
    lower = yy_pred - 1.96 * np.std(y - gam.predict(X))
    upper = yy_pred + 1.96 * np.std(y - gam.predict(X))

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.25, s=18, color='gray', label='data')
plt.plot(xx, yy_pred, color='red', linewidth=2, label='GAM fit')
plt.fill_between(xx.flatten(), lower, upper, color='red', alpha=0.2, label='95% CI')
plt.xlabel('avg_arm_angle (deg)')
plt.ylabel('h / (h + z)')
plt.title(f'GAM fit: Test pseudo-R²={pseudo_r2:.3f}, RMSE={rmse:.3f}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 4) Local slope (derivative) => AME: average slope (average change per 1-degree change)
# pygam provides .derivative(term, X)
deriv_vals = gam.derivative(0, X)  # derivative at each observed X (shape n x 1)
# AME: average derivative across dataset (change in y per 1 unit of angle)
ame = np.nanmean(deriv_vals)
ame_std = np.nanstd(deriv_vals)
print(f"Average Marginal Effect (AME): {ame:.5f} (y change per 1 deg)")
print(f"AME std dev: {ame_std:.5f}")

# 5) Bootstrap confidence interval for the AME (recommended)
# Reduce the number of iterations if the sample is very small
n_boot = 2000
ames = np.empty(n_boot)
n = X.shape[0]
rng = np.random.default_rng(42)
for i in range(n_boot):
    idx = rng.integers(0, n, n)
    Xb = X[idx]
    yb = y[idx]
    try:
        g = LinearGAM(s(0, n_splines=20)).fit(Xb, yb)
        d = g.derivative(0, Xb)
        ames[i] = np.nanmean(d)
    except Exception:
        ames[i] = np.nan

ames = ames[~np.isnan(ames)]
lower_ame = np.percentile(ames, 2.5)
upper_ame = np.percentile(ames, 97.5)
print(f"Bootstrap AME 95% CI: [{lower_ame:.5f}, {upper_ame:.5f}]")