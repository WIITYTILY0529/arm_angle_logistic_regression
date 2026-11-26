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
        metric_cols = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced', 'velocity']
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

def avg_arm_angle(row, df):
    arm_angle = [col for col in df.columns if 'arm_angle' in col]
    row['avg_arm_angle'] = row[arm_angle].mean()
    return row

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

def process_folder(folder_name):
    base_path = str(PROJECT_ROOT / 'data')
    df = merge_csv(f'{base_path}/{folder_name}')
    
    # Store min-max
    minmax = {col: (df[col].min(), df[col].max()) for col in df.columns if 'api_break' in col}
    
    # Compute avg_arm_angle
    df = df.apply(lambda row: avg_arm_angle(row, df), axis=1)
    
    # Compute h_score and z_score
    df = df.apply(lambda row: manipulate(row, df, minmax), axis=1)
    
    # Compute binary_player_type
    df['binary_player_type'] = df.apply(lambda row: 0 if row['h_score'] > row['z_score'] else 1, axis=1)
    
    # Clean up the data
    df_reg = df[['avg_arm_angle', 'h_score', 'z_score', 'binary_player_type']].dropna()
    df_reg = df_reg[df_reg['z_score'] != 0]
    
    # Logistic Regression
    X = df_reg[['avg_arm_angle']].values
    y = df_reg['binary_player_type'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    boundary = -intercept / coef
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'\n{"="*60}')
    print(f'{folder_name.upper()}')
    print(f'{"="*60}')
    print(f'Boundary = {boundary:.4f}')
    print(f'Accuracy Score: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return df_reg, boundary, accuracy

# Process the four folders
folders = ['r_vs_l', 'r_vs_r', 'l_vs_l', 'l_vs_r']
data_dict = {}

for folder in folders:
    df_reg, boundary, accuracy = process_folder(folder)
    data_dict[folder] = {'data': df_reg, 'boundary': boundary, 'accuracy': accuracy}

# Create the 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, folder in enumerate(folders):
    df_reg = data_dict[folder]['data']
    boundary = data_dict[folder]['boundary']
    accuracy = data_dict[folder]['accuracy']
    ax = axes[idx]
    
    x = df_reg['avg_arm_angle'].values
    y = df_reg['binary_player_type'].values
    
    # Compute KDE
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Scatter
    scatter = ax.scatter(x, y, c=z, s=30, cmap='coolwarm', alpha=0.6)
    
    # Add the decision boundary
    ax.axvline(x=boundary, color='green', linestyle='--', linewidth=2.5, 
               label=f'Boundary: {boundary:.2f}')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15, fontsize=9)
    
    # Labels and title
    ax.set_xlabel('avg_arm_angle', fontsize=11)
    ax.set_ylabel('binary_player_type', fontsize=11)
    ax.set_title(f'{folder.upper()}\n(n={len(df_reg)}, Accuracy={accuracy:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')

plt.suptitle('Player Type Classification with Decision Boundaries', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()