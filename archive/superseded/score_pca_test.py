import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# =============================================================================
# 1. Shared utility functions
# =============================================================================

def merge_csv(folderpath):
    folder_path = Path(folderpath)
    df = pd.DataFrame()

    files = list(folder_path.glob('*.csv'))
    if not files:
        print(f"Warning: No CSV files found in {folderpath}")
        return df

    for file_path in files:
        pitch_class = file_path.name[7:9].lower()
        df_raw = pd.read_csv(file_path)

        shared = ['player_id', 'player_name', 'total_pitches']
        metrics = ['pitches', 'arm_angle', 'api_break_x_arm', 'api_break_z_induced']
        metrics = [m for m in metrics if m in df_raw.columns]

        df_adj = df_raw[shared + metrics]
        rename_map = {m: pitch_class + '_' + m for m in metrics}
        df_adj = df_adj.rename(columns=rename_map)

        if df.empty:
            df = df_adj
        else:
            df = pd.merge(df, df_adj, on=shared, how='outer')

    for col in df.columns:
        if 'pitches' in col:
            df[col] = df[col].fillna(0)

    pitch_cols = [c for c in df.columns if 'pitches' in c and c != 'total_pitches']
    if pitch_cols:
        df['tot_pitches'] = df[pitch_cols].sum(axis=1)
        for col in pitch_cols:
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


def add_standardized_breaks_dynamic(df, folder_name):
    df = df.copy()
    folder_lower = folder_name.lower()

    if folder_lower.startswith('l_vs'):
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
    else:
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

    pitch_types = sorted({c[:2] for c in df.columns if c.endswith('api_break_x_arm')})

    for pt in pitch_types:
        x_col = f'{pt}_api_break_x_arm'
        z_col = f'{pt}_api_break_z_induced'

        if x_col in df.columns:
            df[f'{pt}_x_std'] = standardize_series(df[x_col])
        if z_col in df.columns:
            df[f'{pt}_z_std'] = standardize_series(df[z_col])

        if pt in flip_rules:
            if f'{pt}_x_std' in df.columns:
                df[f'{pt}_x_std'] *= flip_rules[pt]['x']
            if f'{pt}_z_std' in df.columns:
                df[f'{pt}_z_std'] *= flip_rules[pt]['z']

    return df


def compute_pitcher_HV(df):
    usage_cols = [c for c in df.columns if c.endswith('%')]
    H_list, V_list = [], []

    for _, row in df.iterrows():
        H, V = 0.0, 0.0
        for col in usage_cols:
            pt = col[:2]
            u = row[col]
            h_col = f'{pt}_x_std'
            v_col = f'{pt}_z_std'

            if h_col in df.columns and v_col in df.columns:
                if pd.isna(u) or pd.isna(row[h_col]) or pd.isna(row[v_col]):
                    continue
                H += u * row[h_col]
                V += u * row[v_col]

        H_list.append(H)
        V_list.append(V)

    df['H_score'] = H_list
    df['V_score'] = V_list
    return df


def HV_to_1D_via_PCA(df):
    hv = df[['H_score', 'V_score']]
    mask = ~hv.isna().any(axis=1)
    hv_clean = hv[mask].values

    if hv_clean.shape[0] < 2:
        return df, None, None

    scaler = StandardScaler()
    hv_std = scaler.fit_transform(hv_clean)

    pca = PCA(n_components=1)
    hv_1d = pca.fit_transform(hv_std).flatten()

    if pca.components_[0, 0] < 0:
        hv_1d *= -1

    out = np.full(len(df), np.nan)
    out[mask] = hv_1d
    df['HV_1D'] = out

    return df, scaler, pca


# =============================================================================
# 2. GAM + slopes per 5-degree interval (HV_1D)
# =============================================================================

def run_explanatory_gam_HV(base_path, folder_name):
    df = merge_csv(Path(base_path) / folder_name)
    if df.empty:
        return None

    df = df.apply(avg_arm_angle, axis=1)
    df = add_standardized_breaks_dynamic(df, folder_name)
    df = compute_pitcher_HV(df)
    df, _, _ = HV_to_1D_via_PCA(df)

    df = df.dropna(subset=['avg_arm_angle', 'HV_1D'])
    if len(df) < 10:
        return None

    X = df[['avg_arm_angle']].values
    y = df['HV_1D'].values

    gam = LinearGAM(s(0, n_splines=20))
    gam.gridsearch(X, y, lam=np.logspace(-3, 2, 20), progress=False)

    xx = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
    yy = gam.predict(xx)
    ci = gam.confidence_intervals(xx)

    pdp = gam.partial_dependence(term=0, X=xx)
    dydx = np.gradient(pdp, xx.flatten())

    angle = xx.flatten()
    bins = np.arange(
        np.floor(angle.min() / 5) * 5,
        np.ceil(angle.max() / 5) * 5 + 5,
        5
    )

    segments = []
    for i in range(len(bins) - 1):
        mask = (angle >= bins[i]) & (angle < bins[i + 1])
        if mask.sum() < 10:
            continue

        slope = dydx[mask].mean()
        x_mid = (bins[i] + bins[i + 1]) / 2
        y_mid = np.interp(x_mid, angle, yy)

        segments.append({
            'x': angle[mask],
            'y': yy[mask],
            'slope': slope,
            'x_mid': x_mid,
            'y_mid': y_mid
        })

    return {
        'X': X,
        'y': y,
        'xx': xx,
        'yy': yy,
        'ci': ci,
        'segments': segments,
        'n': len(df),
        'edf': gam.statistics_['edof']
    }


# =============================================================================
# 3. Visualization
# =============================================================================

BASE_PATH = str(PROJECT_ROOT / 'data')
FOLDERS = ['l_vs_l', 'l_vs_r', 'r_vs_l', 'r_vs_r']

fig, axes = plt.subplots(2, 2, figsize=(24, 16))
axes = axes.flatten()

for ax, folder in zip(axes, FOLDERS):
    res = run_explanatory_gam_HV(BASE_PATH, folder)
    if res is None:
        continue

    ax.scatter(res['X'], res['y'], alpha=0.25, s=12, color='gray')
    ax.plot(res['xx'], res['yy'], color='lightgray', linewidth=3)

    for seg in res['segments']:
        color = 'blue' if seg['slope'] > 0 else 'red'
        ax.plot(seg['x'], seg['y'], color=color, linewidth=4)
        ax.text(
            seg['x_mid'],
            seg['y_mid'],
            f"{seg['slope']:+.2f}",
            fontsize=9,
            ha='center',
            va='bottom',
            color=color
        )

    ax.fill_between(
        res['xx'].flatten(),
        res['ci'][:, 0],
        res['ci'][:, 1],
        color='gray',
        alpha=0.15
    )

    ax.set_title(f"{folder.upper()} | n={res['n']} | EDF={res['edf']:.2f}")
    ax.set_xlabel('Avg Arm Angle')
    ax.set_ylabel('HV_1D')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()