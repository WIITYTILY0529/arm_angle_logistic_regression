import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

    pitch_cols = [c for c in df.columns if 'pitches' in c and c not in ['total_pitches']]
    df['tot_pitches'] = df[pitch_cols].sum(axis=1)

    for col in df.columns:
        if 'pitches' in col and col not in ['total_pitches', 'tot_pitches']:
            df[f'{col[:2]} %'] = df[col] / df['tot_pitches']

    return df

def avg_arm_angle(row):
    arm_angle_cols = [c for c in row.index if 'arm_angle' in c]
    vals = row[arm_angle_cols].dropna()
    row['avg_arm_angle'] = vals.mean() if len(vals) else np.nan
    return row

def standardize_series(s: pd.Series):
    std = s.std()
    mean = s.mean()
    if std == 0 or np.isnan(std):
        return s - mean
    return (s - mean) / std

def add_standardized_breaks(df):
    df = df.copy()
    cols = [c for c in df.columns if ('api_break_x_arm' in c) or ('api_break_z_induced' in c)]
    for c in cols:
        df[f'{c}_std'] = standardize_series(df[c])
    return df

def compute_hz_scores(row, df):
    usage_cols = [c for c in df.columns if c.endswith('%')]
    h_score, z_score = 0.0, 0.0

    for col in usage_cols:
        u = row[col]
        if pd.isna(u) or u <= 0:
            continue

        pitch = col[:2]

        x_std_col = f'{pitch}_api_break_x_arm_std'
        z_std_col = f'{pitch}_api_break_z_induced_std'

        if x_std_col not in df.columns or z_std_col not in df.columns:
            continue

        x_std = row[x_std_col]
        z_std = row[z_std_col]

        if pd.isna(x_std) or pd.isna(z_std):
            continue

        h_score += u * abs(x_std)
        z_score += u * abs(z_std)

    row['h_score'] = h_score
    row['z_score'] = z_score
    row['h/h+z'] = h_score / (h_score + z_score) if (h_score + z_score) > 0 else np.nan
    return row
# the flip wasn't applied here either


def build_pitchwise_pca(df):
    pitch_types = sorted({c[:2] for c in df.columns if c.endswith('api_break_x_arm_std')})

    pca_dict = {}

    for pitch in pitch_types:
        x_col = f'{pitch}_api_break_x_arm_std'
        z_col = f'{pitch}_api_break_z_induced_std'

        if x_col not in df.columns or z_col not in df.columns:
            continue

        x = df[x_col]
        z = df[z_col]
        mask = ~(x.isna() | z.isna())

        data = np.stack([x[mask], z[mask]], axis=1)
        if data.shape[0] < 5:
            continue

        pca = PCA(n_components=2)
        pca.fit(data)
        pca_dict[pitch] = pca
    
    return pca_dict

def compute_pitchwise_weighted_pca(df, pca_dict):
    weighted_scores = []

    for idx, row in df.iterrows():
        num = 0.0
        den = 0.0

        for col in df.columns:
            if col.endswith('%'):
                pitch = col[:2]
                u = row[col]

                if u <= 0:
                    continue
                if pitch not in pca_dict:
                    continue

                x_col = f'{pitch}_api_break_x_arm_std'
                z_col = f'{pitch}_api_break_z_induced_std'

                if pd.isna(row[x_col]) or pd.isna(row[z_col]):
                    continue

                vec = np.array([[row[x_col], row[z_col]]])
                pc2 = pca_dict[pitch].transform(vec)[0, 1]

                num += u * pc2
                den += u

        weighted_scores.append(np.nan if den == 0 else num / den)

    df['pitchwise_pc2_weighted'] = weighted_scores
    return df

df = merge_csv('data/r_vs_l')

df = df.apply(avg_arm_angle, axis=1)

df = add_standardized_breaks(df)
df = df.apply(lambda r: compute_hz_scores(r, df), axis=1)
pca_dict = build_pitchwise_pca(df)
df = compute_pitchwise_weighted_pca(df, pca_dict)
df_final = df.dropna(subset=['pitchwise_pc2_weighted', 'h/h+z']).reset_index(drop=True)

print(df_final[['player_name', 'pitchwise_pc2_weighted', 'avg_arm_angle']].head())

plt.figure(figsize=(8, 6))
plt.scatter(df_final['avg_arm_angle'], df_final['pitchwise_pc2_weighted'], alpha=0.7)
plt.xlabel('arm angle')
plt.ylabel('Pitchwise Weighted PC2 Score')

plt.title('Arm angle vs PC2 score')
plt.grid(True)
plt.show()