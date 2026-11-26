"""Common matchup-data loading and standardization."""

from pathlib import Path

import pandas as pd

MATCHUPS = ("l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r")


def standardize_series(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return series * 0
    return (series - series.mean()) / std


def load_matchup_data(data_dir: str | Path) -> pd.DataFrame:
    frames = []
    root = Path(data_dir)
    for matchup in MATCHUPS:
        for csv_path in sorted((root / matchup).glob("*.csv")):
            frame = pd.read_csv(csv_path)
            frame["matchup"] = matchup
            frame["source_file"] = csv_path.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No matchup CSV files found below {root}")
    return pd.concat(frames, ignore_index=True)
