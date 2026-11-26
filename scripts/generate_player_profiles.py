import numpy as np
import pandas as pd
from pathlib import Path
from _paths import DATA_DIR, FIGURES_DIR
import matplotlib.pyplot as plt
from adjustText import adjust_text

# ======================================================
# Settings
# ======================================================

FLIP_RULES = {
    "ff": {"x": 1,  "z": 1},
    "ch": {"x": 1,  "z": -1},
    "cu": {"x": -1, "z": -1},
    "st": {"x": -1, "z": -1},
    "si": {"x": 1,  "z": -1},
    "fs": {"x": 1,  "z": -1},
    "sl": {"x": -1, "z": -1},
    "fc": {"x": -1, "z": 1},
}

PITCH_COLORS = {
    "ff": "red", "si": "orange", "sl": "gold",
    "ch": "green", "cu": "skyblue",
    "fc": "brown", "fs": "pink", "st": "purple"
}

# ======================================================
# Merge CSVs (usage computed per folder)
# ======================================================

def merge_csv(folderpath):
    files = list(Path(folderpath).glob("*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []

    for f in files:
        pitch = f.stem.split("_")[-1].lower()

        df = pd.read_csv(f)[
            ["player_id", "player_name", "year",
             "pitches", "arm_angle",
             "api_break_x_arm", "api_break_z_induced"]
        ]

        df = df.rename(columns={
            "pitches": f"{pitch}_pitches",
            "arm_angle": f"{pitch}_arm_angle",
            "api_break_x_arm": f"{pitch}_api_break_x_arm",
            "api_break_z_induced": f"{pitch}_api_break_z_induced",
        })

        dfs.append(df)

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on=["player_id", "player_name", "year"], how="outer")

    pitch_cols = [c for c in df.columns if c.endswith("_pitches")]
    df[pitch_cols] = df[pitch_cols].fillna(0)
    df["total_pitches"] = df[pitch_cols].sum(axis=1)

    for c in pitch_cols:
        pt = c[:2]
        df[f"{pt}_pct"] = np.where(
            df["total_pitches"] > 0,
            df[c] / df["total_pitches"],
            0
        )

    return df


# ======================================================
# Global standardization
# ======================================================

def standardize_series(s):
    return (s - s.mean()) / s.std()

def add_standardized_breaks_global(df):
    df = df.copy()

    pitch_types = {
        c[:2] for c in df.columns if c.endswith("_api_break_x_arm")
    }

    for pt in pitch_types:
        x = f"{pt}_api_break_x_arm"
        z = f"{pt}_api_break_z_induced"

        df[f"{pt}_x_std"] = standardize_series(df[x])
        df[f"{pt}_z_std"] = standardize_series(df[z])

        if pt in FLIP_RULES:
            df[f"{pt}_x_std"] *= FLIP_RULES[pt]["x"]
            df[f"{pt}_z_std"] *= FLIP_RULES[pt]["z"]

    return df


# ======================================================
# H / V Score
# ======================================================

def compute_pitcher_HV(df):
    H_list, V_list = [], []
    usage_cols = [c for c in df.columns if c.endswith("_pct")]

    for _, row in df.iterrows():
        H, V = 0.0, 0.0

        for col in usage_cols:
            pt = col[:2]
            u = row[col]

            hx = row.get(f"{pt}_x_std", 0)
            vz = row.get(f"{pt}_z_std", 0)

            if pd.notna(hx):
                H += u * hx
            if pd.notna(vz):
                V += u * vz

        H_list.append(H)
        V_list.append(V)

    df["H_score"] = H_list
    df["V_score"] = V_list
    return df


# ======================================================
# Visualization
# ======================================================

def plot_pitch_profile(row):
    plt.figure(figsize=(7, 7))
    texts = []

    for pt in sorted({c[:2] for c in row.index if c.endswith("_x_std")}):
        x = row[f"{pt}_x_std"]
        y = row[f"{pt}_z_std"]

        if pd.isna(x) or pd.isna(y):
            continue

        plt.scatter(x, y, s=60, color=PITCH_COLORS.get(pt, "black"))
        texts.append(plt.text(x, y + 0.05, pt.upper(), ha="center"))

    adjust_text(texts)
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")
    plt.gca().set_aspect("equal")
    plt.xlim(-4,4)
    plt.ylim(-4,4)

    plt.title(
        f"{row['player_name']} | {row['year']} | {row['matchup']}\n"
        f"H={row['H_score']:.3f}, V={row['V_score']:.3f}"
    )

    plt.savefig(FIGURES_DIR / "player_profiles" / f"{row['player_name']} | {row['year']} | {row['matchup']}.png")
    plt.show()


# ======================================================
# Detailed output
# ======================================================

def print_pitcher_hv_breakdown(df_all, player_name, year, handedness):
    handedness = handedness.lower()
    valid_matchups = [m for m in df_all["matchup"].unique()
                      if m.startswith(handedness)]

    print(f"\n=== {player_name} ({year}) | Handedness: {handedness.upper()} ===")

    for matchup in valid_matchups:
        row = df_all[
            (df_all["player_name"] == player_name) &
            (df_all["year"] == year) &
            (df_all["matchup"] == matchup)
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        print(f"\n[{matchup.upper()}]")
        print("-" * 60)

        H_total, V_total, usage_total = 0, 0, 0

        for col in sorted([c for c in row.index if c.endswith("_pct")]):
            pt = col[:2]
            usage = row[col]

            hx = row.get(f"{pt}_x_std", np.nan)
            vz = row.get(f"{pt}_z_std", np.nan)

            if pd.isna(hx) or pd.isna(vz) or usage == 0:
                continue

            h_contrib = usage * hx
            v_contrib = usage * vz

            usage_total += usage
            H_total += h_contrib
            V_total += v_contrib

            print(
                f"{pt.upper():>3} | "
                f"usage: {usage:.3f} × "
                f"H_std: {hx:+.3f} → {h_contrib:+.3f} | "
                f"V_std: {vz:+.3f} → {v_contrib:+.3f}"
            )

        print("-" * 60)
        print(f"Usage sum : {usage_total:.3f}")
        print(f"H_score   : {H_total:+.4f}")
        print(f"Z_score   : {V_total:+.4f}")


# ======================================================
# Run
# ======================================================

BASE_PATH = DATA_DIR
FOLDERS = ["l_vs_l", "l_vs_r", "r_vs_l", "r_vs_r"]


def main():
    pd.set_option("display.max_columns", None)

    dfs = []

    for f in FOLDERS:
        df = merge_csv(Path(BASE_PATH) / f)
        if df.empty:
            continue
        df["matchup"] = f
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = add_standardized_breaks_global(df_all)
    df_all = compute_pitcher_HV(df_all)

    row = df_all[
        (df_all["player_name"] == "Skenes, Paul") &
        (df_all["year"] == 2025) &
        (df_all["matchup"] == "r_vs_r")
    ].iloc[0]

    print_pitcher_hv_breakdown(
        df_all,
        player_name="Skenes, Paul",
        year=2025,
        handedness="r"
    )

    plot_pitch_profile(row)


if __name__ == "__main__":
    main()
