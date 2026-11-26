import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde

pd.set_option('display.max_columns', None)

# -------------------------------------------------------------
# Merge CSVs
# -------------------------------------------------------------
def merge_csv_folder(folderpath):

    files = list(Path(folderpath).glob("*.csv"))

    if not files:
        return pd.DataFrame()

    dfs = []

    for f in files:

        pitch = f.stem.split("_")[-1].lower()

        raw = pd.read_csv(f)

        needed = [
            "player_id",
            "player_name",
            "pitches",
            "arm_angle",
            "api_break_x_arm",
            "api_break_z_induced"
        ]

        if "year" in raw.columns:
            needed.append("year")

        df = raw[[c for c in needed if c in raw.columns]].copy()

        df = df.rename(columns={
            "pitches":f"{pitch}_pitches",
            "arm_angle":f"{pitch}_arm_angle",
            "api_break_x_arm":f"{pitch}_api_break_x_arm",
            "api_break_z_induced":f"{pitch}_api_break_z_induced"
        })

        dfs.append(df)

    df_merged = dfs[0]

    merge_on = ["player_id","player_name"]

    if "year" in df_merged.columns:
        merge_on.append("year")

    for d in dfs[1:]:
        df_merged = df_merged.merge(d,on=merge_on,how="outer")

    pitch_cols = [c for c in df_merged.columns if c.endswith("_pitches")]

    df_merged[pitch_cols] = df_merged[pitch_cols].fillna(0)

    df_merged["total_pitches"] = df_merged[pitch_cols].sum(axis=1)

    return df_merged


# -------------------------------------------------------------
# Prepare GAM data
# -------------------------------------------------------------
def prepare_gam_data(df_input):

    results = []

    pitch_types = {
        c.split("_")[0]
        for c in df_input.columns
        if c.endswith("_api_break_x_arm")
    }

    for idx,row in df_input.iterrows():

        pitch_values = []

        angle_cols = [c for c in df_input.columns if "arm_angle" in c]

        avg_angle = row[angle_cols].mean()

        for pt in pitch_types:

            x_val = row.get(f"{pt}_api_break_x_arm")
            z_val = row.get(f"{pt}_api_break_z_induced")

            if pd.notnull(x_val) and pd.notnull(z_val) and row.get(f"{pt}_pitches",0) > 0:

                pitch_values.append({
                    "x":x_val,
                    "z":z_val
                })

        if len(pitch_values) >= 2 and pd.notnull(avg_angle):

            max_h_diff = 0
            max_z_diff = 0

            for i in range(len(pitch_values)):
                for j in range(i+1,len(pitch_values)):

                    max_h_diff = max(
                        max_h_diff,
                        abs(pitch_values[i]["x"] - pitch_values[j]["x"])
                    )

                    max_z_diff = max(
                        max_z_diff,
                        abs(pitch_values[i]["z"] - pitch_values[j]["z"])
                    )

            results.append({
                "avg_arm_angle":avg_angle,
                "max_h_break_diff":max_h_diff,
                "max_z_break_diff":max_z_diff
            })

    df_res = pd.DataFrame(results)

    df_res["h_diff_std"] = (
        df_res["max_h_break_diff"] - df_res["max_h_break_diff"].mean()
    ) / df_res["max_h_break_diff"].std()

    df_res["z_diff_std"] = (
        df_res["max_z_break_diff"] - df_res["max_z_break_diff"].mean()
    ) / df_res["max_z_break_diff"].std()

    return df_res


# -------------------------------------------------------------
# Logistic regression (binary classification)
# -------------------------------------------------------------
def run_binary_logistic(df):

    analysis_df = df.dropna(
        subset=["avg_arm_angle","h_diff_std","z_diff_std"]
    ).copy()

    # Build the binary label
    analysis_df["binary_player_type"] = (
        analysis_df["h_diff_std"] <= analysis_df["z_diff_std"]
    ).astype(int)

    X = analysis_df[["avg_arm_angle"]].values
    y = analysis_df["binary_player_type"].values

    model = LogisticRegression(class_weight="balanced")

    model.fit(X,y)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    boundary = -intercept/slope

    acc = accuracy_score(y,model.predict(X))

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(12,8))

    x0 = X[y==0].flatten()
    x1 = X[y==1].flatten()

    colors = np.zeros(len(X))

    if len(x0) > 1:
        kde0 = gaussian_kde(x0)
        colors[y==0] = kde0(x0)

    if len(x1) > 1:
        kde1 = gaussian_kde(x1)
        colors[y==1] = kde1(x1)

    plt.scatter(
        X,
        y,
        c=colors,
        cmap="coolwarm",
        s=35,
        alpha=0.5,
        label="Pitchers"
    )

    plt.axvline(
        boundary,
        color="green",
        linestyle="--",
        linewidth=3,
        label=f"Boundary {boundary:.2f}°"
    )

    X_test = np.linspace(X.min(),X.max(),300).reshape(-1,1)

    probs = model.predict_proba(X_test)[:,1]

    plt.plot(
        X_test,
        probs,
        color="black",
        linewidth=2,
        label="V-dominant probability"
    )

    plt.title(
        f"Logistic Regression: Arm Angle vs Movement Type\nAccuracy: {acc:.3f} | n={len(analysis_df)}"
    )

    plt.xlabel("Average Arm Angle (degrees)")
    plt.ylabel("Probability of V-dominant")

    plt.yticks(
        [0,1],
        ["H-dominant (0)","V-dominant (1)"]
    )

    plt.grid(alpha=0.3)

    plt.legend()

    plt.savefig("binary_logistic_arm_angle.png")

    plt.show()

    print("Decision Boundary:",boundary)
    print("Accuracy:",acc)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
BASE_PATH = str(PROJECT_ROOT / "data")

FOLDERS = ["l_vs_l","l_vs_r","r_vs_l","r_vs_r"]

all_dfs = []

for f in FOLDERS:

    df_f = merge_csv_folder(Path(BASE_PATH)/f)

    if not df_f.empty:

        df_f["matchup"] = f

        all_dfs.append(df_f)

full_df = pd.concat(all_dfs,ignore_index=True)

df_ready = prepare_gam_data(full_df)

run_binary_logistic(df_ready)