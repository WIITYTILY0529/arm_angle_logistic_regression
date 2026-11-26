# Arm Angle and Pitcher Movement Profiles

A research project that quantifies each pitcher's movement profile and measures how much of it is explained
by arm angle, using Statcast data from 2020–2025.

Pitchers are scored on whether their arsenal leans toward horizontal or vertical movement. These scores are
then related to arm angle with logistic regression, linear and polynomial regression, PCA, and generalized
additive models (GAMs).

## Method

1. **Data.** Pitchers with at least 400 pitches in a regular season from 2020 to 2025, split by pitcher–batter
   handedness matchup (`l_vs_l`, `l_vs_r`, `r_vs_l`, `r_vs_r`). The same pitcher is analyzed separately
   against left- and right-handed batters.
2. **Movement scores.** For each pitch type, horizontal (`api_break_x_arm`) and vertical
   (`api_break_z_induced`) movement are standardized against all pitchers who throw that pitch, and flipped
   so that positive values point in the pitch's intended direction. Each pitcher's standardized values are
   weighted by pitch usage and summed into an **H score** (horizontal) and a **Z/V score** (vertical).
3. **Pitcher type.** Pitchers are labeled vertical-dominant or horizontal-dominant by comparing the two
   scores, and logistic regression finds the arm angle that best separates the two groups.
4. **Strength of the relationship.** The two scores are reduced to one dimension with PCA, and a GAM models
   that score as a function of average arm angle.

### Summary of findings

From the write-up in [`writing/reports/Report.docx`](writing/reports/Report.docx):

- An arm angle of about 38–40° separates horizontal- and vertical-dominant pitchers, but the classification
  accuracy is modest (about 0.55–0.7 depending on the matchup).
- The first principal component has loadings of [0.71, −0.71]: the H and V scores trade off against each
  other, and this contrast explains about 60% of their variance.
- Arm angle explains only about 27% of the variance in that component, so many factors other than arm angle
  shape a pitcher's movement profile.

## Repository layout

```text
.
├── data/                 Statcast aggregates by handedness matchup and pitch type
├── src/arm_angle/        Shared data loading and standardization helpers
├── scripts/              Final analysis scripts
├── experiments/          Alternative and earlier analyses (linear, logistic, GAM, scatter)
├── archive/
│   ├── notes/            Early research notes
│   └── superseded/       Earlier versions replaced by the scripts above
├── outputs/figures/      Figures produced by the analyses
├── tests/                Tests for the shared helpers
├── writing/              Report, article draft, abstracts, presentation, and poster
├── LICENSE
├── pyproject.toml
├── requirements-windows.txt
└── requirements-windows.lock.txt
```

See [`data/README.md`](data/README.md) for a description of the input files.

## Scripts

| Script | Purpose | Output |
| --- | --- | --- |
| `scripts/run_horizontal_gam.py` | GAM of the H score on arm angle | plot window |
| `scripts/run_vertical_gam.py` | GAM of the Z score on arm angle | plot window |
| `scripts/run_softsign_gam.py` | GAM of the combined (softsign) movement bias | console summary |
| `scripts/run_integrated_logistic.py` | Logistic regression of pitcher type on arm angle, all matchups combined | `outputs/figures/logistic/integrated_logi_reg.png` |
| `scripts/movement_bias_weighted.py` | Usage-weighted movement bias by pitch type | plot window |
| `scripts/movement_bias_unweighted.py` | Movement bias without usage weighting (FF and CU) | plot window |
| `scripts/compare_scores.py` | Pitchers whose H/V tendency flips between matchups | console |
| `scripts/generate_player_profiles.py` | Movement profile plots for individual pitchers | `outputs/figures/player_profiles/` |
| `scripts/plot_score_density.py` | Distributions of the H and Z scores | `outputs/figures/gam/h_and_v_density.png` |

The scripts under `experiments/` and `archive/superseded/` are kept to document the research process. They
read from `data/` but are not maintained as part of the main workflow, and some save figures to the current
working directory.

## Setup

Python 3.10 or newer is required (developed with Python 3.12 on Windows).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-windows.txt
```

`requirements-windows.txt` lists the direct dependencies; `requirements-windows.lock.txt` pins the exact
versions used. On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Usage

Run the scripts from the project root, for example:

```powershell
python scripts\run_integrated_logistic.py
```

Most scripts open their figures in a plot window. Run the tests with:

```powershell
python -m pytest
```

## Writing

| File | Contents |
| --- | --- |
| `writing/reports/Report.docx` | Research report |
| `writing/reports/Fangraphs.docx` | Article draft |
| `writing/abstracts/Abstract.docx`, `Abstract_final.docx` | Abstract drafts |
| `writing/presentation/Presentation abstract.pptx` / `.pdf` | Slides with speaker notes |
| `writing/presentation/Script_Eng.docx` | Talk script |
| `writing/poster/poster.pptx`, `sabr_analytics_2026_poster.pdf` | Poster |

## Data source

Pitch data comes from [Baseball Savant](https://baseballsavant.mlb.com/) (Statcast).

## License

The code in this repository is released under the [MIT License](LICENSE). Statcast data remains subject to
the terms of its original source.
