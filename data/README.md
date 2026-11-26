# Data

Pitch-level Statcast data from the 2020–2025 regular seasons, aggregated per pitcher, season, and pitch type.
Only pitchers who threw at least 400 pitches in a season are included. The data is split by the
pitcher–batter handedness matchup, so the same pitcher appears separately against left- and right-handed
batters.

| Folder | Matchup |
| --- | --- |
| `l_vs_l/` | Left-handed pitcher vs. left-handed batter |
| `l_vs_r/` | Left-handed pitcher vs. right-handed batter |
| `r_vs_l/` | Right-handed pitcher vs. left-handed batter |
| `r_vs_r/` | Right-handed pitcher vs. right-handed batter |

Each folder has one CSV per pitch type: `ff` (four-seam), `si` (sinker), `fc` (cutter), `sl` (slider),
`st` (sweeper), `cu` (curveball), `ch` (changeup), and `fs` (splitter). Key columns include `pitches`,
`total_pitches`, `arm_angle`, `api_break_x_arm`, and `api_break_z_induced`.

`processed_summary.csv` is an earlier combined analysis table (one row per pitcher, with per-pitch-type
columns).

The source CSVs are kept exactly as exported.
