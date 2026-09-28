import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import os

os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots/References for segregation")

# =========================================================
# FILES
# =========================================================

near_road_file = "Office_of_The_Assistant_Commissioner_of_Police,_Belghoria.csv"

far_road_file  = "Polerhat_Police_Station.csv"

# =========================================================
# PERIOD
# =========================================================

start_date = '2025-01-01'
end_date   = '2025-12-31'

period_label = (
    f"{pd.to_datetime(start_date).strftime('%d-%b-%Y')} "
    f"to "
    f"{pd.to_datetime(end_date).strftime('%d-%b-%Y')}"
)

# =========================================================
# MISSING FRACTION FUNCTION
# =========================================================

def missing_fraction(data):

    data = data.sort_values('timestamp')

    if len(data) < 2:
        return np.nan

    gaps = (
        data['timestamp']
        .diff()
        .dt.total_seconds() / 60
    )

    large_gaps = gaps[gaps > 25]

    missing_obs = (
        ((large_gaps // 10) - 1).sum()
    )

    observed_obs = len(data)

    total_expected = (
        observed_obs + missing_obs
    )

    if total_expected == 0:
        return np.nan

    return (
        missing_obs / total_expected
    )

# =========================================================
# LOAD DATA
# =========================================================

#Station withtraffic close by

near_df_raw = pd.read_csv(near_road_file)

near_ts_here = pd.to_datetime(near_df_raw.iloc[:, 0], errors='coerce').dropna()

mask = near_ts_here.ne(near_ts_here.shift())
near_df = near_df_raw.loc[mask].copy()
near_ts_here= near_ts_here.loc[mask]

##Station farther away

far_df_raw = pd.read_csv(far_road_file)

far_ts_here = pd.to_datetime(far_df_raw.iloc[:, 0], errors='coerce').dropna()

mask = far_ts_here.ne(far_ts_here.shift())
far_df = far_df_raw.loc[mask].copy()
far_ts_here= far_ts_here.loc[mask]

# =========================================================
# DATETIME
# =========================================================

near_df['timestamp'] = pd.to_datetime(
    near_df['timestamp']
)

far_df['timestamp'] = pd.to_datetime(
    far_df['timestamp']
)

# =========================================================
# FILTER PERIOD
# =========================================================

near_mask = (
    (near_df['timestamp'] >= start_date) &
    (near_df['timestamp'] <= end_date)
)

far_mask = (
    (far_df['timestamp'] >= start_date) &
    (far_df['timestamp'] <= end_date)
)

near_data = near_df.loc[
    near_mask
].copy()

far_data = far_df.loc[
    far_mask
].copy()

# =========================================================
# LAeqT VALUES
# =========================================================

near_laeqt = (
    near_data['laeqt']
    .dropna()
    .values
)

far_laeqt = (
    far_data['laeqt']
    .dropna()
    .values
)

# =========================================================
# MISSING FRACTIONS
# =========================================================

near_missing = missing_fraction(
    near_data
)

far_missing = missing_fraction(
    far_data
)

# =========================================================
# KDE
# =========================================================

xmin = min(
    near_laeqt.min(),
    far_laeqt.min()
)

xmax = max(
    near_laeqt.max(),
    far_laeqt.max()
)

x_grid = np.linspace(
    xmin,
    xmax,
    500
)

kde_near = gaussian_kde(
    near_laeqt
)

kde_far = gaussian_kde(
    far_laeqt
)

y_near = kde_near(x_grid)

y_far = kde_far(x_grid)

# =========================================================
# PLOTTING
# =========================================================

plt.figure(figsize=(12,7))

# ---------- histograms ----------

plt.hist(
    near_laeqt,
    bins=40,
    density=True,
    alpha=0.25
)

plt.hist(
    far_laeqt,
    bins=40,
    density=True,
    alpha=0.25
)

# ---------- KDE curves ----------

plt.plot(
    x_grid,
    y_near,
    linewidth=2.5,
    label=(
        'Near Major Road (<20m)\n'
        f'Missing={near_missing:.2%}'
    )
)

plt.plot(
    x_grid,
    y_far,
    linewidth=2.5,
    label=(
        'Far From Major Road (>400m)\n'
        f'Missing={far_missing:.2%}'
    )
)

# =========================================================
# LABELS
# =========================================================

plt.xlabel('LAeqT')

plt.ylabel('Density')

plt.title(
    'LAeqT Density Comparison\n'
    f'({period_label})'
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()