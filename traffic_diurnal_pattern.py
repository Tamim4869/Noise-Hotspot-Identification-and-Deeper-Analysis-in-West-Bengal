import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

near_data = near_df[
    (
        near_df['timestamp'] >= start_date
    ) &
    (
        near_df['timestamp'] <= end_date
    )
].copy()

far_data = far_df[
    (
        far_df['timestamp'] >= start_date
    ) &
    (
        far_df['timestamp'] <= end_date
    )
].copy()

# =========================================================
# HOUR
# =========================================================

near_data['hour'] = (
    near_data['timestamp']
    .dt.hour
)

far_data['hour'] = (
    far_data['timestamp']
    .dt.hour
)

# =========================================================
# HOURLY STATISTICS
# =========================================================

near_stats = (
    near_data
    .groupby('hour')['laeqt']
    .agg([
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])
)

far_stats = (
    far_data
    .groupby('hour')['laeqt']
    .agg([
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])
)

# =========================================================
# PLOTTING
# =========================================================

plt.figure(figsize=(14,7))

# =========================================================
# NEAR ROAD
# =========================================================

plt.plot(
    near_stats.index,
    near_stats['median'],
    linewidth=2.5,
    label='Near Major Road (<20m)'
)

plt.fill_between(
    near_stats.index,
    near_stats['q25'],
    near_stats['q75'],
    alpha=0.25
)

# =========================================================
# FAR ROAD
# =========================================================

plt.plot(
    far_stats.index,
    far_stats['median'],
    linewidth=2.5,
    label='Far From Major Road (>400m)'
)

plt.fill_between(
    far_stats.index,
    far_stats['q25'],
    far_stats['q75'],
    alpha=0.25
)

# =========================================================
# LABELS
# =========================================================

plt.xlabel('Hour of Day')

plt.ylabel('LAeqT')

plt.title(
    'Diurnal LAeqT Profiles\n'
    f'({period_label})'
)

plt.xticks(range(0,24))

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()

