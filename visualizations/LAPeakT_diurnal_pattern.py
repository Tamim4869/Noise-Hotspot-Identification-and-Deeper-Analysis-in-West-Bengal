import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots/References for segregation")


# =========================================================
# STATIONS
# =========================================================

station_files = {
    'Station 1' : 'Titagarh_Municipality.csv',
    'Station 2'  : 'Madhyamgram_Municipality.csv',
    'Station 3': 'Rajarhat_Police_Station.csv', 
    'Station 4': 'Ballygunge_Campus,_C.U.csv'
}

# =========================================================
# FESTIVE PERIOD
# =========================================================

start_date = '2025-09-25'
end_date   = '2026-01-10'

period_label = (
    f"{pd.to_datetime(start_date).strftime('%d-%b-%Y')} "
    f"to "
    f"{pd.to_datetime(end_date).strftime('%d-%b-%Y')}"
)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14,7))

# =========================================================
# LOOP THROUGH STATIONS
# =========================================================

for station_name, file in station_files.items():

    # -----------------------------------------------------
    # load
    # -----------------------------------------------------

    df_raw = pd.read_csv(file)

    near_ts_here = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce').dropna()
    
    mask = near_ts_here.ne(near_ts_here.shift())
    df = df_raw.loc[mask].copy()
    near_ts_here= near_ts_here.loc[mask]

    df['timestamp'] = pd.to_datetime(
        df['timestamp']
    )

    # -----------------------------------------------------
    # filter period
    # -----------------------------------------------------

    data = df[
        (
            df['timestamp'] >= start_date
        ) &
        (
            df['timestamp'] <= end_date
        )
    ].copy()

    # -----------------------------------------------------
    # hour
    # -----------------------------------------------------

    data['hour'] = (
        data['timestamp']
        .dt.hour
    )

    # -----------------------------------------------------
    # hourly stats
    # -----------------------------------------------------

    stats = (
        data
        .groupby('hour')['lapeakt']
        .agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ])
    )

    # -----------------------------------------------------
    # median curve
    # -----------------------------------------------------

    plt.plot(
        stats.index,
        stats['median'],
        linewidth=2.5,
        label=station_name
    )

    # -----------------------------------------------------
    # IQR band
    # -----------------------------------------------------

    plt.fill_between(
        stats.index,
        stats['q25'],
        stats['q75'],
        alpha=0.15
    )

# =========================================================
# LABELS
# =========================================================

plt.xlabel('Hour of Day')

plt.ylabel('LAPeakT')

plt.title(
    'Festive-Period Diurnal LAPeakT Profiles\n'
    f'({period_label})'
)

plt.xticks(range(0,24))

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()
