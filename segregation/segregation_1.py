import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os 

# =============================================================================
# STEP 1 — Load & Clean
# =============================================================================

os.getcwd()
os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots")

trial_dataset= pd.read_csv("WBPCB_Office,_Saltlake.csv")

ts_here = pd.to_datetime(trial_dataset.iloc[:, 0], errors='coerce').dropna()

mask = ts_here.ne(ts_here.shift())
df = trial_dataset.loc[mask].copy()
ts_here= ts_here.loc[mask]

df['ts']= ts_here
df['timestamp'] = pd.to_datetime(df['timestamp'])

df["date"]  = df["timestamp"].dt.normalize()
df["year"]  = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["hour"]  = df["timestamp"].dt.hour

# Window duration & short-window flag
df["window_min"]   = df["timestamp"].diff().dt.total_seconds() / 60
df["short_window"] = df["window_min"] < 8

print(f"Total rows           : {len(df)}")
print(f"Short window rows    : {df['short_window'].sum()}")
print(f"Missing laeqt        : {df['laeqt'].isna().sum()}")
print(f"Missing lapeakt      : {df['lapeakt'].isna().sum()}")

# Drop rows missing key columns
df = df.dropna(subset=["laeqt", "lapeakt", "lceqt", "lzeqt"]).copy()

# =============================================================================
# STEP 1a — Keep only years >= 2023 AND spanning the full calendar year
# =============================================================================

df = df[df["year"] >= 2023].copy()

def spans_full_year(group):
    dates    = group["date"]
    year     = group["year"].iloc[0]
    n_days   = dates.nunique()
    min_ok   = dates.min() <= pd.Timestamp(f"{year}-01-01")
    max_ok   = dates.max() >= pd.Timestamp(f"{year}-12-31")
    cover_ok = n_days >= 300
    return min_ok and max_ok and cover_ok

valid_years = [
    year for year, group in df.groupby("year")
    if spans_full_year(group)
]

print(f"\nYears with full-year coverage: {valid_years}")
df = df[df["year"].isin(valid_years)].copy()

# =============================================================================
# STEP 2 — Feature Engineering
# =============================================================================

df["impulsiveness"] = df["lapeakt"] - df["laeqt"]
df["lc_la"]         = df["lceqt"]   - df["laeqt"]
df["lz_la"]         = df["lzeqt"]   - df["laeqt"]

df["is_night"]    = df["hour"].between(20, 23) | df["hour"].between(0, 4)
df["is_evening"]  = df["hour"].between(17, 20)
df["is_rushhour"] = df["hour"].between(9, 13)  | df["hour"].between(17, 20)

# =============================================================================
# STEP 3 — Year-specific festive calendars
# =============================================================================

festive_periods = {
    2023: [
        ((3,  3),  (3,  14)),
        ((10, 14), (10, 31)),
        ((11, 2),  (11, 17)),
        ((12, 22), (12, 31)),
        ((1,  1),  (1,   5)),
    ],
    2024: [
        ((3,  19), (3,  30)),
        ((10, 2),  (10, 18)),
        ((10, 21), (11, 10)),
        ((12, 20), (12, 31)),
        ((1,  1),  (1,   5)),
    ],
    2025: [
        ((3,  8),  (3,  19)),
        ((9,  21), (10,  9)),
        ((10, 15), (10, 28)),
        ((12, 20), (12, 31)),
        ((1,  1),  (1,   5)),
    ],
}

def is_festive(row):
    year = row["year"]
    md   = (row["date"].month, row["date"].day)
    # Jan 1-5 carry-over: use previous year's festive calendar
    cal_year = year - 1 if md <= (1, 5) else year
    periods  = festive_periods.get(cal_year, [])
    for (m_start, d_start), (m_end, d_end) in periods:
        if (m_start, d_start) <= md <= (m_end, d_end):
            return True
    return False

df["is_festive"] = df.apply(is_festive, axis=1)

print("\nFestive observations by year:")
print(df.groupby("year")["is_festive"].sum())
print("\nNon-festive observations by year:")
print(df.groupby("year")["is_festive"].apply(lambda x: (~x).sum()))

# =============================================================================
# STEP 4 — Per-Station Baseline (non-festive data, per hour-of-day)
# =============================================================================
# Baseline = median value for each hour, computed on non-festive days only

baseline = (
    df[~df["is_festive"]]
    .groupby("hour")[["laeqt", "impulsiveness", "lc_la", "lz_la"]]
    .median()
    .rename(columns={
        "laeqt"        : "base_laeqt",
        "impulsiveness": "base_impulsiveness",
        "lc_la"        : "base_lc_la",
        "lz_la"        : "base_lz_la",
    })
)

df = df.merge(baseline, on="hour", how="left")

# =============================================================================
# STEP 5 — Rule-Based Classification
# =============================================================================

def classify(row):
    if row['lapeakt'] >100 and row["impulsiveness"] > 30 and row["is_night"]:
        return "Fireworks"
    elif row["lc_la"] > 10 and row["impulsiveness"] < 15:
        return "Loud Music"
    elif row["is_rushhour"] and row["laeqt"] > 60 and row["lc_la"] <= 10:
        return "Traffic"
    else:
        return "Unclassified"

df["noise_type"] = df.apply(classify, axis=1)

print("\nNoise type counts:")
print(df["noise_type"].value_counts())

# =============================================================================
# STEP 6 — Analysis & Visualisation
# =============================================================================

colors = {
    "Festive"      : "#E05C5C",
    "Non-Festive"  : "#4C8FC0",
    "Fireworks"    : "#E05C5C",
    "Loud Music"   : "#F0A500",
    "Traffic"      : "#4C8FC0",
    "Unclassified" : "#AAAAAA",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Peak Noise Characteristics — Station XYZ",
    fontsize=14, fontweight="bold"
)

# ── Panel 1: Impulsiveness KDE — festive vs non-festive ──────────────────────
ax = axes[0, 0]
x_grid = np.linspace(
    df["impulsiveness"].min() - 5,
    df["impulsiveness"].max() + 5,
    500
)
for label, mask, color in [
    ("Non-Festive", ~df["is_festive"], colors["Non-Festive"]),
    ("Festive",      df["is_festive"], colors["Festive"]),
]:
    data = df[mask]["impulsiveness"].dropna()
    if len(data) > 10:
        kde = gaussian_kde(data, bw_method="scott")
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)
ax.set_title("Impulsiveness: Festive vs Non-Festive", fontweight="bold")
ax.set_xlabel("LApeak − LAeq  (dB)")
ax.set_ylabel("Density")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# ── Panel 2: CDF of LApeak by noise type ─────────────────────────────────────
ax = axes[0, 1]
for ntype, color in colors.items():
    if ntype in ["Festive", "Non-Festive"]:
        continue
    data = df[df["noise_type"] == ntype]["lapeakt"].dropna().sort_values()
    if len(data) > 10:
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color=color, linewidth=2, label=ntype)
ax.axvline(x=140, color="black", linestyle="--",
           linewidth=1.2, label="140 dB threshold")
ax.set_title("CDF of LApeak by Noise Type", fontweight="bold")
ax.set_xlabel("LApeak  (dB)")
ax.set_ylabel("Cumulative Probability")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# ── Panel 3: Mean LAeq by hour — festive vs non-festive ──────────────────────
ax = axes[1, 0]
for label, mask, color in [
    ("Non-Festive", ~df["is_festive"], colors["Non-Festive"]),
    ("Festive",      df["is_festive"], colors["Festive"]),
]:
    hourly = df[mask].groupby("hour")["laeqt"].mean()
    ax.plot(hourly.index, hourly.values,
            color=color, linewidth=2,
            marker="o", markersize=4, label=label)
ax.set_title("Mean LAeq by Hour of Day", fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean LAeq  (dB)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# ── Panel 4: Classified noise events — bar chart ─────────────────────────────
ax = axes[1, 1]
counts     = df["noise_type"].value_counts()
bar_colors = [colors.get(t, "#AAAAAA") for t in counts.index]
bars       = ax.bar(counts.index, counts.values,
                    color=bar_colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts.values) * 0.01,
        str(val), ha="center", va="bottom",
        fontsize=10, fontweight="bold"
    )
ax.set_title("Classified Noise Events", fontweight="bold")
ax.set_xlabel("Noise Type")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
### plt.savefig("peak_noise_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# STEP 6b — Summary Statistics (printed)
# =============================================================================

print("\n── Summary Statistics: Impulsiveness ──")
for label, mask in [("Festive", df["is_festive"]), ("Non-Festive", ~df["is_festive"])]:
    data = df[mask]["impulsiveness"].dropna()
    print(f"\n  {label} (n={len(data)}):")
    print(f"    Mean   : {data.mean():.2f} dB")
    print(f"    Median : {data.median():.2f} dB")
    print(f"    Std    : {data.std():.2f} dB")
    print(f"    P95    : {data.quantile(0.95):.2f} dB")
    print(f"    P99    : {data.quantile(0.99):.2f} dB")





# =============================================================================
# STEP 7 — Visualise classified observations
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
fig.suptitle(
    "Classified Noise Observations — WBPCB Office, Saltlake",
    fontsize=14, fontweight="bold"
)

noise_types = ["Fireworks", "Loud Music", "Traffic", "Unclassified"]
colors_type = {
    "Fireworks"    : "#E05C5C",
    "Loud Music"   : "#F0A500",
    "Traffic"      : "#4C8FC0",
    "Unclassified" : "#AAAAAA",
}

# ── Panel per noise type: LAeq time series, coloured by classification ────────
for ax, ntype in zip(axes, noise_types):
    subset = df[df["noise_type"] == ntype]
    ax.scatter(
        subset["timestamp"], subset["laeqt"],
        color=colors_type[ntype], alpha=0.4, s=4, label=ntype
    )
    ax.set_ylabel("LAeq  (dB)", fontsize=10)
    ax.set_title(f"{ntype}  (n = {len(subset):,})", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    # Shade festive periods for reference
    for _, day_group in df[df["is_festive"]].groupby("date"):
        ax.axvspan(
            day_group["timestamp"].min(),
            day_group["timestamp"].max(),
            alpha=0.07, color="red"
        )

axes[-1].set_xlabel("Timestamp", fontsize=10)
plt.tight_layout()
plt.savefig("classified_timeseries_wbpcb_saltlake.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Bonus: single combined time series, all types overlaid ───────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for ntype, color in colors_type.items():
    subset = df[df["noise_type"] == ntype]
    ax.scatter(
        subset["timestamp"], subset["laeqt"],
        color=color, alpha=0.4, s=4, label=ntype
    )
# Shade festive periods
for _, day_group in df[df["is_festive"]].groupby("date"):
    ax.axvspan(
        day_group["timestamp"].min(),
        day_group["timestamp"].max(),
        alpha=0.07, color="red"
    )
ax.set_title("All Classified Observations — LAeq Time Series", fontweight="bold")
ax.set_xlabel("Timestamp")
ax.set_ylabel("LAeq  (dB)")
ax.legend(markerscale=3, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
###  plt.savefig("classified_timeseries_combined.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Bonus: tabular export — inspect the actual rows ──────────────────────────
for ntype in noise_types:
    subset = df[df["noise_type"] == ntype][[
        "timestamp", "laeqt", "lceqt", "lzeqt", "lapeakt",
        "impulsiveness", "lc_la", "is_festive", "noise_type"
    ]]
    subset.to_csv(f"classified_{ntype.lower().replace(' ', '_')}.csv", index=False)
    print(f"Saved {len(subset):,} rows → classified_{ntype.lower().replace(' ', '_')}.csv")






