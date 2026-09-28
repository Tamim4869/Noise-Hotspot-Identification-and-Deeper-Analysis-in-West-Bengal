"""
Noise Hotspot Identification — Greater Kolkata Region
======================================================
Pipeline:
  1. Config & constants
  2. Data loading (metadata + station CSVs)
  3. Exceedance proportion computation (festive & monthly, day & night)
  4. Spatial weights matrix (binary, 3 km threshold)
  5. Moran's I (spatial autocorrelation test)
  6. Getis-Ord Gi* (hotspot / coldspot classification)
  7. Visualisation — Gi* classification maps (contextily basemap)
  8. Visualisation — Exceedance proportion maps (contextily basemap)

Dependencies:
    pip install pandas numpy geopandas pysal libpysal esda contextily
                matplotlib shapely tqdm
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import os
import warnings
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran
from libpysal.weights import DistanceBand
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  Config & constants
# ─────────────────────────────────────────────

# ── Paths ────────────────────────────────────
BASE_DIR      = Path("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/Real imputed stuff Copy_4km_z10")                        # adjust if needed
IMPUTED_DIR   = BASE_DIR / "Imputed"
METADATA_PATH = BASE_DIR / "metadata.csv"          # adjust filename if needed
OUTPUT_DIR    = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Spatial filter (Greater Kolkata) ─────────
LAT_MIN, LAT_MAX   = 22.45, 22.75
LON_MIN, LON_MAX   = 88.20, 88.55

# ── CPCB zone limits (LAeq, dB) ──────────────
ZONE_LIMITS = {
    "Industrial":  {"day": 75, "night": 70},
    "Commercial":  {"day": 65, "night": 55},
    "Residential": {"day": 55, "night": 45},
    "Silence zone":     {"day": 50, "night": 40},
}

# ── Spatial weight threshold (km → degrees approx, but we use projected CRS) ─
DISTANCE_THRESHOLD_M = 4_000          # 3 km in metres (after reprojection)

# ── Gi* significance threshold ───────────────
# Classification uses permutation p_sim < 0.05 (two-sided); see classify_gi()
P_THRESHOLD = 0.10

# ── Minimum data points ──────────────────────
MIN_DAYS_MONTH   = 17                 # for monthly analysis
# for festive: exclude if > 50 % missing  (handled per window length)

# ── Festive periods per year ─────────────────
#    Each entry: (year, label, (month,day)_start, (month,day)_end)
#    New-Year carry-over Jan 1-5 is assigned to the *preceding* year's festival.

def _make_festive_periods():
    raw = {
        2023: [
            ("Holi",             (3,  3), (3, 14)),
            ("Durga_Puja",       (10,14), (10,31)),
            ("Diwali",           (11, 2), (11,17)),
            ("Christmas_NewYear",(12,22), (12,31)),
            ("NewYear_Carryover",( 1, 1), ( 1, 5)),   # Jan 2023 → belongs to 2022-23
        ],
        2024: [
            ("Holi",             (3, 19), (3, 30)),
            ("Durga_Puja",       (10, 2), (10,18)),
            ("Diwali",           (10,21), (11,10)),
            ("Christmas_NewYear",(12,20), (12,31)),
            ("NewYear_Carryover",( 1, 1), ( 1, 5)),   # Jan 2024 → belongs to 2023-24
        ],
        2025: [
            ("Holi",             (3,  8), (3, 19)),
            ("Durga_Puja",       (9, 21), (10, 9)),
            ("Diwali",           (10,15), (10,28)),
            ("Christmas_NewYear",(12,20), (12,31)),
            ("NewYear_Carryover",( 1, 1), ( 1, 5)),   # Jan 2025 → belongs to 2024-25
        ],
    }

    records = []
    for year, festivals in raw.items():
        for label, (sm, sd), (em, ed) in festivals:
            # New-Year carry-over: Jan belongs to the same festival-year entry
            # but the actual calendar year of the dates is year itself for 2023
            # (Jan 2023), year itself for 2024 (Jan 2024), etc.
            start = date(year, sm, sd)
            end   = date(year, em, ed)
            records.append({"year": year, "festival": label,
                             "start": start, "end": end})
    return pd.DataFrame(records)

FESTIVE_DF = _make_festive_periods()


# ─────────────────────────────────────────────
# 2.  Data loading
# ─────────────────────────────────────────────

def load_metadata():
    """Load metadata and filter to Greater Kolkata stations."""
    meta = pd.read_csv(METADATA_PATH)
    meta = meta.rename(columns=str.strip)

    # Spatial filter
    mask = (
        meta["lat_x"].between(LAT_MIN, LAT_MAX) &
        meta["long_x"].between(LON_MIN, LON_MAX)
    )
    meta = meta[mask].copy().reset_index(drop=True)
    print(f"[INFO] {len(meta)} stations within Greater Kolkata bounds.")
    return meta


def _folder_name(location: str) -> str:
    """Convert station location name to folder name (spaces → underscores)."""
    return location.replace(" ", "_")


def load_station_data(meta: pd.DataFrame) -> dict:
    """
    Returns a dict keyed by SL (station ID):
        {SL: {"day": df_day, "night": df_night, "zone": zone_str}}
    """
    station_data = {}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading CSVs"):
        sl       = row["SL"]
        location = row["location"]
        zone     = row["zone"]
        folder   = IMPUTED_DIR / _folder_name(location)

        day_path   = folder / f"{sl}_leq_day_imputed.csv"
        night_path = folder / f"{sl}_leq_night_imputed.csv"

        dfs = {}
        for period, path, date_col, val_col in [
            ("day",   day_path,   "date", "laeq_day"),
            ("night", night_path, "date", "laeq_night"),
        ]:
            if not path.exists():
                print(f"[WARN] Missing: {path}")
                dfs[period] = None
                continue

            df = pd.read_csv(path, parse_dates=["date"])
            df = df[["date", val_col]].dropna(subset=["date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            dfs[period] = df

        station_data[sl] = {
            "day":   dfs["day"],
            "night": dfs["night"],
            "zone":  zone,
            "lat":   row["lat_x"],
            "lon":   row["long_x"],
            "location": location,
        }

    return station_data


# ─────────────────────────────────────────────
# 3.  Exceedance proportion helpers
# ─────────────────────────────────────────────

def exceedance_proportion(series: pd.Series, limit: float) -> float:
    """Proportion of non-NaN values that exceed `limit`."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid > limit).sum() / len(valid)


# ── 3a. Festive exceedance table ─────────────

def compute_festive_exceedance(station_data: dict) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with columns:
        SL, festival, year, period, exceedance_prop, lat, lon, zone
    Excludes station-festival-year-period where > 50 % of expected days are missing.
    """
    records = []

    for _, fest_row in FESTIVE_DF.iterrows():
        fest   = fest_row["festival"]
        year   = fest_row["year"]
        start  = fest_row["start"]
        end    = fest_row["end"]

        # All calendar dates in this window
        all_dates = pd.date_range(start, end, freq="D")
        n_expected = len(all_dates)

        for sl, info in station_data.items():
            zone = info["zone"]
            limits = ZONE_LIMITS.get(zone, None)
            if limits is None:
                continue

            for period in ["day", "night"]:
                df = info[period]
                limit = limits[period]

                if df is None:
                    continue

                # Filter to festive window
                mask = df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
                subset = df.loc[mask]

                val_col = f"laeq_{period}"
                n_valid = subset[val_col].notna().sum()

                # Exclude if > 50 % missing
                if n_valid < n_expected / 2:
                    continue

                prop = exceedance_proportion(subset[val_col], limit)

                records.append({
                    "SL":              sl,
                    "festival":        fest,
                    "year":            year,
                    "period":          period,
                    "exceedance_prop": prop,
                    "lat":             info["lat"],
                    "lon":             info["lon"],
                    "zone":            zone,
                    "location":        info["location"],
                })

    return pd.DataFrame(records)


# ── 3b. Monthly exceedance table 

def compute_monthly_exceedance(station_data: dict) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with columns:
        SL, year, month, period, exceedance_prop, lat, lon, zone
    Excludes station-year-month-period where fewer than 20 valid days.
    """
    records = []

    for sl, info in tqdm(station_data.items(), desc="Monthly exceedance"):
        zone   = info["zone"]
        limits = ZONE_LIMITS.get(zone, None)
        if limits is None:
            continue

        for period in ["day", "night"]:
            df = info[period]
            if df is None:
                continue

            val_col = f"laeq_{period}"
            limit   = limits[period]

            df = df.copy()
            df["year"]  = df["date"].dt.year
            df["month"] = df["date"].dt.month

            for (yr, mo), grp in df.groupby(["year", "month"]):
                n_valid = grp[val_col].notna().sum()
                if n_valid < MIN_DAYS_MONTH:
                    continue

                prop = exceedance_proportion(grp[val_col], limit)

                records.append({
                    "SL":              sl,
                    "year":            yr,
                    "month":           mo,
                    "period":          period,
                    "exceedance_prop": prop,
                    "lat":             info["lat"],
                    "lon":             info["lon"],
                    "zone":            zone,
                    "location":        info["location"],
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 4.  Spatial weights (binary, 3 km)
# ─────────────────────────────────────────────

def build_weights(gdf: gpd.GeoDataFrame):
    """
    Build a binary DistanceBand weight matrix (threshold = 3 km).
    Input GDF must be in a projected CRS (metres).

    Stations with NO neighbours within 3 km (islands) are dropped —
    they cause a reshape error in esda's crand permutation engine.

    Returns (gdf_filtered, weights) or (None, None) if < 3 stations remain.
    """
    if len(gdf) < 3:
        return None, None

    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    w = DistanceBand(coords, threshold=DISTANCE_THRESHOLD_M, binary=True)

    # Identify and remove islands (cardinality == 0)
    islands = [i for i, card in w.cardinalities.items() if card == 0]
    if islands:
        keep_idx = [i for i in range(len(gdf)) if i not in islands]
        if len(keep_idx) < 3:
            return None, None
        gdf    = gdf.iloc[keep_idx].reset_index(drop=True)
        coords = list(zip(gdf.geometry.x, gdf.geometry.y))
        w      = DistanceBand(coords, threshold=DISTANCE_THRESHOLD_M, binary=True)

    w.transform = "r"   # row-standardise for Moran's I
    return gdf, w


# ─────────────────────────────────────────────
# 5.  Moran's I
# ─────────────────────────────────────────────

def run_morans_i(values: np.ndarray, w: DistanceBand) -> dict:
    """
    Run Moran's I on `values` with weights `w`.
    Returns dict with I, p_sim, z_sim.
    """
    mi = Moran(values, w, permutations=999)
    return {
        "moran_I":  mi.I,
        "p_sim":    mi.p_sim,
        "z_sim":    mi.z_sim,
        "significant": mi.p_sim < 0.10,
    }


# ─────────────────────────────────────────────
# 6.  Getis-Ord Gi*
# ─────────────────────────────────────────────

def run_gi_star(values: np.ndarray, w: DistanceBand) -> tuple:
    """
    Run Getis-Ord Gi* and return (Zs, p_sim).

    - Zs   : analytical z-scores (used only to determine direction:
              positive => potential hotspot, negative => potential coldspot)
    - p_sim: permutation-based two-sided p-value for each location
             (999 permutations; no distributional assumption)

    Classification (applied externally via classify_gi):
        p_sim < 0.05 AND Zs > 0 => Hotspot
        p_sim < 0.05 AND Zs < 0 => Coldspot
        else                     => Not significant
    """
    g = G_Local(values, w, transform="r", permutations=999, star=True)
    return g.Zs, g.p_sim


def classify_gi(Zs: np.ndarray, p_sim: np.ndarray) -> np.ndarray:
    """
    Classify each location using permutation-based p-values.
    Direction (hot vs cold) is determined by the sign of the z-score.
    Significance threshold: p_sim < 0.05 (two-sided).
    """
    labels = np.full(len(Zs), "Not significant", dtype=object)
    labels[(p_sim < 0.10) & (Zs > 0)] = "Hotspot"
    labels[(p_sim < 0.10) & (Zs < 0)] = "Coldspot"
    return labels


# ─────────────────────────────────────────────
# 7 & 8.  Visualisation helpers
# ─────────────────────────────────────────────

# Colour maps
GI_COLORS = {
    "Hotspot":         "#d73027",   # red
    "Coldspot":        "#4575b4",   # blue
    "Not significant": "#d3d3d3",   # light grey
}

PROP_CMAP = "YlOrRd"

CRS_GEO  = "EPSG:4326"
CRS_PROJ = "EPSG:32645"    # UTM Zone 45N — appropriate for Kolkata


def _prepare_gdf(sub_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a subset DataFrame to a projected GeoDataFrame."""
    gdf = gpd.GeoDataFrame(
        sub_df.copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(sub_df["lat"], sub_df["lon"])],
        crs=CRS_GEO,
    )
    return gdf.to_crs(CRS_PROJ)


def _base_map_ax(gdf_proj: gpd.GeoDataFrame, title: str, figsize=(9, 8)):
    """Create a figure + axes with contextily basemap."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # Add a small buffer around the points for the map extent
    buf = 3_000   # metres
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    ax.set_xlim(minx - buf, maxx + buf)
    ax.set_ylim(miny - buf, maxy + buf)

    cx.add_basemap(ax, crs=CRS_PROJ, source=cx.providers.OpenStreetMap.Mapnik,
                   zoom="auto", attribution_size=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_axis_off()
    return fig, ax


# ── Plot: Gi* classification ─────────────────

def plot_gi_classification(gdf_proj: gpd.GeoDataFrame, labels: np.ndarray,
                            title: str, save_path: Path):
    gdf_proj = gdf_proj.copy()
    gdf_proj["gi_class"] = labels

    fig, ax = _base_map_ax(gdf_proj, title)

    for cls, color in GI_COLORS.items():
        subset = gdf_proj[gdf_proj["gi_class"] == cls]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=color, markersize=60, edgecolor="white",
                    linewidth=0.5, zorder=3, label=cls)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in GI_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.85, title="Gi* Class", title_fontsize=8)

    # Annotate station names
    for _, row in gdf_proj.iterrows():
        ax.annotate(str(row["SL"]), xy=(row.geometry.x, row.geometry.y),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=5, color="#333333", zorder=4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot: Exceedance proportion ──────────────

def plot_exceedance_prop(gdf_proj: gpd.GeoDataFrame, title: str, save_path: Path):
    fig, ax = _base_map_ax(gdf_proj, title)

    sc = gdf_proj.plot(
        ax=ax,
        column="exceedance_prop",
        cmap=PROP_CMAP,
        vmin=0, vmax=1,
        markersize=70,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
        zorder=3,
    )

    # Colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=PROP_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Exceedance proportion", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for _, row in gdf_proj.iterrows():
        ax.annotate(str(row["SL"]), xy=(row.geometry.x, row.geometry.y),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=5, color="#333333", zorder=4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# 9.  Main pipeline — Purpose 1 (Festive)
# ─────────────────────────────────────────────

def run_festive_analysis(festive_exc: pd.DataFrame):
    """
    For each (festival, year, period) combination:
      - Build spatial weights
      - Run Moran's I
      - Run Gi* → classify
      - Save both Gi* and exceedance proportion maps
    """
    print("\n" + "="*60)
    print("PURPOSE 1 — Festive Hotspot / Coldspot Analysis")
    print("="*60)

    moran_records = []
    out_dir = OUTPUT_DIR / "festive"
    out_dir.mkdir(exist_ok=True)

    combos = festive_exc.groupby(["festival", "year", "period"])

    for (festival, year, period), sub in tqdm(combos, desc="Festive combos"):
        sub = sub.dropna(subset=["exceedance_prop"]).reset_index(drop=True)

        if len(sub) < 3:
            print(f"[SKIP] {festival} {year} {period}: only {len(sub)} stations.")
            continue

        gdf_raw  = _prepare_gdf(sub)
        gdf, w   = build_weights(gdf_raw)

        if w is None or w.n < 3:
            print(f"[SKIP] {festival} {year} {period}: insufficient neighbours after dropping islands.")
            continue

        # Use filtered gdf (islands removed) for both values and plots
        values = gdf["exceedance_prop"].values

        # Moran's I
        mi = run_morans_i(values, w)
        moran_records.append({"festival": festival, "year": year,
                               "period": period, **mi})

        # Gi*
        Zs, p_sim = run_gi_star(values, w)
        labels    = classify_gi(Zs, p_sim)

        tag   = f"{festival}_{year}_{period}"
        title_base = f"{festival.replace('_',' ')} {year} — {period.capitalize()} period"

        # Gi* map
        plot_gi_classification(
            gdf, labels,
            title=f"Gi* Hotspots/Coldspots\n{title_base}",
            save_path=out_dir / f"gi_{tag}.png",
        )

        # Exceedance proportion map
        plot_exceedance_prop(
            gdf,
            title=f"LAeq Exceedance Proportion\n{title_base}",
            save_path=out_dir / f"exceedance_{tag}.png",
        )

    # Save Moran's I summary
    moran_df = pd.DataFrame(moran_records)
    moran_df.to_csv(out_dir / "morans_i_festive.csv", index=False)
    print(f"\n[DONE] Festive maps saved to: {out_dir}")
    print(f"[DONE] Moran's I results saved to: {out_dir / 'morans_i_festive.csv'}")
    return moran_df


# ─────────────────────────────────────────────
# 10.  Main pipeline — Purpose 2 (Monthly)
# ─────────────────────────────────────────────

def run_monthly_analysis(monthly_exc: pd.DataFrame):
    """
    For each (year, month, period) combination:
      - Build spatial weights
      - Run Moran's I
      - Run Gi* → classify
      - Save both Gi* and exceedance proportion maps
    """
    print("\n" + "="*60)
    print("PURPOSE 2 — Monthly Hotspot / Coldspot Analysis")
    print("="*60)

    moran_records = []
    out_dir = OUTPUT_DIR / "monthly"
    out_dir.mkdir(exist_ok=True)

    combos = monthly_exc.groupby(["year", "month", "period"])

    for (year, month, period), sub in tqdm(combos, desc="Monthly combos"):
        sub = sub.dropna(subset=["exceedance_prop"]).reset_index(drop=True)

        if len(sub) < 3:
            print(f"[SKIP] {year}-{month:02d} {period}: only {len(sub)} stations.")
            continue

        gdf_raw  = _prepare_gdf(sub)
        gdf, w   = build_weights(gdf_raw)

        if w is None or w.n < 3:
            print(f"[SKIP] {year}-{month:02d} {period}: insufficient neighbours after dropping islands.")
            continue

        # Use filtered gdf (islands removed) for both values and plots
        values = gdf["exceedance_prop"].values

        # Moran's I
        mi = run_morans_i(values, w)
        moran_records.append({"year": year, "month": month,
                               "period": period, **mi})

        # Gi*
        Zs, p_sim = run_gi_star(values, w)
        labels    = classify_gi(Zs, p_sim)

        month_name = date(year, month, 1).strftime("%b")
        tag        = f"{year}_{month:02d}_{period}"
        title_base = f"{month_name} {year} — {period.capitalize()} period"

        # Gi* map
        plot_gi_classification(
            gdf, labels,
            title=f"Gi* Hotspots/Coldspots\n{title_base}",
            save_path=out_dir / f"gi_{tag}.png",
        )

        # Exceedance proportion map
        plot_exceedance_prop(
            gdf,
            title=f"LAeq Exceedance Proportion\n{title_base}",
            save_path=out_dir / f"exceedance_{tag}.png",
        )

    # Save Moran's I summary
    moran_df = pd.DataFrame(moran_records)
    moran_df.to_csv(out_dir / "morans_i_monthly.csv", index=False)
    print(f"\n[DONE] Monthly maps saved to: {out_dir}")
    print(f"[DONE] Moran's I results saved to: {out_dir / 'morans_i_monthly.csv'}")
    return moran_df


# ─────────────────────────────────────────────
# 11.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Load data ────────────────────
    meta         = load_metadata()
    station_data = load_station_data(meta)

    # ── Step 2: Compute exceedance proportions
    print("\n[INFO] Computing festive exceedance proportions...")
    festive_exc  = compute_festive_exceedance(station_data)
    festive_exc.to_csv(OUTPUT_DIR / "festive_exceedance.csv", index=False)
    print(f"[INFO] Festive exceedance table: {festive_exc.shape}")

    print("\n[INFO] Computing monthly exceedance proportions...")
    monthly_exc  = compute_monthly_exceedance(station_data)
    monthly_exc.to_csv(OUTPUT_DIR / "monthly_exceedance.csv", index=False)
    print(f"[INFO] Monthly exceedance table: {monthly_exc.shape}")

    # ── Step 3: Hotspot analyses ─────────────
    moran_festive = run_festive_analysis(festive_exc)
    moran_monthly = run_monthly_analysis(monthly_exc)

    print("\n[ALL DONE] ✓")






