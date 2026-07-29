import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots")

# --- paths ---
root_dir = "D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots/Imputed"
meta_path = "locations_coords_zones_2_2.csv"

# --- load metadata ---
meta = pd.read_csv(meta_path)

# --- thresholds ---
thresholds = {
    "industrial": 75,
    "commercial": 65,
    "residential": 55,
    "silent": 50
}

results = []

# --- loop over folders ---
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    
    if not os.path.isdir(folder_path):
        continue
    
    # --- find day file ---
    for file in os.listdir(folder_path):
        if file.endswith("_leq_day_imputed.csv"):
            
            file_path = os.path.join(folder_path, file)
            
            # extract slno from filename
            slno = int(file.split("_")[0])
            
            df = pd.read_csv(file_path)
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['laeq_day'])
            
            if df.empty:
                continue
            
            mean_leq = df['laeq_day'].mean()
            
            row = meta[meta['SL'] == slno]
            if row.empty:
                continue
            
            zone_val = row['zone'].values[0]

            if pd.isna(zone_val):
                continue  # skip if zone missing
            
            zone = str(zone_val).lower()
            
            thresh = thresholds.get(zone, np.nan)
            
            exceed_rate = np.mean(df['laeq_day'] > thresh) if not np.isnan(thresh) else np.nan
            
            results.append({
                "SL": slno,
                "mean_leq": mean_leq,
                "exceed_rate": exceed_rate
            })

# --- combine ---
res_df = pd.DataFrame(results)
res_df = res_df.merge(meta, on="SL", how="left")

# --- plot ---
plt.figure(figsize=(8,30))

sc = plt.scatter(
    res_df['long_x'],
    res_df['lat_x'],
    c=res_df['mean_leq'],
    cmap= 'plasma',
    s=200 * res_df['exceed_rate']
)

plt.colorbar(sc, label="Mean LAeq (Day)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Noise Map (Mean Day LAeq)")

plt.show()



























