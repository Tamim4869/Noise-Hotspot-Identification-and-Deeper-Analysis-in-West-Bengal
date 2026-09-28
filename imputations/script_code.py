import pandas as pd
import requests
import os
import re

import json

os.getcwd()
os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data")

df= pd.read_csv("anms_device_master (1).csv")

url_col = "cdn_url"
location_col = "location"
district_col = "district"   # <-- new column

base_dir = "noise data"
os.makedirs(base_dir, exist_ok=True)

def clean_name(name):
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))

seen = set()

for _, row in df.iterrows():
    url = row[url_col]
    location = clean_name(row[location_col]).strip().replace(" ", "_")
    district = clean_name(row[district_col]).strip().replace(" ", "_")

    # create province folder
    district_path = os.path.join(base_dir, district)
    os.makedirs(district_path, exist_ok=True)

    # ensure unique filename (across all files)
    filename = location
    count = 1
    while (district, filename) in seen:
        filename = f"{location}_{count}"
        count += 1
    seen.add((district, filename))

    filepath = os.path.join(district_path, filename + ".csv")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Saved: {filepath}")

    except Exception as e:
        print(f"Failed for {location}: {e}")


with open("get_device_list.json") as f:
    device_list = json.load(f)

devs= pd.DataFrame(device_list)

devs_flat = pd.json_normalize(devs['data'])

# pick required columns
devs_final = devs_flat[['location', 'district', 'lat', 'long']]

merged_dummy= df.merge(devs_final, on="location", how="left")

merged_dummy.to_csv("merged.csv", index=False)

merged_dummy[merged_dummy[['lat','long']].isna().all(axis=1)].index
merged_dummy[merged_dummy[['lat','long']].isna().any(axis=1)]

missing = df[~df['location'].isin(devs_final['location'])]

missing['location']
missing.shape[0]

missing2= devs_final[~devs_final['location'].isin(df['location'])]
missing2['location']
missing2.shape[0]

new_df= pd.read_csv("location_coords.csv")

devs_final2= devs_flat[['location', 'lat', 'long', 'zone']]

new_merged_dummy= new_df.merge(devs_final2, on="location", how="left")

new_merged_dummy.to_csv("new_merged.csv", index= False)









































