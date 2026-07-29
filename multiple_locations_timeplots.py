import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots/Imputed")

selected_ids = {84, 91, 163}  # <-- your chosen stations

plt.figure(figsize=(12,6))

for station_folder in base_path.iterdir():
    if station_folder.is_dir():
        
        day_files = list(station_folder.glob("*_leq_day_imputed.csv"))
        
        if day_files:
            file = day_files[0]
            
            # extract serial number from filename
            sl_no = int(file.name.split('_')[0])
            
            if sl_no in selected_ids:
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values('date')

                plt.plot(df['date'], df['laeq_day'], label=f"{sl_no}")

plt.xlabel("Date")
plt.ylabel("LAeq Day")
plt.title("Selected Stations")
plt.legend()
plt.tight_layout()
plt.show()




