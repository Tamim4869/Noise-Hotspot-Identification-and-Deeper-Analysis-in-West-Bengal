import pandas as pd
import numpy as np
import os
from urllib.parse import quote

os.getcwd()
os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots")

def day_period_thres(ts_series, d=15):
    day = ts_series.dt.date.iloc[0]
    
    start = pd.Timestamp(f"{day} 06:00:00")
    end   = pd.Timestamp(f"{day} 22:00:00")
    
    ts_sorted = ts_series.sort_values()
    g = ts_sorted[(ts_sorted >= start) & (ts_sorted <= end)]
    
    ts_ext = pd.concat([pd.Series([start]), g, pd.Series([end])])
    dt = ts_ext.diff().dropna().dt.total_seconds() / 60
    
    if len(dt) == 0:
        return np.nan
    
    return ((dt - d).clip(lower=0)).sum() + dt.max()


def night_period_thres(ts_series, d=15):
    day = ts_series.dt.date.iloc[0]
    
    start1 = pd.Timestamp(f"{day} 00:01:00")
    end1   = pd.Timestamp(f"{day} 06:00:00")
    start2 = pd.Timestamp(f"{day} 22:00:00")
    end2   = pd.Timestamp(f"{day} 23:59:00")
    
    ts_sorted = ts_series.sort_values()
    
    g1 = ts_sorted[(ts_sorted >= start1) & (ts_sorted <= end1)]
    ts1 = pd.concat([pd.Series([start1]), g1, pd.Series([end1])])
    dt1 = ts1.diff().dropna().dt.total_seconds() / 60
    
    g2 = ts_sorted[(ts_sorted >= start2) & (ts_sorted <= end2)]
    ts2 = pd.concat([pd.Series([start2]), g2, pd.Series([end2])])
    dt2 = ts2.diff().dropna().dt.total_seconds() / 60
    
    dt_all = pd.concat([dt1, dt2])
    
    if len(dt_all) == 0:
        return np.nan
    
    return ((dt_all - d).clip(lower=0)).sum() + dt_all.max()



def compute_leq_day(group, col):
    day = group['ts'].dt.date.iloc[0]
    
    start = pd.Timestamp(f"{day} 06:00:00")
    end   = pd.Timestamp(f"{day} 22:00:00")
    
    g = group[(group['ts'] >= start) & (group['ts'] <= end)].sort_values('ts')
    
    if g.empty:
        return np.nan
    
    # timestamps + values
    ts_vals = g['ts'].reset_index(drop=True)
    s_vals = g[col].reset_index(drop=True)
    
    # prepend 6am, append 10pm
    ts_ext = pd.concat([pd.Series([start]), ts_vals, pd.Series([end])])
    
    # compute Δ_i
    dt = ts_ext.diff().dropna().dt.total_seconds() / 60
    
    # 🔴 EXPLICITLY DROP LAST GAP
    dt = dt.iloc[:-1]
    
    # now lengths match correctly
    num = np.sum(dt * 10**(s_vals / 10))
    den = np.sum(dt)
    
    return 10 * np.log10(num / den) if den > 0 else np.nan


def compute_leq_night(group, col):
    day = group['ts'].dt.date.iloc[0]
    
    ts_sorted = group['ts'].sort_values().reset_index(drop=True)
    s_all = group[col].reset_index(drop=True)
    
    # ---- helper for one window ----
    def window_leq(start, end):
        g = group[(group['ts'] >= start) & (group['ts'] <= end)].sort_values('ts')
        
        if g.empty:
            return 0.0, 0.0  # (numerator, denominator)
        
        ts_vals = g['ts'].reset_index(drop=True)
        s_vals = g[col].reset_index(drop=True)
        
        ts_ext = pd.concat([pd.Series([start]), ts_vals, pd.Series([end])])
        dt = ts_ext.diff().dropna().dt.total_seconds() / 60
        
        # 🔴 drop last gap
        dt = dt.iloc[:-1]
        
        if len(dt) == 0:
            return 0.0, 0.0
        
        num = np.sum(dt * 10**(s_vals / 10))
        den = np.sum(dt)
        
        return num, den
    
    # ---- define windows ----
    start1 = pd.Timestamp(f"{day} 00:01:00")
    end1   = pd.Timestamp(f"{day} 06:00:00")
    
    start2 = pd.Timestamp(f"{day} 22:00:00")
    end2   = pd.Timestamp(f"{day} 23:59:00")
    
    # compute both
    num1, den1 = window_leq(start1, end1)
    num2, den2 = window_leq(start2, end2)
    
    num_total = num1 + num2
    den_total = den1 + den2
    
    return 10 * np.log10(num_total / den_total) if den_total > 0 else np.nan



def process_location(url, sl_no, location):

    url = quote(url, safe=':/')

    try:
        trial_dataset = pd.read_csv(url)
    except Exception as e:
        print(f"Error reading {url}: {e}")
        return

    # ----- remove repeated timestamps -----
    ts_here = pd.to_datetime(trial_dataset.iloc[:,0], errors='coerce')
    mask_year = ts_here.dt.year >= 2021

    trial_dataset = trial_dataset.loc[mask_year].copy()
    ts_here = ts_here.loc[mask_year]
    
    # Now create duplicate-removal mask
    mask = ts_here.notna() & ts_here.ne(ts_here.shift())
    
    df_clean = trial_dataset.loc[mask].copy()
    ts_here = ts_here.loc[mask]

    df_clean['ts'] = ts_here
    df_clean['date'] = ts_here.dt.date

    # ----- compute day/night filtered results -----
    day_result = df_clean.groupby('date').apply(
        lambda g: pd.Series({
            'laeq_day': compute_leq_day(g,'laeqt'),
            'lceq_day': compute_leq_day(g,'lceqt'),
            'lzeq_day': compute_leq_day(g,'lzeqt'),
            'day_thres': day_period_thres(g['ts'])
        })
    ).reset_index()

    day_result = day_result[day_result['day_thres'] <= 300].drop(columns='day_thres')

    night_result = df_clean.groupby('date').apply(
        lambda g: pd.Series({
            'laeq_night': compute_leq_night(g,'laeqt'),
            'lceq_night': compute_leq_night(g,'lceqt'),
            'lzeq_night': compute_leq_night(g,'lzeqt'),
            'night_thres': night_period_thres(g['ts'])
        })
    ).reset_index()

    night_result = night_result[night_result['night_thres'] <= 150].drop(columns='night_thres')

    # ----- interpolation -----

    def interpolate_df(df):
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').asfreq('D')
    
        # compute mask on THIS df
        is_nan = df.isna().all(axis=1)
        group = (is_nan != is_nan.shift()).cumsum()
        group_sizes = is_nan.groupby(group).transform('sum')
    
        fill_mask = is_nan & (group_sizes <= 2)
    
        # interpolate on SAME df (preserves index)
        df_interp = df.interpolate(method='linear', limit= 2)
    
        # 🔴 use .loc (prevents alignment issues)
        df.loc[fill_mask] = df_interp.loc[fill_mask]
    
        return df.reset_index()

    day_imputed = interpolate_df(day_result)
    night_imputed = interpolate_df(night_result)

    # ----- save to location folder -----

    base_dir = "Imputed"

    # create main folder once
    os.makedirs(base_dir, exist_ok=True)
    
    # location-specific folder inside it
    folder = os.path.join(base_dir, location.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    day_file = os.path.join(folder, f"{sl_no}_leq_day_imputed.csv")
    night_file = os.path.join(folder, f"{sl_no}_leq_night_imputed.csv")

    day_imputed.to_csv(day_file, index=False)
    night_imputed.to_csv(night_file, index=False)

    print(f"{location} finished.")
    
    
 
big_df = pd.read_csv("lcz_urls.csv")

for _, row in big_df.iterrows():

    process_location(
        row['cdn_url'],
        row['SL'],
        row['location']
    )
    

    
    