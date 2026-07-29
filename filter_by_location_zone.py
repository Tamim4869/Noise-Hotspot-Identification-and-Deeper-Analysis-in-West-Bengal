import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os 

os.getcwd()
os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots")

df= pd.read_csv("locations_coords_zones_2.csv")

# Example:
district = "North 24 Pgs."
zone = "Commercial"


stations = df.loc[
    (df['district_x'] == district) &
    (df['zone'] == zone),
    'location'
]

print(stations)

import osmnx
import geopandas as gpd
from shapely.geometry import Point

# Filter districts
target_df = df[
    df['district_x'].isin(['Kolkata', 'North 24 Pgs.'])
].copy()

# Create geometry
target_df['geometry'] = target_df.apply(
    lambda row: Point(row['long_x'], row['lat_x']),
    axis=1
)

stations_gdf = gpd.GeoDataFrame(
    target_df,
    geometry='geometry',
    crs='EPSG:4326'
)


import osmnx as ox

place = "Kolkata, West Bengal, India"

G = ox.graph_from_place(
    place,
    network_type='drive'
)

nodes, edges = ox.graph_to_gdfs(G)

major_types = [
    'motorway',
    'trunk',
    'primary',
    'secondary'
]

major_roads = edges[
    edges['highway'].apply(
        lambda x: any(
            road in major_types
            for road in (x if isinstance(x, list) else [x])
        )
    )
]

stations_gdf = stations_gdf.to_crs(epsg=32645)
major_roads = major_roads.to_crs(epsg=32645)

road_buffer = major_roads.buffer(200)

buffer_union = road_buffer.union_all()

near_traffic = stations_gdf[
    stations_gdf.geometry.within(buffer_union)
]

print(
    near_traffic[
        ['district_x', 'location']
    ]
)










