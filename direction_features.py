import pandas as pd
import numpy as np   
from tqdm import tqdm


df = pd.read_csv('../input/train.csv', nrows=100000)

num_points = 10 # split the line between pickup and dropoff to 10 parts

lon_cut = (df.dropoff_longitude - df.pickup_longitude)/num_points
lat_cut = (df.dropoff_latitude - df.pickup_latitude)/num_points

lon_dict = {'Manhattan': -73.9712,
            'Brooklyn': -73.944,
            'Queens': -73.7949,
            'Bronx': -73.8648,
            'Staten': -74.1502,
            'JFK': -73.778889,
            'EWR': -74.168611,
            'LGA': -73.872611,
            'liberty': -74.0445}

lat_dict = {'Manhattan': 40.7831,
            'Brooklyn': 40.6782,
            'Queens': 40.7282,
            'Bronx': 40.8448,
            'Staten': 40.5795,
            'JFK': 40.639722,
            'EWR': 40.6925,
            'LGA': 40.77725,
            'liberty': 40.6892}

dist_dict = {}
for i in tqdm(range(num_points)):
    lons = df.pickup_longitude + lon_cut*i
    lats = df.pickup_latitude + lat_cut*i
    
    for sub in lon_dict.keys():
        dist = np.sqrt((lons - lon_dict[sub])**2 + (lats - lat_dict[sub])**2)
        dist_dict[sub+str(i)] = dist


def get_stats(df):
    mean_area = df.mean(axis=1)
    max_area = df.max(axis=1)
    min_area = df.min(axis=1)
    min_idx = df.idxmin(axis=1).values
    direction_into = min_idx == df.columns[-1]
    direction_out = min_idx == df.columns[0]
    
    new_df = pd.DataFrame()
    new_df['mean_'] = mean_area
    new_df['max_'] = max_area
    new_df['min_'] = min_area
    new_df['direction_in_'] = direction_into
    new_df['direction_out_'] = direction_out
    return new_df

res_df = pd.DataFrame()
for ki in dist_dict.keys():
    res_df[ki] = dist_dict[ki]

container = []
for ki in tqdm(lon_dict.keys()):
    cols = [col for col in res_df.columns if ki in col]
    new_feature = get_stats(res_df.loc[:, cols])
    new_feature.columns = new_feature.columns + ki
    container.append(new_feature)
        
for df_buf in container:
    res_df = pd.concat([res_df, df_buf], axis=1)

    