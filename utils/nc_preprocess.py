try:
    import cdsapi
except:
    pass
import subprocess
try:
    from netCDF4 import Dataset
except:
    pass
import numpy as np
import pandas as pd
from numba import njit
from tidypath import storage
from phdu import savedata
import os
import sys
from pathlib import Path
from time import time as get_time

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
sys.path.append(os.path.join(RootDir, "utils"))
fullpath = lambda x: os.path.join(RootDir, x)

variables_and_depths = {'potential_temperature': [10, 100, 1000],
                        'rotated_meridional_velocity': [0, 100, 1500],
                        'rotated_zonal_velocity': [0, 100, 1500],
                        'salinity': [0, 10, 100]
                        }

int_parser = lambda i: f'0{i}' if i < 10 else str(i)

def cds_downloader(year, month, variable):
    """
    Downloads data from Copernicus Climate Data Store.
    Returns path to downloaded data.
    """
    nc_dir = fullpath('ERA5-Land-data-analysis-main/nc_data')
    year_dir = os.path.join(nc_dir, str(year))
    var_dir = os.path.join(year_dir, variable)
    Path(var_dir).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(var_dir, f'{month}.zip')

    if os.path.exists(data_path):
        return data_path, var_dir
    else:
        if year > 2014:
            product_type = 'operational'
        else:
            product_type = 'consolidated'

        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-oras5',
            {
                'format': 'zip',
                'product_type': product_type,
                'vertical_resolution': 'all_levels',
                'variable': variable,
                'year': year,
                'month': [
                    int_parser(month)
                ],
            },
            data_path
        )
        return data_path, var_dir

def get_depths_idx(nc_data, depths):
    depths_idx = np.empty(len(depths), dtype=int)
    for i, depth in enumerate(depths):
        depths_idx[i] = np.argmin(np.abs(nc_data['deptht'][:] - depth))
    return depths_idx

def load_data(year, month):
    data = storage.load_lzma(fullpath('data/preprocessing/weather_data_base_v2/_.lzma'))
    data['Year'] = data.DATE_TIME.dt.year
    data['Month'] = data.DATE_TIME.dt.month
    data_year = data[data.Year == year]
    data_month = data_year[data_year.Month == month]
    return data_month

@njit
def filter_var(LAT, LON, nc_lat, nc_lon, nc_var_data):
    num_depths = nc_var_data.shape[0]
    var_filtered = np.empty((num_depths, LAT.shape[0]))

    def find_first_true_value(x):
        """
        Returns first true value or -1 if no true value is found.
        """
        idx = -1
        for i, xi in enumerate(x):
            if xi:
                idx = i
                break
        return idx

    for k, (lat, lon) in enumerate(zip(LAT, LON)):
        lat_diff = np.abs(nc_lat - lat)
        lon_diff = np.abs(nc_lon - lon)
        total_diff = lat_diff + lon_diff
        valid = total_diff < 1.5
        total_diff = total_diff[valid]
        nc_var_data_k = nc_var_data[:, valid]

        closest_idxs = np.argsort(total_diff)
        nc_var_data_k_closest= nc_var_data_k[:, closest_idxs]
        nc_var_nans = np.isnan(nc_var_data_k_closest)
        for depth_idx, is_nan in enumerate(nc_var_nans):
            if is_nan.all():
                var_filtered[depth_idx, k] = np.NaN
            else:
                var_filtered[depth_idx, k] = nc_var_data_k_closest[depth_idx, find_first_true_value(~is_nan)]
    return var_filtered

def process_nc_data(nc_data, data_month, depths):
    depth_idxs = get_depths_idx(nc_data, depths)
    nc_vars = list(nc_data.keys())
    nc_var = list(set(nc_vars) - set(['deptht', 'time_counter', 'nav_lat', 'nav_lon', 'time_counter_bnds']))[0]
    nc_var_data = nc_data[nc_var][0, depth_idxs, :, :]
    assert nc_var_data.shape == (len(depths), 1021, 1442)
    nc_var_data.data[nc_var_data.mask] = np.nan
    nc_var_data = nc_var_data.data.reshape((len(depths), -1))
    nc_lat = nc_data['nav_lat'][:].data.ravel()
    nc_lon = nc_data['nav_lon'][:].data.ravel()

    t1 = get_time()
    var_filtered = filter_var(data_month.LATITUDE.values, data_month.LONGITUDE.values, nc_lat, nc_lon, nc_var_data)
    t2 = get_time()
    print(f"Time elapsed in minutes: {(t2 - t1) / 60}")

    data_month_prunned = data_month.drop(columns=['Year', 'Month'])
    depths_rounded = np.round(nc_data['deptht'][depth_idxs].data).astype(int)
    for k, depth in enumerate(depths_rounded):
        data_month_prunned[f'{nc_var}_{depth}m'] = var_filtered[k]
    return data_month_prunned

@savedata('all')
def env_data(year, month, variable):
    """
    Downloads, processes and saves environmental data for a given month and year.
    """
    print("Loading trajectory data")
    data_month = load_data(year, month)
    if data_month.shape[0] == 0:
        print(f"No data for {year}-{month}")
        return pd.DataFrame()
    else:
        print(f"Donwloading {variable} data")
        data_path, var_dir = cds_downloader(year, month, variable)
        print("Extracting data")
        subprocess.run(['unzip', data_path, '-d', var_dir])
        os.remove(data_path)
        print("Loading environmental data")
        month_str = int_parser(month)
        month_file = [f for f in os.listdir(var_dir) if month_str in f and f.endswith('.nc')][0]
        nc_data = Dataset(os.path.join(var_dir, month_file), 'r').variables
        print("Filtering environmental data")
        depths = variables_and_depths[variable]
        data_month_processed = process_nc_data(nc_data, data_month, depths)
        os.remove(os.path.join(var_dir, month_file))
        return data_month_processed

def env_data_year(year):
    variables = list(variables_and_depths.keys())
    dfs_variables = []
    for variable in variables:
        dfs_variables.append(pd.concat([env_data(year, month, variable) for month in range(1, 13)], axis=0))
    df = pd.concat(dfs_variables, axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    return df

@savedata
def env_data_full():
    return pd.concat([env_data_year(year) for year in range(1985, 2020)], axis=0)
