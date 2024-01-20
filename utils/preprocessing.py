"""
- Format data for compatibility with the model.
- Preprocess added variables (environment)
"""
import numpy as np
import pandas as pd
import math
from numba import njit
import gc
import calendar
import warnings
from collections.abc import Iterable
from collections import defaultdict
from scipy.stats import norm, levy, wrapcauchy
from scipy.special import erf, erfinv, erfc, erfcinv
from scipy.cluster import hierarchy
from . import file_management, other_utils, data_visualization, analysis, nc_preprocess, params
import datetime
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from copy import deepcopy
import os
from pathlib import Path
import sys
import warnings
sys.stdout.flush()
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)

try:
    import xgboost as xgb
except:
    pass
from tidypath import storage
from phdu import savedata
from .nn_utils import class_counter
from . import models, geometry
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    num_cores = 0
    tf.config.threading.set_inter_op_parallelism_threads(2) #num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.set_soft_device_placement(True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
except:
    pass

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)


##############################################################################################################################
"""                                                    I. Parameters                                                       """
##############################################################################################################################

interesting_cols = ["ID", "COMMON_NAME", "Taxa", "Class", "SEX", "DATABASE", "TAG", "NumberOfSatellites", "Length", "Mean_latitude", "Mean_longitude", "Mean_year"]
updated_cols = ['Cluster ID', 'Cluster ID confidence', 'Cluster ID confidence interval', 'Animals in dataset', 'Animals in dataset interval', 'Length interval', "Mean year interval"]
secondary_cols = ["Order", "Family", "SatelliteProg", "TAG_TYPE", "ResidualError", "Stage", "AGE", "BODY_LENGTH"]
totally_unimportant_cols = ["Colour"]
all_cols = interesting_cols + secondary_cols + totally_unimportant_cols + updated_cols
#NOTE: ID != UniqueAnimal_ID, but equally identify the animals I think

discarded_weather_cols = ["2 metre temperature",
                          '100 metre U wind component', '100 metre V wind component', 'Neutral wind at 10 m u-component', 'Neutral wind at 10 m v-component',
                          'Surface pressure'
                         ]

weather_col_selection = ["Bathymetry", 'Sea ice area fraction', 'Sea surface temperature', 'Surface net solar radiation', 'Surface net thermal radiation',
                         'Mean sea level pressure', 'Significant height of combined wind waves and swell']

weather_cols = dict(temperature = ['Sea ice area fraction', 'Sea surface temperature',
                                   'Surface net solar radiation', 'Surface net thermal radiation'
                                  ],
                    wind = ['10 metre U wind component', '10 metre V wind component','K index',
                            'Mean sea level pressure', "Total precipitation"
                           ] ,
                    waves = ['Mean wave period', 'Significant height of combined wind waves and swell',
                             'Mean wave direction_x', 'Mean wave direction_y'],
                    bathymetry = ["Bathymetry"]
)
weather_cols_idxs = {}
pointer = 0
for k, v in weather_cols.items():
    list_len = len(v)
    weather_cols_idxs[k] = np.arange(pointer, pointer + list_len)
    pointer += list_len

weather_cols.update(dict(all=[col for value in weather_cols.values() for col in value]))
weather_cols_idxs["all"] = np.arange(len(weather_cols["all"]))

weather_cols_v2 = {}
weather_cols_v2['all-depth'] = ['votemper_10m', 'votemper_97m', 'votemper_1046m',
                                'vomecrtn_1m', 'vomecrtn_97m', 'vomecrtn_1516m',
                                'vozocrte_1m', 'vozocrte_97m', 'vozocrte_1516m',
                                'vosaline_1m', 'vosaline_10m', 'vosaline_97m',
                                '10 metre U wind component', '10 metre V wind component',
                                'Mean sea level pressure', 'Sea surface temperature',
                                'Mean wave period',
                                'Significant height of combined wind waves and swell',
                                'Total precipitation', 'K index', 'Sea ice area fraction',
                                'Surface net short-wave (solar) radiation',
                                'Surface net long-wave (thermal) radiation',
                                'Near IR albedo for diffuse radiation',
                                'Near IR albedo for direct radiation',
                                'Downward UV radiation at the surface', 'Evaporation', 'Geopotential',
                                'Mean wave direction_x', 'Mean wave direction_y', 'coast-d',
                                'bathymetry']
weather_cols_v2['all'] = [col for col in weather_cols_v2['all-depth'] if not col.startswith("vo") or (not col.startswith('votemper') and '1m' in col)]
weather_cols_v2[None] = []
weather_cols_v2['pruned'] = ['Sea surface temperature', # pruned using multi-collinearity
                             'votemper_1046m',
                             'vomecrtn_1m',
                             'vomecrtn_97m',
                             'vomecrtn_1516m',
                             'vozocrte_1m',
                             'vozocrte_97m',
                             'vozocrte_1516m',
                             'vosaline_1m',
                             '10 metre U wind component',
                             '10 metre V wind component',
                             'Mean sea level pressure',
                             'Mean wave period',
                             'Significant height of combined wind waves and swell',
                             'Total precipitation',
                             'K index',
                             'Sea ice area fraction',
                             'Surface net short-wave (solar) radiation',
                             'Surface net long-wave (thermal) radiation',
                             'Near IR albedo for diffuse radiation',
                             'Near IR albedo for direct radiation',
                             'Evaporation',
                             'Geopotential',
                             'Mean wave direction_x',
                             'Mean wave direction_y',
                             'coast-d',
                             'bathymetry']

weather_cols_v2['mrmr+collinear'] = ['votemper_1046m', # TODO: to be use with delete_features=['sin t', 'cos t']
                                     'vosaline_1m',
                                     'vozocrte_1516m',
                                     'vomecrtn_1516m',
                                     'vozocrte_97m',
                                     'vomecrtn_97m',
                                     'Geopotential',
                                     'Sea surface temperature',
                                     'vomecrtn_1m',
                                     'vozocrte_1m',
                                     'Mean wave period',
                                     'Significant height of combined wind waves and swell',
                                     'bathymetry',
                                     'Mean wave direction_x',
                                     'Mean wave direction_y',
                                     'coast-d']

weather_cols_v2['vif'] = ['Sea surface temperature',
                          'votemper_1046m',
                          'vomecrtn_1m',
                          'vomecrtn_97m',
                          'vomecrtn_1516m',
                          'vozocrte_1m',
                          'vozocrte_97m',
                          'vozocrte_1516m',
                          '10 metre U wind component',
                          '10 metre V wind component',
                          'Total precipitation',
                          'K index',
                          'Sea ice area fraction',
                          'Surface net short-wave (solar) radiation',
                          'Evaporation',
                          'Geopotential',
                          'Mean wave direction_x',
                          'Mean wave direction_y',
                          'bathymetry']

weather_cols_v2['mrmr+vif'] = ['Sea surface temperature', # TODO: to be use with delete_features=['sin t', 'cos t']
                               'votemper_1046m',
                               'vomecrtn_1m',
                               'vomecrtn_97m',
                               'vomecrtn_1516m',
                               'vozocrte_1m',
                               'vozocrte_97m',
                               'vozocrte_1516m',
                               'Geopotential',
                               'Mean wave direction_x',
                               'Mean wave direction_y',
                               'bathymetry']

weather_cols_v2['mrmrloop+vif'] = ['Sea surface temperature', # TODO: to be use with delete_features=['sin t', 'cos t']
                                   'votemper_1046m',
                                   'vomecrtn_1516m',
                                   'vozocrte_97m',
                                   'vozocrte_1516m',
                                   'Geopotential',
                                   ]

weather_cols_v2['mrmr'] = ['votemper_1046m', # TODO: delete 'sin t' only
                           'Sea surface temperature',
                           'vosaline_97m',
                           'votemper_97m',
                           'vozocrte_1516m',
                           'vomecrtn_1516m',
                           'votemper_10m',
                           'vosaline_10m',
                           'vosaline_1m',
                           'Geopotential',
                           'vozocrte_97m',
                           'vomecrtn_97m',
                           'Significant height of combined wind waves and swell',
                           'Mean wave period',
                           'Mean wave direction_x',
                           'vomecrtn_1m',
                           'vozocrte_1m']

spatiotemporal_cols = ['x', 'y', 'z', 'sin t', 'cos t']
spatiotemporal_cols_raw = ['lat', 'lon', 't']


##############################################################################################################################
"""                                                     II. Dataset                                                        """
##############################################################################################################################

def add_stage_type(df):
    df["Stage-type"] = df["Stage"].map(lambda c: c.split(":")[0].capitalize())
    return df

def split_trajectories(split_by='day', save=True, groupby="ID", weather=False, dataname="default", savingDir=fullPath('utils/data'), data_path=fullPath('utils/data/mobility_data.lzma')):
    """
    Select certain columns of the full dataframe. Optionally, saves that dataframe or the trajectories data.
    Trajectories data: dict containing the info from each col in usecols and the trajectory as a dataframe with columns 'LATITUDE', 'LONGITUD', 'HOUR'/'DAY' (a multivariate time series).
    - split_by = one of 'day', 'hour'. Determines the third component of the MV series.
    """
    print('Loading data')
    if weather:
        data = file_management.load_lzma(fullPath("utils/data/mobility_weather_data.lzma"))
        weather_vars = weather_cols["all"]
        weather_str = "_weather"
    else:
        data = file_management.load_lzma(fullPath('utils/data/mobility_data.lzma'))
        weather_vars = []
        weather_str = ""
    data["NumberOfSatellites"][data["NumberOfSatellites"] == ">5"] = 6
    float_cols = ["NumberOfSatellites", "ResidualError", "LONGITUDE", "LATITUDE"]
    data = data.astype({col: np.float64 for col in float_cols})
    print("Computing year and sorting by ID, DATE_TIME")
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    data["YEAR"] = data["DATE_TIME"].apply(lambda x: x.year)
    data['MONTH'] = [d.month for d in data['DATE_TIME']]
    del data['DATE'], data['TIME']
    gc.collect()

    data = data.sort_values(by=[groupby, "DATE_TIME"], ignore_index=True)

    if split_by == 'day':
        days = []
        for date in data['DATE_TIME']:
            d = date.timetuple()
            days.append(d.tm_yday - 1 + (d.tm_hour + d.tm_min/60 + d.tm_sec/3600) / 24) # Day in [0, 365]
        data['DAY'] = days
        training_cols = ['LATITUDE', 'LONGITUDE', 'DAY'] + weather_vars
    else:
        training_cols = ['LATITUDE', 'LONGITUDE', 'HOUR', 'MONTH'] + weather_vars

    label_cols = [col for col in data.columns if col not in training_cols and col != 'DATE_TIME']

    print('Splitting in trajectories')

    def time_diff_in_hours(dates):
        date_diffs = dates - dates[0]
        hour_diff = [date_diff.days * 24 + date_diff.seconds / 3600 for date_diff in date_diffs]
        return hour_diff

    trajectories = []
    labels = []
    year = []

    def base_processing(df):
        trajectories.append(df[training_cols].values.astype(np.float64).T)
        year.append(df["YEAR"].values)
        label_df = df[label_cols].iloc[:1]
        label_df = label_df.assign(Length = df.shape[0],
                                   Mean_latitude = df["LATITUDE"].mean(),
                                   Mean_longitude = df["LONGITUDE"].mean(),
                                   Mean_year = 0.5 * (df["DATE_TIME"].iloc[0].year + df["DATE_TIME"].iloc[-1].year),
                                   Mean_NumberOfSatellites = df["NumberOfSatellites"].mean(),
                                   Mean_ResidualError = df["ResidualError"].mean()
                                  )
        labels.append(label_df)
        return

    if split_by == "hour":
        def processing(df):
            df['HOUR'] = time_diff_in_hours(df['DATE_TIME'])
            base_processing(df)
            return
    elif split_by == "day":
        processing = base_processing
    else:
        raise ValueError("split_by must be one of: 'day', 'hour'.")

    for _, df in tqdm(data.groupby(by=groupby, sort=False)):
        #if len(set(df["DATE_TIME"])) > 1:
        #    df.sort_values(by='DATE_TIME', inplace=True, ignore_index=True)
        #else:
        #    df.sort_index(inplace=True)
        processing(df)

    labels = pd.concat(labels, ignore_index=True, axis=0)

    if save:
        file_management.save_lzma(trajectories, 'trajectories{}_split-by-{}_groupby-{}_{}'.format(weather_str, split_by, groupby, dataname), savingDir)
        file_management.save_lzma(labels, 'labels{}_split-by-{}_groupby-{}_{}'.format(weather_str, split_by, groupby, dataname), savingDir)
        file_management.save_lzma(year, "year{}_split-by-{}_groupby-{}_{}".format(weather_str, split_by, groupby, dataname), savingDir)
    return trajectories, labels

def update_time_extension(y):
    X = file_management.load_lzma(fullPath("utils/data/trajectories_split-by-day_groupby-ID_default.lzma"))
    Year = file_management.load_lzma(fullPath("utils/data/year_split-by-day_groupby-ID_default.lzma"))
    time_edges = [0, 1, 10, 50, 100, 200, 500]
    for mode in ["all", "year"]:
        ndays = temporal_extension(X, Year, mode=mode)
        ndays_interval = pd.cut(ndays, time_edges)
        y[f'Days in trajectory ({mode})'] = ndays
        y[f'Days in trajectory ({mode}) - interval'] = ndays_interval
    return y

def update_label_cols(epsilon=0.0015, min_size=10000, save=False):
    X = file_management.load_lzma(fullPath("utils/data/trajectories_split-by-day_groupby-ID_default.lzma"))
    Year = file_management.load_lzma(fullPath("utils/data/year_split-by-day_groupby-ID_default.lzma"))
    ndays_all = temporal_extension(X, Year, "all")
    ndays_year = temporal_extension(X, Year, "year")

    y = file_management.load_lzma(fullPath("utils/data/labels_split-by-day_groupby-ID_default.lzma"))
    file_management.save_lzma(y, "labels_split-by-day_groupby-ID_default_old")
    y2 = file_management.load_lzma(fullPath("utils/data/labels_weather_split-by-day_groupby-ID_default.lzma"))
    file_management.save_lzma(y2, "labels_weather_split-by-day_groupby-ID_default")

    y = update_time_extension(y)
    epsilon_str = other_utils.encoder(epsilon)
    min_size_str = other_utils.encoder(min_size)

    cluster = file_management.load_lzma(fullPath(f"utils/data/cluster/cluster_labels_hdbscan_eps-{epsilon_str}_cluster-size-{min_size_str}.lzma"))
    cluster_map_neg = cluster.max() + 1
    cluster[cluster == -1] = cluster_map_neg
    length_edges = np.hstack([0, y["Length"].cumsum().values])

    clusters_trajectory = np.empty((y.shape[0]), dtype=np.int32)
    clusters_trajectory_confidence = np.empty((y.shape[0]))

    for i, (start, end) in enumerate(zip(length_edges[:-1], length_edges[1:])):
        cluster_vals = np.bincount((cluster[start:end]))
        clusters_trajectory[i] = cluster_vals.argmax()
        clusters_trajectory_confidence[i] = cluster_vals[cluster_vals.argmax()] / cluster_vals.sum()

    clusters_trajectory[clusters_trajectory == cluster_map_neg] = -1

    y["Cluster ID"] = clusters_trajectory
    y["Cluster ID confidence"] = clusters_trajectory_confidence
    y["Cluster ID confidence interval"] = pd.qcut(y["Cluster ID confidence"], 10, duplicates="drop")

    # FILL data
    animals_per_species = y["COMMON_NAME"].value_counts(sort=False)
    y["Animals in dataset"] = animals_per_species.loc[y["COMMON_NAME"].values].values
    y["Animals in dataset interval"] = pd.qcut(y["Animals in dataset"], 10)
    y["Length interval"] = pd.qcut(y["Length"], 5, duplicates="drop")
    y["Mean year interval"] = pd.qcut(y["Mean_year"], 10)

    # Update weather df
    cols = ['Cluster ID', 'Cluster ID confidence', 'Cluster ID confidence interval', 'Animals in dataset', 'Animals in dataset interval', 'Length interval', "Mean year interval"]
    cols += [f'Days in trajectory ({mode}){interval_str}' for mode in ["all", "year"] for interval_str in ["", " - interval"]]
    y2_helper = y2.copy()
    y2_helper.index = y2["ID"].values # ID as index
    subset_y = y[cols].copy()
    subset_y.index = y["ID"].values
    y2_helper = pd.concat([y2_helper, subset_y], axis=1) # as ID is the index, it will concat the columns properly
    y2_helper.index = y2.index # reset index
    y2 = y2_helper.copy()

    if save:
        file_management.save_lzma(y, "labels_split-by-day_groupby-ID_default")
        file_management.save_lzma(y2, "labels_weather_split-by-day_groupby-ID_default")
    return y

def add_bathymetry_data(Z, weather=None, pad_day_rate=None, Z_as_base=False):
    weather_str = "_weather" if weather is not None else ""
    equal_spacing_str = "" if pad_day_rate is None else f'_equally-spaced-local-dr{pad_day_rate}'
    if Z_as_base:
        X = deepcopy(Z)
    else:
        X = file_management.load_lzma(fullPath(f'utils/data/trajectories{weather_str}_split-by-day_groupby-ID_default{equal_spacing_str}.lzma'))
    bathymetry_data = np.genfromtxt(fullPath('utils/data/BathymetryData.dat'),
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     delimiter=' ')
    ground = bathymetry_data > 0
    bathymetry_data -= 1
    bathymetry_data[~ground] *= -1
    bathymetry_data[~ground] = np.log10(bathymetry_data[~ground])
    bathymetry_data[ground] = -1

    lon_edges = np.arange(-180, 180.25, 0.25)
    lon_centers = 0.5 * (lon_edges[1:] + lon_edges[:-1])
    lat_edges = np.arange(90, -90.25, -0.25)
    lat_centers = 0.5 * (lat_edges[1:] + lat_edges[:-1])

    def find_closest(lat, lon):
        i = np.abs(lat - lat_centers).argmin()
        j = np.abs(lon - lon_centers).argmin()
        return bathymetry_data[i,j]

    print("Adding bathymetry...")
    Z_new = []
    for z, x in zip(Z, tqdm(X)):
        bathymetry_x = np.array([find_closest(*x[:2, i]) for i in range(x.shape[1])])
        bathymetry_x[(z == 0).all(axis=0)] = 0
        Z_new.append(np.vstack([z, bathymetry_x]))
    return Z_new




##############################################################################################################################
"""                                                III. Trajectory shift                                                   """
##############################################################################################################################

def to_cartesian(lat, lon):
    """lat lon in rads"""
    if type(lat) == np.ndarray:
        r_cartesian = np.empty((lat.size, 3))
        r_cartesian[:, 0] = np.cos(lon)*np.cos(lat)
        r_cartesian[:, 1] = np.sin(lon)*np.cos(lat)
        r_cartesian[:, 2] = np.sin(lat)
    else:
        r_cartesian = np.empty((3))
        r_cartesian[0] = math.cos(lon)*math.cos(lat)
        r_cartesian[1] = math.sin(lon)*math.cos(lat)
        r_cartesian[2] = math.sin(lat)
    return r_cartesian

def to_spherical(r):
    if len(r.shape) > 1:
        lat = np.arctan2(r[:,2], np.sqrt(np.square(r[:,:2]).sum(axis=1)))
        lon = np.arctan2(r[:,1], r[:,0])
    else:
        lat = math.atan2(r[2], math.sqrt(np.square(r[:2]).sum()))
        lon = math.atan2(r[1], r[0])
    return lat, lon

def great_circle_distance(lat, lon, lat_f=None, lon_f=None):
    if lat_f is None:
        lat_f, lat_0 = lat[1:], lat[:-1]
        lon_f, lon_0 = lon[1:], lon[:-1]
    else:
        lat_0, lon_0 = lat, lon
    sigma = 2*np.arcsin(np.sqrt(np.sin(0.5*(lat_f-lat_0))**2 + np.cos(lat_f)*np.cos(lat_0)*np.sin(0.5*(lon_f - lon_0))**2))
    return sigma

def great_circle_distance_cartesian(r):
    r_spherical = to_spherical(r)
    return great_circle_distance(*r_spherical)

def conversion_matrix(lat, lon):
    """For a vector V:
    V = (Vr, Vlat, Vlon) = A (Vx, Vy, Vz)
    """
    sin_lat = math.sin(lat)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)

    e_r = np.array([cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                   ])
    e_lat = np.array([sin_lat * cos_lon,
                      sin_lat * sin_lon,
                      -cos_lat
    ])
    e_phi = np.array([-sin_lon,
                      cos_lon,
                      0
    ])
    A = np.vstack([e_r, e_lat, e_phi])
    return A

def conversion_matrix_vec(Lat, Lon):
    """
    Vectorized version of 'conversion_matrix'.
    Returns conversion matrices A_i stacked on axis 0.
    """
    lat = Lat[:, None]
    lon = Lon[:, None]
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    e_r = np.hstack([cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                   ])
    e_lat = np.hstack([sin_lat * cos_lon,
                      sin_lat * sin_lon,
                      -cos_lat
    ])
    e_phi = np.hstack([-sin_lon,
                      cos_lon,
                      np.zeros((lat.shape[0], 1))
    ])
    A = np.stack([e_r, e_lat, e_phi], axis=1)
    return A

def exponential_map(P, v):
    """Exponential map. From point P in Riemannian fold, returns the point Q resulting from the shift of P according to a vector v in the Euclidean tangent space."""
    if len(P.shape) > 1:
        v_norm = np.sqrt(np.square(v).sum(axis=1))[:, None]
        Q = P*np.cos(v_norm) + v*np.sin(v_norm)/v_norm
        not_displacing = v_norm.squeeze() == 0
        Q[not_displacing] = P[not_displacing]
    else:
        v_norm = math.sqrt(v.dot(v))
        if v_norm > 0:
            Q = P*math.cos(v_norm) + v*math.sin(v_norm)/v_norm
        else:
            Q = P
    return Q

def log_map(P, Q):
    """Logarithmic map. From two points P, Q in the manifold, determine the vector v in the tangent space of P, such that an exponential map Exp(P,v) takes point P to point Q
    in the manifold."""
    d = great_circle_distance_cartesian(np.vstack([P, Q]))
    if d > 0:
        u = Q - P.dot(Q)*P
        v = d * u / math.sqrt(u.dot(u))
    else:
        v = np.zeros((3))
    return v

def log_map_vec(r):
    """Logarithmic map. From two points P, Q in the manifold, determine the vector v in the tangent space of P, such that an exponential map Exp(P,v) takes point P to point Q
    in the manifold. v verifies ||v|| = d(P,Q).
    Returns v for each pair of points in the trajectory.
    """
    d = great_circle_distance_cartesian(r)[:, None]
    P = r[:-1]
    Q = r[1:]
    u = Q - (P*Q).sum(axis=1)[:,None]*P
    v = d * u / np.sqrt(np.square(u).sum(axis=1))[:, None]
    v[d.squeeze() == 0] = np.zeros((3))
    return v

def transport_vector(P, Q, v):
    v_spherical = conversion_matrix(*to_spherical(P)).dot(v)
    v_Q = conversion_matrix(*to_spherical(Q)).T.dot(v_spherical)
    return v_Q

def sphere_translation_riemann(original_lat, original_lon, origin=(0,0)):
    """
    Translation of the latitude and longitude to the origin O (1,0,0).
    Returns: Shifted lat, lon coordinates.
    Both input and output are in rads (!).
    """

    original_trajectory = to_cartesian(original_lat, original_lon)
    tangent_vectors = log_map_vec(original_trajectory)
    P = to_cartesian(*origin)
    new_trajectory = np.empty((original_lat.size, 3))
    new_trajectory[0] = P
    for i, (P_original, v) in enumerate(zip(original_trajectory, tangent_vectors), start=1):
        v_shifted = transport_vector(P_original, P, v)
        P = exponential_map(P, v_shifted)
        new_trajectory[i] = P

    lats, lons = to_spherical(new_trajectory)
    return lats, lons

def shift_trajectories(weather=False, split_by="day", groupby="ID", pad_day_rate=None, savingDir=fullPath('utils/data')):
    """Displaces each trajectory to (lat,lon) = (0,0) and saves the result as a lzma file."""
    weather_str = "_weather" if weather else ""
    equal_spacing_str = "" if pad_day_rate is None else f'_equally-spaced-local-dr{pad_day_rate}'

    X = file_management.load_lzma('utils/data/trajectories{}_split-by-{}_groupby-{}_default{}.lzma'.format(weather_str, split_by, groupby, equal_spacing_str))
    X_shifted = []
    for x in tqdm(X):
        lat, lon = sphere_translation_riemann(*(x[:2]*np.pi/180))
        lat *= 180/np.pi
        lon *= 180/np.pi
        x_total = np.vstack((lat, lon, x[2:]))
        X_shifted.append(x_total)

    print("Saving")
    file_management.save_lzma(X_shifted, 'trajectories{}_split-by-{}_groupby-{}_to-origin{}'.format(weather_str, split_by, groupby, equal_spacing_str), savingDir)
    return




##############################################################################################################################
"""                                               IV. Artificial trajectories                                              """
##############################################################################################################################

def velocity_norm_stats(X_b=None, Year_b=None, y_b=None, labels_b=None,
                        time_bining="constant", std_multiplier=1, size=1, random_state=0, label_idx=0, model_angle=False, biased_angle=None):
    """
    Returns a dict containing statistical features of the trajectories and a list containing the time arrays.

    Attributes:
        - X_b:               Data to use as base to generate the aritificial trajectories.
        - time_bining:       'original': time bining of the original trajectory.
                             'pseudo-random': random time bining, based on the original trajectory.
                             'constant': constant time bining.
                             I think the only one rigurous is the constant bining. Else, the properties of levy flights and random walks do not appear.
        - std_multplier:     multiplier for the std of the velocity norm.
        - size:              % of the original trajectories to use, stratified by species.
        - random_state:      random seed for the trajectories selection.
        - label_idx:         Idx of artificial trajectory (label_artificial = max_label + 1 + label_idx).
        - model_angle:       True => wrapped cauchy distribution. False => uniform.
        - biased_angle:      if None => angle modeled is fit with the data. Else => distribution centered at 0, with random std.
    """
    if X_b is None:
        tree = models.DecisionTree(prunning_function=get_min_len_prunner(2))
        X, Year, y, labels = tree.X, tree.Year, tree.y, tree.labels
    else:
        X, Year, y, labels = [deepcopy(_) for _ in [X_b, Year_b, y_b, labels_b]]
    if size < 1:
        _, X, _, Year, _, label_vals = train_test_split(X, Year, labels.values, stratify=y, test_size=size, random_state=random_state)
        labels = pd.DataFrame({col: arr for col, arr in zip(labels.columns, label_vals.T)})
    else:
        X, labels = X, labels

    trajectory_data = dict(X=X, labels=labels)
    v_stats = defaultdict(lambda: np.empty((len(X))))
    if model_angle:
        v_stats["d_alpha"] = np.empty((len(X), 3))
    T = []
    DT = []
    Y = []
    #dt_base = 0.11 # mean dt of all trajs
    def get_year(t, dt, year_prev):
        new_year_expected = np.argwhere(t[1:] < t[:-1]) # +1 for idx in t array. As it is, for idx in dt array
        new_year = np.argwhere(dt > 365)[:,0]
        new_year_all = np.unique(np.hstack([0, new_year_expected, new_year, t.size]))
        possible_years = np.hstack([np.unique(year_prev), year_prev.max()+1]) # add 1 just in case the new trajectory dt vals imply an extra year
        year = np.empty((t.size))
        is_leap = np.empty((t.size), np.bool)
        for start, end, y in zip([new_year_all[:-1], new_year_all[1:], possible_years]):
            year[start:end] = y
            is_leap[start:end] = calendar.isleap(y)
        return year, is_leap

    def process_time(t, dt, year_prev):
        new_year_unexpected = np.argwhere(dt > 365)[:,0]
        possible_years = np.hstack([np.unique(year_prev), year_prev.max()+1])
        is_leap = np.array([calendar.isleap(y) for y in possible_years])
        num_days = 365 * np.ones((is_leap.size))
        num_days[is_leap] += 1
        year = year_prev[0] * np.ones((t.size))
        idx_year_start = 0
        for y in possible_years:
            num_days = (366 if calendar.isleap(y) else 365)
            idxs_shift = np.argwhere(t[idx_year_start:] > num_days)[:,0]
            if idxs_shift.size > 0:
                year[idx_year_start:idxs_shift[0]] = y
                idx_year_start = idxs_shift[0]
                t[idx_year_start:] -= num_days
        return t, year

    is_cartesian = X[0].shape[0] > 3
    for i, (x, year) in enumerate(zip(X, Year)):
        if is_cartesian:
            d = great_circle_distance_cartesian(x[:3].T)
            x_np = undo_periodic(x[:5], year)
            t_i = x_np[2]
            x_rad = (np.pi/180) * x_np[:2]
        else:
            x_np = x.copy()
            x_rad = (np.pi/180) * x[:2]
            d = great_circle_distance(*x_rad)
            t_i = x[2]

        dt = compute_dt(t_i, year)
        v_norm = d / dt

        if time_bining == "original":
            t_artificial = t_i
            dt_artificial = dt
            year_artificial = year
        else:
            if time_bining == "constant":
                if t_i.size > 1:
                    # t_equal_dt = (np.linspace(t_i[0], t_i[-1], t_i.size) + random.gauss(0, 10))
                    # The previous one does not work because t is cyclical.
                    dt_mean = dt.mean()
                    t_artificial = np.arange(t_i[0], t_i[0] + (t_i.size +2)*dt_mean, dt_mean)[:t_i.size] + random.gauss(0, 10)
                else:
                    t_artificial = t_i + random.gauss(0, 10)
                dt_artificial = dt_mean * np.ones((dt.size))
            elif time_bining == "pseudo-random":
                t0 = t_i[0] + random.gauss(0, 10)
                dt_artificial = dt * np.random.uniform(low=0.8, high=1.2, size=dt.size)
                t_artificial = t0 + np.hstack([0, dt_artificial.cumsum()])
            else:
                raise ValueError(f"time_bining '{time_bining}' not valid. Available: 'original', 'constant', 'pseudo-random'")
            t_artificial, year_artificial = process_time(t_artificial, dt_artificial, year)
            T.append(t_artificial)
            DT.append(dt_artificial)
            Y.append(year_artificial)

        v_stats["mean"][i] = v_norm.mean()
        v_stats["median"][i] = np.median(v_norm)
        v_stats["std"][i] = v_norm.std()
        v_stats["max"][i] = v_norm.max()
        v_stats["length"][i] = x.shape[1] - 1
        v_stats["lat0"][i] = x_np[0,0] * np.pi/180  # rads
        v_stats["lon0"][i] = x_np[1,0] * np.pi/180  # rads

        if model_angle:
            if biased_angle is None:
                x_cartesian = x if is_cartesian else to_cartesian(*x_rad)
                u_cartesian = log_map_vec(x_cartesian)
                u_spherical = np.vstack([conversion_matrix(*P).dot(u) for P, u in zip(x_rad.T, u_cartesian)])
                u_spherical_unit = u_spherical / d[:, None] # np.linalg.norm(u_spherical)
                u_spherical_unit = np.vstack([np.array([0,0,1]), u_spherical_unit])
                if np.isnan(u_spherical_unit).any():
                    for k, (unit_i, unit_end) in enumerate(zip(u_spherical_unit[:-1], u_spherical_unit[1:]), start=1):
                        if np.isnan(unit_end).all():
                            u_spherical_unit[k] = unit_i.copy()
                d_alpha = np.arccos(np.sum(u_spherical_unit[1:] * u_spherical_unit[:-1], axis=1))
                normal_vectors = np.cross(u_spherical_unit[:-1], u_spherical_unit[1:])
                sign = np.sign(-np.dot(normal_vectors, np.array([1, 0, 0])[:, None]).squeeze())
                d_alpha *= sign
                d_alpha += np.pi # adding the external angle between the vectors, since wrapcauchy has domain [-pi, pi] and arccos has domain [0, pi]
                d_alpha[np.isnan(d_alpha)] = 0
                alpha0 = d_alpha[0] + random.gauss(0, 0.1) # sigma approx 5.8 degrees

                d_alpha_not_init = d_alpha[1:].copy()
                c, loc, scale = wrapcauchy.fit(d_alpha_not_init, floc=d_alpha_not_init.mean(), fscale=d_alpha_not_init.std())
                if c == 1 or loc == 0 or scale == 0:
                    print(f"c={c}, loc={loc}, scale={scale}. Substituting for c=0.5, scale=1:")
                    c, scale = 0.5, 1
            else:
                loc = biased_angle
                scale = 0.1 + random.random()
                c = 0.1 + 0.8*random.random()
                x0_cartesian = x if is_cartesian else to_cartesian(*x_rad[:,:2])
                u0_cartesian = log_map_vec(x0_cartesian).squeeze()
                u0_spherical = conversion_matrix(*x_rad.T[0]).dot(u0_cartesian)
                u0_spherical_unit = u0_spherical / d[0]
                u0_spherical_unit = np.vstack([np.array([0,0,1]), u0_spherical_unit])
                d_alpha = np.arccos(np.sum(u0_spherical_unit[1:] * u0_spherical_unit[:-1], axis=1))
                normal_vectors = np.cross(u0_spherical_unit[:-1], u0_spherical_unit[1:])
                sign = np.sign(-np.dot(normal_vectors, np.array([1, 0, 0])[:, None]).squeeze())
                alpha0 = (d_alpha * sign)[0]
            v_stats["d_alpha"][i] = c, loc, scale

            v_stats["alpha0"][i] = alpha0

    v_stats["length"] = v_stats["length"].astype(np.int64)
    v_stats["std"] *= std_multiplier
    return v_stats, T, DT, Y, trajectory_data

def velocity_to_trajectory(v_stats, T, DT, V, spread_origin=1, random_state_origin=0, add_dt=False):
    """
    (t, v)   =>  (lat, lon, t).

    v must be in terms of the basis {e_r, e_lat, e_phi}.
    (lat = pi/2 - theta,    theta = polar angle).
    """

    X = [] # Arr containing the trajectories
    np.random.seed(random_state_origin)
    spread_origin *= np.pi / 180

    for lat0, lon0, t, dt, v in zip(tqdm(v_stats["lat0"]), v_stats["lon0"], T, DT, V):
        if spread_origin > 0:
            lat0 += spread_origin * np.random.normal()
            lat0 = lat0 if abs(lat0) < np.pi/2 else lat0*0.95
            lon0 += spread_origin * np.random.normal()

        #dt = (t[1:] - t[:-1])
        #dt[dt < -365] %= 366
        #dt[dt < 0] %= 365
        #dt = np.hstack([dt[0], dt]) # Approximate the first dt equal to the second
        dS = v * dt[:, None]

        r = np.empty((t.size, 3)) #np.empty((t.size + 1, 3))
        r[0] = to_cartesian(lat0, lon0)
        P = r[0].copy()

        for i, ds in enumerate(dS, start=1):
            ds_cartesian = conversion_matrix(*to_spherical(P)).T.dot(ds)
            P = exponential_map(P, ds_cartesian)
            r[i] = P.copy()

        #t = np.hstack([t[0]-dt[0], t])

        lat, lon = to_spherical(r)
        lat *= 180/np.pi
        lon *= 180/np.pi
        x_arr = [lat, lon, t, np.hstack([dt, 0])] if add_dt else [lat, lon, t]
        X.append(np.vstack(x_arr))
    return X

def phi(x):
    return np.exp(-0.5*x**2) / math.sqrt(2*np.pi)
def capital_phi(x):
    return 0.5 * (1 + erf(x/math.sqrt(2)))

mu_sample = np.linspace(-10, 0, 500)
sigma_sample = np.linspace(-10, 0, 500)
mu = np.exp(mu_sample)
sigma = np.exp(sigma_sample)
mu, sigma = np.meshgrid(mu, sigma, indexing="ij")
alpha = -mu/sigma
phi_alpha = phi(alpha)
capital_alpha = capital_phi(alpha)
base_vals = dict(mu=mu, sigma=sigma, capital_alpha=capital_alpha, alpha=alpha, phi_alpha=phi_alpha)
def get_gaussian_params(mean, std, cutoff, base_vals=base_vals):
    mu = base_vals["mu"]
    sigma = base_vals["sigma"]
    capital_alpha = base_vals["capital_alpha"]
    beta = (cutoff-mu) / sigma
    capital_beta = capital_phi(beta)
    Z = capital_beta - capital_alpha
    Z[Z == 0] = 1e-12
    G = mu + math.sqrt(2)*sigma*(capital_alpha + capital_beta - 1)
    return G, mu, sigma

def optimize_gaussian(G, mu, sigma, median):
    H = np.abs(G - median)
    minimizer = np.unravel_index(np.argmin(H, axis=None), H.shape)
    loc = mu[minimizer]
    scale = sigma[minimizer]
    return loc, scale

def get_levy_params(mean, std, cutoff, c=np.exp(np.linspace(-15, 3, 100000))):
    N = erfc(np.sqrt(0.5*c/cutoff))
    G = 2*np.square(erfcinv(N/2))
    return G, c

def optimize_levy(G, c, median):
    H = np.abs(c - median*G)
    minimizer = np.unravel_index(np.argmin(H, axis=None), H.shape)
    scale = c[minimizer]
    loc = 0
    return loc, scale

def update_label_info(labels, trajectory, individual_label=False):
    """
    Modify labels to indicate the artificial trajectory type.
    - individual_label:      True  => label (x_i_artificial) = label(x_i) + art.traj.type
                             False => label(x_i_artificial) = art.traj.type
    """
    labels["ID"] += f"_{trajectory}"
    if individual_label:
        for col in models.artificial_label_cols:
            labels[col] += f"_{trajectory}"
        animals_per_species = labels["COMMON_NAME"].value_counts(sort=False)
        labels["Animals in dataset"] = animals_per_species.loc[labels["COMMON_NAME"].values].values
        #intervals = []
        #for interval_str in ):
        #    interval = interval_str.replace("(", "").replace("]", "").split(", ")
        #    intervals.append(pd.Interval(float(interval[0]), float(interval[1])))
        labels["Animals in dataset interval"] = pd.cut(labels["Animals in dataset"], bins=pd.IntervalIndex(np.unique(labels["Animals in dataset interval"].values)))
    else:
        num_artificial = labels.shape[0]
        labels[models.artificial_label_cols] = trajectory
        labels["Animals in dataset"] = num_artificial
        intervals = np.array([*set(labels["Animals in dataset interval"])])
        interval = intervals[[i.overlaps(pd.Interval(left=num_artificial-1., right=num_artificial)) for i in intervals]]
        if interval.size == 0:
            max_lim_dataset = max([i.right for i in set(labels["Animals in dataset interval"])])
            interval = pd.Interval(left=max_lim_dataset, right=num_artificial)
        labels["Animals in dataset interval"] = [interval] * labels.shape[0]
    return labels

def artificial_velocity(mode="individual", trajectory="levy-flight", individual_label=False, return_stats=False, zero_mean=False, **kwargs):
    """
    Sample velocities from normal or levy distributions, with parameters mean, sigma, cutoff emulating those of the animals.
    - individual_label:      True  => label (x_i_artificial) = label(x_i) + art.traj.type
                             False => label(x_i_artificial) = art.traj.type
    """
    model_angle = False
    biased_angle = None
    if trajectory == "random-walk":
        distribution_type = norm
        sigma_multiplier = 0.5
        param_getter = get_gaussian_params
        param_optimizer = optimize_gaussian
    elif trajectory == "levy-flight":
        distribution_type = levy
        sigma_multiplier = 0.2
        param_getter = get_levy_params
        param_optimizer = optimize_levy
    elif trajectory == "correlated-random-walk": # similar directions as the original trajectory
        distribution_type = norm
        sigma_multiplier = 0.5
        param_getter = get_gaussian_params
        param_optimizer = optimize_gaussian
        model_angle = True
        biased_angle = None
    elif trajectory == "biased-random-walk": # tends to go forward
        distribution_type = norm
        sigma_multiplier = 0.5
        param_getter = get_gaussian_params
        param_optimizer = optimize_gaussian
        model_angle = True
        biased_angle = 0
    else:
        raise ValueError(f"Trajectory {trajectory} not valid. Available: 'random-walk', 'levy-flight', 'correlated-random-walk'")

    v_stats, t, dt, year, trajectory_data = velocity_norm_stats(model_angle=model_angle, biased_angle=biased_angle, **kwargs)
    trajectory_data["labels"] = update_label_info(trajectory_data["labels"], trajectory, individual_label=individual_label)

    mean_v = v_stats["mean"].mean()
    mean_median = v_stats["median"].mean()
    mean_sigma =  v_stats["std"].mean() # sigma_multiplier *
    mean_cutoff = v_stats["max"].mean()
    length_total = int(v_stats["length"].sum())

    if not model_angle:
        np.random.seed(0)
        theta = 2*np.pi * np.random.rand(length_total)
    else:
        theta = []
        for alpha0, (c, loc, scale), length in zip(v_stats["alpha0"], v_stats["d_alpha"], v_stats["length"]):
            theta.append(alpha0)
            d_alpha = wrapcauchy(c, loc=loc, scale=scale).rvs(length - 1 if length > 1 else 1, random_state=0)
            theta += list(alpha0 + d_alpha.cumsum())
        theta = np.array(theta)
    SN_component = np.sin(theta)
    WE_component = np.cos(theta)

    def velocity_sample(mean, median, sigma, cutoff, length):
        params = param_getter(mean, sigma, cutoff)
        loc, scale = param_optimizer(*params, median)
        scale = scale if scale > 0 else mean_sigma * sigma_multiplier
        loc = 0 if zero_mean else loc
        distribution = distribution_type(loc=loc, scale=scale)
        v = distribution.rvs(length)
        prune_v = lambda v: v[(v < cutoff) & (v >= 0)]
        v = prune_v(v)
        counter = 0
        generation_length = deepcopy(length)
        while v.size < length:
            v_new = distribution.rvs(generation_length)
            v_new = prune_v(v_new)
            v = np.hstack([v, v_new])[:length]
            if counter % 10 == 0:
                generation_length *= 100
                counter = 0
            else:
                counter += 1
            if generation_length > 1e8:
                v = np.hstack([v, np.repeat(median, length - v.size + 1)])[:length]
                print(f"error in generation at mean {mean}, median {median}, sigma {sigma}, cutoff {cutoff}, length {length}")
        return v

    if mode == "mean":
        v = velocity_sample(mean_v, mean_median, mean_sigma, mean_cutoff, length_total)
    elif mode == "individual":
        v = []
        for mean, median, sigma, cutoff, length in zip(tqdm(v_stats["mean"]), v_stats["median"], v_stats["std"], v_stats["max"], v_stats["length"]):
            sigma = sigma if sigma > 0 else mean_sigma
            cutoff = cutoff if cutoff > 0 else mean_cutoff
            median = median if median > 0 else mean_median
            sigma *= sigma_multiplier
            v += list(velocity_sample(mean, median, sigma, cutoff, length)) ### DELETE 1 for other types
        v = np.array(v)

    V = np.empty((v.size, 3))
    V[:, 0] = 0
    V[:, 1] = v * SN_component
    V[:, 2] = v * WE_component
    # split in trajectories
    trajectory_lims = v_stats["length"].cumsum()[:-1]
    V = np.split(V, trajectory_lims)
    assert len(V) == v_stats["length"].size

    if return_stats:
        return v_stats, t, dt, year, trajectory_data, V
    else:
        return V

def artificial_trajectory(X_b=None, Year_b=None, y_b=None, labels_b=None, individual_label=False, spread_origin=1, random_state_origin=0, trajectory="levy-flight",
                          add_dt=False, add_bathymetry=False, **kwargs):
    """
    Generate random walks or levy flights emulating the animal trajectories.
    If X_b, Year_b, etc are provided, compute similar trajectories. Else compute trajectories similar to some selected randomly over the dataset.
    """
    def aux(label_idx, artificial_trajectory):
        print(f"Trajectory type {label_idx + 1}: {artificial_trajectory}. \n Generating velocities ...")
        v_stats, t, dt, Year, trajectory_data, V = artificial_velocity(return_stats=True, trajectory=artificial_trajectory, individual_label=individual_label,
                                                                       X_b=X_b, Year_b=Year_b, y_b=y_b, labels_b=labels_b, **kwargs)
        print("Computing trajectories ...")
        X = velocity_to_trajectory(v_stats, t, dt, V, spread_origin=spread_origin, random_state_origin=random_state_origin, add_dt=add_dt)
        if add_bathymetry:
            X = add_bathymetry_data(X, Z_as_base=True)
        return  X, Year, trajectory_data["labels"]

    if isinstance(trajectory, (list, tuple, np.ndarray)):
        X_total, Year_total, labels_total = [], [], []
        for label_idx, artificial_trajectory in enumerate(trajectory):
            X, Year, labels = aux(label_idx, artificial_trajectory)
            X_total += X
            Year_total += Year
            labels_total.append(labels)
        labels_total = pd.concat(labels_total, axis=0, ignore_index=True)
        return X_total, Year_total, labels_total
    else:
        return aux(0, trajectory)




##############################################################################################################################
"""                                           V. Interpolated trajectories                                                 """
##############################################################################################################################

def get_maximal_year(t, year, prune_by="time"):
    """
    prune_by:   'data': gets year with max amount of data.
                'time': gets year with max length of time interval.
    """
    #new_year = np.unique(np.hstack([0, 1 + np.argwhere(t[1:] < t[:-1])[:,0], t.size]))
    new_year = np.unique(np.hstack([0, 1 + np.argwhere(year[:-1] != year[1:])[:,0], t.size]))
    if prune_by == "data":
        data_per_year = new_year[1:] - new_year[:-1]
        maximal_data = data_per_year.argmax()
    elif prune_by == "time":
        time_per_year = t[new_year[1:]-1] - t[new_year[:-1]]
        maximal_data = time_per_year.argmax()

    year_idxs = slice(new_year[maximal_data], new_year[maximal_data+1])
    return year_idxs

def equally_spaced_trajectories(X, Year, **kwargs):
    X_new = []
    Year_new = []
    print("Computing equally spaced trajectories ...")
    for x, year in zip(X, tqdm(Year)):
        x_new, year_new, _ = equally_spaced_trajectory(x, year, **kwargs)
        X_new.append(x_new)
        Year_new.append(year_new)
    return X_new, Year_new

def equally_spaced_trajectory(x, year, day_rate=3, prune_by="time", imputation="local", error_log=None, add_dt=False):
    """
    Returns a trajectory sampled at equal time steps.

    Attributes:
        - x:             trajectory
        - is_leap:       bool array. True if time data belongs to a leap year.
        - day_rate:      number of samples per day.
        - prune_by:      'time':   Interpolate from trajectory segment with longest time interval within a single year.
                         'data':                  ''                        maximum amount of data         ''
        - imputation:    'local':  Perform linear interpolation between the region of two known data points.
                         'global': Perform linear regression with degree such that error is minimized, using all known data points.
        - error_log:     dict containing the estimated error in lat and lon when performing global regression.

    """
    non_t_vars = 3 if add_dt else 2
    t = x[2]
    new_year = np.unique(np.hstack([0, 1 + np.argwhere(year[:-1] != year[1:])[:,0], t.size]))
    # idxs = get_maximal_year(x[2], year, prune_by=prune_by)
    def computer(idx_start, idx_end):
        x_pruned = x[:, idx_start:idx_end].copy()
        year_pruned = year[idx_start:idx_end].copy()
        #is_leap = is_leap[idxs]
        # x_pruned[1] += 180
        # x_pruned[1][x_pruned[1] < 180] += 360

        t = x_pruned[-1]
        t_unique = np.unique(t)
        while t.size > t_unique.size and t.size > 1:
            discrepancy = np.argwhere(t != np.hstack([t_unique, np.zeros((t.size - t_unique.size))]))[0,0]
            repeated = np.argwhere(t == t[discrepancy])[:,0]
            x_pruned = np.delete(x_pruned, repeated[1:], axis=1)
            year_pruned = np.delete(year_pruned, repeated[1:])
            #is_leap = np.delete(is_leap, repeated[1:])
            t = x_pruned[-1]
            t_unique = np.unique(t)

        decimal_candidates = np.arange(0, 1 + 1/day_rate, 1/day_rate)
        def get_boundary(t_b, mode):
            """Get boundaries for time such that all the custom points are interpolated (none are extrapolated)."""
            integer_part = int(t_b)
            t_b_decimal_candidates = (t_b - integer_part) - decimal_candidates
            if mode == "low":
                valid_low = t_b_decimal_candidates <= 0
                if valid_low.any():
                    decimal_part = decimal_candidates[valid_low].min()
                else:
                    raise ValueError(f"t_b_decimal_candidates: {t_b_decimal_candidates}")
            elif mode == "high":
                valid_high = t_b_decimal_candidates >= 0
                if valid_high.any():
                    decimal_part = decimal_candidates[valid_high].max()
                else:
                    raise ValueError(f"t_b_decimal_candidates: {t_b_decimal_candidates}")
            return integer_part + decimal_part

        t0 = get_boundary(x_pruned[2,0], "low")
        tf = get_boundary(x_pruned[2,-1], "high")
        num_days = tf - t0
        #t_custom = np.arange(t0, t0 + num_days + 1/day_rate, 1/day_rate)
        t_custom = np.arange(t0, t0 + num_days + 1/day_rate, 1/day_rate)
        #t_custom[-3:] %= 1e-5 + (366 if calendar.isleap(year_pruned[0]) else 365) #ensure everything is ok
        t_custom = t_custom[t_custom <= x_pruned[2,-1]] # <= 1e-5 + (366 if calendar.isleap(year_pruned[0]) else 365)] # ensure t only involves one year.

        if t_custom.size > 0:
            if add_dt:
                dt = np.hstack([compute_dt(x_pruned[2], year_pruned), 0]) # 0: base value, later substituted by np.nan
                x_pruned = np.vstack([dt, x_pruned])
            x_custom = np.vstack([np.NaN * np.ones((non_t_vars, t_custom.size)), t_custom])
            x_full = np.hstack([x_pruned, x_custom])
            sort_t = np.argsort(x_full[-1])
            x_full_sorted = x_full[:, sort_t]
            nan_values = np.isnan(x_full_sorted[0])

            if imputation == "local":
                x_custom = impute_trajectory_local(x_full_sorted, nan_values)
                error_log = {}
            elif imputation == "global":
                x_custom, error_log = impute_trajectory_global(x_pruned, x_full_sorted, nan_values, error_log=error_log)
            else:
                raise ValueError(f"imputation {imputation} not valid. Available: 'global', 'local'.")

            year_custom = np.unique(year_pruned) * np.ones((x_custom.shape[1]))
        else:
            x_custom = np.nan * np.empty((non_t_vars+1, 1))
            year_custom = np.nan * np.empty((1))
            error_log = {}
            print(f"T_custom size 0. Time_boundaries: {x_pruned[2,0]} - {x_pruned[2,-1]}")
        return x_custom, year_custom, error_log
    xs = []
    ys = []
    errors = []
    for idx_start, idx_end in zip(new_year[:-1], new_year[1:]):
        x_custom, year_custom, error_log = computer(idx_start, idx_end)
        xs.append(x_custom)
        ys.append(year_custom)
        errors.append(error_log)
    x_custom = np.hstack(xs)
    year_custom = np.hstack(ys)
    error_log = {}
    for k in errors[0].keys():
        error_log[k] = [e[k] for e in errors]
    return x_custom, year_custom, error_log

def equally_spaced_trajectory_iter(weather=False, to_origin=False, groupby="ID", imputation="local", day_rate=3, save=True, savingDir=fullPath('utils/data'), add_dt=False, **kwargs):
    weather_str = "_weather" if weather else ""
    to_origin_str = "to-origin" if to_origin else "default"
    X = file_management.load_lzma(fullPath(f'utils/data/trajectories{weather_str}_split-by-day_groupby-{groupby}_{to_origin_str}.lzma'))
    Year = file_management.load_lzma(fullPath(f'utils/data/year{weather_str}_split-by-day_groupby-{groupby}_default.lzma'))

    X_custom = []
    Year_custom = []
    DT_custom = []
    error_log = defaultdict(lambda: defaultdict(list)) if imputation == "global" else None
    for x, year in zip(tqdm(X), Year):
        x_custom, year_custom, error_log = equally_spaced_trajectory(x, year, imputation=imputation, error_log=error_log, day_rate=day_rate, add_dt=add_dt, **kwargs)
        if add_dt:
            dt = x_custom[0]
            dt[-1] = np.nan
            DT_custom.append(dt)
        X_custom.append(x_custom[1:])
        Year_custom.append(year_custom)
    if save:
        print("Saving")
        file_management.save_lzma(X_custom, f"trajectories{weather_str}_split-by-day_groupby-{groupby}_{to_origin_str}_equally-spaced-{imputation}-dr{day_rate}.lzma", savingDir)
        file_management.save_lzma(Year_custom, f"year{weather_str}_split-by-day_groupby-{groupby}_default_equally-spaced-{imputation}-dr{day_rate}.lzma", savingDir)
        file_management.save_lzma(DT_custom, f"dt{weather_str}_split-by-day_groupby-{groupby}_default_equally-spaced-{imputation}-dr{day_rate}.lzma", savingDir)
    return X_custom, Year_custom, DT_custom, error_log

def impute_trajectory_local(x_full_sorted, is_nan):
    idxs = np.argwhere(~is_nan)[:,0]
    t = x_full_sorted[-1]

    def update_x_full_sorted(f):
        for i in range(x_full_sorted.shape[0] - 1):
            f(x_full_sorted[i])
        return
    # Extrapolating begining and end
    def updater(y, idxs_fit, idxs_update):
        p = np.polyfit(t[idxs_fit], y[idxs_fit], 1)
        y[idxs_update] = np.polyval(p, t[idxs_update])
        return
    def update_begining(y):
        if idxs[0] > 0:
            updater(y, idxs[:2], slice(0, idxs[0]))
        return
    def update_end(y):
        if idxs[-1] < y.size-1:
            updater(y, idxs[-2:], slice(idxs[-1], y.size))
        return
    def update_mid(y):
        for idx_start, idx_end in zip(idxs[:-1], idxs[1:]):
            updater(y, [idx_start, idx_end], slice(idx_start+1, idx_end))
        return

    for f in [update_begining, update_end, update_mid]:
        update_x_full_sorted(f)

    x_custom = x_full_sorted[:, is_nan]
    x_custom[1] %= 360
    x_custom[1] -= 180

    return x_custom

def impute_trajectory_global(x_pruned, x_full_sorted, idxs, error_log=None, max_deg=None):
    error_log = defaultdict(lambda: defaultdict(list)) if error_log is None else error_log
    t = x_pruned[-1]
    t_custom = x_full_sorted[-1, idxs]
    if t.size < 20:
        max_deg = 6 if t.size > 10 else t.size - 1
    else:
        max_deg = t.size//10 if max_deg is None else max_deg
        max_deg = min([max_deg, 40]) #100])

    n_splits = 5 if t.size > 5 else t.size

    if n_splits > 1:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for i, out in enumerate(["Latitude", "Longitude"]):
            y = x_pruned[i]
            error = np.infty
            best_deg = np.NaN
            error_std = np.NaN
            counter = 0
            broke = False
            for deg in range(max_deg):
                errors = np.empty((n_splits))
                for fold_number, (train, test) in enumerate(kfold.split(y)):
                    x_train = t[train]
                    x_test = t[test]
                    y_train = y[train]
                    y_test = y[test]

                    try:
                        p = np.polyfit(x_train, y_train, deg)
                        y_pred = np.polyval(p, x_test)
                        errors[fold_number] = np.abs(y_test - y_pred).mean()
                    except:
                        broke = True
                        break

                if broke:
                    break
                mean_err = errors.mean()

                if mean_err < error:
                    error = mean_err
                    best_deg = deg
                    error_std = errors.std()
                    counter = 0
                else:
                    counter += 1
                    if counter > 20:
                        break

            error_log[out]["mean"].append(error)
            error_log[out]["std"].append(error_std)
            error_log[out]["deg"].append(best_deg)

            p = np.polyfit(t, y, best_deg)
            imputed_y = np.polyval(p, t_custom)
            x_full_sorted[i, idxs] = imputed_y
    else:
        x_full_sorted[:2, idxs] = x_pruned[:2]
        for out in ["Latitude", "Longitude"]:
            error_log[out]["mean"].append(np.nan)
            error_log[out]["std"].append(np.nan)
            error_log[out]["deg"].append(np.nan)

    x_custom = x_full_sorted[:, idxs]
    x_custom[1] %= 360
    x_custom[1] -= 180

    return x_custom, error_log



##############################################################################################################################
"""                                                     VI. Training                                                       """
##############################################################################################################################

def spherical_velocity(x, dt=None):
    """
    Returns: velocity (if dt is provided) or distance in terms of the spherical unit vectors.
    Attributes:
        - x:  array containing latitude and longitude in the first 2 rows.
        - dt: array of time increments between points.
    """
    d_cartesian = log_map_vec(to_cartesian(*x[:2]))
    A = conversion_matrix_vec(*x[:2])
    d_spherical = np.array([a.dot(d) for a, d in zip(A, d_cartesian)])
    if dt is None:
        return d_spherical
    else:
        v = d_spherical / dt[:, None]
        return v

def get_leap_year(y):
    """Returns bool array. True if time data belongs to a leap year"""
    year_change = np.argwhere(y[:-1] != y[1:])[:,0]
    is_leap = np.empty((y.size), dtype=np.bool)
    year_change_edges = np.hstack([0, year_change, y.size])
    for start, end in zip(year_change_edges[:-1], year_change_edges[1:]):
        is_leap[start:end] = calendar.isleap(y[start])
    return is_leap

def rescale_dt(dt, is_leap_year):
    """Rescaling dt has to take into account wether there is a leap year."""
    is_leap = is_leap_year[:dt.size]
    leap_group = [is_leap, ~is_leap]
    end_of_year = [366, 365]
    for leap, end_of_year in zip(leap_group, end_of_year):
        dt[leap & (dt < -end_of_year)] %= (end_of_year + 1)
        dt[leap & (dt < 0)] %= end_of_year
    return dt

def temporal_extension(X, Year, mode="all", **kwargs):
    """
    mode:   'all':  Time of all the trajectory
            'year': Maximal time extension within one year.
    """
    if mode == "all":
        time_ext = lambda x, year: compute_dt(x[2], year, **kwargs).sum()
    elif mode == "year":
        def time_ext(x, year):
            if x.shape[1] == 1:
                return 0
            else:
                idxs = get_maximal_year(x[2], year, **kwargs)
                t_pruned = x[2, idxs]
                return t_pruned[-1] - t_pruned[0]
    else:
        raise ValueError(f"mode {mode} not valid. Available: 'all', 'year'")
    ndays = np.array([time_ext(x, year) for x, year in zip(X, Year)])
    return ndays

def replace_dt_zeros(delta_t, by="mean", threshold=1e-8):
    """
    by: "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
    """
    zero_dt_bool = delta_t < threshold
    zero_dt = np.argwhere(zero_dt_bool)[:,0]
    num_zeros = zero_dt.size
    default_dt = 0.11 # median dt of the whole dataset

    if num_zeros > 0:
        idxs = set(range(delta_t.size))
        if by == "closest":
            valid_idxs = idxs - set(zero_dt)
            candidates = np.empty((num_zeros))
            start = 0
            end = num_zeros
            if zero_dt[0] == 0:
                candidate_idxs = valid_idxs - set(range(2))
                if len(candidate_idxs) > 0:
                    candidates[0] = delta_t[min(candidate_idxs)]
                else:
                    candidates[0] = default_dt
                start = 1
            if zero_dt[-1] == (delta_t.size - 1) and delta_t.size > 3:
                candidate_idxs = valid_idxs - set(range(num_zeros - 3, num_zeros))
                if len(candidate_idxs) > 0:
                    candidates[-1] = delta_t[max(candidate_idxs)]
                else:
                    candidates[-1] = default_dt
                end -= 1
            for i, idx in enumerate(zero_dt[slice(start, end)], start=start):
                candidates_upper = valid_idxs - set(range(idx))
                candidates_lower = valid_idxs - set(range(idx, num_zeros))

                if len(candidates_upper) > 0 and len(candidates_lower) > 0:
                    closest_upper = min(candidates_upper) if len(candidates_upper) > 0 else max(candidates_lower)
                    closest_lower =  max(candidates_lower) if len(candidates_lower) > 0 else min(candidates_upper)
                    candidates[i] = delta_t[[closest_upper, closest_lower]].mean()
                elif len(candidates_upper) > 0:
                    candidates[i] = delta_t[min(candidates_upper)]
                elif len(candidates_lower) > 0:
                    candidates[i] = delta_t[max(candidates_lower)]
                else:
                    candidates[i] = default_dt

            delta_t[zero_dt] = candidates

        elif by == "mean":
            non_zeros = delta_t.size - num_zeros
            if non_zeros > 0:
                delta_t[zero_dt_bool] = delta_t[~zero_dt_bool].mean()
            else:
                delta_t[zero_dt] = default_dt # IDEA: Declare as NANS in the preprocessing step and try to infer the value using imputation.

    return delta_t

def compute_dt(t, year, replace_zero_by="mean"):
    if t.size < 2:
        return np.array([0])
    else:
        dt = (t[1:] - t[:-1])
        is_leap = get_leap_year(year)
        new_year_expected = np.argwhere(t[1:] < t[:-1])[:,0] # +1 for idx in t array. As it is, for idx in dt array
        new_year = np.argwhere(year[:-1] != year[1:])[:,0]
        new_year_unexpected = set(new_year) - set(new_year_expected)

        dt = rescale_dt(dt, is_leap)
        dt = replace_dt_zeros(dt, by=replace_zero_by)
        for idx in new_year_unexpected:
            dt[idx] += (366 if is_leap[idx-1] else 365)
        return dt


def make_periodic(z, year, added_dt=False, to_origin=None, velocity=None, replace_zero_by="mean", diff=True, diff_idxs=None):
    """
    Attributes:

        - z:                     (lat, lon, t) vector with shape (3, length)

        - year:                  array of year values. Used to compute the days per year and dt values.

        - added_dt:              Bool. If true, returns the vector except for the last point, where dt is undefined. Mainly thought for the equal-time case.

        - to_origin:             "time":             Shift initial time to 1 Jan.

        - velocity:              None:               Does not add velocity.
                                 "arch-segment":     Velocities in the SN-WE components.
                                 "x-y-z":            Velocities as the derivatives w.r.t. time of x, y, z.

        - replace_zero_by:       "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
        - diff:                  Bool. If True, returns the difference in each magnitude (except for the velocity)

    Returns:

        periodic_x := (x, y, z, sin t, cos t, {weather vars},  {dt}, {velocity_vars})

    Considerations:
    {x, y, z} = {cos(theta) cos(phi), cos(theta) sin(phi), sin(theta)}
    theta = lat (not the polar angle)
    """
    x = z.copy()
    is_leap = get_leap_year(year)
    if to_origin in ["time", "all"]:
        x[2] = (x[2] - x[2,0])
        x[2] = rescale_dt(x[2], is_leap)

    t_angle = (2*np.pi) * x[2] # maybe / 366 for leap
    t_angle[is_leap] /= 366
    t_angle[~is_leap] /= 365
    theta = (np.pi/180) * x[0]
    phi = (np.pi/180) * x[1]
    cos_theta = np.cos(theta)

    periodic_x = np.empty((5, x.shape[1]))
    periodic_x[0] = cos_theta * np.cos(phi)
    periodic_x[1] = cos_theta * np.sin(phi)
    periodic_x[2] = np.sin(theta)
    periodic_x[3] = np.sin(t_angle)
    periodic_x[4] = np.cos(t_angle)

    if x.shape[0] > 3: # there are weather variables or/and dt
        periodic_x = np.vstack([periodic_x, x[3:]])

    if velocity is not None:
        delta_t = compute_dt(x[2], year, replace_zero_by=replace_zero_by)

        if velocity == "x-y-z":
            v = (periodic_x[:3, 1:] - periodic_x[:3, :-1]) / delta_t
        elif velocity == "arch-segment":
            v = spherical_velocity((np.pi/180) * x, delta_t).T[-2:] # take only components in theta, phi. (v_r=0)
        else:
            raise ValueError(f"velocity_mode: {velocity} not valid. Available: 'arch-segment', 'x-y-z'.")
        if diff_idxs is None:
            periodic_x = np.vstack([np.diff(periodic_x, axis=1) if diff else periodic_x[:, :-1], # velocity undefined for the last point.
                                    v])
        elif len(diff_idxs) > 0:
            no_diff_idxs = np.array([i for i in np.arange(periodic_x.shape[0]) if i not in diff_idxs])
            periodic_x = np.vstack([periodic_x[no_diff_idxs, :-1],
                                    np.diff(periodic_x[diff_idxs], axis=1) if diff else periodic_x[diff_idxs, :-1],
                                    v])
        else:
            periodic_x = np.vstack([periodic_x[:, :-1], v])
    else:
        if added_dt:
            if diff_idxs is None:
                periodic_x = np.diff(periodic_x, axis=1)
            elif len(diff_idxs) > 0:
                no_diff_idxs = np.array([i for i in np.arange(periodic_x.shape[0]) if i not in diff_idxs])
                periodic_x = np.vstack([periodic_x[no_diff_idxs, :-1],
                                        np.diff(periodic_x[diff_idxs], axis=1) if diff else periodic_x[diff_idxs, :-1]
                                       ])
            else:
                periodic_x = periodic_x[:,:-1]
    return periodic_x, year[:periodic_x.shape[1]]

def apply_scaling(z, scaler, scale_idxs, scale_padded, axis, fit=False, side=False, base_scaler=None):
    if z is None:
        return
    if isinstance(scaler, dict):
        S = pd.Series(dtype=object)
        if fit:
            for s, idxs_train, _, idxs_side in scaler.values():
                idxs = idxs_train if len(idxs_train) > 0 else idxs_side if side else None
                if idxs is not None and not hasattr(s, "n_features_in_"): # scaler is fitted
                    Z = np.concatenate(tuple(z[i] for i in idxs), axis=1).T
                    _ = apply_scaling(Z, s, scale_idxs, scale_padded, axis, fit=True)
            #if base_scaler is not None and not side and len(scale_idxs) > 0:
            #    Z = np.concatenate(tuple(z), axis=1).T
            #    scale_pts = np.ones((Z.shape[0]), dtype=np.bool) if scale_padded else  ~ ((np.abs(Z) < 1e-8).all(axis=1))
            #    base_scaler.fit(Z[np.ix_(scale_pts, scale_idxs)])
        for s, idxs_train, idxs_test, idxs_side in scaler.values():
            idxs = idxs_side if side else idxs_train if fit else idxs_test
            for i in idxs:
                S.loc[i] = apply_scaling(z[i].T, s, scale_idxs, scale_padded, axis, fit=False)
            #for i in set(np.arange(len(z))) - set(idxs):
            #    S.loc[i] = apply_scaling(z[i].T, base_scaler, scale_idxs, scale_padded, axis, fit=False)

        return np.vstack(S.loc[np.arange(S.size)].values)
    else:
        x = z.copy()
        if axis == 1:
            x = x.T # feature_axis: 0 -> 1
        if scaler is not None:
            scale_pts = np.ones((x.shape[0]), dtype=np.bool) if scale_padded else  ~ ((np.abs(x) < 1e-8).all(axis=1))
            if len(scale_idxs) > 0:
                x[np.ix_(scale_pts, scale_idxs)] = scaler.fit_transform(x[np.ix_(scale_pts, scale_idxs)]) if fit else scaler.transform(x[np.ix_(scale_pts, scale_idxs)])
        if axis == 1:
            x = x.T
        return x

def undo_periodic(periodic_z, year, scaler=None, scale_idxs=None, scale_padded=False):
    """
    (x, y, z, sin t, cos t)   -->   (lat, lon, t)
    (x, y, z, sin t, cos t, v_SN, V_WE)   -->   (lat, lon, t, v_SN, V_WE)
    """
    periodic_x = periodic_z.copy()
    is_leap = get_leap_year(year)
    periodic_x = apply_scaling(periodic_x, scaler, scale_idxs, scale_padded, axis=1)
    num_cds = periodic_x.shape[0]
    x = np.empty((num_cds, periodic_x.shape[1]))
    x[0] = (180/np.pi) * np.arctan2(periodic_x[2], np.sqrt(periodic_x[0]**2 + periodic_x[1]**2))
    x[1] = (180/np.pi) * np.arctan2(periodic_x[1], periodic_x[0])

    if scale_idxs.min() > 4: # Time data is present
        for partition, num_days in zip([is_leap, ~is_leap], [366, 365]):
            x[2][partition] = ((num_days/ (2*np.pi)) * np.arctan2(periodic_x[3][partition], periodic_x[4][partition])) % num_days # not adding one. ([0, 365]) undo_periodic(periodic) != identity. Maybe 365 -> 366 for leap
    return x

def periodic_time(x, year, to_origin=True):
    """Converts only time to periodic."""
    is_leap = get_leap_year(year)
    t_angle = (2*np.pi) * x[2]
    t_angle[is_leap] /= 366
    t_angle[~is_leap] /= 365

    periodic_x = np.empty((x.shape[0] + (1 if to_origin else 2), x.shape[1]))
    sep = 2 if to_origin else 3
    periodic_x[:sep] = x[:sep]
    periodic_x[sep] = np.sin(t_angle)
    periodic_x[sep+1] = np.cos(t_angle)
    if x.shape[0] > 3:
        periodic_x[sep+2:] = x[3:]
    return periodic_x

class NotEnoughData(Exception):
    pass

def preprocessing(label="COMMON_NAME", groupby="ID", partitions=1, min_animals=5, prunning_function=None, seed=0, partition_number=1, periodic=True, weather=None, velocity=None, to_origin=None, replace_zero_by="mean", pad_day_rate=None, fill_pad_day_rate=True, common_len=0,
                  delete_features=[], add_dt=False, add_bathymetry=False, diff=False, split_by=dict(column=None, colvalue=None), assert_side_in_train=False,
                  common_origin_distance=False, minlen=5, as_image=False,
                  v2 = True, species_stage=None, species_train=None, invert_lat=False,
                  artificial_trajectory_type=None, artificial_percentage=None, **artificial_traj_kwargs):
    if weather is not None and weather == 'mrmr+collinear' or weather == 'mrmr+vif' or weather == 'mrmrloop+vif':
        delete_features = ['sin t', 'cos t']
    elif weather == 'mrmr':
        delete_features = ['sin t']

    if common_origin_distance or as_image:
        df, labels, Year = compute_df_distance_weather_spec(v2=v2, weather=weather, species_stage=species_stage, pad_day_rate=pad_day_rate, species_train=species_train, invert_lat=invert_lat)
        v, v_norm, dt = compute_df_velocity(v2=v2, weather=weather, species_stage=species_stage, pad_day_rate=pad_day_rate, species_train=species_train, invert_lat=invert_lat)
        # if pad_day_rate > 0:
        #     X, Year = equally_spaced_trajectories(df.values, Year, day_rate=pad_day_rate)
        X = [periodic_time(x, y, to_origin=True) for x, y in zip(df.values, Year)]
        X, Year, y, labels, label = get_prunning_function(minlen=minlen, min_animals=min_animals, label=label)(X, Year, labels.reset_index())
        if prunning_function is not None:
            X, Year, y, labels, label = prunning_function(X, Year, labels)
        X_original = deepcopy(X)
        y = remap_labels(y)
        labels = labels.set_index(["COMMON_NAME", "ID"])
        v = v.loc[labels.index]
        v_norm = v_norm.loc[labels.index]
        dt = dt.loc[labels.index]
        labels = labels.reset_index()
        if species_train is not None:
            if species_train not in labels["COMMON_NAME"].unique():
                raise NotEnoughData(f"Species {species_train} not present in the dataset.")
        if common_origin_distance:
            cdts_features = ["SN", "WE", "sin t", "cos t"]
            if weather is not None:
                if weather in weather_cols_v2:
                    weather_features = weather_cols_v2[weather]
                else:
                    weather_features = weather.split("+")
                features = cdts_features + weather_features
            else:
                features = cdts_features
            if len(delete_features) > 0:
                remaining_idxs = [i for i, f in enumerate(features) if f not in delete_features]
                features = [*np.array(features)[remaining_idxs]]
                X = [x[remaining_idxs] for x in X]
            scale_idxs = np.array([i for i, f in enumerate(features) if f not in cdts_features])
        elif as_image:
            features = ["SN", "WE"]
            X = [x[:2] for x in X]
            scale_idxs = np.array([])
            if len(delete_features) > 0:
                raise ValueError("delete_features not implemented for as_image=True")
        if split_by["column"] is not None:
            side_data, remain_data = split_dataset_by_value(X, X_original, Year, v, v_norm, dt, labels=labels, label_col=label, **split_by, assert_side_in_train=assert_side_in_train)
            data = {k: v for k, v in zip(["X", "X_original", "Year","v", "v_norm", "dt", "labels", "y"], side_data)}
            data.update({f"{k}_side": v for k, v in zip(["X", "X_original", "Year", "v", "v_norm", "dt", "labels", "y"], remain_data)})
        else:
            data = dict(X=X, X_original=X_original, Year=Year, y=y, labels=labels, v=v, v_norm=v_norm, dt=dt)
        return data, label, features, scale_idxs

    else:
        to_origin_str = "default" if to_origin is None else "to-origin"
        equal_spacing_str = "" if pad_day_rate is None else f'_equally-spaced-local-dr{pad_day_rate}'
        spatio_temporal_idxs = np.arange(5 if add_dt and add_bathymetry else 4 if add_dt or add_bathymetry else 3)
        cdts_features = ["x", "y", "z", "sin t", "cos t"] if periodic else ["lat", "lon", "t"]

        if artificial_percentage is None:
            artificial_percentage = 0.1 if isinstance(artificial_trajectory_type, (list, tuple, np.ndarray)) else 0.2

        if weather is None:
            weather_str = ""
            var_idxs = spatio_temporal_idxs
            weather_features = []
        else:
            if v2:
                if weather in weather_cols_v2:
                    weather_features = weather_cols_v2[weather]
                    weather_idxs = np.arange(len(weather_features))
                else:
                    all_cols = np.array(weather_cols_v2["all"])
                    weather_features = weather.split("+")
                    weather_idxs = np.array([i for i, c in enumerate(all_cols) if c in weather_features])
            else:
                weather_str = "_weather"
                if isinstance(weather, (list, tuple, np.ndarray)):
                    all_cols = np.array(weather_cols["all"])
                    weather_idxs = []
                    for col in weather:
                        idxs = np.argwhere(all_cols == col)
                        if idxs.size > 0:
                            weather_idxs.append(idxs[0,0])
                    weather_idxs = np.array(weather_idxs)
                    weather_features = [*weather]
                else:
                    weather_idxs = weather_cols_idxs[weather]
                    weather_features = weather_cols[weather]
            var_idxs = np.hstack([spatio_temporal_idxs, weather_idxs + spatio_temporal_idxs.size])

        if velocity is None:
            v_features = []
        else:
            v_features = ["v (SN)", "v (WE)"] if velocity == "arch-segment" else ["v_x", "v_y", "v_z"]

        if v2:
            X, labels, Year = load_all_data(v2=v2, weather=weather, return_labels=True, species_stage=species_stage, species_train=species_train, invert_lat=invert_lat)
            X = list(X.values)
            labels = labels.reset_index()
        else:
            X = file_management.load_lzma(fullPath(f'utils/data/trajectories{weather_str}_split-by-day_groupby-{groupby}_{to_origin_str}{equal_spacing_str}.lzma'))
            labels = file_management.load_lzma(fullPath(f'utils/data/labels{weather_str}_split-by-day_groupby-{groupby}_default.lzma'))
            Year = file_management.load_lzma(fullPath(f'utils/data/year{weather_str}_split-by-day_groupby-{groupby}_default{equal_spacing_str}.lzma'))

        if add_bathymetry: # and weather is None:
            bathymetry = ["Bathymetry-add"]
            X = add_bathymetry_data(X, weather=weather, pad_day_rate=pad_day_rate)
        else:
            bathymetry = []

        if add_dt and not v2:
            dt_features = ["dt"]
            if pad_day_rate is None:
                X = [np.vstack([x, np.hstack([compute_dt(x[2], y), 0])]) for x, y in zip(X, Year)] # added 0 to be able to concatenate. Later will be removed
            else:
                DT = file_management.load_lzma(fullPath(f'utils/data/dt{weather_str}_split-by-day_groupby-{groupby}_default{equal_spacing_str}.lzma'))
                X = [np.vstack([x, dt]) for x, dt in zip(X, DT)]
        else:
            dt_features = []

        if v2:
            features = cdts_features + weather_features
            diff_idxs = np.arange(len(cdts_features), len(features))
        else:
            features = cdts_features + bathymetry + weather_features + dt_features + v_features
            diff_idxs = np.array([i for i, f in enumerate(cdts_features + bathymetry + weather_features + dt_features) if f not in cdts_features + dt_features])

        if prunning_function is not None:
            X, Year, y, labels, label = prunning_function(X, Year, labels)

        # if pad_day_rate > 0:
        #     X, Year = equally_spaced_trajectories(X, Year, day_rate=pad_day_rate)

        X, Year, y, labels, label = get_prunning_function(minlen=minlen, min_animals=min_animals, label=label)(X, Year, labels)

        if partitions > 1:
            np.random.seed(seed)
            #idxs = np.random.choice(np.arange(len(X)), int(datasize*len(X)), replace=False)
            idxs_shuffle = np.random.choice(np.arange(len(X)), len(X), replace=False)
            X = [X[i] for i in idxs_shuffle]
            Year = [Year[i] for i in idxs_shuffle]
            y = y[idxs_shuffle]
            labels = labels.iloc[idxs_shuffle]
            partitions_start = np.linspace(0, len(X), partitions, endpoint=False, dtype=np.int32)
            partition_length = partitions_start[1] - partitions_start[0]
            start = partitions_start[partition_number-1]
            idxs_partition = slice(start, start + partition_length)
            X = X[idxs_partition]
            Year = Year[idxs_partition]
            y = y[idxs_partition]
            labels = labels.iloc[idxs_partition]

        if common_len > 0:
            X, Year, y, labels = prune_by_common_len(X, Year, labels, label_col=label, L=common_len)

        # X, Year, y, labels, label = prune_by_min_animals(X, Year, labels, min_animals=min_animals, label_col=label) # ensure there is at least one animal/pecies in the training and test set. DONE ABOVE, with get_min_len_prunner
        y = remap_labels(y)

        if artificial_trajectory_type is not None:
            if weather is not None:
                raise ValueError("Weather variables incompatible with artificial trajectories. No data available")

            X_artificial, Year_artificial, labels_artificial = artificial_trajectory(X_b=X, Year_b=Year, y_b=y, labels_b=labels,
                                                                                     trajectory=artificial_trajectory_type, size=artificial_percentage,
                                                                                     add_dt=add_dt, add_bathymetry=add_bathymetry,
                                                                                     **artificial_traj_kwargs)
            X += X_artificial
            Year += Year_artificial
            labels = pd.concat([labels, labels_artificial], axis=0, ignore_index=True)
            y = remap_labels(labels[label])

        T = [x[2].copy() for x in X]
        X_original = deepcopy(X)

        if periodic:
            if v2: # var chosens via weather_cols_v2
                X, Year = zip(*[make_periodic(x, year, added_dt=add_dt, velocity=velocity, to_origin=to_origin, replace_zero_by=replace_zero_by, diff=diff, diff_idxs=diff_idxs) for (x, year) in zip(X, Year)])
            else:
                X, Year = zip(*[make_periodic(x[var_idxs], year, added_dt=add_dt, velocity=velocity, to_origin=to_origin, replace_zero_by=replace_zero_by, diff=diff, diff_idxs=diff_idxs) for (x, year) in zip(X, Year)])

        if pad_day_rate is not None and fill_pad_day_rate:
            pad_trajectory = get_pad_trajectory(day_rate=pad_day_rate)
            valid_trajectories = ~ np.array([np.isnan(x).any() for x in X])
            X = [pad_trajectory(x, t) for x, t, c in zip(X, T, valid_trajectories) if c]
            Year = [pad_trajectory(y[None], t)[0] for y, t, c in zip(Year, T, valid_trajectories) if c]

        if len(delete_features) > 0:
            remaining_idxs = [i for i, f in enumerate(features) if f not in delete_features]
            features = [*np.array(features)[remaining_idxs]]
            X = [x[remaining_idxs] for x in X]
        scale_idxs = np.array([i for i, f in enumerate(features) if f not in cdts_features])

        if split_by["column"] is not None:
            side_data, remain_data = split_dataset_by_value(X, X_original, Year, labels=labels, label_col=label, **split_by, assert_side_in_train=assert_side_in_train)
            data = {k: v for k, v in zip(["X", "X_original", "Year", "labels", "y"], side_data)}
            data.update({f"{k}_side": v for k, v in zip(["X", "X_original", "Year", "labels", "y"], remain_data)})
        else:
            data = dict(X=X, X_original=X_original, Year=Year, y=y, labels=labels)
        return data, label, features, scale_idxs

def get_pad_trajectory(day_rate=3, keep_NA=False):
    """Discarded.py"""
    t_base = np.arange(0, 365 + 1/day_rate, 1/day_rate)
    epsilon = 1e-5
    multiplyer = np.NAN if keep_NA else 1
    def pad_trajectory(z, t):
        """Pads trajectory from left and right to fill the time frame t_base."""
        x = z.copy()
        low_lim = 1 + np.argwhere(t_base < t.min() - epsilon)[:,0]
        high_lim = np.argwhere(t_base > t.max() + epsilon)[:,0]
        if low_lim.size > 0:
            x = np.hstack([multiplyer * np.zeros((x.shape[0], low_lim[-1])), x])
        if high_lim.size > 0:
            x = np.hstack([x, multiplyer * np.zeros((x.shape[0], t_base.size - high_lim[0]))])
        return x[:, :t_base.size]
    return pad_trajectory

def pad(X, scaler=None, idx_to_scaler=None, maxlen=400, step=1, expand_dim=True, n_steps=1, insert_row=False, scale_idxs=None, scale_padded=False, side=False, base_scaler=None):
    if X is None:
        return
    else:
        num_signals = X[0].shape[0]
        data = np.zeros((len(X), maxlen, num_signals))
        pruner = lambda x, step: x[::step]
        prune = step > 1
        for i, x in enumerate(X):
            x = pruner(x.T, step) if prune and x.T.shape[0] > maxlen else x.T
            if idx_to_scaler is None:
                x = apply_scaling(x, scaler, scale_idxs, scale_padded, axis=0, side=side, base_scaler=base_scaler)
            else:
                x = apply_scaling(x, idx_to_scaler[i], scale_idxs, scale_padded, axis=0, side=side, base_scaler=base_scaler)
            l = x.shape[0]
            length = l if l < maxlen else maxlen
            data[i,:length] = x[:length]

        X_tf = tf.convert_to_tensor(data, tf.float32)
        if n_steps > 1:
            n_length = maxlen // n_steps
            if insert_row:
                X_tf = tf.reshape(X_tf, [X_tf.shape[0], n_steps, 1, n_length, X_tf.shape[-1]])
            else:
                X_tf = tf.reshape(X_tf, [X_tf.shape[0], n_steps, n_length, X_tf.shape[-1]])
        if expand_dim:
            X_tf = tf.expand_dims(X_tf, axis=-1)
        return X_tf

def ravel_data(X_train_list, X_test_list,  X_original_train, X_original_test, y_train_short, y_test_short, scaler, scale_idxs=None, scale_padded=False,
               side=False, base_scaler=None, **kwargs):
    """Ravel target and feature lists. Scaling can be applied only to the raveled feature matrices (to X_partition, not X_partition_list)."""
    y_test = [[y] * x.shape[1] for x,y in zip(X_test_list, y_test_short)]
    y_test = np.array([item for sublist in y_test for item in sublist])
    X_test = np.concatenate(tuple(X_test_list), axis=1).T
    if X_train_list is not None:
        y_train = [[y] * x.shape[1] for x,y in zip(X_train_list, y_train_short)]
        y_train = np.array([item for sublist in y_train for item in sublist])
        X_train = np.concatenate(tuple(X_train_list), axis=1).T
    else:
        y_train, X_train = None, None
    # Scale (except x, y, z, sin t, cos t)
    if isinstance(scaler, dict):
        X_train = apply_scaling(X_train_list, scaler, scale_idxs, scale_padded, axis=0, fit=True and not side, side=side, base_scaler=base_scaler)
        X_test = apply_scaling(X_test_list, scaler, scale_idxs, scale_padded, axis=0, fit=False or side, side=side, base_scaler=base_scaler)
    else:
        X_train = apply_scaling(X_train, scaler, scale_idxs, scale_padded, axis=0, fit=True and not side, side=side, base_scaler=base_scaler)
        X_test = apply_scaling(X_test, scaler, scale_idxs, scale_padded, axis=0, side=side, base_scaler=base_scaler)
    data = {'X_train_list': X_train_list,
            'X_train': X_train,
            'X_test_list': X_test_list,
            'X_test': X_test,
            "X_original_train": X_original_train,
            "X_original_test": X_original_test,
            'y_train_short': y_train_short,
            'y_train': y_train,
            'y_test_short': y_test_short,
            'y_test': y_test
       }
    return data

def DMatrix(*args, **kwargs):
    """Ravel target and feature lists. Scaling can be applied only to the raveled feature matrices (to X_partition, not X_partition_list). Groups X_test, y_test in DMatrices."""
    data = ravel_data(*args, **kwargs)
    if data["X_train"] is not None:
        data["X_train"] = xgb.DMatrix(data["X_train"], label=data["y_train"])
    data["X_test"] = xgb.DMatrix(data["X_test"], label=data["y_test"])
    return data

def preprocessing_NN(X_train_list, X_test_list, X_original_train, X_original_test, y_train_short, y_test_short, scaler, categorical=None, pad_sequence=False, pad_kwargs={},
                     raw_sequences=False, scale_idxs=None, scale_padded=False, idx_to_scaler_train=None, idx_to_scaler_test=None, side=False, base_scaler=None):
    X_train_list, X_train, X_test_list, X_test, X_original_train, X_original_test, y_train_short, y_train, y_test_short, y_test = ravel_data(X_train_list, X_test_list,
                                                                                                                                             X_original_train, X_original_test,
                                                                                                                                             y_train_short, y_test_short,
                                                                                                                                             scaler, scale_idxs=scale_idxs,
                                                                                                                                             scale_padded=scale_padded,
                                                                                                                                             side=side,
                                                                                                                                             base_scaler=base_scaler).values()

    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.expand_dims(tf.convert_to_tensor(y_test, dtype=tf.int32), axis=1)
    if X_train is not None:
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.expand_dims(tf.convert_to_tensor(y_train, dtype=tf.int32), axis=1)


    if categorical == 'months':
        onehot = OneHotEncoder(sparse=False)
        _ = onehot.fit(np.arange(1, 13)[:, None])
        base = datetime.datetime(2020,1,1, 0,0,0)
        def month_encoder(X):
            days = X[:, -1].numpy().astype('timedelta64[D]')
            dates = np.array([datetime.datetime(2020,1,1, 0,0,0)], dtype='datetime64[D]') + days
            months = tf.convert_to_tensor(dates.astype('datetime64[M]').astype(int) % 12 + 1)
            #months = tf.constant([(base + datetime.timedelta(days=float(d))).month for d in X[:,-1]], dtype=tf.float32)
            months_onehot = onehot.transform(months[:, None])
            X_new = X[:, :-1]
            X_new = tf.concat([X_new, months_onehot], axis=1)
            return X_new
        X_test = month_encoder(X_test)
        if X_train is not None:
            X_train = month_encoder(X_train)

    elif categorical == 'weeks':
        onehot = OneHotEncoder(sparse=False)
        _ = onehot.fit(np.arange(53)[:, None])
        def week_encoder(X):
            days = X[:, -1]
            weeks = tf.cast(days // 7, dtype=tf.int32)
            weeks_onehot = onehot.transform(weeks[:, None])
            X_new = X[:, :-1]
            X_new = tf.concat([X_new, weeks_onehot], axis=1)
            return X_new
        X_test = week_encoder(X_test)
        if X_train is not None:
            X_train = week_encoder(X_train)

    if pad_sequence:
        X_train = pad(X_train_list, scaler=scaler, scale_idxs=scale_idxs, scale_padded=scale_padded, idx_to_scaler=idx_to_scaler_train,side=side, base_scaler=None,
                      **pad_kwargs)
        X_test = pad(X_test_list, scaler=scaler, scale_idxs=scale_idxs, scale_padded=scale_padded, idx_to_scaler=idx_to_scaler_test, side=side,
                     base_scaler=base_scaler, **pad_kwargs)
        y_train = None if side else tf.expand_dims(tf.convert_to_tensor(y_train_short, dtype=tf.int32), axis=1)
        y_test = tf.expand_dims(tf.convert_to_tensor(y_test_short, dtype=tf.int32), axis=1)
    elif raw_sequences:
        X_train = [tf.expand_dims(tf.convert_to_tensor(x.T), axis=0) for x in X_train_list]
        X_test = [tf.expand_dims(tf.convert_to_tensor(x.T), axis=0) for x in X_test_list]
        y_train = [tf.expand_dims(tf.constant(y), axis=0) for y in y_train_short]
        y_test = [tf.expand_dims(tf.constant(y), axis=0) for y in y_test_short]

    data = {'X_train_list': X_train_list,
            'X_train': X_train,
            'X_test_list': X_test_list,
            'X_test': X_test,
            "X_original_train": X_original_train,
            "X_original_test": X_original_test,
            'y_train_short': y_train_short,
            'y_train': y_train,
            'y_test_short': y_test_short,
            'y_test': y_test
       }

    return data



##############################################################################################################################
"""                                                 VII. Filtering funcs                                                   """
##############################################################################################################################

def split_dataset_by_value(*iterables, labels=None, column=None, colvalue=None, label_col=None, label_side_step=False, assert_side_in_train=False):
    side = pd.concat([labels[column] == val for val in colvalue], axis=1).any(axis=1).values
    remaining = ~side

    labels_remain = labels[remaining]
    labels_side = labels[side]
    if label_side_step:
        y_remain = remap_labels(labels_remain[label_col].values)
        y_side = remap_labels(labels_side[label_col].values) + y_remain.max() + 1
    else:
        labels_remain_unique = labels_remain[label_col].unique()
        labels_side_unique = labels_side[label_col].unique()
        cat_to_label_remain = {c:i for i, c in enumerate(labels_remain_unique)}
        y_remain = labels_remain[label_col].map(cat_to_label_remain).values
        cat_to_label_side = deepcopy(cat_to_label_remain)
        not_common_labels = [c for c in labels_side_unique if c not in labels_remain_unique]
        max_label_remain = max(cat_to_label_remain.values())
        if not_common_labels:
            for i, c in enumerate(not_common_labels, start=1):
                cat_to_label_side[c] = max_label_remain + i
        y_side = labels_side[label_col].map(cat_to_label_side).values
        if assert_side_in_train:
            side = side & np.isin(labels[label_col].values, labels_remain_unique)
            y_side = y_side[y_side <= max_label_remain]
            labels_side = labels[side]

    def extract_partition(Y, mask):
        if isinstance(Y, (list, tuple)):
            return [x for x,t in zip(Y, mask) if t]
        elif isinstance(Y, (np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series)):
            return Y[mask]
        else:
            raise ValueError("Y must be a list, tuple, np.ndarray, pd.core.frame.DataFrame or pd.core.series.Series")

    data_side = [extract_partition(X, side) for X in iterables] + [labels_side, y_side]
    data_remain = [extract_partition(X, remaining) for X in iterables] + [labels_remain, y_remain]

    return data_remain, data_side

def remap_labels(y):
    labels = np.unique(y)
    if labels.dtype in [np.int32, np.int64, np.int16, np.int8, np.int]:
        is_binary = labels.size == 2
    else: # str label
        is_binary = any(["Not" in l for l in labels])
    if is_binary:
        for l in labels:
            if "Not" in l:
                label_false = l
            else:
                label_true = l
        y_remap = {label_false: 0, label_true: 1}
    else:
        y_remap = {old:new for new, old in enumerate(sorted(set(y)))} # labels from 0 to the number of labels.
    y_mapped = np.array([y_remap[l] for l in y], dtype=np.int32)
    return y_mapped

def trajectories_common_time(L, X=None, Year=None, pad_day_rate=3, to_origin="space", fill_pad_day_rate=False):
    if isinstance(L, int):
        L = [L]
    if X is None:
        inception = models.InceptionTime(nb_filters=128, delete_features=[], prunning_function=preprocessing.get_min_len_prunner(L),
                                  add_bathymetry=False, velocity=None,
                                  to_origin=to_origin, fill_pad_day_rate=fill_pad_day_rate, pad_day_rate=pad_day_rate)
        inception.preprocess()
        X = deepcopy(inception.X_original)
        Year = deepcopy(inception.Year)

    pad_trajectory = get_pad_trajectory(day_rate=3, keep_NA=True)
    valid_trajectories = ~ np.array([np.isnan(x).any() for x in X])
    T = [x[2].copy() for x in X]
    X = np.array([pad_trajectory(x, t) for x, t, c in zip(X, T, valid_trajectories) if c])
    Year = np.array([pad_trajectory(y[None], t)[0] for y, t, c in zip(Year, T, valid_trajectories) if c])

    non_zero = (~np.isnan(X)).all(axis=1).T
    #t = np.unique(X[:,-1].round(decimals=3))[:-1]
    valid_trajs = {}
    for l in L:
        d_l = {}
        for i in range(non_zero.shape[0] - l):
            d_l[(i, i+l)] = np.where(non_zero[i:i+l].all(axis=0))[0].astype(np.int32)
        valid_trajs[l] = d_l
    return valid_trajs, X, Year

def trajectories_common_time_max(L, **kwargs):
    valid_trajectories, X, Year = trajectories_common_time(L, **kwargs)
    trajectories_maxlen = dict()
    for l, d_l in valid_trajectories.items():
        maxlen = 0
        for k, idxs in d_l.items():
            if idxs.size > maxlen:
                maxlen = idxs.size
                trajectories_maxlen[l] = (k, idxs)
    return trajectories_maxlen, X, Year

def prune_by_common_len(X, Year, labels, L=50, label_col="COMMON_NAME"):
    specs, X, Year = trajectories_common_time_max(L, X=X, Year=Year)
    (start, end), trajectories = [*specs.values()][0]

    X_pruned = X[trajectories, :, start:end]
    Year_pruned = Year[trajectories, start:end]
    y_pruned = labels[label_col].values[trajectories]
    labels_pruned = labels.iloc[trajectories]
    return X_pruned, Year_pruned, y_pruned, labels_pruned

def prune_by_min_animals(X, Year, labels, min_animals=2, label_col="COMMON_NAME"):
    animals_per_species = labels[label_col].value_counts().to_dict()
    good_args = [animals_per_species[species] >= min_animals for species in labels[label_col].values]

    X_pruned = [x for x, c in zip(X, good_args) if c]
    Year_pruned = [year for year, c in zip(Year, good_args) if c]
    y_pruned = labels[label_col][good_args].values
    labels_pruned = labels[good_args]
    return X_pruned, Year_pruned, y_pruned, labels_pruned, label_col

def get_prunning_function(column=None, colvalue=None, label="COMMON_NAME", NaN_value=None, min_animals=5, minlen=5, min_days=0, mode="year", mapping=None, func=None,
                          vertices=[]):
    """
    Returns function for prunning the dataset.
    Chooses those rows such that the column "column" has value "colvalue" and sets the classification target on the label "label".
    NaN_value: character for recognizing missing values.
    min_animals: minimum number of animals per species.
    minlen: minimum length of the trajectory.
    min_days: minimum number of days in the trajectory.
    func: function to apply to the metadata.
    vertices: iterable of vertices to prune the data. Only the longest part of the trajectory inside the polygon will be kept.
    """
    def prunning_function(X, Year, y):
        nonlocal colvalue, NaN_value
        if NaN_value is not None:
            if not isinstance(NaN_value, (list, tuple, np.ndarray)):
                NaN_value = [NaN_value]
            for nan_v in NaN_value:
                y[label].replace(nan_v, np.NaN, inplace=True)
        if func is not None:
            y = func(y)
        if colvalue is None:
            good_measurements = (~y[label].isna()) & (y[label] != NaN_value)
        else:
            if isinstance(colvalue, str):
                colvalue = [colvalue]
            good_measurements = pd.concat([y[column] == val for val in colvalue], axis=1).any(axis=1) & ((y[label] != NaN_value) if NaN_value is not None else ~y[label].isna())
        xlens = np.array([x.shape[1] for x in X])
        cat_greater_than_minlen = y[label][xlens > minlen]
        animals_per_cat = defaultdict(int)
        animals_per_cat.update(cat_greater_than_minlen.value_counts(dropna=False).to_dict())
        animals_gt_min = np.array([np.NaN if pd.isna(cat) else animals_per_cat[cat] >= min_animals for cat in y[label].values])
        good_measurements = good_measurements & animals_gt_min & (y[f"Days in trajectory ({mode})"] > min_days)

        labels_pruned = y[good_measurements]
        if mapping is not None:
            labels_pruned[f"{label}-original"] = labels_pruned[label].values
            if callable(mapping):
                labels_pruned.loc[:, label].replace(mapping(labels_pruned[label]), inplace=True)
            elif isinstance(mapping, dict):
                labels_pruned.loc[:, label].replace(mapping, inplace=True)
            else:
                raise TypeError("'mapping' must be a function, dict or None.")
        X_pruned = [x for x,t in zip(X, good_measurements) if t]
        Year_pruned = [year for year, t in zip(Year, good_measurements) if t]

        if len(vertices) > 0:
            X_pruned, Year_pruned, labels_pruned = geometry.prune_trajectories_inside_vertices(X_pruned, Year_pruned, labels_pruned, vertices)

        y_pruned = labels_pruned[label].values

        return X_pruned, Year_pruned, y_pruned, labels_pruned, label
    return prunning_function

shearwater_stage = get_prunning_function("COMMON_NAME", "Corys shearwater", label="Stage", NaN_value="unknown")
shearwater_sex = get_prunning_function("COMMON_NAME", "Corys shearwater", label="SEX", NaN_value="U")

def get_min_len_prunner(minlen, **prunning_kwargs):
    """Returns function for prunning trajectories with length < minlen."""
    def prunning_func(X, Year, y):
        base_prunner = get_prunning_function(**prunning_kwargs)
        X, Year, y, labels, label = base_prunner(X, Year, y)

        xlens = np.array([x.shape[1] for x in X])
        good_args = xlens > minlen

        X_pruned = [x for x, c in zip(X, good_args) if c]
        Year_pruned = [year for year, c in zip(Year, good_args) if c]
        y_pruned = y[good_args]
        labels_pruned = labels[good_args]
        return X_pruned, Year_pruned, y_pruned, labels_pruned, label
    return prunning_func

min_len_100 = get_min_len_prunner(100)

def get_min_days_prunner(min_days, mode="year", **prunning_kwargs):
    """Returns function for prunning trajectories with length < minlen."""
    def prunning_func(X, Year, y):
        base_prunner = get_prunning_function(**prunning_kwargs)
        X, Year, y, labels, label = base_prunner(X, Year, y)

        good_args = labels[f"Days in trajectory ({mode})"] > min_days

        X_pruned = [x for x, c in zip(X, good_args) if c]
        Year_pruned = [year for year, c in zip(Year, good_args) if c]
        y_pruned = y[good_args]
        labels_pruned = labels[good_args]
        return X_pruned, Year_pruned, y_pruned, labels_pruned, label
    return prunning_func

min_days_100 = get_min_days_prunner(100)

def get_trajectory_deleter(percentage, mode, random_state, species=None, taxa=None, N=None, **prunning_kwargs):
    """
    Returns function for prunning a % of the trajectories.
    - Attributes:
        - percentage:                   Proportion of trajectories deleted ([0,1]).
        - random_state:                 Random seed for deleting trajectories.
        - mode:         'random':       Delete randomly from all trajectories.
                        'species':      Choose randomly a species, and then delete randomly an individual from that species.
                        'stratified':   Delete the % stratifying by species (the % of individuals belonging to each species ~ constant).
                        'specific':     Delete the % of individuals from a specific species or taxa.
                                        If N is not None, delete individuals until the number of individuals of that species is N.
        - species, taxa:                Species or taxa to delete (only for mode='specific').

    """
    random.seed(random_state)

    get_df = lambda vals, cols: pd.DataFrame({col: arr for col, arr in zip(cols, vals.T)})
    def random_deleter(*args):
        return train_test_split(*args, test_size=percentage, random_state=random_state)[::2]
    def stratified_deleter(X, Year, y, label_vals):
        return train_test_split(X, Year, y, label_vals, stratify=y, test_size=percentage, random_state=random_state)[::2]
    def species_deleter(X, Year, y, label_vals):
        num_trajs = len(X)
        goal_size = num_trajs * (1 - percentage)
        species = np.unique(y)
        while num_trajs > goal_size:
            random_species = random.choice(species)
            species_idxs = np.argwhere(y == random_species)[:,0]
            deleted_idx = random.choice(species_idxs)
            del X[deleted_idx], Year[deleted_idx]
            y = np.delete(y, deleted_idx)
            label_vals = np.delete(label_vals, deleted_idx, axis=0)
            species = np.unique(y)
            num_trajs -= 1
        return X, Year, y, label_vals
    def specific_deleter(X, Year, y, label_vals):
        if species is not None:
            valid = (label_vals == species).any(axis=1)
        elif taxa is not None:
            valid = (label_vals == taxa).any(axis=1)
        else:
            raise ValueError("Must specify species or taxa.")
        idxs_species = np.where(valid)[0]
        idxs_rest = np.where(~valid)[0]
        num_trajs_species = idxs_species.size
        if N is not None:
            if N > num_trajs_species:
                raise ValueError(f"Number of individuals of {species} is {num_trajs_species}, smaller than N={N}.")
            else:
                goal_size = N
        else:
            goal_size = num_trajs_species * (1 - percentage)
        while num_trajs_species > goal_size:
            idxs_species = np.delete(idxs_species, random.randint(0, idxs_species.size - 1))
            num_trajs_species -= 1
        idxs = np.concatenate([idxs_species, idxs_rest])
        select_indices = lambda x, idxs: [x[i] for i in idxs]
        return select_indices(X, idxs), select_indices(Year, idxs), y[idxs], label_vals[idxs]

    if mode == "random":
        deleter = random_deleter
    elif mode == "species":
        deleter = species_deleter
    elif mode == "stratified":
        deleter = stratified_deleter
    elif mode == "specific":
        deleter = specific_deleter
    else:
        raise ValueError(f"mode {mode} not valid. Available: 'random', 'species', 'stratified', 'specific'.")

    def prunning_func(X, Year, y):
        base_prunner = get_prunning_function(**prunning_kwargs)
        X, Year, y, labels, label = base_prunner(X, Year, y)
        X, Year, y, label_vals = deleter(X, Year, y, labels.values)
        labels = get_df(label_vals, labels.columns)

        return X, Year, y, labels, label
    return prunning_func

def get_binary_prunning_function(column="Taxa", colvalue="Sharks"):
    """
    Returns function for prunning the dataset. The target is a binary variable in {0, 1}.
    Attributes:
        - column: binary variable.
        - colvalue: value for which the variable is 1.
    """
    def prunning_function(X, Year, y):
        label_col = f"{column} (binary)"
        y[label_col] = y[column].copy()
        y[label_col][y[label_col] != colvalue] = f"Not {colvalue}"
        y_target = y[label_col].map({colvalue: 1}).fillna(0).astype(np.int32).values

        return X, Year, y_target, y, label_col
    return prunning_function

def get_stage_prunner_all(mapping=None, NaN_value="unknown", **kwargs):
    """Splits all species by stage (breading, non-breading)."""
    if mapping is None:
        def mapping(s):
            non_breeding_keys = ["non-breeding", "migration", "hivernage"]
            stage_map = {k: "non-breeding" if any(nbk in k.lower() for nbk in non_breeding_keys) else "breeding" for k in s.unique()}
            return stage_map
    elif isinstance(mapping, bool) and not mapping:
        mapping = None
    def prunning_func(X, Year, y):
        base_prunner = get_prunning_function(column="Stage", colvalue=None, label="Stage", NaN_value=NaN_value, mapping=mapping,
                                             **kwargs
                                            )
        return base_prunner(X, Year, y)
    return prunning_func



##############################################################################################################################
"""                                                   VIII. Shapelets                                                      """
##############################################################################################################################

def prune_shapelets(threshold=1, zero_pad=False, d=0, minlen=3, distance="euclidean", parentDir=fullPath("utils/data/shapelet/pruned"), overwrite=False):
    """Prunes the shapelets using hierarchical clustering, with distance=DTW."""
    kwds = {k: v for k, v in locals().items() if k not in ["parentDir", "overwrite"]}
    filename = f"{other_utils.dict_to_id(kwds)}.lzma"
    path = os.path.join(parentDir, filename)
    if Path(path).exists() and not overwrite:
        return file_management.load_lzma(path)
    else:
        shapelets = analysis.load_shapelets(zero_pad=[zero_pad], d=d, minlen=minlen)
        corr_linkage, _ = data_visualization.shapelet_dendrogram(d=d, distance=distance)
        cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(lambda: defaultdict(list))
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id]["idxs"].append(idx)
            cluster_id_to_feature_ids[cluster_id]["l"].append(shapelets[idx].length)

        inception = models.InceptionTime(pad_day_rate=3, prunning_function=get_min_days_prunner(180), to_origin="space")
        inception.preprocess()
        X_train = tf_to_df(inception.X_train, zero_pad=zero_pad, d=d)

        selected = []
        for data in cluster_id_to_feature_ids.values():
            s = shapelets[data["idxs"][np.argmin(np.abs(data["l"] - np.median(data["l"])))]]
            selected.append(X_train.iloc[s.series_id, 0][s.start_pos: s.start_pos+s.length].values)

        Path(parentDir).mkdir(exist_ok=True, parents=True)
        file_management.save_lzma(selected, filename, parentDir)
        return selected



##############################################################################################################################
"""                                                      IX. Other                                                         """
##############################################################################################################################

def lat_lon_bins(n_lat=20, n_lon=40, r=1.0, out="deg"):
    """
    Longitude-latitude bins such that each cell has equal surface.
    Retrieved from: http://notmatthancock.github.io/2017/12/26/regular-area-sphere-partitioning.html
    """
    r = int(r)
    phi, delta_phi = np.linspace(0, 2*np.pi, n_lon, retstep=True)
    delta_S = delta_phi / n_lat

    theta = 1 - np.arange(2*(r**2)*n_lat+1) * delta_S / (r**2 * delta_phi)
    theta = np.arccos(theta)

    if out == "deg":
        phi = phi*180/np.pi - 180
        theta = theta*180/np.pi - 90
    return theta, phi

def tf_to_df(X_tf, zero_pad=False, d=0):
    """
    Tensorflow dataset to pandas DataFrame.
    Attributes:
        - X_tf: dataset as with shape (N, length, num_series, 1).
        - zero_pad: pad with zeros so that the signal has length 1 year.
        - d: univariate series index.
    """
    X = X_tf.numpy().squeeze().copy()
    if zero_pad:
        X_list = [x[:, d] for x in X]
    else:
        X_list = [x[(x != 0).all(axis=1), d] for x in X]
        l = min([x.size for x in X_list])
        X_list = [x[:l] for x in X_list]
    X_list = [pd.Series(x) for x in X_list]
    X_df = pd.Series(X_list).reset_index().iloc[:, 1:]
    return X_df

def extract_stage_data_v2(species, weather=None, pad_day_rate=None):
    if isinstance(species, str):
        species = [species]
    df_species_pruned, labels_species_pruned, pruned_years= extract_stage_data_v2_full(weather=weather, pad_day_rate=pad_day_rate)
    pruned_years = pd.Series(pruned_years, index=labels_species_pruned.index).loc[species].tolist()
    df_species_pruned = df_species_pruned.loc[species]
    labels_species_pruned = labels_species_pruned.loc[species]
    return df_species_pruned, labels_species_pruned, pruned_years

@savedata
def extract_stage_data_v2_full(weather=None, pad_day_rate=None):
    """
    Data from v2 has several breeding stages per individual. This function extracts each stage as a different individual.
    """
    species = params.stage_species_v2
    if pad_day_rate is None:
        df, labels, year = load_all_data(weather=weather, return_labels=True, v2=True, pad_day_rate=pad_day_rate)
        stage = breeding_stage_v2(pad_day_rate=pad_day_rate)
    else:
        stage = equally_spaced_trajectories_avg_v2(weather=weather, pad_day_rate=pad_day_rate)
        IDs = stage.ID.unique()
        xcols = [c for c in stage.columns if c.startswith('x')]
        X = [s[xcols].values.T.astype(float) for _, s in stage.groupby("ID")]
        year = [s.year.values for _, s in stage.groupby("ID")]
        labels_v2 = load_data_v2()[2]
        labels = labels_v2.set_index("ID").loc[IDs].reset_index().set_index(["COMMON_NAME", "ID"])
        df = pd.Series(X, index=labels.index)
        stage = stage[["stage", "ID", "COMMON_NAME"]]
        stage.columns = ["Stage", "ID", "COMMON_NAME"]

    df_species = df.loc[species]
    year_species = pd.Series(year, index=df.index).loc[species]
    stage_species = stage.query(f'COMMON_NAME in {species}')
    labels_species = labels.loc[species]

    pruned_trajectories = []
    pruned_IDs = []
    pruned_labels = []
    pruned_years = []
    for common_name, ID in tqdm(df_species.index):
        idx = (common_name, ID)
        stage_ID = stage_species.query(f"COMMON_NAME == '{common_name}' and ID == '{ID}'").drop(columns=['ID', 'COMMON_NAME'])
        stage_types = stage_ID.Stage.dropna().drop_duplicates().values
        if stage_types.size == 1:
            pruned_trajectories.append(df_species.loc[idx])
            pruned_IDs.append(ID)
            new_label = labels_species.loc[idx].copy()
            new_label['Stage'] = stage_types[0]
            new_label['ID-real'] = ID
            new_label['Length'] = df_species.loc[idx].shape[1]
            new_label['COMMON_NAME'] = common_name
            pruned_labels.append(new_label)
            pruned_years.append(year_species.loc[idx])
        else:
            for t in stage_types: # case t == 0 included
                is_stage = (stage_ID.Stage == t).values
                # extract each sequence of consecutive True values
                stage_changes = np.diff(np.hstack([False, is_stage, False]).astype(int))
                stage_starts = np.where(stage_changes == 1)[0]
                stage_end = np.where(stage_changes == -1)[0]
                for i, (start, end) in enumerate(zip(stage_starts, stage_end)):
                    pruned_trajectories.append(df_species.loc[idx][:, start:end])
                    pruned_IDs.append(f"{ID}-{t}-{i+1}")

                    # update new label
                    new_label = labels_species.loc[idx].copy()
                    year_i = year_species.loc[idx][start:end]
                    pruned_years.append(year_i)
                    new_label['Length'] = end - start
                    new_label['Stage'] = t
                    new_label['ID-real'] = ID
                    new_label['COMMON_NAME'] = common_name
                    pruned_labels.append(new_label)

    labels_species_pruned = pd.concat(pruned_labels, axis=1).T
    labels_species_pruned['ID'] = pruned_IDs
    labels_species_pruned = labels_species_pruned.set_index(['COMMON_NAME', 'ID'])
    df_species_pruned = pd.Series(pruned_trajectories, index=labels_species_pruned.index)

    valid = labels_species_pruned.Length > 1
    pruned_years = pd.Series(pruned_years)[valid.values].tolist()
    labels_species_pruned = labels_species_pruned[valid]
    df_species_pruned = df_species_pruned[valid]

    time_edges = [0, 1, 10, 50, 100, 200, 500]
    for mode in ["all", "year"]:
        ndays = temporal_extension(list(df_species_pruned.values), pruned_years, mode=mode)
        ndays_interval = pd.cut(ndays, time_edges)
        labels_species_pruned[f'Days in trajectory ({mode})'] = ndays
        labels_species_pruned[f'Days in trajectory ({mode}) - interval'] = ndays_interval

    return df_species_pruned, labels_species_pruned, pruned_years

def relabel_stages(labels, species, remap_breeding=False, generic_to_nan=False):
    stage = labels.Stage
    stage_type = stage.value_counts().index.tolist()
    NaN_value = ['unknown', 'Summer loops']
    if generic_to_nan:
        NaN_value += ['breeding', 'REPRO']

    non_breeding = [t for t in stage_type if 'non-breeding' in t] + ['Migration', 'HIVERNAGE', 'Foraging']
    breeding = ['Internesting']

    for nan_val in NaN_value:
        stage = stage.replace(nan_val, np.NaN)
    for nb in non_breeding:
        stage = stage.replace(nb, 'non-breeding')
    for b in breeding:
        stage = stage.replace(b, f'breeding: {b.lower()}')
    stage = stage.replace('REPRO', 'breeding')
    # if the species has specific breeding stages, map the general to NaN to avoid confusion.
    species_stages = labels.query("COMMON_NAME == @species").Stage.unique()
    if any("breeding:" in st for st in species_stages):
        is_species = labels['COMMON_NAME'] == species
        stage[is_species] = stage[is_species].replace({'breeding': np.NaN, 'REPRO': np.NaN})

    if remap_breeding:
        stage = stage.replace(params.breeding_remaps)
    labels['Stage'] = stage
    return labels

def relabel_stages_binary(labels):
    stage = labels.Stage
    stage_type = stage.value_counts().index.tolist()
    NaN_value = ['unknown', 'Summer loops']

    non_breeding = [t for t in stage_type if 'non-breeding' in t] + ['Migration', 'HIVERNAGE', 'Foraging']
    breeding = ['Internesting'] + [t for t in stage_type if 'breeding:' in t] + ['REPRO']

    for nan_val in NaN_value:
        stage = stage.replace(nan_val, np.NaN)
    for nb in non_breeding:
        stage = stage.replace(nb, 'non-breeding')
    for b in breeding:
        stage = stage.replace(b, 'breeding')
    labels['Stage'] = stage
    return labels

def compute_num_stages():
    """
    Computes the number of stages per species.
    """
    species = params.stage_species_v2

    func = lambda df: df.groupby('COMMON_NAME').apply(lambda df: relabel_stages(df, df.COMMON_NAME.iloc[0], remap_breeding=False))
    prunning_function=get_prunning_function(label='Stage', func=func)
    kwargs = dict(v2=True, weather=None, prunning_function=prunning_function, common_origin_distance=False, species_stage=species, delete_features=[], pad_day_rate=None, fill_pad_day_rate=False, scale_by_velocity=True, fit_scale_side=True, invert_lat=False)
    tree = models.DecisionTree(**kwargs)

    num_stages = tree.labels.groupby('COMMON_NAME').apply(lambda S: S.Stage.nunique())
    binary = num_stages[num_stages == 2].index.to_list()
    multilabel = num_stages[num_stages > 2].index.to_list()
    one_class = num_stages[num_stages == 1].index.to_list()
    return multilabel, binary, one_class

def load_all_data(weather=None, return_labels=True, v2=True, species_stage=None, pad_day_rate=None, species_train=None, invert_lat=False, expand_df=False):
    """
    If species_train is provided, the data is inverted for those species that are in the opposite hemisphere.
    expand_df: if True, the Series containing ID -> trajectory is expanded in a single DataFrame.
    """
    if v2:
        weather_specific = weather is not None and weather not in ['all', 'all-depth', 'pruned', 'mrmr+collinear', 'vif', 'mrmr+vif', 'mrmrloop+vif', 'mrmr']
        if weather_specific:
            weather_input = 'all'
        else:
            weather_input = weather
        if species_stage is not None:
            df, labels, year = extract_stage_data_v2(species_stage, weather=weather_input, pad_day_rate=pad_day_rate)
            if species_train is not None and invert_lat:
                print("Inverting latitudes for species in the opposite hemisphere.")
                lats = df.groupby(level=0).apply(lambda S: S.apply(lambda x: x[0].mean()).median())
                lat_hemisphere = np.sign(lats)
                lat_train = lat_hemisphere[species_train]
                switch_lat = lat_hemisphere != lat_train
                species_switch = switch_lat[switch_lat].index.tolist()
                def invert_lat(x):
                    y = x.copy()
                    y[0] = -y[0]
                    return y

                for s in species_switch:
                    df.loc[s] = df.loc[s].apply(invert_lat).values
        else:
            X, year, labels = load_data_v2(weather=weather_input, pad_day_rate=pad_day_rate)
            labels = labels.set_index(["COMMON_NAME", "ID"])
            df = pd.Series(X, index=labels.index)
        if weather_specific:
            all_cols = np.array(weather_cols_v2["all"])
            weather_features = weather.split("+")
            weather_idxs = np.array([i for i, c in enumerate(all_cols) if c in weather_features])
            idxs = np.concatenate([np.arange(3), weather_idxs + 3]) # 3 for lat, lon, day
            df = df.apply(lambda x: x[idxs])
    else:
        if pad_day_rate is not None:
            raise ValueError("pad_day_rate is only available for v2 data.")

    if expand_df:
        df.index = df.index.droplevel(0)
        X = np.concatenate(tuple(df.values), axis=1).T
        L = df.apply(lambda x: x.shape[1])
        ID = np.repeat(df.index, L)
        columns = ['lat', 'lon', 'day']
        if weather is not None:
            colmap = weather_cols_v2 if v2 else weather_cols
            columns += colmap[weather]
        df = pd.DataFrame(X, index=ID, columns=columns)
    if return_labels:
        return df, labels, year
    else:
        return df

def load_features():
    df = load_all_data(weather='all-depth', return_labels=False, v2=True, expand_df=True)
    X_periodic = make_periodic(df.values.T[:3], year=2022*np.ones(df.shape[0]))[0]
    df_periodic = pd.DataFrame(X_periodic.T, columns=['x', 'y', 'z', 'sin t', 'cos t'], index=df.index)
    df = pd.concat([df_periodic, df.iloc[:, 3:]], axis=1)
    df.columns = [params.feature_map[f] for f in df.columns]
    return df

@savedata
def feature_corr(method='spearman'):
    df = load_features()
    return df.corr(method=method)

@savedata
def feature_vif(exclude_cols='collinear'):
    """
    Computes the variance inflation factor for each feature.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df = load_features()
    if exclude_cols is not None and 'collinear' in exclude_cols:
        cols = spatiotemporal_cols + weather_cols_v2['pruned']
        cols_mapped = [params.feature_map[c] for c in cols]
        df = df[cols_mapped]
        if '+' in exclude_cols:
            exclude = exclude_cols.split('+')
            cols = [c for c in df.columns if c not in exclude]
            excluded = set(df.columns) - set(cols)
            print(f"Excluded: {excluded}")
            df = df[cols]
    elif exclude_cols is not None:
        raise NotImplementedError
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    VIF = []
    for i in tqdm(range(df.shape[1])):
        VIF.append(variance_inflation_factor(df.values, i))
    vif["VIF"] = VIF
    return vif

@savedata
def features_redundancy(feature='x', maxlen=300, n_neighbors=10):
    """
    Computes the redundancy between the one feature and the rest.
    """
    from sklearn.feature_selection import mutual_info_regression
    df = load_features()
    if maxlen is not None:
        df = df.groupby(df.index, group_keys=False).apply(lambda S: S if S.shape[0] < maxlen else S.iloc[np.linspace(0, S.shape[0]-1, maxlen).astype(int)])

    X = df.values.astype(np.float32) # keep target feature for normalization
    y = df[feature].values.astype(np.float32).squeeze()

    mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=0)
    mi_sorted = pd.Series(mi, index=df.columns).sort_values(ascending=False)
    mi_sorted.name = feature
    return mi_sorted

@savedata
def features_mi(exclude_cols=None, maxlen=300, target_type='species', n_neighbors=10, maxlen_species=None, maxlen_taxa=None):
    """
    Computes the mutual information between the features and the target.
    """
    from sklearn.feature_selection import mutual_info_classif
    df = load_features()
    if maxlen is not None:
        df = df.groupby(df.index, group_keys=False).apply(lambda S: S if S.shape[0] < maxlen else S.iloc[np.linspace(0, S.shape[0]-1, maxlen).astype(int)])
    if exclude_cols is not None and 'collinear' in exclude_cols:
            cols = spatiotemporal_cols + weather_cols_v2['pruned']
            cols_mapped = [params.feature_map[c] for c in cols]
            df = df[cols_mapped]
    elif exclude_cols is not None:
        raise NotImplementedError

    if maxlen_species is not None:
        df['species'] = map_ID_to_species(df.index)
        df = df.groupby('species', group_keys=False).apply(lambda S: S if S.shape[0] < maxlen_species else S.iloc[np.linspace(0, S.shape[0]-1, maxlen_species).astype(int)])
        df = df.drop(columns=['species'])
    if maxlen_taxa is not None:
        df['species'] = map_ID_to_species(df.index)
        species_to_taxa = get_species_to_taxa()
        df['Taxa'] = np.array([species_to_taxa[s] for s in df['species']])
        df = df.groupby('Taxa', group_keys=False).apply(lambda S: S if S.shape[0] < maxlen_taxa else S.iloc[np.linspace(0, S.shape[0]-1, maxlen_taxa).astype(int)])
        df = df.drop(columns=['Taxa', 'species'])

    print(df.shape)
    target = map_ID_to_species(df.index).squeeze()
    if target_type == 'taxa':
        species_to_taxa = get_species_to_taxa()
        target = np.array([species_to_taxa[s] for s in target])
    target_to_int = {v: i for i, v in enumerate(np.unique(target))}
    target = np.array([target_to_int[t] for t in target])
    mi = mutual_info_classif(df.values.astype(np.float32), target, discrete_features=False, n_neighbors=n_neighbors)
    mi_sorted = pd.Series(mi, index=df.columns).sort_values(ascending=False)
    return mi_sorted

def features_mi_normalized(maxlen=300, maxlen_species=None, maxlen_taxa=None, **kwargs):
    mi = features_mi(maxlen=maxlen, **kwargs)
    mi /= features_mi_upper_bound(maxlen=maxlen, maxlen_species=maxlen_species, maxlen_taxa=maxlen_taxa)
    return mi

@savedata
def features_mi_upper_bound(maxlen=300, maxlen_species=None, maxlen_taxa=None):
    from scipy.stats import entropy

    df = load_features()
    df = df.groupby(df.index, group_keys=False).apply(lambda S: S if S.shape[0] < maxlen else S.iloc[np.linspace(0, S.shape[0]-1, maxlen).astype(int)])
    if maxlen_species is not None:
        df['species'] = map_ID_to_species(df.index)
        df = df.groupby('species', group_keys=False).apply(lambda S: S if S.shape[0] < maxlen_species else S.iloc[np.linspace(0, S.shape[0]-1, maxlen_species).astype(int)])
        df = df.drop(columns=['species'])
    if maxlen_taxa is not None:
        df['species'] = map_ID_to_species(df.index)
        species_to_taxa = get_species_to_taxa()
        df['Taxa'] = np.array([species_to_taxa[s] for s in df['species']])
        df = df.groupby('Taxa', group_keys=False).apply(lambda S: S if S.shape[0] < maxlen_taxa else S.iloc[np.linspace(0, S.shape[0]-1, maxlen_taxa).astype(int)])
        df = df.drop(columns=['Taxa', 'species'])

    target = map_ID_to_species(df.index).squeeze()
    target_to_int = {s: i for i, s in enumerate(np.unique(target))}
    target_int = np.array([target_to_int[s] for s in target])

    value_counts = np.bincount(target_int)
    probs = value_counts / value_counts.sum()
    max_mi = entropy(probs, base=2)
    return max_mi

@savedata
def load_data_v2(weather=None, pad_day_rate=None):
    meta = pd.read_csv(fullPath("data/metadata.csv"))
    colmap = {c: c for c in meta.columns}
    colmap['Species'] = 'COMMON_NAME'
    meta.columns = meta.columns.map(colmap)
    if pad_day_rate is not None:
        df = equally_spaced_trajectories_avg_v2(weather=weather, pad_day_rate=pad_day_rate)
        xcols = [c for c in df.columns if c.startswith('x')]
        X = [s[xcols].values.T.astype(float) for _, s in df.groupby("ID")]
        # df from equally_spaced_trajectories_avg_v2 is NOT sorted
        meta = meta.reset_index().set_index("ID").loc[df.ID.unique()]
        meta = meta.reset_index().set_index(["NewID"])
        assert (df.ID.unique() == meta.ID).all(), "IDs in df and meta are not the same."
    elif weather is None:
        df = pd.read_csv(fullPath("data/dataset.csv"))
        df['date'] = pd.to_datetime(df['DATE_TIME'])
        df['day'] = (df.date.dt.dayofyear + df.date.dt.hour / 24 + df.date.dt.minute / (24 * 60) + df.date.dt.second / (24 * 60 * 60)) - 1
        df['year'] = df.date.dt.year
        X = [s[['LATITUDE', 'LONGITUDE', 'day']].values.T for _, s in df.groupby("ID")]
    else:
        df = env_data_v2_imputed()
        df = df.drop(columns=['index'])
        df['day'] = (df.DATE_TIME.dt.dayofyear + df.DATE_TIME.dt.hour / 24 + df.DATE_TIME.dt.minute / (24 * 60) + df.DATE_TIME.dt.second / (24 * 60 * 60)) - 1
        df['year'] = df.DATE_TIME.dt.year
        X = [s[['LATITUDE', 'LONGITUDE', 'day'] + weather_cols_v2[weather]].values.T for _, s in df.groupby("ID")]
        meta = meta.set_index("ID").loc[df.ID.unique()]
        assert (df.ID.unique() == meta.ID).all(), "IDs in df and meta are not the same."

    year = [s.year.values for _, s in df.groupby("ID")]
    # if pad_day_rate is not None:
    #     X, year = equally_spaced_trajectories(X, year, day_rate=pad_day_rate)
    #     is_nan = np.array([np.isnan(x).all() for x in X])
    #     if is_nan.any():
    #         num_removed = is_nan.sum()
    #         warnings.warn(f"Removed {num_removed} trajectories with only NaN values after padding.", UserWarning)
    #         X = [x for x, nan in zip(X, is_nan) if not nan]
    #         year = [y for y, nan in zip(year, is_nan) if not nan]
    #         meta = meta[~is_nan]

    # update days in trajectory in meta
    time_edges = [0, 1, 10, 50, 100, 200, 500]
    for mode in ["all", "year"]:
        ndays = temporal_extension(X, year, mode=mode)
        ndays_interval = pd.cut(ndays, time_edges)
        meta[f'Days in trajectory ({mode})'] = ndays
        meta[f'Days in trajectory ({mode}) - interval'] = ndays_interval
    meta['Length'] = [x.shape[1] for x in X]
    return X, year, meta

@savedata
def weather_data_base_v2():
    """
    DataFrame with columns ['DATE_TIME', 'LATITUDE', 'LONGITUDE'] to be used as base for weather data.
    """
    df = pd.read_csv(fullPath("data/dataset.csv"))
    if df['DATE_TIME'].dtype in [str, object]:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    columns = ['DATE_TIME', 'LATITUDE', 'LONGITUDE']
    df = df[columns]
    return df

def weather_data_base_v2_julia(csv=True, overwrite=False):
    """
    Stores weather_data_base_v2() in csv or zstd compression to be used in Julia.
    """
    if csv:
        output_path = 'utils/data/mobility_data_v2_jl.csv'
        if not os.path.exists(output_path) or overwrite:
            print ('Storing')
            df = weather_data_base_v2()
            df['DATE_TIME'] = df.DATE_TIME.dt.floor('ms')
            df.to_csv(output_path, index=False)
        return pd.read_csv(output_path)

    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path = 'utils/data/mobility_data_v2_jl.zstd'
        if not os.path.exists(output_path) or overwrite:
            df = weather_data_base_v2()
            df['DATE_TIME'] = df.DATE_TIME.dt.floor('ms')
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path, compression='zstd')
            return df
        else:
            print("File already exists.")
            return pq.read_table(output_path).to_pandas()


@savedata
def compute_df_distance(add_time=True, **kwargs):
    sys.path.append(RootDir)
    from forecasting.preprocessing import space

    kwargs.pop('return_labels', None)
    df, labels, year = load_all_data(return_labels=True, **kwargs)
    df = pd.Series([(t, y) for t, y in zip(df.values, year)], index=labels.index)
    T = df.apply(lambda x: x[0][2])
    lat_lon = df.apply(lambda x: x[0][:2] * np.pi / 180)
    df_dx = pd.Series([space.spherical_velocity(cds)[:,-2:].T for cds in lat_lon], index=labels.index)
    def initial_location_zero(x):
        origin = np.zeros((2, 1))
        if np.isnan(x).all():
            return origin
        else:
            return np.cumsum(np.hstack([np.zeros((2, 1)),  x]), axis=1)

    df_distance = df_dx.apply(initial_location_zero)
    if add_time:
        df_distance = pd.Series([np.vstack([cds, t[None]]) for (cds, t) in zip(df_distance.values, T.values)], index=labels.index)
    return df_distance, labels, year

def compute_df_distance_weather_spec(weather=None, **kwargs):
    weather_specific = weather is not None
    if weather_specific:
        weather_input = 'all'
        df, labels, year = compute_df_distance(weather=weather_input, **kwargs)
        df_weather_vars = load_all_data(weather=weather, return_labels=False, **kwargs)
        V = df_weather_vars.apply(lambda x: x[3:])
        df_distance = pd.Series([np.vstack([cds, v]) for (cds, v) in zip(df.values, V.values)], index=labels.index)
        return df_distance, labels, year
    else:
        return compute_df_distance(weather=weather, **kwargs)

def compute_df_velocity(return_dt=True, **kwargs):
    weather = kwargs.pop('weather', None)
    if weather is not None and weather not in ['all', 'all-depth']:
        weather_input = 'all'
    else:
        weather_input = weather
    df_distance, _, year = compute_df_distance(weather=weather_input, **kwargs)
    df_dx = df_distance.apply(lambda x: np.diff(x[:2], axis=1))
    dt = pd.Series([compute_dt(x[2], y) for x, y in zip(df_distance.values, year)], index=df_dx.index)
    v = df_dx / dt
    v_norm = v.apply(np.linalg.norm, axis=0)
    if return_dt:
        return v, v_norm, dt
    else:
        return v, v_norm

def rescale_by_velocity(X, v, v_norm, dt, scaler_v, fit=False, remove_outliers=True, side=False):
    S = []
    for (scaler, i_train, i_test, i_side) in scaler_v.values():
        if side:
            idxs = i_side
        elif fit:
            idxs = i_train
        else:
            idxs = i_test
        if len(idxs) > 0:
            v_norm_i = v_norm.iloc[idxs]
            v_i = v.iloc[idxs]
            dt_i = dt.iloc[idxs]
            v_angle = v_i / v_norm_i
            v_norm_species = np.hstack(v_norm_i)
            mu = v_norm_species.mean()
            sigma = v_norm_species.std()
            if remove_outliers:
                Z_score = (v_norm_species - mu) / sigma
                v_zscore3 = 3 * sigma + mu
                v_norm_species[Z_score > 3] = v_zscore3
                def smooth_outliers(x):
                    Z = (x - mu) / sigma
                    x[Z > 3] = v_zscore3
                    return x
                v_norm_i = v_norm_i.apply(smooth_outliers)
            if fit:
                scaler.fit(v_norm_species[:, None])

            v_rescaled = v_angle * v_norm_i.apply(lambda x: scaler.transform(x[:, None])[:,0])
            # when v_norm == 0, v_angle is np.NaN. Replace by zeros
            def nan_to_zero(x):
                x[:, np.isnan(x).all(axis=0)] = 0
                return x
            v_rescaled = v_rescaled.apply(nan_to_zero)
            rescaled_trajectory = (v_rescaled * dt_i).apply(lambda x: np.hstack([np.zeros((2, 1)), np.cumsum(x, axis=1)]))
            rescaled_trajectory.index = idxs
            S.append(rescaled_trajectory)

    S = pd.concat(S, axis=0, ignore_index=False)
    S = S.loc[S.index.intersection(np.arange(len(X)))]
    X_rescaled = []
    for i, x in enumerate(X):
        if i in S.index:
            s = S.loc[i]
            x_rescaled = x.copy()
            x_rescaled[:2] = s
            X_rescaled.append(x_rescaled)
        else:
            warnings.warn(f"Scaling not available for index {i}", RuntimeWarning)
            X_rescaled.append(x)
    return X_rescaled

def rescale_to_img(X, v, v_norm, dt, scaler_v, fit=False, remove_outliers=True, indiv_scaling=True,
                   # bins_ns=np.linspace(-1.91, 1.29, 201), bins_we=np.linspace(-2.22, 8.58, 601),
                   bins_ns = np.linspace(0, 1, 101), bins_we = np.linspace(0, 1, 101),
                   density=0):
    """
    This function rescales the trajectory such that the trajectory fits in a 2D image.
    rescale_by_velocity rescales the velocity
    rescale_to_img      rescales the trajectory.
    """
    S = []
    X_rescaled = []
    for (scaler, i_train, i_test, _ ) in scaler_v.values():
        if fit:
            idxs = i_train
        else:
            idxs = i_test
        if len(idxs) > 0:
            if remove_outliers:
                v_norm_i = v_norm.iloc[idxs]
                v_i = v.iloc[idxs]
                dt_i = dt.iloc[idxs]
                v_angle = v_i / v_norm_i
                v_norm_species = np.hstack(v_norm_i)
                mu = v_norm_species.mean()
                sigma = v_norm_species.std()
                Z_score = (v_norm_species - mu) / sigma
                v_zscore3 = 3 * sigma + mu
                v_norm_species[Z_score > 3] = v_zscore3
                def smooth_outliers(x):
                    Z = (x - mu) / sigma
                    x[Z > 3] = v_zscore3
                    return x
                v_norm_i = v_norm_i.apply(smooth_outliers)
                v_rescaled = v_angle * v_norm_i
                def nan_to_zero(x):  # when v_norm == 0, v_angle is np.NaN. Replace by zeros
                    x[:, np.isnan(x).all(axis=0)] = 0
                    return x
                v_rescaled = v_rescaled.apply(nan_to_zero)
                x_rescaled = (v_rescaled * dt_i).apply(lambda x: np.hstack([np.zeros((2, 1)), np.cumsum(x, axis=1)]))
                x_rescaled = list(x_rescaled.values)
            else:
                x_rescaled = [X[i] for i in idxs]
            if indiv_scaling:
                if fit:
                    x_rescaled_full = np.hstack(x_rescaled)
                    scaler.fit(x_rescaled_full.T)
                x_rescaled = pd.Series([scaler.transform(x.T).T for x in x_rescaled], index=idxs)
                S.append(x_rescaled)
            else:
                X_rescaled.append(pd.Series(x_rescaled, index=idxs))
    if not indiv_scaling:
        scaler = next(iter(scaler_v.values()))[0]
        X_rescaled = pd.concat(X_rescaled, axis=0, ignore_index=False)
        if fit:
            X_rescaled_full = np.hstack(X_rescaled.values)
            scaler.fit(X_rescaled_full.T)
        X_rescaled = X_rescaled.apply(lambda x: scaler.transform(x.T).T)
        S = X_rescaled
    else:
        S = pd.concat(S, axis=0, ignore_index=False)
    scaling_not_available = pd.Index(np.arange(len(X))).difference(S.index)
    for i in scaling_not_available:
        warnings.warn(f"Scaling not available for index {i}", RuntimeWarning)
        S.loc[i] = X[i]
    S = S.loc[np.arange(len(X))]
    bins = [bins_ns, bins_we]
    X_rescaled = [np.histogram2d(*x, bins=bins, density=density)[0] for x in S.values]
    if not density:
        def binary_histogram(x):
            x[x > 0] = 1
            return x
        X_rescaled = [binary_histogram(x) for x in X_rescaled]
    return X_rescaled


def load_fishing_df():
    """
    The latitude and longitude width have spacing 1 degree.
    Check it with:

        lat, lon = fishing_data[['lat', 'lon']].values.T
        pd.Series(np.diff(lon)).value_counts()
        pd.Series(np.diff(lat)).value_counts()

    Bins are 1 degree wide, so they cover from lat - 0.5 to lat + 0.5 and lon - 0.5 to lon + 0.5
    To link one position to its fishing effort, we need to find the bin that contains it.
    The bin is the rounded value of the position.
    """
    fishing_data = pd.read_csv(fullPath('data/v2/Data AIAM/fishing_agg2016.csv'), index_col=0).sort_values(by=['lat', 'lon']).iloc[:, 1:]
    fishing_data.columns = ['effort', 'lat', 'lon']
    fishing_data['effort'][fishing_data['effort'] == 0] = np.NaN
    fishing_data['lat'] = fishing_data['lat'].astype(int)
    fishing_data['lon'] = fishing_data['lon'].astype(int)
    return fishing_data

def fishing_effort_grid():
    """
    Returns a dictionary containing the fishing effort data for each bin.
    """
    fishing_data = load_fishing_df()
    effort = fishing_data.set_index(['lat', 'lon']).to_dict()['effort']

    grid = [(lat_i, lon_i) for lat_i in range(-90, 91) for lon_i in range(-180, 181)]
    grid_has_effort_values = set(effort.keys())
    grid_no_effort_values = set(grid) - grid_has_effort_values
    for cell in grid_no_effort_values:
        effort[cell] = np.NaN

    # fill the gap at longitude -180 and 180.
    for lat_i in range(-90, 91):
        eff_left = effort[(lat_i, -180)]
        eff_right = effort[(lat_i, 180)]
        if math.isnan(eff_left):
            effort[(lat_i, -180)] = eff_right
        elif math.isnan(eff_right):
            effort[(lat_i, 180)] = eff_left
        else:
            # take the average for both
            avg_eff = (eff_left + eff_right) / 2
            effort[(lat_i, -180)] = avg_eff
            effort[(lat_i, 180)] = avg_eff
    return effort

@savedata
def fishing_effort(v2=True):
    """
    Returns a pandas series containing the fishing effort data for each trajectory.
    Each latitude,longitude pair of the trajectory is mapped to a fishing effort value.
    """
    effort = fishing_effort_grid()
    trajectories = load_all_data(v2=v2, return_labels=False)
    trajectories = trajectories.reset_index().set_index('ID').drop(columns='COMMON_NAME').iloc[:,0]
    def find_effort(x):
        x_rounded = np.round(x[:2]).astype(int)
        eff = np.empty(x.shape[1])
        for i, x_i in enumerate(x_rounded.T):
            eff[i] = effort[tuple(x_i)]
        return eff

    effort_trajectories = trajectories.apply(find_effort)
    return effort_trajectories

def get_fishing_effort(v2=True, nan_to_zero=True):
    """
    Returns a pandas series containing the fishing effort data for each trajectory.
    Each latitude,longitude pair of the trajectory is mapped to a fishing effort value.
    """
    eff = fishing_effort(v2=v2)
    if nan_to_zero:
        def map_NaN_to_zero(x):
            y = x.copy()
            y[np.isnan(y)] = 0
            return y
        eff = eff.apply(map_NaN_to_zero)
    return eff

def biodiversity_data():
    biodiversity = pd.read_csv(fullPath('data/v2/Data AIAM/effectivespeciespergridcell.csv')).sort_values(by=['Lat', 'Lon'])
    biodiversity = biodiversity[['EffectiveNorm', 'Lat', 'Lon']]
    return biodiversity

def biodiversity_grid():
    """
    Lat and lon represent the left point of the grid cell.
    Example:
    lat = 1 comprises latitudes within [1, 2)
    lon = -1 comprises longitudes within [-1, 0)
    """
    biodiversity = biodiversity_data()
    biodiversity['Lat'] = np.floor(biodiversity['Lat'].values).astype(int)
    biodiversity['Lon'] = np.floor(biodiversity['Lon'].values).astype(int)
    grid = biodiversity.set_index(['Lat', 'Lon']).to_dict()['EffectiveNorm']
    # map longitude 180 to 179 (in the grid, lon=179 comprises long in [179, 180])
    for lat in range(-90, 90):
        grid[(lat, 180)] = grid[(lat, 179)]
    return grid

def sst_anomaly_grid():
    from netCDF4 import Dataset
    path = fullPath('data/v2/Data AIAM/mean_non_extended.nc')
    data = Dataset(path, 'r').variables

    sst_masked = data['sst_anomaly'][:]
    sst = sst_masked.data
    sst[sst_masked.mask] = np.nan
    nc_lat = np.floor(data['lat'][:].data).astype(int)
    nc_lon = np.floor(data['lon'][:].data).astype(int)

    sst_df = pd.DataFrame(sst, index=nc_lat, columns=nc_lon)
    sst_df = sst_df.melt(ignore_index=False, var_name='lon', value_name='sst').reset_index().rename(columns={'index': 'lat'})
    grid = sst_df.set_index(['lat', 'lon']).to_dict()['sst']
    # map longitude 180 to 179 (in the grid, lon=179 comprises long in [179, 180])
    for lat in range(-90, 90):
        grid[(lat, 180)] = grid[(lat, 179)]
    return grid

def find_values_in_grid(grid, v2=True):
    trajectories = load_all_data(v2=v2, return_labels=False)
    trajectories = trajectories.reset_index().set_index('ID').drop(columns='COMMON_NAME').iloc[:,0]
    def find_value(x):
        x_rounded = np.floor(x[:2]).astype(int)
        v = np.empty(x.shape[1])
        for i, x_i in enumerate(x_rounded.T):
            v[i] = grid[tuple(x_i)]
        return v

    results = trajectories.apply(find_value)
    return results

@savedata
def biodiversity(v2=True):
    grid = biodiversity_grid()
    return find_values_in_grid(grid, v2=v2)

@savedata
def sst_anomaly(v2=True):
    grid = sst_anomaly_grid()
    return find_values_in_grid(grid, v2=v2)

@savedata
def trajectory_complementary_data(v2=True, weather=None):
    """
    Data about trajectories to explore for error analysis.
    """
    from forecasting.preprocessing import space

    def ID_as_index(df):
        return df.reset_index().set_index('ID').drop(columns='COMMON_NAME').iloc[:,0]

    X, labels, Year = load_all_data(v2=v2, weather=weather)
    X = ID_as_index(X)
    labels = labels.reset_index().set_index('ID')
    Year = pd.Series(Year, index=X.index)

    print("Computing days in trajectory")
    ndays_total = temporal_extension(X.values, Year.values, mode='all')
    ndays_year = temporal_extension(X.values, Year.values, mode='year')
    ndays_total = pd.Series(ndays_total, index=X.index, name='Days in trajectory')
    ndays_year = pd.Series(ndays_year, index=X.index, name='Days in trajectory (year)')
    ndays_data = pd.concat([ndays_total, ndays_year], axis=1)

    print("Computing distance")

    lat_lon = X.apply(lambda x: x[:2, [0, -1]] * np.pi / 180)
    df_dx = pd.Series([space.spherical_velocity(cds)[:,-2:].T for cds in lat_lon], index=X.index)
    SN_delta = df_dx.apply(lambda x: x.squeeze()[0]).abs()
    WE_delta = df_dx.apply(lambda x: x.squeeze()[1]).abs()
    SN_delta = SN_delta.to_frame(name='SN delta')
    WE_delta = WE_delta.to_frame(name='WE delta')

    distance = compute_df_distance_weather_spec(v2=v2, weather=weather)[0]
    distance = ID_as_index(distance)

    def SN_max_extension(x):
        return np.abs(np.cumsum(x[0])).max()
    def WE_max_extension(x):
        return np.abs(np.cumsum(x[1])).max()

    SN_ext = df_dx.apply(SN_max_extension).to_frame(name='SN extension')
    WE_ext = df_dx.apply(WE_max_extension).to_frame(name='WE extension')

    distance_data = pd.concat([SN_delta, WE_delta, SN_ext, WE_ext], axis=1)

    def compute_length(x):
        return x.shape[1]
    def median_latitude(x):
        return np.median(x[0])
    def median_longitude(x):
        """
        Uses median instead of mean to avoid problems at +-180
        """
        return np.median(x[1])

    length = X.apply(compute_length).to_frame(name='# of observations')
    median_lat = X.apply(median_latitude).to_frame(name='median latitude')
    median_lon = X.apply(median_longitude).to_frame(name='median longitude')
    mean_year = Year.apply(np.mean).to_frame(name='mean year')

    space_time_data = pd.concat([length, median_lat, median_lon, mean_year], axis=1)

    print("Computing fishing effort")
    effort = get_fishing_effort(v2=v2)

    def mean_effort(x):
        return np.mean(x)

    def percentile_effort(x, p=70):
        return np.percentile(x, p)

    def trimmed_mean_effort(x, p=50):
        x_prunned = x[x >= np.percentile(x, p)]
        if x_prunned.size == 0:
            return np.NaN
        else:
            return x_prunned.mean()

    def mean_effort_excluding_nans(x):
        y = x[x > 0]
        if y.size == 0:
            return np.NaN
        else:
            return y.mean()

    em = effort.apply(mean_effort).to_frame(name='mean effort')
    em_excluding_nans = effort.apply(mean_effort_excluding_nans).to_frame(name='mean effort excluding NaNs')
    etm = effort.apply(trimmed_mean_effort).to_frame(name='trimmed mean effort')
    ep25 = effort.apply(percentile_effort, p=25).to_frame(name='25th percentile effort')
    ep50 = effort.apply(percentile_effort, p=50).to_frame(name='50th percentile effort')
    ep75 = effort.apply(percentile_effort, p=75).to_frame(name='75th percentile effort')
    ep_max = effort.apply(np.max).to_frame(name='max effort')
    eff_std = effort.apply(np.std).to_frame(name='effort std')

    effort_data = pd.concat([em, em_excluding_nans, etm, ep25, ep50, ep75, ep_max, eff_std], axis=1)
    effort_data = effort_data.fillna(0)

    print("Computing biodiversity")
    biodiversity_data = biodiversity(v2=v2)
    bm = biodiversity_data.apply(np.mean).to_frame(name='mean biodiversity')
    bstd = biodiversity_data.apply(np.std).to_frame(name='biodiversity std')
    bp25 = biodiversity_data.apply(percentile_effort, p=25).to_frame(name='25th percentile biodiversity')
    bp50 = biodiversity_data.apply(percentile_effort, p=50).to_frame(name='50th percentile biodiversity')
    bp75 = biodiversity_data.apply(percentile_effort, p=75).to_frame(name='75th percentile biodiversity')
    bp_max = biodiversity_data.apply(np.max).to_frame(name='max biodiversity')

    biodiversity_data = pd.concat([bm, bstd, bp25, bp50, bp75, bp_max], axis=1)

    print("Computing SST anomaly")
    sst_anomaly_data = sst_anomaly(v2=v2)
    sst_anomaly_not_nans = sst_anomaly_data.apply(lambda x: x[~np.isnan(x)])
    sst_anomaly_mean = sst_anomaly_not_nans.apply(np.mean).to_frame(name='mean SST anomaly')
    sst_anomaly_std = sst_anomaly_not_nans.apply(np.std).to_frame(name='SST anomaly std')
    sst_anomaly_p25 = sst_anomaly_not_nans.apply(percentile_effort, p=25).to_frame(name='25th percentile SST anomaly')
    sst_anomaly_p50 = sst_anomaly_not_nans.apply(percentile_effort, p=50).to_frame(name='50th percentile SST anomaly')
    sst_anomaly_p75 = sst_anomaly_not_nans.apply(percentile_effort, p=75).to_frame(name='75th percentile SST anomaly')
    sst_anomaly_p_max = sst_anomaly_not_nans.apply(np.max).to_frame(name='max SST anomaly')

    sst_anomaly_data = pd.concat([sst_anomaly_mean, sst_anomaly_std, sst_anomaly_p25, sst_anomaly_p50, sst_anomaly_p75, sst_anomaly_p_max], axis=1)

    print("Computing metadata")
    species_to_animal_count = labels["COMMON_NAME"].value_counts(sort=False)
    N_species = species_to_animal_count.loc[labels["COMMON_NAME"].values]
    N_species.index = labels.index
    N_species = N_species.to_frame(name='N species')
    taxa = labels['Taxa'].str.get_dummies()
    tag = labels['Tag'].str.get_dummies()
    sex = labels['Sex'].str.get_dummies()
    metadata = pd.concat([N_species, taxa, tag, sex], axis=1)

    print("Computing sampling frequency")
    dt = get_dt(v2=v2, weather=weather)
    f = 1 / dt
    f_mean = f.apply(np.mean).to_frame(name='sampling frequency (mean)')
    f_std = f.apply(np.std).to_frame(name='sampling frequency (std)')
    f_data = pd.concat([f_mean, f_std], axis=1)

    print("Computing overlap")
    overlap_data = analysis.occurrences_count_vs_acc(v2=v2)
    overlap_data = overlap_data[[col for col in overlap_data.columns if "Accuracy" not in col]]

    dataset = pd.concat([distance_data, ndays_data, space_time_data, f_data, effort_data, biodiversity_data, sst_anomaly_data, overlap_data, metadata], axis=1)
    return dataset

@savedata
def distance_to_coast(v2=False):
    coast = storage.load_lzma(fullPath('utils/data/coastline/coast_vptree.lzma'))
    if v2:
        data = weather_data_base_v2()
    else:
        data = storage.load_lzma(fullPath("utils/data/mobility_data.lzma"))
    data = data[['LONGITUDE', 'LATITUDE']]
    data.columns = ['lon', 'lat']
    data['lon'] = data['lon'].astype(np.float64).values
    data['lat'] = data['lat'].astype(np.float64).values
    lon_lat = data[["lon", 'lat']].values.astype(np.float64)
    lon_lat *= np.pi / 180

    coast_data = defaultdict(lambda : np.empty((lon_lat.shape[0])))
    for i, p in enumerate(tqdm(lon_lat)):
        d, (lon, lat) = coast.get_nearest_neighbor(p)
        coast_data["d"][i] = d
        coast_data["lon"][i] = lon
        coast_data["lat"][i] = lat
    coast_df = pd.DataFrame(coast_data, index=data.index)
    coast_df.columns = ['coast-d', 'coast-lon', 'coast-lat']
    DF = pd.concat([data, coast_df], axis=1, ignore_index=True)
    DF.columns = ["LONGITUDE", "LATITUDE", 'coast-d', 'coast-lon', 'coast-lat']
    return DF

@savedata
def bathymetry(v2=False):
    if v2:
        data = weather_data_base_v2()
    else:
        data = storage.load_lzma(fullPath("utils/data/mobility_data.lzma"))
    data = data[['LONGITUDE', 'LATITUDE']]

    bathymetry_data = np.genfromtxt(fullPath('utils/data/BathymetryData.dat'),
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     delimiter=' ')

    ground = bathymetry_data > 0
    bathymetry_data[ground] = 0

    lon_edges = np.arange(-180, 180.25, 0.25)
    lon_centers = 0.5 * (lon_edges[1:] + lon_edges[:-1])
    lat_edges = np.arange(90, -90.25, -0.25)
    lat_centers = 0.5 * (lat_edges[1:] + lat_edges[:-1])

    @njit
    def find_closest(lat, lon):
        i = np.abs(lat - lat_centers).argmin()
        j = np.abs(lon - lon_centers).argmin()
        return bathymetry_data[i,j]

    print("Adding bathymetry...")
    X = data[["LONGITUDE", "LATITUDE"]].values
    bathymetry_X = np.empty((X.shape[0]))
    for i, (lon, lat) in enumerate(tqdm(X)):
        bathymetry_X[i] = find_closest(lat, lon)
    data['bathymetry'] = bathymetry_X
    return data

@savedata
def get_dt(v2=True, weather=None):
    df, _, Year = load_all_data(v2=v2, weather=weather)
    T = df.apply(lambda x: x[2]).to_frame(name='time')
    T['Year'] = Year
    dt = T.apply(lambda x: compute_dt(x['time'], x['Year']), axis=1)
    dt.index = dt.index.droplevel(0)
    return dt

@savedata
def merge_env_data_v2():
    print("Loading env data")
    distance_to_coast_data = distance_to_coast(v2=True)
    bathymetry_data = bathymetry(v2=True)
    df_env = nc_preprocess.env_data_full()
    df_env_julia = pd.read_csv(fullPath("utils/data/weather/v2/full_df.csv"))
    df_env_julia = df_env_julia.replace({9999.0: np.nan})
    df_env_julia["Mean wave direction_x"] = np.sin(df_env_julia["Mean wave direction"].values * np.pi / 180) # the angle is w.r.t. the north pole. 90 is East, 0 is North
    df_env_julia["Mean wave direction_y"] = np.cos(df_env_julia["Mean wave direction"].values * np.pi / 180)
    df_env_julia = df_env_julia.drop(columns=["Mean wave direction"])

    print("Asserting dates coincide")
    dates_julia = df_env_julia['DATE_TIME']
    dates_julia = pd.to_datetime(dates_julia, format="%Y-%m-%d %H:%M:%S")
    dates = df_env['DATE_TIME'].reset_index(drop=True)
    dates_julia_rounded = dates_julia.round('S')
    dates_rounded = dates.round('S')
    assert (dates_rounded.sort_values().reset_index(drop=True) == dates_julia_rounded.sort_values().reset_index(drop=True)).all()

    print("Map (date, lat, lon) to ID")
    df_with_ID = pd.read_csv(fullPath("data/dataset.csv"))
    cds_to_ID = defaultdict(list)
    for ID, date, lat, lon in tqdm(df_with_ID.values):
        cds_to_ID[(date, lat, lon)].append(ID)

    print("Map (date, lat, lon) to ID for python env data")
    df_env_index = df_env.set_index(['DATE_TIME', 'LATITUDE', 'LONGITUDE']).index
    env_IDs = []
    appearances_per_ID = defaultdict(int)
    for (date, lat, lon) in tqdm(df_env_index):
        env_IDs.append(cds_to_ID[(date, lat, lon)][appearances_per_ID[(date, lat, lon)]])
        appearances_per_ID[(date, lat, lon)] += 1
    df_env['ID'] = env_IDs

    print("Map (date, lat, lon) to ID for rounded dates")
    dates_rounded.index = dates.values
    dates_to_dates_rounded = dates_rounded.to_dict()
    cds_to_ID_rounded = {}
    for (date, lat, lon), IDs in tqdm(cds_to_ID.items()):
        cds_to_ID_rounded[(dates_to_dates_rounded[date], lat, lon)] = IDs

    print("Map (date, lat, lon) to ID for julia env data")
    df_env_julia['DATE_TIME_rounded'] = dates_julia_rounded.values
    df_env_julia_index = df_env_julia.set_index(['DATE_TIME_rounded', 'LATITUDE', 'LONGITUDE']).index

    def find_closest_lat_lon(date, lat, lon):
        out = []
        for (_date, lat_, lon_) in tqdm(cds_to_ID_rounded.keys()):
            if date == _date:
                if math.isclose(lat, lat_) and math.isclose(lon, lon_):
                    out.append((_date, lat_, lon_))
        return out
    env_IDs_julia = []
    appearances_per_ID_rounded = defaultdict(int)
    for (date, lat, lon) in tqdm(df_env_julia_index):
        try:
            env_IDs_julia.append(cds_to_ID_rounded[(date, lat, lon)][appearances_per_ID_rounded[(date, lat, lon)]])
        except KeyError:
            closest = find_closest_lat_lon(date, lat, lon)
            if len(closest) > 1:
                raise ValueError("More than one closest")
            else:
                date, lat, lon = closest[0]
                env_IDs_julia.append(cds_to_ID_rounded[(date, lat, lon)][appearances_per_ID_rounded[(date, lat, lon)]])
        appearances_per_ID_rounded[(date, lat, lon)] += 1

    df_env_julia['ID'] = env_IDs_julia

    print("Asserting IDs coincide")
    df_env = df_env.reset_index().sort_values(['ID', 'DATE_TIME'])
    df_env_julia = df_env_julia.reset_index().drop(columns=['DATE_TIME_rounded', 'LATITUDE', 'LONGITUDE']).sort_values(['ID', 'DATE_TIME'])
    assert (df_env['ID'].values == df_env_julia['ID'].values).all()

    print("Merging python and julia env data")
    df_env_julia = df_env_julia.drop(columns=['DATE_TIME', 'ID']).reset_index(drop=True)
    df_env = df_env.reset_index(drop=True)
    df_env_total = pd.concat([df_env, df_env_julia], axis=1)

    print("Asserting bathymetry and distance to coast are aligned")
    assert (distance_to_coast_data[['LONGITUDE', 'LATITUDE']].values == bathymetry_data[['LONGITUDE', 'LATITUDE']].values).all()

    df_coast_batyhmetry = pd.concat([bathymetry_data, distance_to_coast_data['coast-d']], axis=1)

    print("Asserting bathymetry+distance-to-coast and the rest of env data have same coordinates")
    idx_origin = df_coast_batyhmetry[['LONGITUDE', 'LATITUDE']].values
    idx_destination = df_env_total[['LONGITUDE', 'LATITUDE']].values
    assert (np.sort(idx_origin, axis=0) == np.sort(idx_destination, axis=0)).all()

    print("Map (lon, lat) -> (coast-d, bathymetry)")
    df_coast_batyhmetry = df_coast_batyhmetry.drop_duplicates()
    lon_lat_to_coast_bathymetry = {}
    for lon, lat, b, cd in tqdm(df_coast_batyhmetry.values):
        lon_lat_to_coast_bathymetry[(lon, lat)] = np.array([cd, b])

    print("Add (coast-d, bathymetry) to env data")
    coast_d_bathymetry_mapped = []
    for lon, lat in tqdm(idx_destination):
        coast_d_bathymetry_mapped.append(lon_lat_to_coast_bathymetry[(lon, lat)])
    coast_d_bathymetry_mapped = np.array(coast_d_bathymetry_mapped)
    df_env_total['coast-d'] = coast_d_bathymetry_mapped[:, 0]
    df_env_total['bathymetry'] = coast_d_bathymetry_mapped[:, 1]

    print("Setting order of columns")
    first_cols = ["ID", "LATITUDE", "LONGITUDE", "DATE_TIME"]
    cols = first_cols + [col for col in df_env_total.columns if col not in first_cols]
    df_env_total = df_env_total[cols]

    print("Storing data")
    return df_env_total

def impute_nans_with_closest_geographically(df):
    """
    Impute nan values with the closest geographically.
    df: Dataframe with LATITUDE and LONGITUDE columns.
    """
    df['Lat_binned'] = pd.cut(df.LATITUDE, bins=np.arange(-90.25, 90.25, 0.25), right=True).apply(lambda x: x.mid).astype(float)
    df['Lon_binned'] = pd.cut(df.LONGITUDE, bins=np.arange(-180.25, 180.25, 0.25), right=True).apply(lambda x: x.mid).astype(float)
    nan_cols = df.columns[df.isna().any()]
    print(f"NaN values found in columns: {nan_cols}")
    print("Computing mean values for each (lat, lon) cell")
    binned_vals = df.groupby(['Lat_binned', 'Lon_binned']).apply(lambda S: S[nan_cols].mean())
    binned_vals = binned_vals.reset_index()

    print("Imputing NaN values with the closest geographically")

    @njit
    def impute_nans_with_closest_location(lat, lon, x):

        is_nan = np.isnan(x)
        lat_0 = lat[is_nan]
        lon_0 = lon[is_nan]
        x_valid = x[~is_nan]
        lat_valid = lat[~is_nan]
        lon_valid = lon[~is_nan]

        y = np.empty((is_nan.sum()))
        for i, (lat_0_i, lon_0_i) in enumerate(zip(lat_0, lon_0)):
            lat_diff = np.abs(lat_0_i - lat_valid)
            lon_diff = np.abs(lon_0_i - lon_valid)
            total_diff = lat_diff + lon_diff

            y[i] = x_valid[np.argmin(total_diff)]
        z = x.copy()
        z[is_nan] = y
        return z

    for col in tqdm(binned_vals.columns):
        if col not in ['Lat_binned', 'Lon_binned']:
            binned_vals[col] = impute_nans_with_closest_location(binned_vals.Lat_binned.values, binned_vals.Lon_binned.values, binned_vals[col].values)

    print("Merging imputed values with original dataframe")
    binned_vals = binned_vals.set_index(['Lat_binned', 'Lon_binned'])
    df = df.set_index(['Lat_binned', 'Lon_binned'])
    for col in tqdm(nan_cols):
        df[col] = df[col].fillna(binned_vals[col])
    return df.reset_index(drop=True)

@savedata
def env_data_v2_imputed():
    """
    Impute missing values in env data.
    NOTE: For some reason the dataframe has columns 'index' that are useless. Dropping them is advised.
    """
    df_env = merge_env_data_v2()
    df_env = impute_nans_with_closest_geographically(df_env)
    return df_env

@savedata
def breeding_stage_v2(pad_day_rate=None):
    df =  pd.read_csv(fullPath("data/dataset.csv")).set_index("ID")
    stage = df['Stage'].to_frame()

    if pad_day_rate is not None:
        df['date'] = pd.to_datetime(df['DATE_TIME'])
        df = df.drop(columns=['dt_f0'])
        df['day'] = (df.date.dt.dayofyear + df.date.dt.hour / 24 + df.date.dt.minute / (24 * 60) + df.date.dt.second / (24 * 60 * 60)) - 1
        df['year'] = df.date.dt.year
        df['Stage'] = stage['Stage'].values

        valid = df['Stage'].notna()
        df_pruned = df[valid]
        df_pruned['day_interval'] = pd.cut(df_pruned['day'], bins=np.arange(0, 367 + 1/pad_day_rate, 1/pad_day_rate))

        def most_frequent_stage(S):
            s = S.Stage.value_counts(ascending=False)
            if len(s) > 0:
                return s.index[0]
            else:
                raise ValueError("No stage found")
        stage = {}
        for ID_i in tqdm(df_pruned['ID'].unique()):
            df_i = df_pruned.query("ID == @ID_i")
            for (y, day_interval), S in df_i.groupby(['year', 'day_interval']):
                if S.size > 0:
                    stage[(ID_i, y, day_interval)] = most_frequent_stage(S)

        stage = pd.Series(stage).reset_index()
        stage.columns = ['ID', 'year', 'day_interval', 'Stage']
        stage['COMMON_NAME'] = map_ID_to_species(stage.ID.values)
    else:
        stage['COMMON_NAME'] = map_ID_to_species(stage.ID.values)
    return stage

def breeding_stage_v2_counts(pad_day_rate=None):
    """
    Count the number of times each stage appears for each species
    """
    stage = breeding_stage_v2(pad_day_rate=pad_day_rate)
    stage_count = defaultdict(int)
    for (s, ID), df in stage.groupby(['COMMON_NAME', 'ID']):
        stage_types = df.Stage.dropna().drop_duplicates().values
        for t in stage_types:
            stage_count[(s, ID, t)] += 1
    stage_count = pd.Series(stage_count)
    stage_by_species = stage_count.groupby(level=[0, 2]).sum()
    stage_by_species_total = stage_by_species.groupby(level=0).sum().sort_values(ascending=False)
    return stage_count, stage_by_species, stage_by_species_total

@savedata
def get_species_to_taxa(v2=True):
    if v2:
        meta = load_data_v2()[-1]
    else:
        meta = storage.load_lzma(fullPath('utils/data/labels_weather_split-by-day_groupby-ID_default.lzma'))
    species_to_taxa = meta.set_index("COMMON_NAME").Taxa.to_dict()
    return species_to_taxa

def get_taxa_to_species(v2=True):
    species_to_taxa = get_species_to_taxa(v2)
    taxa_to_species = defaultdict(list)
    for k, v in species_to_taxa.items():
        taxa_to_species[v].append(k)
    return taxa_to_species

def map_ID_to_species(ID, v2=True):
    labels = load_all_data(v2=v2)[1]
    index = labels.index.swaplevel(0, 1)
    species = index.to_frame().set_index('ID').loc[ID].values
    return species

@savedata
def equally_spaced_trajectories_avg_v2(weather=None, pad_day_rate=3):
    """
    Computes equally spaced trajectories by averaging the values of the original trajectories within the same time bin.
    Does not interpolate.
    """
    stage = breeding_stage_v2()
    df, _, year = load_all_data(weather=weather, return_labels=True, v2=True, pad_day_rate=None)
    year = pd.Series(year, index=df.index)
    if weather is not None:
        order = load_all_data(weather=None, return_labels=False, v2=True, pad_day_rate=None).index
        df = df.loc[order]
        year = year.loc[order]

    df = df.reset_index()
    df.columns = ['COMMON_NAME', 'ID', 'data']
    df['year'] = year.values

    # Create a new DataFrame with expanded arrays
    expanded_data = []
    for _, row in tqdm(df.iterrows()):
        cm = row['COMMON_NAME']
        ID = row['ID']
        array_values = row['data']
        year_values = row['year']
        for i in range(len(array_values[0])):
            expanded_data.append([cm, ID, year_values[i]] + [x[i] for x in array_values])
    columns = ['COMMON_NAME', 'ID', 'year'] + [f'x{i}' for i in range(array_values.shape[0])]
    df = pd.DataFrame(expanded_data, columns=columns)

    df['Stage'] = stage.Stage.values # data is aligned
    df['day_interval'] = pd.cut(df['x2'], bins=np.arange(0, 367 + 1/pad_day_rate, 1/pad_day_rate))

    def most_frequent_stage(S):
        s = S.Stage.value_counts(ascending=False)
        if len(s) > 0:
            return s.index[0]
        else:
            return np.NaN
    stage = {}
    xcolumns = [f'x{i}' for i in range(array_values.shape[0]) if i != 2]
    for ID_i in tqdm(df['ID'].unique()):
        df_i = df.query("ID == @ID_i")
        for (y, day_interval), S in df_i.groupby(['year', 'day_interval']):
            if S.size > 0:
                stage[(ID_i, y, day_interval, 'stage')] = most_frequent_stage(S)
                for col in xcolumns:
                    if col == 'x1':
                        longitude = S[col]
                        if longitude.max() > 0 and longitude.min() < 0:
                            stage[(ID_i, y, day_interval, col)] = S[col].median()
                        else:
                            stage[(ID_i, y, day_interval, col)] = S[col].mean()
                    else:
                        stage[(ID_i, y, day_interval, col)] = S[col].mean()

    stage = pd.Series(stage)
    stage = stage.unstack().reset_index()
    stage.columns = ['ID', 'year', 'day_interval'] + ['stage'] + xcolumns
    stage['x2'] = stage['day_interval'].apply(lambda x: x.mid)
    stage['COMMON_NAME'] = map_ID_to_species(stage['ID'].values)
    # sort the columns
    xcolumns_full = [f'x{i}' for i in range(array_values.shape[0])]
    stage = stage[['COMMON_NAME', 'ID', 'year', 'day_interval'] + ['stage'] + xcolumns_full]
    stage[xcolumns_full] = stage[xcolumns_full].astype(np.float64)
    return stage

@savedata
def occurrences_count(v2=True, reduce='mean', lat_width=1, lon_width=1):
    df = load_all_data(v2=v2, return_labels=False, expand_df=True)
    lat_bins = np.arange(-90 - lat_width, 90 + lat_width, lat_width)
    lon_bins = np.arange(-180 - lon_width, 180 + lon_width, lon_width)
    df['lat_bin'] = pd.cut(df.lat, lat_bins, right=True).apply(lambda x: x.mid)
    df['lon_bin'] = pd.cut(df.lon, lon_bins, right=True).apply(lambda x: x.mid)
    df_prunned = df[['lat_bin', 'lon_bin']].reset_index() # index = ID (trajectory)
    df_prunned = df_prunned.drop_duplicates().reset_index(drop=True) # only keep one observation per location and trajectory
    df_prunned['species'] = map_ID_to_species(df_prunned.ID.values)
    species_loc_counts = df_prunned[['lat_bin', 'lon_bin', 'species']].value_counts()
    species_loc_counts_dict = species_loc_counts.to_dict()
    species_to_taxa = get_species_to_taxa()

    def counter(trajectory):
        """
        Count the number of ocurrences of the same species, other species and other species within the same taxa.
        reduce: function to apply to the counts
        """
        ID = trajectory.ID.iloc[0]
        trajectory = trajectory.drop(columns='ID')
        trajectory['counts-same-species'] = trajectory.apply(lambda row: species_loc_counts_dict[tuple(row.values)], axis=1) - 1

        # count only species different than the one in the trajectory
        species = trajectory.species.iloc[0]
        taxa = species_to_taxa[species]
        counts_other_species = species_loc_counts[species_loc_counts.index.get_level_values('species') != species]
        counts_other_species_sum = counts_other_species.groupby(['lat_bin', 'lon_bin']).sum().to_dict()
        counts_other_species_sum = defaultdict(int, counts_other_species_sum) # default value is 0
        trajectory['counts-other-species'] = trajectory.apply(lambda row: counts_other_species_sum[(row.lat_bin, row.lon_bin)], axis=1)

        # count other species within the same taxa
        counts_other_species = counts_other_species.to_frame()
        counts_other_species['taxa'] = counts_other_species.index.get_level_values('species').map(species_to_taxa)
        counts_same_taxa_sum = counts_other_species.query("taxa == @taxa")[0].groupby(['lat_bin', 'lon_bin']).sum().to_dict()
        counts_same_taxa_sum = defaultdict(int, counts_same_taxa_sum) # default value is 0
        trajectory['counts-other-species-same-taxa'] = trajectory.apply(lambda row: counts_same_taxa_sum[(row.lat_bin, row.lon_bin)], axis=1)
        if reduce is not None:
            trajectory = trajectory[['counts-same-species', 'counts-other-species', 'counts-other-species-same-taxa']]
            if reduce in ['mean', 'median']:
                trajectory = getattr(trajectory, reduce)()
            else:
                trajectory = trajectory.apply(reduce, axis=0)
        else:
            trajectory = trajectory.assign(ID=ID)
        return trajectory

    results = {}
    for ID, S in tqdm(df_prunned.groupby('ID')):
        results[ID] = counter(S)
    return pd.Series(results)
