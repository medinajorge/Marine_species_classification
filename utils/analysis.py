"""
Main computations for species and breeding stage classification
"""
import numpy as np
import math
import pandas as pd
try:
    import shap
except:
    pass
try:
    import xgboost as xgb
except:
    pass
import os
from copy import deepcopy
from collections import defaultdict
from collections.abc import Iterable
import shutil
import warnings
from . import file_management, other_utils, preprocessing, models, data_visualization, params, nn_utils, nb_funcs, shap_plots_mod
from numba import njit
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import mode
import scipy.stats as ss
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
try:
    import dtw
except:
    pass
import sys
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
    import tensorflow as tf
    #import tensorflow.keras.backend as K
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

import plotly.graph_objects as go
import plotly.express as px
from phdu import savedata, savefig, SavedataSkippedComputation
from phdu.plots.plotly_utils import get_figure, CI_plot
from phdu.plots.base import plotly_default_colors, color_gradient, color_std
from phdu import clustering, _helper, bootstrap, pd_utils

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)


def nan_by_year(df_full, usecols=None):
    """Returns dictionary with key = df columns and values = 2D array. row 0 = years, row 1 = nan %."""
    if usecols is None:
        df = df_full.loc[:, "DATE_TIME":"Total precipitation"]
    else:
        df = df_full[usecols]
    if "Year" not in df.columns:
        df["Year"] = df["DATE_TIME"].apply(lambda x: x.year)

    num_years = len(set(df["Year"]))
    nan_values = defaultdict(lambda: np.zeros((2, num_years)))

    for i, (year, df_year) in enumerate(df.groupby("Year")):
        num_data = df_year.shape[0]
        for col in df_year.columns:
            df_col = df_year[col]
            num_nan = np.sum(df_col.isna().values)
            nan_values[col][0, i] = year
            nan_values[col][1, i] = num_nan / num_data
    return nan_values

def var_to_grid(full_df, magnitude="Accuracy", lat_boxes=250, lon_boxes=500):
    """Returns mean value of the magnitude in each cell of the grid."""
    df = full_df.copy()
    df.columns = [col.capitalize() for col in df.columns]

    lat = df["Latitude"].astype(np.float64, copy=False)
    lon = df["Longitude"].astype(np.float64, copy=False)
    splitter = lambda col, num_boxes: pd.cut(df[col].values, np.linspace(df[col].min(), df[col].max(), num_boxes), right=True)
    df['Lat-lon interval'] = [*zip(splitter("Latitude", lat_boxes), splitter("Longitude", lon_boxes))]
    density_data = defaultdict(lambda: np.empty((len(set(df['Lat-lon interval'])))))

    if magnitude.capitalize() == "Density":
        data_iterable = df['Lat-lon interval'].value_counts(sort=False, normalize=True).to_dict().items()
        get_val = lambda x: x
    else:
        data_iterable = df.groupby("Lat-lon interval")
        get_val = lambda df: df[magnitude].mean()

    for i, ((lats, lons), data) in enumerate(data_iterable):
        try:
            lat = lats.mid
            lon = lons.mid
            m = get_val(data)
        except:
            lat, lon, m = 0, 0, np.NaN
        density_data['Latitude'][i] = lat
        density_data['Longitude'][i] = lon
        density_data[magnitude][i] = m
    density_df = pd.DataFrame(density_data)
    density_df.dropna(inplace=True)

    return density_df

def compute_confusion_matrix(df=None, cols=["Taxa", "COMMON_NAME"], artificial_trajectory=["random-walk"], classifier="InceptionTime", velocity="arch-segment", to_origin="", weather="", size=None):
    allcols = cols + ["Predicted"]
    if df is None:
        size_str = "" if size is None else f'_size-{other_utils.encoder(size)}'
        if artificial_trajectory is not None:
            artificial_trajectory = artificial_trajectory if isinstance(artificial_trajectory, (list, tuple, np.ndarray)) else [artificial_trajectory]
            artificial_trajectory_formatted = [t.capitalize().replace("-", " ") for t in artificial_trajectory]
            artificial_trajectory_str = "_" + "--".join(artificial_trajectory)
        else:
            artificial_trajectory_str = ""
        df = pd.read_csv(f"utils/data/classification_report/report_{classifier}_5-kfold_species_clf_velocity-{velocity}_to-origin-{to_origin}_weather-{weather}{artificial_trajectory_str}{size_str}.csv")
    df = df[allcols].sort_values(by=allcols)

    name_to_cat = {}
    i = 0

    if len(cols) > 1:
        categories = defaultdict(list)
        for c, df_c in df.groupby(cols[0]):
            if c not in artificial_trajectory:
                for c_int in set(df_c[cols[-1]]):
                    name_to_cat[c_int] = i
                    i += 1
                    categories[c].append(c_int)
    else:
        categories = sorted(set(df[cols[-1]]))
        name_to_cat = {v: k for k, v in enumerate(categories)}

    if len(artificial_trajectory) > 0:
        for j, trajectory_type in enumerate(artificial_trajectory, start=i):
            name_to_cat[trajectory_type] = j
        for trajectory_formatted in artificial_trajectory_formatted:
            categories[trajectory_formatted].append(trajectory_formatted)
    cat_to_name = {v:k for k,v in name_to_cat.items()}
    if len(artificial_trajectory) > 0:
        for trajectory_type, trajectory_formatted in zip(artificial_trajectory, artificial_trajectory_formatted):
            cat_to_name[name_to_cat[trajectory_type]] = trajectory_formatted
    y = df[cols[-1]].map(name_to_cat)
    y_pred = df["Predicted"].map(name_to_cat)
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y, y_pred, normalize="true")).rename(columns=cat_to_name, index=cat_to_name)

    return confusion_matrix_df, categories

@savefig
def confusion_matrix_plot(hierarchical=False, diagonal_out=False, **kwargs):
    df = classification_report_random_states(**kwargs)
    cm, _ = compute_confusion_matrix(df=df, artificial_trajectory=[])
    species = cm.columns
    if hierarchical:
        species_to_taxa = preprocessing.get_species_to_taxa()
        taxas = [species_to_taxa[s] for s in species]
        new_index = pd.MultiIndex.from_arrays([taxas, species], names=["Taxa", "Species"])
        new_index = new_index.get_level_values(0) + "--" + new_index.get_level_values(1)
        cm_multi = cm.copy()
        cm_multi.index = new_index
        cm_multi.columns = new_index
        fig = clustering.hierarchical_cluster_matrix(1 - cm_multi, '1 - P', cmin=0, ticksize=8., colorbar_x=1, cmap='plasma_r')
        colorscale = data_visualization.transparent_colorscale(fig, upper=True)
        fig.update_layout(coloraxis=dict(colorscale=colorscale, colorbar=dict(title=dict(text='1 - P', font_size=20), tickfont_size=18, x=1, len=0.7, y=0.55)),
                          height=1500, width=1600)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

        # def ex(height, width):
        #     fig.update_layout(height=height, width=width)
        #     fig.write_image('ex.png')
        # import pdb; pdb.set_trace()

    else:
        # Sort species by taxa
        order = ['Polar bears', 'Penguins', 'Birds', 'Seals', 'Sirenians', 'Cetaceans', 'Fishes', 'Turtles']
        fig = data_visualization.confusion_matrix(df=df, artificial_trajectory=[], order=order, diagonal_out=diagonal_out, return_fig='matrix')
    return fig

@savefig
def confusion_matrix_plot_diagonal(colorscale_diag='plasma', **kwargs):
    df = classification_report_random_states(**kwargs)
    order = ['Polar bears', 'Penguins', 'Birds', 'Seals', 'Sirenians', 'Cetaceans', 'Fishes', 'Turtles']
    return data_visualization.confusion_matrix(df=df, artificial_trajectory=[], order=order, diagonal_out=True, return_fig='diagonal', colorscale_diag=colorscale_diag)

def confusion_matrix_hierarchy(weather='all', common_origin_distance=False, random_states=range(1,6), method='average', distance='min'):
    df = classification_report_random_states(weather=weather, common_origin_distance=common_origin_distance, random_states=random_states)
    cm, _ = compute_confusion_matrix(df=df, artificial_trajectory=[])
    species = cm.columns.to_list()
    if distance == 'mean':
        cm_symmetric = (cm + cm.T) / 2
        np.fill_diagonal(cm_symmetric.values, 1)
        d = squareform(1 - cm_symmetric)
    elif distance == 'harmonic':
        cm_harmonic_mean = 2 * cm * cm.T / (cm + cm.T)
        np.fill_diagonal(cm_harmonic_mean.values, 1)
        cm_harmonic_mean.fillna(0, inplace=True)
        d = squareform(1 - cm_harmonic_mean)
    elif distance == 'min': # min distance between A and B chooses the highest misclassification probability between A and B.
        cm_max = np.maximum(cm, cm.T)
        np.fill_diagonal(cm_max.values, 1)
        cm_max.fillna(0, inplace=True)
        d = squareform(1 - cm_max)
    elif distance == 'jaccard':
        cm_jaccard = cm / (cm + cm.T - cm * cm.T)
        np.fill_diagonal(cm_jaccard.values, 1)
        cm_jaccard.fillna(0, inplace=True)
        d = squareform(1 - cm_jaccard)
    else:
        raise ValueError(f"distance={distance} not recognized.")
    corr_linkage = hierarchy.linkage(d, method=method)
    dendro = hierarchy.dendrogram(corr_linkage, labels=species, leaf_font_size=16, leaf_rotation=90, no_plot=True)
    order = dendro['leaves']
    cm = cm.iloc[order, order]
    species_sorted = cm.columns.to_list()

    return cm, corr_linkage, species_sorted, species

@savefig
def confusion_matrix_hierarchy_plot(**kwargs):
    from scipy.cluster import hierarchy
    _, corr_linkage, _2, species = confusion_matrix_hierarchy(**kwargs)
    fig = plt.figure(figsize=(25, 10))
    _ = hierarchy.dendrogram(corr_linkage, labels=species, leaf_font_size=16, leaf_rotation=90)
    return fig

def confusion_matrix_hierarchy_clusters(height=1, **kwargs):
    cm, corr_linkage, species_sorted, species = confusion_matrix_hierarchy(**kwargs)
    clusters = cluster_prunning(corr_linkage, species, height).sort_values()
    valid_clusters = clusters.value_counts() > 1
    valid_clusters = valid_clusters[valid_clusters].index
    clusters = clusters[clusters.isin(valid_clusters)]
    clusters = clusters.groupby(clusters).apply(lambda x: x.index.to_list())
    species_to_int = {s: i for i, s in enumerate(species_sorted)}
    cluster_idxs = clusters.apply(lambda x: np.array([species_to_int[s] for s in x]))
    cluster_order = cluster_idxs.apply(np.min).sort_values().index
    clusters = clusters.loc[cluster_order]
    clusters.index = range(1, len(clusters)+1)
    cluster_idxs = cluster_idxs.loc[cluster_order]
    cluster_idxs.index = range(1, len(clusters)+1)
    return clusters, cluster_idxs, cm

@savefig('all-cluster_color-cmap')
def confusion_matrix_hierarchy_cluster_plot(height=1, cluster_color='red', cmap='plasma', method='average', diag_to_zero=True, **kwargs):
    ticksize=8
    clusters, cluster_idxs, cm = confusion_matrix_hierarchy_clusters(height=height, method=method, **kwargs)
    if diag_to_zero:
        np.fill_diagonal(cm.values, 0)
    fig = px.imshow(cm, color_continuous_scale=cmap)
    colorscale = data_visualization.transparent_colorscale(fig, upper=False)
    fig.update_layout(margin=dict(l=0, b=30, r=60, t=0, pad=0),
                      height=1500, width=1600, hovermode=False,
                      xaxis_tickfont_size=ticksize, yaxis_tickfont_size=ticksize,
                      coloraxis=dict(cmin=0, cmax=int(not diag_to_zero), colorscale=colorscale,
                                     colorbar=dict(title_text='P', tickfont_size=32, title_font_size=40, x=1, len=0.7, y=0.55)),
                      xaxis_showgrid=False, yaxis_showgrid=False)

    cluster_height_pos = {1: {1: (0, 2),
                              2: (0, -4.5),
                              3: (0, -6.),
                              4: (-1.5, 1.7),
                              5: (0, 2.5),
                              6: (0, -4.5),
                              7: (0.5, -4.25),
                              8: (0, 2),
                              9: (0, 2)
                              }
                          }
    if height in cluster_height_pos:
        cluster_pos = cluster_height_pos[height]
        cluster_pos = defaultdict(lambda: (0, 2), cluster_pos)
    else:
        cluster_pos = {cluster_ID: (0, 2) for cluster_ID in clusters.index}

    for cluster_ID in clusters.index:
        initial = cluster_idxs[cluster_ID].min() - 0.5
        end = cluster_idxs[cluster_ID].max() + 0.5
        dx, dy = cluster_pos[cluster_ID]
        fig.add_shape(type="rect", xref="x", yref="y", x0=initial, y0=initial, x1=end, y1=end,
                      line=dict(color=cluster_color,width=6, dash="solid"))
        fig.add_annotation(x=(initial+end)/2 + dx, y=end+dy, text=f"<span style='font-weight: 900;'>C{cluster_ID}</span>", showarrow=False, font=dict(size=40, color=cluster_color, family="sans-serif"))
    fig.update_layout(xaxis=dict(showticklabels=False
        # title=dict(text='Predicted species',font_family='sans-serif', font_size=44),  showticklabels=False), # xaxis title wrongly plotted at the top for some reason.
                                 ),
                      yaxis=dict(title=dict(text='Real species',font_family='sans-serif', font_size=44, standoff=40), showticklabels=False),
                      # paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        )
    fig.add_annotation(x=0.5, y=0, text='Predicted species', showarrow=False, font=dict(size=44, family="sans-serif"), xref="paper", yref="paper", yshift=-60)
    fig.update_layout(margin_b=60)
    return fig

def get_f1_scores(df, col_true, col_pred="Predicted"):
    count = df[col_true].value_counts()
    f1 = f1_score(df[col_true], df[col_pred], labels=count.index, average=None)
    probs = count / df.shape[0]
    f1_base = 2*probs / (probs+1)
    return pd.DataFrame(dict(N=count.values, f1=f1, f1_base=f1_base), index=count.index)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, norm=False, clip=False):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) #add 2 if the input is 2D

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # equivalent heatmap = last_conv_layer_output.dot(pooled_grads[:, tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    if clip:
        heatmap = tf.clip_by_value(heatmap, 0, np.inf)
    if norm:
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap


##############################################################################################################################
"""                                                        II. DTW                                                         """
##############################################################################################################################

def load_D(len_X=None, min_days=180, mode="D", parentDir="utils/data/dtw/", velocity=None, delete_time_vars=False, add_dt=False, scale=False, pad_day_rate=3, to_origin="space"):
    """
    scaler:     - True: z-normalize each time series for each feature.
                - False: z-normalize features different than (x,y,z,sin(t), cos(t)) across the whole dataset.

    """
    velocity_str = f"_velocity-{other_utils.encoder(velocity)}"
    add_dt_str = "_add-dt" if add_dt else ""
    scaler_str = "_scaled" if scale else ""
    pad_day_rate_str = f"_pad-day-rate-{other_utils.encoder(pad_day_rate)}"
    min_days_str = f"min-days-{other_utils.encoder(min_days)}"
    delete_time_vars_str = "_delete-time" if delete_time_vars else ""
    to_origin_str = f"_to-origin-{other_utils.encoder(to_origin)}"

    if len_X is None:
        len_X = len(models.DecisionTree(prunning_function=preprocessing.get_min_days_prunner(min_days), velocity=velocity, delete_time_vars=delete_time_vars,
                                        add_dt=add_dt, scale=scale, pad_day_rate=pad_day_rate).X)
    D = []
    for i in range(len_X):
        try:
            D.append(np.load(os.path.join(parentDir, f"dtw_{min_days_str}_mode-{mode}_idx-{i}{velocity_str}{add_dt_str}{scaler_str}{pad_day_rate_str}{delete_time_vars_str}{to_origin_str}.npz"))["D"])
        except:
            print(i)
    D = np.vstack(D)
    return D

def reduced_D(func=np.median, **kwargs):
    """
    Return pooled version of the DTW matrix. A_ij = func(D[species i, species j]).
    Defaults:
                velocity=None,
                delete_time_vars=True,
                add_dt=False,
                scale=False,
                pad_day_rate=3,
                min_days=180,
                mode="D"
    """
    d = dict(velocity=None, delete_time_vars=True, add_dt=False, scale=False, pad_day_rate=3, min_days=180, mode="D")
    d.update(kwargs)
    min_days = d["min_days"]
    inception_kwargs = ["velocity", "delete_time_vars", "add_dt", "scale", "pad_day_rate"]
    prunning_function = prunning_function = preprocessing.get_min_len_prunner(2) if min_days is None else preprocessing.get_min_days_prunner(min_days)
    inception = models.InceptionTime(prunning_function=prunning_function, **{k: v for k, v in d.items() if k in inception_kwargs})

    D = load_D(len_X=len(inception.X), **d)
    nan_idxs = np.unique(np.where(np.isnan(D))[0])
    if nan_idxs.size > 0:
        D = np.delete(np.delete(D, nan_idxs, axis=0), nan_idxs, axis=1)
        inception.X = [inception.X[i] for i in set(range(len(inception.X))) - set(nan_idxs)]
        inception.y = np.delete(inception.y, nan_idxs)
        inception.Year = [inception.Year[i] for i in set(range(len(inception.Year))) - set(nan_idxs)]
        inception.labels.drop(inception.labels.index[nan_idxs], inplace=True)
    inception.preprocess()
    inception.get_cat_to_label()

    taxa_string = inception.labels["Taxa"].values
    species = inception.y.copy()
    Taxas = sorted(inception.labels.Taxa.value_counts().index)
    taxa_to_idx = {t: i for i, t in enumerate(Taxas)}
    taxas = np.array([taxa_to_idx[t] for t in taxa_string])
    order = np.lexsort((species, taxas))
    species = species[order]
    taxas = taxas[order]
    unique_taxa, idxs_taxa = np.unique(taxas, return_index=True)
    _, idxs = np.unique(species, return_index=True)
    idxs.sort()

    idx_to_taxa = {v:k for k, v in taxa_to_idx.items()}
    from_idx = lambda l, d: [d[li] for li in l]
    taxas_str = from_idx(taxas, idx_to_taxa)
    species_str = from_idx(species, inception.cat_to_label)
    species_to_taxa = {}
    for i in range(len(species)):
        species_to_taxa[species_str[i]] = taxas_str[i]
    unique_species = [inception.cat_to_label[species[i]] for i in idxs]
    taxa_species_str = [species_to_taxa[i] for i in unique_species]
    multilabel = [taxa_species_str, unique_species]

    D_ordered = D[order][:,order]
    D_reduced = np.empty((idxs.size, idxs.size))
    idxs_upper = np.hstack([idxs[1:], species.size])
    for i, (il, iu) in enumerate(zip(idxs, idxs_upper)):
        for j, (il2, iu2) in enumerate(zip(idxs, idxs_upper)):
            D_reduced[i, j] = func([D_ordered[il:iu, il2:iu2]])
    return D_reduced, multilabel

def dtw_KNN(target, max_k=100, use_tqdm=False, **kwargs):
    y = target.copy()
    dm = load_D(y.size, **kwargs)
    nan_values = np.isnan(dm)
    if nan_values.any():
        warnings.warn("Nan values in distance matrix. Performing k-NN for non-nan values.", RuntimeWarning)
        i, _ = np.where(np.isnan(dm))
        i = np.unique(i)
        dm = np.delete(np.delete(dm, i, axis=0), i, axis=1)
        y = np.delete(y, i)
    accs = np.empty((max_k))
    k_iter = tqdm(range(1, max_k + 1)) if use_tqdm else range(1, max_k + 1)
    for n_neighbors in k_iter:
        knn_labels = y.copy()
        knn_idx = dm.argsort()[:, :n_neighbors]
        for i in range(dm.shape[0]):
            if i in knn_idx[i]:
                knn_idx[i][knn_idx[i] == i] = dm[i].argsort()[n_neighbors+1]

        # Identify k nearest labels
        knn_labels = y[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0].squeeze()
        mode_proba = mode_data[1]/n_neighbors
        accs[n_neighbors-1] = (mode_label == y).mean()
    acc = accs.max()
    best_k = accs.argmax() + 1
    return accs, best_k

def dtw_accs_by_params(y, max_k=50, min_days=180, pad_day_rate=3, modes=["I", "D"], velocities=[None, "arch-segment"], delete_time_vars=[True, False],
                   add_dts=[True, False], scales=[True, False], to_origin="space", savingDir=fullPath("utils/data/dtw/")):
    results = {}
    pbar = tqdm(range(len(modes) * len(velocities) * len(delete_time_vars) * len(add_dts) * len(scales)))
    id_dict = dict(min_days=min_days, pad_day_rate=pad_day_rate, to_origin=to_origin)
    for mode in modes:
        for v in velocities:
            for del_t in delete_time_vars:
                for add_dt in add_dts:
                    for scale in scales:
                        accs, best_k = dtw_KNN(y, max_k=max_k, mode=mode, velocity=v, delete_time_vars=del_t, add_dt=add_dt, scale=scale,
                                               **id_dict)
                        results[(mode, v, del_t, add_dt, scale)] = [accs.max(), best_k]
                        pbar.refresh()
                        print(pbar.update())

    df = pd.DataFrame(results, index=["Accuracy", "K"])
    df.columns.names = ["DTW type", "Velocity", "Delete time", "Add dt", "Scale"]
    Path(savingDir).mkdir(exist_ok=True, parents=True)
    df.to_csv(os.path.join(savingDir, f"results_{other_utils.dict_to_id(id_dict)}.csv"), index=True)

    return df

def load_dtw_results(min_days=180, pad_day_rate=3, savingDir=fullPath("utils/data/dtw/")):
    df = pd.read_csv(os.path.join(savingDir, f"results_min-days-{min_days}_pad-day-rate-{pad_day_rate}.csv"), header=[*range(5)], index_col=0)
    return df



##############################################################################################################################
"""                                                       II. Shap                                                         """
##############################################################################################################################

def merge_shap_values(size=0.3, velocity="arch-segment", to_origin="space", taxa=None, add_dt=True, use_tqdm=False):
    size_str = other_utils.encoder(size)
    to_origin_str = "" if to_origin is None else to_origin
    add_dt_str = "_add-dt" if add_dt else ""
    full_data = defaultdict(list)
    if taxa is None:
        shap_dir = fullPath("utils/data/shap_values/XGB")
    else:
        shap_dir = fullPath(f"utils/data/shap_values/{taxa}/XGB")

    fold_iter = tqdm(range(1, 4)) if use_tqdm else range(1, 4)
    for fold in fold_iter:
        shap_data = file_management.load_lzma(os.path.join(shap_dir,
                                                           f"shap_size-{size_str}_20-trees_depth-30_v-{velocity}_{to_origin_str}_size-{size_str}{add_dt_str}fold-{fold}.lzma")
                                             )
        for k, v in shap_data.items():
            full_data[k].append(v)

    print("Merging results")
    for k, v in full_data.items():
        if type(v[0]) == np.ndarray:
            full_data[k] = np.concatenate(v, axis=0)
    return full_data



##############################################################################################################################
"""                                                     III. Shapelets                                                     """
##############################################################################################################################

def load_shapelets(zero_pad=[False, True], time=[34, 48], d=0, seed=range(31), pad_day_rate=3, minlen=1,
                   pruned=False, threshold=1, parentDir=fullPath("utils/data/shapelet/"), distance="euclidean"):
    """Needs sktime installed"""
    if pruned:
        var_dict = dict(d=d, minlen=minlen, threshold=threshold, zero_pad=zero_pad[0], distance=distance)
        return file_management.load_lzma(os.path.join(parentDir, f"pruned/{other_utils.dict_to_id(var_dict)}.lzma"))
    else:
        shapelets = []
        var_dict = dict(d=d, pad_day_rate=pad_day_rate)
        for zp in zero_pad:
            for t in time:
                for s in seed:
                    var_dict.update(dict(zero_pad=zp, time=t, seed=s))
                    try:
                        st = file_management.load_lzma(os.path.join(parentDir, f"sktime_{other_utils.dict_to_id(var_dict)}.lzma"))
                        shapelets += st.shapelets
                    except:
                        continue
        if minlen > 1:
            shapelets = [s for s in shapelets if s.length > minlen]
        return shapelets

@njit
def min_euclidean_distance_numba(x, y):
    """
    Min Euclidean distance between two pd.Series, normalized by the length of the shortest one. Computation:
    Naming the shortest series S, with size l, and the longest L:
        d = min([distance(S, L_subsequence(length=l)) for all subsequences with length l in L]) / L
    """
    if x.size == y.size:
        return np.sqrt(((x-y)**2).sum())
    elif x.size > y.size:
        long = x
        short = y
        L = y.size
    else:
        long = y
        short = x
        L = x.size
    arr_size = long.size - L
    d = np.empty((arr_size))
    for i in range(arr_size):
        d[i] = np.sqrt(np.square(long[i:i+L] - short).sum())
    return d.min() / np.sqrt(L)

def shapelet_distance(zero_pad=False, d=0, minlen=3, idxs=None, distance="euclidean", verbose=False, compare_to="trajectories",
                 pruned=False, threshold=1, overwrite=False, use_tqdm=False):
    """
    Compute DTW between shapelets (X) and Y determined by compare_to.
        compare_to:   - "shapelets":    Y = X
                      - "trajectories": Y = trajectories with min_days 180.
    """
    if distance == "dtw":
        d_computer = lambda x, y: dtw.dtw(x, y, distance_only=True).normalizedDistance
    elif distance == "euclidean":
        d_computer = min_euclidean_distance_numba
    else:
        raise ValueError(f"distance {distance} not valid. Available: 'dtw', 'euclidean'.")

    parentDir=fullPath(f"utils/data/shapelet/{distance}")
    inception = models.InceptionTime(pad_day_rate=3, prunning_function=preprocessing.get_min_days_prunner(180), to_origin="space")
    inception.preprocess()
    X_train = preprocessing.tf_to_df(inception.X_train, zero_pad=zero_pad, d=d)
    X_test = preprocessing.tf_to_df(inception.X_test, zero_pad=zero_pad, d=d)
    X = [x[0].values for x in X_train.values] + [x[0].values for x in X_test.values]
    labels = pd.concat([inception.labels_train, inception.labels_test], axis=0)

    shapelets = load_shapelets(zero_pad=[zero_pad], d=d, minlen=minlen, pruned=pruned, threshold=threshold, distance=distance)

    if compare_to == "shapelets":
        Y = [X_train.iloc[s.series_id, 0][s.start_pos: s.start_pos+s.length].values for s in shapelets]
    elif compare_to == "trajectories":
        Y = X
    else:
        raise ValueError(f"compare_to: {compare_to} not valid. Available: 'shapelets', 'trajectories'.")

    Path(parentDir).mkdir(exist_ok=True, parents=True)
    var_dict = dict(zero_pad=zero_pad, d=d, minlen=minlen, compare_to=compare_to, pruned=pruned, threshold=threshold)
    path = os.path.join(parentDir, f"shapelet_{distance}_{other_utils.dict_to_id(var_dict)}.npz")

    idxs = range(len(shapelets)) if idxs is None else idxs
    if Path(path).exists() and not overwrite:
        D_total = np.load(path)["D"]
    else:
        D_total = []
        iterable = tqdm(idxs) if use_tqdm else idxs
        for i in iterable:
            path_full = other_utils.id_updater(path, dict(idx=i))
            if Path(path_full).exists() and not overwrite:
                D_total.append(np.load(path_full)["D"])
                if verbose:
                    print(f"idx {i} already existed.")
            else:
                s = shapelets[i]
                if not pruned:
                    s = X_train.iloc[s.series_id, 0][s.start_pos: s.start_pos+s.length].values
                D = np.empty((len(Y)))
                for j, y in enumerate(Y):
                    D[j] = d_computer(s, y)
                D_total.append(D)
                np.savez_compressed(path_full, D=D)
        D_total = np.vstack(D_total)
        np.savez_compressed(path, D=D_total)
        for i in idxs:
            os.remove(other_utils.id_updater(path, dict(idx=i)))
    path_labels = os.path.join(parentDir, "labels_shapelet.lzma")
    if not Path(path_labels).exists() or overwrite:
        file_management.save_lzma(labels, path_labels.split("/")[-1], parentDir)
    return D_total

@savedata
def classification_report(classifier='inception', get_prunning_function=None, prunning_kws={}, use_kfold=True, n_splits=5, random_state=1, **kwargs):
    """
    Compute classification report for a given classifier.
    Uses parameters from models.classifier_params
    """
    if get_prunning_function is not None:
        prunning_function = get_prunning_function(**prunning_kws)
    else:
        prunning_function = None
    clf_params = models.classifier_params[classifier]
    model = clf_params['clf'](**kwargs, **clf_params['model'], prunning_function=prunning_function)

    print("number of features: {}".format(model.X[0].shape[0]))

    print("Training ...")
    _ = model.train(use_kfold=use_kfold, n_splits=n_splits, random_state=random_state, **clf_params["train"])

    print("Storing results ...")
    df = model.classification_report(fold_idx=range(n_splits), save=False)
    return df

def get_classification_report_default_kwargs():
    return dict(use_kfold = True,
                n_splits = 5,
                min_animals = 5,
                minlen = 5,
                random_state = 1,
                scale_by_velocity = False,
                as_image = False,
                as_image_density = False,
                as_image_indiv_scaling = False,
                remove_outliers = True,
                common_origin_distance = False,
                weather = None,
                to_origin = None,
                v2 = True,
                classifier = 'inception'
                )

def classification_report_random_states(random_states=range(1, 6), **specs):
    kwargs = get_classification_report_default_kwargs()
    kwargs.update(specs)
    dfs = []
    for random_state in random_states:
        kwargs['random_state'] = random_state
        df = classification_report(**kwargs)
        df = df.assign(random_state=random_state)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df

def get_classification_report_prunning_kws_species():
    return dict(species = None,
                taxa = None,
                percentage_delete = None,
                N = None,
                mode = None,
                random_state = 1
                )

@savedata
def stage_clf_report(weather=None, common_origin_distance=False, pad_day_rate=None, species_train='Black-browed albatross', invert_lat=False, delete_features=[], model='inception', nf=32):
    """
    nf = nb_filters for inception or n_feature_maps for resnet
    The goal is to measure the accuracy of the model when trained with one species and tested with the rest.
    (transfer learning)
    """
    species = params.stage_species_v2
    assert species_train in species, "species_train not in species with stage data."
    species_side = [s for s in species if s != species_train]

    func = lambda df: preprocessing.relabel_stages(df, species_train, remap_breeding=False)
    prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
    split_by = dict(column=['COMMON_NAME'], colvalue=species_side)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, common_origin_distance=common_origin_distance, species_stage=species, split_by=split_by, delete_features=delete_features, pad_day_rate=pad_day_rate, fill_pad_day_rate=False, scale_by_velocity=True, fit_scale_side=True, species_train=species_train, invert_lat=invert_lat)
    try:
        tree = models.DecisionTree(**kwargs)
    except preprocessing.NotEnoughData:
        warnings.warn("Not enough data, returning empty dataframe.", RuntimeWarning)
        return pd.DataFrame(), pd.DataFrame()

    if pad_day_rate in params.pad_day_rate_to_maxlen:
        maxlen = params.pad_day_rate_to_maxlen[pad_day_rate]
    else:
        maxlen_data = tree.labels_side.groupby('COMMON_NAME').apply(lambda S: np.percentile(S.Length.values, 90))
        maxlen_data[species_train] = np.percentile(tree.labels.Length.values, 90)
        maxlen = int(maxlen_data.median())
        warnings.warn(f"maxlen not in params.pad_day_rate_to_maxlen. Using median of 90th percentile of lengths: {maxlen}", RuntimeWarning)

    if model == 'inception':
        clf = models.InceptionTime(**kwargs, nb_filters=nf, maxlen=maxlen, get_input_len_from='maxlen')
        epochs = 80
        try:
            _ = clf.train(test_size=0.1, epochs=epochs, verbose=0)
        except:
            warnings.warn("Failed training, probabily due to small test size,. Trying again with test_size=0.2.", RuntimeWarning)
            _ = clf.train(test_size=0.2, epochs=epochs, verbose=0)
    elif model == 'tree':
        clf = models.DecisionTree(**kwargs)
        try:
            _ = clf.train(test_size=0.1)
        except:
            warnings.warn("Failed training, probabily due to small test size,. Trying again with test_size=0.2.", RuntimeWarning)
            _ = clf.train(test_size=0.2)
    acc = clf.evaluator()[-1]
    print(f"Accuracy: {acc}")
    df_side = clf.classification_report(partition='side')
    df_train = clf.classification_report()
    return df_side, df_train

@savedata
def stage_clf_report_multi(weather=None, common_origin_distance=True, pad_day_rate=None, species_train='Black-browed albatross', invert_lat=False, delete_features=[], nf=32):
    """
    nf = nb_filters for inception
    The goal is to measure the accuracy of the model when trained with one species and tested with the rest.
    Task: multilabel classification: breeding(several types),  not breeding
    """
    species = params.stage_species_v2_multilabel
    assert species_train in species, "species_train not in species with stage data."
    species_side = [s for s in species if s != species_train]

    func = lambda df: preprocessing.relabel_stages(df, species_train, remap_breeding=True, generic_to_nan=True)
    prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
    split_by = dict(column=['COMMON_NAME'], colvalue=species_side)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, common_origin_distance=common_origin_distance, species_stage=species, split_by=split_by, delete_features=delete_features, pad_day_rate=pad_day_rate, fill_pad_day_rate=False, scale_by_velocity=True, fit_scale_side=True, species_train=species_train, invert_lat=invert_lat, assert_side_in_train=True)
    try:
        tree = models.DecisionTree(**kwargs)
    except preprocessing.NotEnoughData:
        raise RuntimeError("Not enough data.")

    if pad_day_rate in params.pad_day_rate_to_maxlen:
        maxlen = params.pad_day_rate_to_maxlen[pad_day_rate]
    else:
        maxlen_data = tree.labels_side.groupby('COMMON_NAME').apply(lambda S: np.percentile(S.Length.values, 90))
        maxlen_data[species_train] = np.percentile(tree.labels.Length.values, 90)
        maxlen = int(maxlen_data.median())
        warnings.warn(f"maxlen not in params.pad_day_rate_to_maxlen. Using median of 90th percentile of lengths: {maxlen}", RuntimeWarning)

    clf = models.InceptionTime(**kwargs, nb_filters=nf, maxlen=maxlen, get_input_len_from='maxlen')
    epochs = 80
    try:
        _ = clf.train(test_size=0.1, epochs=epochs, verbose=0)
    except:
        warnings.warn("Failed training, probabily due to small test size,. Trying again with test_size=0.2.", RuntimeWarning)
        _ = clf.train(test_size=0.2, epochs=epochs, verbose=0)
    acc = clf.evaluator()[-1]
    print(f"Accuracy: {acc}")
    print("Computing loss and accuracy for each species")
    df_side = clf.classification_report(partition='side')
    df_train = clf.classification_report()

    species_to_idx = clf.labels_side.reset_index().groupby('COMMON_NAME').apply(lambda S: S.index.values)
    species_to_id = clf.labels_side.reset_index().groupby('COMMON_NAME').apply(lambda S: S.ID.values)
    loss = defaultdict(list)
    acc = defaultdict(list)
    df_side = df_side.set_index('ID')
    df_train = df_train.set_index('ID')
    for species, idxs in species_to_idx.items():
        y = tf.gather(clf.y_side, idxs)
        X = tf.gather(clf.X_side, idxs)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        df_species = df_side.loc[species_to_id[species]]
        for _, test in kfold.split(X, y.numpy().squeeze()):
            X_test = tf.gather(X, test)
            y_test = tf.gather(y, test)
            loss[(species, 'real')].append(clf.loss_fn(y_test, clf.model(X_test)).numpy())
            loss[(species, 'switched-labels')].append(np.nan)

            df_species_test = df_species.iloc[test]
            accuracy = (df_species_test.Prediction == 'Correct').mean()
            acc[(species, 'real')].append(accuracy)
            acc[(species, 'switched-labels')].append(np.nan) #not implemented

    loss = pd.Series(loss).unstack()
    acc = pd.Series(acc).unstack()

    return loss, acc, df_side, df_train

@savedata
def stage_clf_report_binary(weather=None, common_origin_distance=False, pad_day_rate=None, species_train='Black-browed albatross', invert_lat=False, delete_features=[], nf=32):
    """
    nf = nb_filters for inception
    The goal is to measure the accuracy of the model when trained with one species and tested with the rest.
    Task: binary classification: breeding / not breeding
    """
    species = params.stage_species_v2
    assert species_train in species, "species_train not in species with stage data."
    species_side = [s for s in species if s != species_train]

    func = preprocessing.relabel_stages_binary
    prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
    split_by = dict(column=['COMMON_NAME'], colvalue=species_side)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, common_origin_distance=common_origin_distance, species_stage=species, split_by=split_by, delete_features=delete_features, pad_day_rate=pad_day_rate, fill_pad_day_rate=False, scale_by_velocity=True, fit_scale_side=True, species_train=species_train, invert_lat=invert_lat)
    try:
        tree = models.DecisionTree(**kwargs)
    except preprocessing.NotEnoughData:
        raise RuntimeError("Not enough data.")

    if pad_day_rate in params.pad_day_rate_to_maxlen:
        maxlen = params.pad_day_rate_to_maxlen[pad_day_rate]
    else:
        maxlen_data = tree.labels_side.groupby('COMMON_NAME').apply(lambda S: np.percentile(S.Length.values, 90))
        maxlen_data[species_train] = np.percentile(tree.labels.Length.values, 90)
        maxlen = int(maxlen_data.median())
        warnings.warn(f"maxlen not in params.pad_day_rate_to_maxlen. Using median of 90th percentile of lengths: {maxlen}", RuntimeWarning)

    clf = models.InceptionTime(**kwargs, nb_filters=nf, maxlen=maxlen, get_input_len_from='maxlen')
    epochs = 80
    try:
        _ = clf.train(test_size=0.1, epochs=epochs, verbose=0)
    except:
        warnings.warn("Failed training, probabily due to small test size,. Trying again with test_size=0.2.", RuntimeWarning)
        _ = clf.train(test_size=0.2, epochs=epochs, verbose=0)
    acc = clf.evaluator()[-1]
    print(f"Accuracy: {acc}")
    print("Computing loss and accuracy for each species")
    df_side = clf.classification_report(partition='side')
    df_train = clf.classification_report()

    species_to_idx = clf.labels_side.reset_index().groupby('COMMON_NAME').apply(lambda S: S.index.values)
    species_to_id = clf.labels_side.reset_index().groupby('COMMON_NAME').apply(lambda S: S.ID.values)
    loss = defaultdict(list)
    acc = defaultdict(list)
    df_side = df_side.set_index('ID')
    df_train = df_train.set_index('ID')
    for species, idxs in species_to_idx.items():
        y = tf.gather(clf.y_side, idxs)
        X = tf.gather(clf.X_side, idxs)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        df_species = df_side.loc[species_to_id[species]]
        for _, test in kfold.split(X, y.numpy().squeeze()):
            X_test = tf.gather(X, test)
            y_test = tf.gather(y, test)
            loss[(species, 'real')].append(clf.loss_fn(y_test, clf.model(X_test)).numpy())
            loss[(species, 'switched-labels')].append(clf.loss_fn(1 - y_test, clf.model(X_test)).numpy())

            df_species_test = df_species.iloc[test]
            accuracy = (df_species_test.Prediction == 'Correct').mean()
            acc[(species, 'real')].append(accuracy)
            acc[(species, 'switched-labels')].append(1 - accuracy)

    loss = pd.Series(loss).unstack()
    acc = pd.Series(acc).unstack()

    return loss, acc, df_side, df_train

def stage_transfer_learning_binary_matrix(transfer=True, **kwargs):
    df_loss = []
    df_acc = []
    is_switched = []
    df_loss_std = []
    df_acc_std = []
    for species_train in params.stage_species_v2:
        result = stage_clf_report_binary(species_train=species_train, skip_computation=True, **kwargs)
        if not isinstance(result, SavedataSkippedComputation):
            loss, acc, *_ = result
            lm = loss.applymap(np.mean)
            ls = loss.applymap(np.std)
            accm = acc.applymap(np.mean)
            accs = acc.applymap(np.std)
            best_acc_idxs = np.argmax(accm.values, axis=1)
            row_iter = np.arange(accm.shape[0])
            is_switched.append(pd.Series(best_acc_idxs.astype(bool), index=accm.index, name=species_train))
            df_acc.append(pd.Series(accm.values[row_iter, best_acc_idxs], index=accm.index, name=species_train))
            df_loss.append(pd.Series(lm.values[row_iter, best_acc_idxs], index=lm.index, name=species_train))
            df_loss_std.append(pd.Series(ls.values[row_iter, best_acc_idxs], index=ls.index, name=species_train))
            df_acc_std.append(pd.Series(accs.values[row_iter, best_acc_idxs], index=accs.index, name=species_train))

    df_acc = pd.concat(df_acc, axis=1).T # rows: species_train, columns: species_test
    df_loss = pd.concat(df_loss, axis=1).T
    df_loss_std = pd.concat(df_loss_std, axis=1).T
    df_acc_std = pd.concat(df_acc_std, axis=1).T
    is_switched = pd.concat(is_switched, axis=1).T
    is_switched = is_switched.fillna(False)

    if transfer:
        acc_base = stage_binary_baseline_acc()
        loss_base = stage_binary_baseline_loss()
        transfer_acc = df_acc - acc_base
        transfer_loss = df_loss - loss_base
        return transfer_acc, transfer_loss, is_switched, df_loss_std, df_acc_std
    else:
        return df_acc, df_loss, is_switched, df_loss_std, df_acc_std

def stage_transfer_learning_multi_matrix(**kwargs):
    df_loss = []
    df_acc = []
    df_loss_std = []
    df_acc_std = []
    df_baseline_acc = []
    df_baseline_loss = []
    df_num_categories = []

    for species_train in params.stage_species_v2:
        result = stage_clf_report_multi(species_train=species_train, skip_computation=True, **kwargs)
        if not isinstance(result, SavedataSkippedComputation):
            loss, acc, df_side, _ = result
            lm = loss.applymap(np.mean).values[:,0]
            ls = loss.applymap(np.std).values[:,0]
            accm = acc.applymap(np.mean).values[:,0]
            accs = acc.applymap(np.std).values[:,0]
            baseline_acc = df_side.groupby("COMMON_NAME").apply(compute_binary_baseline_acc).loc[acc.index]
            baseline_acc.name = species_train
            baseline_loss = df_side.groupby("COMMON_NAME").apply(compute_binary_baseline_loss).loc[loss.index]
            baseline_loss.name = species_train
            num_categories = df_side.groupby('COMMON_NAME').apply(lambda df: df.Stage.nunique())
            num_categories.name = species_train

            df_acc.append(pd.Series(accm, index=acc.index, name=species_train))
            df_loss.append(pd.Series(lm, index=loss.index, name=species_train))
            df_loss_std.append(pd.Series(ls, index=loss.index, name=species_train))
            df_acc_std.append(pd.Series(accs, index=acc.index, name=species_train))
            df_baseline_acc.append(baseline_acc)
            df_baseline_loss.append(baseline_loss)
            df_num_categories.append(num_categories)


    df_acc = pd.concat(df_acc, axis=1).T # rows: species_train, columns: species_test
    df_loss = pd.concat(df_loss, axis=1).T
    df_loss_std = pd.concat(df_loss_std, axis=1).T
    df_acc_std = pd.concat(df_acc_std, axis=1).T
    df_baseline_acc = pd.concat(df_baseline_acc, axis=1).T
    df_baseline_loss = pd.concat(df_baseline_loss, axis=1).T
    df_num_categories = pd.concat(df_num_categories, axis=1).T.fillna(0).astype(int)

    return df_acc, df_loss, df_loss_std, df_acc_std, df_baseline_acc, df_baseline_loss, df_num_categories


def species_clf_taxa_confusion_matrix(cluster=False, **kwargs):
    """
    Returns the confusion in species prediction among taxa groups.
    """
    df = classification_report_random_states(**kwargs)
    species_to_taxa = preprocessing.get_species_to_taxa()
    df['Taxa-predicted'] = df['Predicted'].apply(lambda x: species_to_taxa[x])
    N = df.value_counts('Taxa')
    df_wrong = df.query("Prediction == 'Wrong'")
    conf_matrix = df_wrong.groupby("Taxa").apply(lambda S: S.value_counts("Taxa-predicted", normalize=False) / N[S.Taxa.iloc[0]])
    conf_matrix = conf_matrix.unstack().fillna(0)
    conf_matrix.index.name = "Real Taxa"
    conf_matrix.columns.name = "Predicted Taxa"
    if cluster:
        conf_matrix = clustering.dendrogram_sort(conf_matrix, to_distance=lambda x: 1 - x)
    else:
        # default_order = ['Polar bears', 'Penguins', 'Birds', 'Seals',  'Cetaceans', 'Sirenians','Turtles', 'Fishes']
        default_order = ['Polar bears', 'Penguins', 'Birds', 'Seals', 'Sirenians', 'Cetaceans', 'Fishes', 'Turtles']
        conf_matrix = conf_matrix.loc[default_order, default_order]
    return conf_matrix

def matrix_plot(df, legend_title, cmid=None, cmap='Blues', cmin=None, cmax=None, ticksize=28):
    fig = px.imshow(df, x=df.index, y=df.columns, color_continuous_scale=cmap)
    fig.update_layout(margin=dict(l=80, b=30, r=60, t=10, pad=1), xaxis_tickfont_size=ticksize, yaxis_tickfont_size=ticksize,
                          coloraxis=dict(cmid=cmid, colorbar=dict(title_text=legend_title, tickfont_size=26, title_font_size=30, x=1.02, y=0.5, yanchor='middle', len=0.8, nticks=5)),
                          height=750, width=1000, font_size=32, hovermode=False,
                      xaxis=dict(tickangle=90))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@savefig
def species_clf_taxa_confusion_matrix_plot(cmap='Blues', cmin=None, cmax=None, ticksize=28, **kwargs):
    df = species_clf_taxa_confusion_matrix(**kwargs)
    fig = matrix_plot(df, legend_title='Error Probability', cmap=cmap, cmin=cmin, cmax=cmax, ticksize=ticksize)
    fig.update_layout(coloraxis_colorscale = data_visualization.transparent_colorscale(fig, upper=False))
    return fig

@savefig
def species_clf_avg_accuracy_taxa():
    @njit
    def nb_mean(x):
        return x.mean()

    def compute_acc_CI(df):
        x = df.Accuracy.values
        return bootstrap.CI_bca(x, nb_mean)[0]
    order = ['Polar bears', 'Penguins', 'Birds', 'Seals', 'Turtles',  'Cetaceans', 'Sirenians', 'Fishes']
    species_to_taxa = preprocessing.get_species_to_taxa()
    specs = {'Geo+env': dict(random_states=range(1, 6), common_origin_distance=False, weather='all'),
             'Common origin': dict(random_states=range(1, 6), common_origin_distance=True, weather=None)}
    colors = plotly_default_colors(2)
    colors = {'Geo+env': colors[1], 'Common origin': colors[0]}

    taxas = order
    fig = None
    for label, kws in specs.items():
        c = colors[label]
        if fig is None:
            fig_kwargs = dict(simple_axes=True, x_title='Taxa', y_title='Mean accuracy')
        else:
            fig_kwargs = {}
        df = classification_report_random_states(v2=True, **kws)
        acc = df.groupby('COMMON_NAME').apply(lambda x: (x.Prediction == 'Correct').mean()).to_frame('Accuracy')
        acc['Taxa'] = acc.index.map(species_to_taxa)
        acc_mean_sample = acc.groupby('Taxa').mean().loc[order]
        acc_mean_CI = acc.groupby('Taxa').apply(compute_acc_CI)[order]
        CIs = np.vstack([np.array([np.nan, np.nan]) if isinstance(ci, float) else ci for ci in acc_mean_CI.values])
        fig = CI_plot(taxas, acc_mean_sample.Accuracy, CIs, fig=fig, width=0.1, color=color_std(c), color_sample_stat=c, label=label, color_legend=color_std(c, opacity=0.75),
                      **fig_kwargs)

    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                      xaxis_tickangle=-90, yaxis_range=[0, 1])
    return fig

@savefig
def species_clf_taxa_confusion_reduction(cluster=False):
    """
    Reduction in confusion between real and predicted species for each taxa, when using environmental features
    """
    df = species_clf_taxa_confusion_matrix(weather=None, cluster=False)
    df_env = species_clf_taxa_confusion_matrix(weather='all', cluster=False)
    conf_change = df_env - df
    if cluster:
        conf_change = clustering.dendrogram_sort(conf_change, to_distance=lambda x: 1 - x)
    else:
        default_order = ['Polar bears', 'Penguins', 'Birds', 'Seals', 'Sirenians', 'Cetaceans', 'Fishes', 'Turtles']
        conf_change = conf_change.loc[default_order, default_order]
    return matrix_plot(conf_change, legend_title='Error Probability<br>(Geo → Geo+Env)', cmap='RdBu_r', cmid=0)

def stage_transfer_learning_binary_matrix_acc(max_base_acc=0.95, cluster=True, **kwargs):
    transfer_acc, _transfer_loss, is_switched, _df_loss_std, _df_acc_std = stage_transfer_learning_binary_matrix(**kwargs)
    df = transfer_acc.copy()
    df[df < 0] = np.NaN
    df[is_switched] *= -1

    species = df.columns
    for s in species:
        if s not in df.index:
            df.loc[s] = 0
    df = df.loc[df.columns]
    np.fill_diagonal(df.values, np.nan)
    df[df.abs() < 0.01] = 0
    df.replace(0, np.nan, inplace=True)

    acc_base = stage_binary_baseline_acc()
    valid = acc_base  < max_base_acc
    valid_species = valid[valid].index
    df = df.loc[valid_species, valid_species]

    if cluster:
        df.fillna(0, inplace=True)
        _, dendro = clustering.hierarchy_dendrogram(1 - df) # 1 - df because we want to cluster on distance. The higher the transfer learning, the smaller the distance.
        # abs because we allow for transfer learning with real and switched labels
        order = dendro["leaves"]

        df = df.iloc[order,:].iloc[:, order]
        df.replace(0, np.nan, inplace=True)

    df = df[~df.isna().all(axis=1)]
    df = df.T
    df = df[~df.isna().all(axis=1)]
    df = df.T # rows: species_train, columns: species_test
    df.index.name = 'species_train'
    df.columns.name = 'species_test'
    return df

def stage_transfer_learning_multi_matrix_acc(max_base_acc=0.95, cluster=False, **kwargs):
    df_acc, df_loss, df_loss_std, df_acc_std, df_baseline_acc, df_baseline_loss, df_num_categories = stage_transfer_learning_multi_matrix(**kwargs)
    df = df_acc.copy()
    df -= df_baseline_acc
    df[df < 0] = np.NaN

    species = df.columns
    for s in species:
        if s not in df.index:
            df.loc[s] = 0
    df = df.loc[df.columns]
    np.fill_diagonal(df.values, np.nan)
    df[df.abs() < 0.01] = 0
    df.replace(0, np.nan, inplace=True)

    not_valid = (df_baseline_acc >= max_base_acc) | (df_num_categories < 2)
    df[not_valid] = np.nan

    if cluster:
        method = 'average'
        df.fillna(0, inplace=True)
        # symmetrize
        cm_symmetric = (df + df.T) / 2
        np.fill_diagonal(cm_symmetric.values, 1)
        d = squareform(1 - cm_symmetric)
        corr_linkage = hierarchy.linkage(d, method=method)
        dendro = hierarchy.dendrogram(corr_linkage, labels=df.columns, leaf_font_size=16, leaf_rotation=90, no_plot=True)
        order = dendro["leaves"]

        df = df.iloc[order,:].iloc[:, order]
        df.replace(0, np.nan, inplace=True)

    df = df[~df.isna().all(axis=1)]
    df = df.T
    df = df[~df.isna().all(axis=1)]
    df = df.T # rows: species_train, columns: species_test
    df.index.name = 'species_train'
    df.columns.name = 'species_test'

    df_num_categories = df_num_categories.loc[df.index, df.columns]
    df_num_categories.index.name = 'species_train'
    df_num_categories.columns.name = 'species_test'
    return df, df_num_categories

def remap_prediction(df):
    stages = df.Stage.unique()
    predicted = df.Predicted.unique()
    remap = {}
    for p in predicted:
        if p not in stages:
            if p.startswith('breeding:'):
                remap[p] = 'breeding'
    df['Predicted-v2'] = df.Predicted.replace(remap)
    return df

def remap_binary(df):
    stages = df.Stage.unique()
    predicted = df.Predicted.unique()
    remap_stages = {k: 'breeding' if k.startswith('breeding:') else k for k in stages}
    remap_predicted = {k: 'breeding' if k.startswith('breeding:') else k for k in predicted}
    df['Stage-binary'] = df.Stage.replace(remap_stages)
    df['Predicted-binary'] = df.Predicted.replace(remap_predicted)
    return df


def error_analysis_dataset(v2=True, weather=None, map_wrong_to=-1, random_states=[1], species_level=False, exclude_cols=[], replace_infinities=False, reduce='mean', **kwargs):
    """
    Returns data and target (prediction = correct or wrong) to later perform error analysis
    """
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    elif not isinstance(exclude_cols, Iterable):
        raise ValueError(f"exclude_cols {exclude_cols} not valid. Available: str, float, Iterable.")

    colgroup = dict(taxa=['Birds', 'Cetaceans', 'Fishes', 'Penguins', 'Polar bears', 'Seals', 'Sirenians', 'Turtles'],
                    sex=['M', 'F', 'U'])
    target = species_clf_acc(v2=v2, weather=weather, map_wrong_to=map_wrong_to, random_states=random_states, **kwargs)
    dataset = preprocessing.trajectory_complementary_data(v2=v2, weather=weather)
    dataset = dataset.loc[target.index]

    for group in exclude_cols:
        dataset.drop(columns=colgroup[group], inplace=True)

    if species_level:
        df = pd.concat([dataset, target], axis=1)
        df['species'] = preprocessing.map_ID_to_species(df.index.values)
        if reduce in ['mean', 'median']:
            df_species = getattr(df.groupby('species'), reduce)()
        elif callable(reduce):
            df_species = df.groupby('species').apply(reduce)
        else:
            raise ValueError(f"reduce {reduce} not valid. Available: 'mean', 'median', callable.")
        df_species['overlap-same/overlap-different'] = df_species['counts-same-species (median)'] / df_species['counts-other-species (median)']
        df_species['overlap-same/overlap-different-same-taxa'] = df_species['counts-same-species (median)'] / df_species['counts-other-species-same-taxa (median)']
        dataset, target = df_species.drop(columns='Accuracy'), df_species['Accuracy']

    if replace_infinities and np.isinf(dataset).any().any():
        warnings.warn("Infinite values in dataset. Replacing with highest finite value.", RuntimeWarning)
        def replace_infinite_values(x):
            if x.dtype == np.float64:
                x[np.isinf(x)] = np.finfo(np.float32).max
                return x
        dataset = dataset.apply(replace_infinite_values, axis=0)

    return dataset, target

def species_clf_acc(v2=True, weather=None, map_wrong_to=-1, random_states=range(1, 6), **kwargs):
    kws = get_classification_report_default_kwargs()
    kws.update(kwargs)
    kws['v2'] = v2
    kws['weather'] = weather

    df = classification_report_random_states(**kws, random_states=random_states)
    cols = ['COMMON_NAME', 'ID', 'Taxa', 'Tag', 'Sex', 'Weight', 'Length',
           'Days in trajectory (all)',
           'Days in trajectory (year)',
           'Predicted', 'Prediction']
    df = df[cols].set_index('ID')
    if len(random_states) == 1:
        accuracy = df['Prediction'].replace({'Correct': 1, 'Wrong': map_wrong_to})
    else:
        accuracy = df.groupby('ID').apply(lambda x: (x.Prediction == 'Correct').mean())
    accuracy.name = 'Accuracy'
    return accuracy

@savedata
def occurrences_count_vs_acc(v2=True, lat_width=0.5, lon_width=0.5, **kwargs):
    if 'weather' in kwargs:
        del kwargs['weather']
        warnings.warn("Count vs acc is computed in both weather settings")
    accuracy = species_clf_acc(v2=v2, weather=None, **kwargs)
    accuracy_env = species_clf_acc(v2=v2, weather='all', **kwargs)
    accuracy_env.name = 'Accuracy (env)'

    S = preprocessing.occurrences_count(reduce=None, v2=v2, lat_width=lat_width, lon_width=lon_width)
    cols_remaining = ['counts-same-species', 'counts-other-species', 'counts-other-species-same-taxa', 'counts-ratio', 'counts-ratio-same-taxa']
    idx = {reduce: [f'{i} ({reduce})' for i in cols_remaining] for reduce in ['mean', 'median']}

    def get_counts_ratio(s, reduce='median'):
        s['counts-ratio'] = s['counts-same-species'] / s['counts-other-species']
        s['counts-ratio-same-taxa'] = s['counts-same-species'] / s['counts-other-species-same-taxa']
        s = s[cols_remaining]
        s = getattr(s, reduce)()
        s.index = idx[reduce]
        return s.to_frame()
    S_me = S.apply(get_counts_ratio, reduce='median')
    S_mu = S.apply(get_counts_ratio, reduce='mean')
    df_me = pd.concat(S_me.values, axis=1).T
    df_me.index = S.index
    df_mu = pd.concat(S_mu.values, axis=1).T
    df_mu.index = S.index

    return pd.concat([df_me, df_mu, accuracy, accuracy_env], axis=1)

@savefig
def overlap_vs_acc_plot(lat_width=0.5, lon_width=0.5, weather='all', v2=True, **kwargs):
    """
    Plot overlap vs accuracy. Overlap is computed as the ratio between the number of occurrences of the same species and the number of occurrences of other species.
    """
    if 'common_origin_distance' in kwargs and kwargs['common_origin_distance']:
        raise NotImplementedError # TODO

    df = occurrences_count_vs_acc(lat_width=lat_width, lon_width=lon_width, v2=v2, **kwargs)
    df['species'] = preprocessing.map_ID_to_species(df.index.values)
    df = df.groupby('species').mean()
    species_to_taxa = preprocessing.get_species_to_taxa()
    df['taxa'] = [species_to_taxa[x] for x in df.index]

    df['ratio'] = df['counts-same-species (median)'] / df['counts-other-species (median)']
    # df['ratio-same-taxa'] = df['counts-same-species (median)'] / df['counts-other-species-same-taxa (median)']
    # df.corr('spearman')[['Accuracy', 'Accuracy (env)']].sort_values('Accuracy')
    # df.isna().sum()

    if weather == 'all':
        acc = df['Accuracy (env)']
    elif weather is None:
        acc = df['Accuracy']
    else:
        raise ValueError(f"weather={weather} not supported. Available options: [None, 'all']")
    taxa_to_color = {taxa: color for taxa, color in zip(df.taxa.unique(), plotly_default_colors())}
    fig = get_figure(xaxis_title='Overlap<sub>same species</sub> / Overlap<sub>different species</sub>', yaxis_title='Accuracy', xaxis_type='log', simple_axes=True, width=1200)
    for taxa, df_taxa in df.groupby('taxa'):
        fig.add_trace(go.Scatter(x=df_taxa['ratio'], y=acc.loc[df_taxa.index], name=taxa, marker=dict(color=taxa_to_color[taxa], size=24, line=dict(color='black', width=3)), mode='markers'))
    fig.update_layout(xaxis=dict(tickvals=[0.01, 0.1, 1, 10, 100],
                                 ticktext=['0.01', '0.1', '1', '10', '100']
                                 ))
    return fig

def clusterize_redundant_features(X, y, mode='regression', **kwargs):
    """
    Clusterize redundant features (features with similar predictive power)
    """
    if mode == 'regression':
        return shap.utils.hclust(X, y)
    elif mode == 'classification':
        from . import shap_cluster_mod
        return shap_cluster_mod.hclust(X, y, mode='classification', **kwargs)

def cluster_prunning(cluster, labels=None, height=0.2):
    """
    Prune the cluster tree
    """
    cluster_ID = hierarchy.cut_tree(cluster, height=height).squeeze()
    if labels is not None:
        cluster_ID = pd.Series(cluster_ID, index=labels).sort_values()
    return cluster_ID

def error_analysis_shap_feature_cluster(species_level=False, height=0.2,  **kwargs):
    """
    Clusterize redundant features (features with similar predictive power).

    Height = 0 for fully redundant features
    Height = 1 for independent features
    """
    @savedata
    def _error_analysis_shap_feature_cluster(species_level=False, **kwargs):
        dataset, target = error_analysis_dataset(species_level=species_level, replace_infinities=True, **kwargs)
        cluster = clusterize_redundant_features(dataset, target)
        labels = dataset.columns
        return cluster, labels

    feature_cluster, labels = _error_analysis_shap_feature_cluster(species_level=species_level, **kwargs)
    cluster_ID = cluster_prunning(feature_cluster, labels=labels, height=height)
    return cluster_ID

def feature_clustering(height=0.2, N_taxa=None, magnitude='accuracy', min_N_species=200, gpu=False, **kwargs):
    """
    Returns a Series with the cluster ID for each feature used in the dataset.
    """
    @savedata
    def _feature_clustering(N_taxa=N_taxa, magnitude=magnitude, **kwargs):
        print("Loading data")
        df = preprocessing.load_all_data(v2=True, return_labels=False, weather='all', expand_df=True)
        target = preprocessing.map_ID_to_species(df.index)
        target = pd.Series(target.squeeze(), name='Species', index=df.index)
        df = pd.concat([df, target], axis=1)
        species_to_taxa = preprocessing.get_species_to_taxa()
        df.index = [species_to_taxa[s] for s in df.Species]
        df.index.name = 'taxa'

        if isinstance(N_taxa, int):
            print(f"Sampling {N_taxa} observations per taxa")
            df = df.groupby('taxa').apply(lambda S: S if S.shape[0] < N_taxa else S.sample(N_taxa, random_state=42))
        df = df.reset_index(drop=True)

        N_species = df.Species.value_counts()
        valid_species = N_species[N_species >= min_N_species].index
        df = df[df.Species.isin(valid_species)]
        # target to category
        target = pd.Categorical(df.Species).codes
        df = df.drop(columns='Species')

        print("Clustering features")
        cluster = clusterize_redundant_features(df, target, mode='classification', func=magnitude, gpu=gpu)
        labels = df.columns
        return cluster, labels

    feature_cluster, labels = _feature_clustering(N_taxa=N_taxa, magnitude=magnitude, **kwargs)
    cluster_ID = cluster_prunning(feature_cluster, labels=labels, height=height)
    return cluster_ID

@savedata
def error_analysis_shap(exclude_cols='taxa', v2=True, weather='all', common_origin_distance=False, random_states=range(1, 6), model='xgboost', stratify=True, **model_kwargs):
    cols_remain = None
    if isinstance(exclude_cols, float): # height of a hierarchical clustering dendrogram
        cluster_ID = error_analysis_shap_feature_cluster(species_level=False, weather=weather, random_states=random_states, exclude_cols=['taxa', 'sex'], height=exclude_cols)
        cols_remain = cluster_ID.drop_duplicates().index.tolist()
        exclude_cols = ['taxa', 'sex']
    dataset, target = error_analysis_dataset(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=random_states, exclude_cols=exclude_cols, replace_infinities=True, species_level=False)
    if cols_remain is not None:
        dataset = dataset[cols_remain]
    if stratify:
        species = preprocessing.map_ID_to_species(target.index)
        species_categories = preprocessing.remap_labels(species.squeeze())
        kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(dataset, species_categories)
    else:
        kfold_split = KFold(n_splits=5, shuffle=True, random_state=42).split(dataset)

    data = defaultdict(list)
    pbar = tqdm(range(5))
    for train, test in kfold_split:
        pbar.update()
        train_data = dataset.iloc[train]
        test_data = dataset.iloc[test]
        train_target = target.iloc[train]
        test_target = target.iloc[test]

        if model == 'random-forest':
            kwargs = dict(n_estimators=100, max_depth=None, random_state=42)
            kwargs.update(model_kwargs)
            tree = RandomForestRegressor(**kwargs)
            # TODO: handle missing values
            tree.fit(train_data.values, train_target.values)
            data['test_predictions'].append(tree.predict(test_data.values))
        elif model == 'xgboost':
            dtrain = xgb.DMatrix(train_data.values, label=train_target.values)
            dtest = xgb.DMatrix(test_data.values, label=test_target.values)
            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': None, 'eta': 0.1}
            params.update(model_kwargs)
            evals = [(dtest, 'eval')]
            tree = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=10)
            data['test_predictions'].append(tree.predict(dtest))

        explainer = shap.TreeExplainer(tree)
        shap_values = explainer.shap_values(test_data.values)

        data['shap_values'].append(shap_values)
        data['test_data'].append(test_data)
        data['test_target'].append(test_target)
    return data

@savedata
def error_analysis_shap_species_level(exclude_cols=['taxa','sex'], v2=True, weather='all', common_origin_distance=False, random_states=range(1, 6), model='xgboost', stratify=True, **model_kwargs):
    cols_remain = None
    if isinstance(exclude_cols, float): # height of a hierarchical clustering dendrogram
        cluster_ID = error_analysis_shap_feature_cluster(species_level=True, weather=weather, random_states=random_states, exclude_cols=['taxa', 'sex'], height=exclude_cols)
        cols_remain = cluster_ID.drop_duplicates().index.tolist()
        exclude_cols = ['taxa', 'sex']

    dataset, target = error_analysis_dataset(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=random_states, exclude_cols=exclude_cols, replace_infinities=True, species_level=True)
    if cols_remain is not None:
        dataset = dataset[cols_remain]

    if stratify:
        species_to_taxa = preprocessing.get_species_to_taxa()
        taxa = np.array([species_to_taxa[s] for s in target.index])
        taxa_categories = preprocessing.remap_labels(taxa)
        kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(dataset, taxa_categories)
    else:
        kfold_split = KFold(n_splits=5, shuffle=True, random_state=42).split(dataset)

    data = defaultdict(list)
    pbar = tqdm(range(5))
    for train, test in kfold_split:
        pbar.update()
        train_data = dataset.iloc[train]
        test_data = dataset.iloc[test]
        train_target = target.iloc[train]
        test_target = target.iloc[test]

        if model == 'random-forest':
            kwargs = dict(n_estimators=100, max_depth=None, random_state=42)
            kwargs.update(model_kwargs)
            tree = RandomForestRegressor(**kwargs)
            # TODO: handle missing values
            tree.fit(train_data.values, train_target.values)
            data['test_predictions'].append(tree.predict(test_data.values))
        elif model == 'xgboost':
            dtrain = xgb.DMatrix(train_data.values, label=train_target.values)
            dtest = xgb.DMatrix(test_data.values, label=test_target.values)
            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': None, 'eta': 0.1}
            params.update(model_kwargs)
            evals = [(dtest, 'eval')]
            tree = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=10)
            data['test_predictions'].append(tree.predict(dtest))
        else:
            raise ValueError(f"Unknown model: {model}")

        explainer = shap.TreeExplainer(tree)
        shap_values = explainer.shap_values(test_data.values)

        data['shap_values'].append(shap_values)
        data['test_data'].append(test_data)
        data['test_target'].append(test_target)
    return data

def error_analysis_shap_performance(species_level=False, **kwargs):
    if species_level:
        data = error_analysis_shap_species_level(**kwargs)
    else:
        data = error_analysis_shap(**kwargs)
    test_target = pd.concat(data['test_target']).values
    test_predictions = np.concatenate(data['test_predictions'])
    test_baseline = np.ones((test_target.size)) * test_target.mean()
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    error_baseline = rmse(test_baseline, test_target)
    error = rmse(test_predictions, test_target)
    return dict(error=error, error_baseline=error_baseline)

@savefig
def error_analysis_shap_plot(max_display=10, species_level=False, **kwargs):
    if species_level:
        data = error_analysis_shap_species_level(**kwargs)
    else:
        data = error_analysis_shap(**kwargs)
    shap_values = np.vstack(data['shap_values'])
    test_data = pd.concat(data['test_data'])
    feature_map = params.get_error_feature_map_mpl()
    features = [feature_map[c] for c in test_data.columns]
    shap.summary_plot(shap_values, test_data, show=False, feature_names=features, max_display=max_display)
    fig = plt.gcf()
    plt.xlabel('Change in predicted accuracy', fontsize=22, fontname='sans-serif')
    plt.xticks(fontsize=16, fontname='sans-serif')
    plt.yticks(fontsize=16, fontname='sans-serif')
    return fig

def corr_matrix_pruned(df, method='spearman', alpha=0.05):
    compute_corr = getattr(ss, f'{method}r')
    C = {}
    p = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                corr, pval = 1, 0
            else:
                corr, pval = compute_corr(*(df[[col1, col2]].dropna().values.T))
            C[(col1, col2)] = corr
            p[(col1, col2)] = pval
    C = pd.Series(C).unstack()
    p = pd.Series(p).unstack()
    C_pruned = C.copy()
    C_pruned[p > alpha] = np.NaN
    return C_pruned

def prune_features_by_key(S):
    keys = ['SST', 'biodiversity', 'effort', 'Days in traj', 'SN', 'WE', 'counts-other-species', 'counts-other-species-same-taxa', 'counts-ratio', 'sampling', 'counts-same-species']
    def find_highest_in_group(S, k):
        group = [i for i in S.index if k in i]
        if group:
            return S.loc[group].abs().idxmax()
        else:
            return None
    representative_idxs = [find_highest_in_group(S, k) for k in keys]
    representative_idxs = [i for i in representative_idxs if i is not None]
    non_grouped = [i for i in S.index if not any(k in i for k in keys)]
    return S.loc[representative_idxs + non_grouped].drop_duplicates().sort_values(ascending=True)

@savefig('all-height-prune-random_states-prune_features')
def corr_with_accuracy_plot(method='spearman', species_level=False, prune=0.1, v2=True, weather='all', common_origin_distance=False, random_states=range(1,6), height=1000, prune_features=True, **kwargs):
    dataset, target = error_analysis_dataset(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=random_states, species_level=species_level, **kwargs)
    target.name = 'Accuracy'
    df = pd.concat([dataset, target], axis=1)

    cm = corr_matrix_pruned(df, method=method)
    acc_cor = cm['Accuracy'].dropna().sort_values().iloc[:-1] # drop acc-acc correlation
    acc_cor = acc_cor[acc_cor.abs() > prune]
    if prune_features:
        acc_cor = prune_features_by_key(acc_cor)
    features = [params.error_feature_map[c] for c in acc_cor.index]
    fig = get_figure(xaxis_title='Corr with accuracy', yaxis_title='Feature', xaxis_range=[-1, 1],
                     margin=dict(t=0,b=0,l=0,r=0),  height=height, width=900)
    fig.add_bar(x=acc_cor.values, y=features, orientation='h')
    return fig

@savedata
def corr_with_accuracy_micro(v2=True, weather='all', common_origin_distance=False, random_states=range(1,6), **kwargs):
    dataset, target = error_analysis_dataset(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=random_states, **kwargs)
    target.name = 'Accuracy'
    df = pd.concat([dataset, target], axis=1)
    df['COMMON_NAME'] = preprocessing.map_ID_to_species(df.index.values, v2=True)

    corr_micro = {}
    for species, df_species in tqdm(df.groupby('COMMON_NAME')):
        if df_species.Accuracy.nunique() == 1:
            continue
        cm = corr_matrix_pruned(df_species)
        acc_cor = cm['Accuracy'].dropna().sort_values().iloc[:-1] # drop acc-acc correlation
        # acc_cor = acc_cor[acc_cor.abs() > prune]
        map = {'N species': '# animals'}
        acc_cor.index = acc_cor.index.map(lambda x: map.get(x, x))
        for feature, corr in acc_cor.items():
            corr_micro[(species, feature)] = corr
    corr_micro = pd.Series(corr_micro).unstack()
    return corr_micro

@savefig('all-height-prune_features')
def corr_with_accuracy_micro_plot(v2=True, weather='all', common_origin_distance=False, random_states=range(1,6), prune=0.1, prune_features=True, height=1000, **kwargs):
    corr_micro = corr_with_accuracy_micro(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=random_states, **kwargs)
    corr_micro = corr_micro.mean().sort_values()
    corr_micro = corr_micro[corr_micro.abs() > prune]
    if prune_features:
        corr_micro = prune_features_by_key(corr_micro)
    features = [params.error_feature_map[c] for c in corr_micro.index]
    fig = get_figure(xaxis_title='Corr with accuracy', yaxis_title='Feature', xaxis_range=[-1, 1],
                     margin=dict(t=0,b=0,l=0,r=0),  height=height, width=900)
    fig.add_bar(x=corr_micro.values, y=features, orientation='h')
    return fig

@savefig('all')
def corr_with_accuracy_micro_feature_plot(feature, **kwargs):
    c = corr_with_accuracy_micro(**kwargs)
    fig = c[feature].sort_values(ascending=False).dropna().plot(kind='bar').get_figure()
    plt.ylim(-1, 1)
    return fig

@savedata('all')
def stage_clf_performance(species, model, weather=None, common_origin_distance=True, epochs=140, maxlen=None, **model_kwargs):
    vertices = []
    split_by = dict(column=None, colvalue=None)
    if not common_origin_distance:
        if species in params.location_prunning:
            vertices = params.location_prunning[species]
        if species in params.stage_split_by:
            split_by = params.stage_split_by[species]

    prunning_function=preprocessing.get_prunning_function(column='COMMON_NAME', colvalue=species, label='Stage', vertices=vertices)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, split_by=split_by, common_origin_distance=common_origin_distance, species_stage=species)
    clf_params = models.classifier_params[model]
    if model is None:
        tree = models.DecisionTree(**kwargs)
        acc = labels.Stage.value_counts(normalize=True).max() # base accuracy
    elif model in ['tree', 'forest', 'xgb']:
        clf = clf_params['clf'](**clf_params['model'], **model_kwargs, **kwargs)
        _ = clf.train(**clf_params['train'])
        acc = clf.evaluator()[-1]
    else:
        if model == 'resnet':
            if weather is not None:
                maxlen = 128
            else:
                maxlen = 256
        else:
            maxlen = 512
        tree = models.DecisionTree(**kwargs)
        maxlen_data = int(np.percentile(tree.labels.Length.values, 90))
        maxlen = min(maxlen, maxlen_data)
        model_params = clf_params['model']
        model_params.update(model_kwargs)
        model_params['maxlen'] = maxlen
        print("Model params:\n", model_params)
        clf = clf_params['clf'](**model_params, **kwargs)
        _ = clf.train(epochs=epochs, batch_size=128)
        acc = clf.evaluator()[-1]
    return acc

@savedata
def stage_clf_performance_summary_aux(weather=None, common_origin_distance=True):
    kwargs = dict(weather=weather, common_origin_distance=common_origin_distance)
    classifier_specs = dict(tree={}, inception=dict(nb_filters=[16, 32,64,128]))#, resnet=dict(n_feature_maps=[32,64,128]))
    def get_acc(species, classifier, kwargs, **specs):
        acc_species_classifier = stage_clf_performance(species, model=classifier, **kwargs, **specs, skip_computation=True)
        if isinstance(acc_species_classifier, SavedataSkippedComputation):
            acc_species_classifier = np.NAN
        return acc_species_classifier

    acc = {}
    for species in tqdm(params.stage_species_v2):
        for classifier, specs in classifier_specs.items():
            if specs:
                k, vs = list(specs.items())[0]
                if isinstance(vs, (list, tuple)):
                    results = [get_acc(species, classifier, kwargs, **{k: v}) for v in vs]
                else:
                    results = [get_acc(species, classifier, kwargs, **specs)]
            else:
                results = [get_acc(species, classifier, kwargs)]

            acc[(species, classifier)] = results

        # base accuracy
        species_train = deepcopy(species)
        species = params.stage_species_v2
        species_side = [s for s in species if s != species_train]

        func = lambda df: preprocessing.relabel_stages(df, species_train, remap_breeding=False)
        prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
        split_by = dict(column=['COMMON_NAME'], colvalue=species_side)
        kwargs_base = dict(v2=True, prunning_function=prunning_function, species_stage=species, split_by=split_by, species_train=species_train, **kwargs)
        tree = models.DecisionTree(**kwargs_base)
        acc_base = tree.labels.Stage.value_counts(normalize=True).max()
        acc[(species_train, 'base')] = acc_base

    acc = pd.Series(acc)
    acc_summary = acc.unstack().applymap(np.round, na_action='ignore', decimals=2)
    return acc_summary

def stage_clf_performance_summary(weather=None, common_origin_distance=True, max_base_acc=0.9):
    acc_summary = stage_clf_performance_summary_aux(weather=weather, common_origin_distance=common_origin_distance)
    def compute_max(x):
        if isinstance(x, float):
            return x
        else:
            x = np.array(x)
            return x[~np.isnan(x)].max()
    acc_summary = acc_summary[acc_summary.base < max_base_acc].applymap(compute_max)

    return acc_summary

@savefig
def stage_clf_performance_summary_plot(clf='inception', multilabel_only=True):
    acc_summary = stage_clf_performance_summary(weather=None, common_origin_distance=True)
    acc_summary_env = stage_clf_performance_summary(weather='all', common_origin_distance=False)
    acc_common_origin = acc_summary[clf] - acc_summary.base
    acc_env = acc_summary_env[clf] - acc_summary_env.base
    if multilabel_only:
        acc_common_origin = acc_common_origin.loc[params.stage_species_v2_multilabel]
        acc_env = acc_env.loc[params.stage_species_v2_multilabel]

    max_improvement = pd.concat([acc_common_origin, acc_env], axis=1).max(axis=1).max() + 0.05
    fig = get_figure(yaxis_title="Species", xaxis_title="Improvement in accuracy")
    # horizontal bars
    fig.add_trace(go.Bar(
        y=acc_common_origin.index,
        x=acc_common_origin.values,
        name='Common origin',
        orientation='h',
        marker=dict(
            color='rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=acc_env.index,
        x=acc_env.values,
        name='Environment',
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    fig.update_layout(
        barmode='group',
        xaxis_range=[-max_improvement, max_improvement],
        xaxis_title="Improvement in accuracy",
        yaxis_title="Species",
        legend=dict(
            x=0.06,
            y=0.85,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=20,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=1
        )
    )
    return fig

def remap_stage_not_in_training_set(df, training_stages):
    """
    Remaps predictions that were not in the training set to the general breeding stage if the general stage is in the training set. Otherwise, map it to NaN and drop it.

    Predictions depend on the species_train
    Stages correspond to the species_test
    """
    # TODO: Define a biological-based relation between breeding stages for different species. This way, ther
    stages = df.Stage.unique()

    remap = {}
    for p in training_stages:
        if p not in stages:
            if p.startswith('breeding:'):
                if 'breeding' in training_stages:
                    remap[p] = 'breeding'
                else:
                    remap[p] = np.NAN
    df['Predicted'] = df.Predicted.replace(remap)
    df = df.dropna(subset=['Predicted'])
    return df

def remap_binary(df):
    stages = df.Stage.unique()
    predicted = df.Predicted.unique()
    remap_stages = {k: 'breeding' if k.startswith('breeding:') else k for k in stages}
    remap_predicted = {k: 'breeding' if k.startswith('breeding:') else k for k in predicted}
    df['Stage'] = df.Stage.replace(remap_stages)
    df['Predicted'] = df.Predicted.replace(remap_predicted)
    return df

def stage_transfer_learning(gpt_remap=True, stage_remap='not-in-train', max_base_acc=0.9, skip_computation=True, **kwargs):
    result = stage_clf_report(**kwargs, skip_computation=skip_computation)
    if isinstance(result, SavedataSkippedComputation):
        return pd.DataFrame()
    else:
        df, dft = result
        training_stages = dft.Stage.unique()
    df = df[['COMMON_NAME', 'ID', 'Predicted', 'Stage', 'Prediction']]
    if gpt_remap:
        df['Stage'] = df['Stage'].replace(params.breeding_remaps)
    if stage_remap == 'not-in-train':
        df = df.groupby("COMMON_NAME").apply(remap_stage_not_in_training_set, training_stages=training_stages)
    elif stage_remap == 'binary':
        df = remap_binary(df)
    elif stage_remap == 'bio':
        # TODO: Define a biological-based relation between breeding stages for different species.
        raise NotImplementedError
    else:
        raise ValueError(f"stage_remap: {stage_remap} not valid. Available: 'not-in-train', 'binary'.")

    df = df.reset_index(drop=True)
    acc = df.groupby('COMMON_NAME').apply(lambda df: (df['Stage'] == df['Predicted']).mean())
    acc_base = df.groupby('COMMON_NAME').apply(lambda df: df.Stage.value_counts(normalize=True).max())
    results = pd.concat([acc, acc_base], axis=1)
    results.columns = ['acc', 'acc_base']
    results['improvement'] = results['acc'] - results['acc_base']
    if max_base_acc:
        results = results[results.acc_base < max_base_acc]
    return results

def stage_transfer_learning_matrix(model='tree', pad_day_rate=3, weather=None, common_origin_distance=True, stage_remap='not-in-train', delete_features=[], negative_to=np.NaN, common_only=False, cluster=True, invert_lat=False):
    """
    rows: species_train, columns: species_test
    common_only: only include species that are in both species_train and species_test
    """
    kwargs = {k:v for k, v in locals().items() if k not in ['negative_to', 'common_only', 'cluster']}

    dfs = []
    for species_train in params.stage_species_v2:
        if stage_remap == 'binary':
            transfer = stage_transfer_learning(species_train=species_train, **kwargs)
            transfer['acc_switched_binary'] = 1 - transfer.acc
            transfer['improvement_switched_binary'] = transfer['acc_switched_binary'] - transfer['acc_base']
            improvement = transfer[['improvement', 'improvement_switched_binary']].max(axis=1)
            improvement[improvement < 0] = 0
            improvement[(improvement > 0) & (transfer['improvement_switched_binary'] > transfer['improvement'])] *= -1
        else:
            improvement = stage_transfer_learning(species_train=species_train, **kwargs).improvement
        improvement.name = species_train
        dfs.append(improvement)
    df = pd.concat(dfs, axis=1)
    if common_only:
        common = df.index
        df = df[common]

    if negative_to is not None:
        if stage_remap == 'binary':
            warnings.warn("If negative values are changed, the complementary transfer is hidden (label 0 of species A being similar to label 1 of species B)", UserWarning)
        df[df < 0] = negative_to

    species = df.columns
    for s in species:
        if s not in df.index:
            df.loc[s] = 0
    df = df.loc[df.columns]
    np.fill_diagonal(df.values, np.nan)
    df[df.abs() < 0.01] = 0
    df.replace(0, np.nan, inplace=True)

    if cluster:
        df.fillna(0, inplace=True)
        _, dendro = clustering.hierarchy_dendrogram(1 - df) # 1 - df because we want to cluster on distance. The higher the transfer learning, the smaller the distance.
        order = dendro["leaves"]

        df = df.iloc[order,:].iloc[:, order]
        df.replace(0, np.nan, inplace=True)

    df = df[~df.isna().all(axis=1)]
    df = df.T # rows: species_train, columns: species_test
    df = df[~df.isna().all(axis=1)]
    df.index.name = 'species_train'
    df.columns.name = 'species_test'

    return df

@savefig
def stage_transfer_learning_plot(stage_remap='not-in-train', **kwargs):
    df = stage_transfer_learning_matrix(stage_remap='not-in-train', **kwargs)
    if stage_remap == 'not-in-train':
        # exclude binary
        df = df[[c for c in df.columns if c in params.stage_species_v2_multilabel]]
        df = df.loc[[i for i in df.index if i in params.stage_species_v2_multilabel]]
        title = 'Breeding stage transfer learning (multilabel)'
    else:
        title = 'Breeding stage transfer learning'

    fig = px.imshow(df)
    ticksize = 24
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=100, pad=0),
        font_size=32,
        xaxis=dict(title='Species test', tickfont=dict(size=ticksize)),
        yaxis=dict(title='Species train<br>', tickfont=dict(size=ticksize)),
        title=title,
        coloraxis=dict(cmin=0, cmax=0.3, colorbar=dict(title='Accuracy gain   <br>   (zero-shot)<br> <br>', tickfont=dict(size=ticksize+4),
                                                       x=1.02, y=0.5, yanchor='middle', len=1, thickness=20, tickvals=[0, 0.09, 0.19, 0.29], ticktext=["0", "0.1", "0.2", ">0.3"]
                                                       )),
    )
    fig.update_xaxes(tickangle=90)
    return fig

@savefig('all-order')
def stage_transfer_learning_bidirectional_plot(remapped=False, order='raw', **kwargs):
    """
    Only implemented for the binary case. Considers also transfer learning from species with opposite labeling. This is, if label 1 of species A is similar to label 0 of species B, then the transfer learning from A to B is considered negative
    """
    if 'stage_remap' in kwargs and kwargs['stage_remap'] != 'binary':
        raise NotImplementedError
    if 'negative_to' in kwargs and kwargs['negative_to'] is not None:
        del kwargs['negative_to']
        warnings.warn("negative_to must be None for bidirectional plot. Setting it to None.")
    if remapped:
        df = stage_transfer_learning_matrix(stage_remap='binary', negative_to=None, **kwargs)
    else:
        df = stage_transfer_learning_binary_matrix_acc(**kwargs)

    def color_HTML(color, text):
        # color: hexadecimal
        s = "<span style='color:" + str(color) + "'>" + str(text) + "</span>"
        return s
    if order == 'mean-transfer':
        mean_transfer = df.abs().fillna(0).mean(axis=1)
        order = mean_transfer.sort_values(ascending=False).index
        df = df.loc[order]
        mean_transfer_received = df.abs().fillna(0).mean(axis=0)
        order_received = mean_transfer_received.sort_values(ascending=False).index
        df = df[order_received]
    tickvals = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
    ticktext = [">0.3", "0.2", "0.1", "0.1", "0.2", ">0.3"]
    colors = ['red' if v < 0 else 'blue' for v in tickvals]
    ticktext_colored = [color_HTML(c, t) for c, t in zip(colors, ticktext)]
    fig = px.imshow(df)
    ticksize = 24
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=100, pad=0),
        font_size=32,
        xaxis=dict(title='Species test', tickfont=dict(size=ticksize)),
        yaxis=dict(title='Species train<br>', tickfont=dict(size=ticksize)),
        title='Breeding stage transfer learning (binary)',
        coloraxis=dict(cmin=-0.3, cmax=0.3, colorscale='RdBu', colorbar=dict(title='Accuracy gain   <br>   (zero-shot)<br> <br>',
                                                           tickfont=dict(size=ticksize+4), tickvals=tickvals, ticktext=ticktext_colored,
                                                           x=1.05, y=0.5, yanchor='middle', len=1, thickness=20)
                       ))
    fig.add_annotation(text="real label", x=1.55, y=0.42, showarrow=False, font_size=ticksize, font_color='blue', xref='paper', yref='paper', xanchor="center")
    fig.add_annotation(text="switched label", x=1.55, y=0.17, showarrow=False, font_size=ticksize, font_color='red', xref='paper', yref='paper', xanchor="center")
    fig.update_xaxes(tickangle=90)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@savefig('all-order')
def stage_transfer_learning_multi_plot(pad_day_rate=None, weather=None, common_origin_distance=True, order='mean-transfer', **kwargs):
    df, df_num_categories = stage_transfer_learning_multi_matrix_acc(pad_day_rate=pad_day_rate, weather=weather, common_origin_distance=common_origin_distance, **kwargs)
    if order == 'mean-transfer':
        mean_transfer = df.abs().fillna(0).mean(axis=1)
        order = mean_transfer.sort_values(ascending=False).index
        order_cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
        df = df.loc[order, order_cols]
        df_num_categories = df_num_categories.loc[order, order_cols]
        # mean_transfer_received = df.abs().fillna(0).mean(axis=0)
        # order_received = mean_transfer_received.sort_values(ascending=False).index
        # df = df[order_received]
        # df_num_categories = df_num_categories[order_received]
    tickvals = [0, 0.1, 0.2]
    ticktext = ["0", "0.1", ">0.2"]
    fig = px.imshow(df, color_continuous_scale='Blues')
    # plot df_num_categories as text in red
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if not math.isnan(df.iloc[i, j]):
                text = str(df_num_categories.iloc[i, j])
                text = f"<span style='font-weight: 900;'>{text}</span>"
                fig.add_annotation(x=j, y=i, text=text, showarrow=False, font=dict(family='sans-serif', size=34, color='red'))

    ticksize = 24
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=100, pad=0),
        font_size=32,
        xaxis=dict(title='Species test', tickfont=dict(size=ticksize)),
        yaxis=dict(title='Species train<br>', tickfont=dict(size=ticksize)),
        title='Breeding stage transfer learning (multi-label)',
        coloraxis=dict(cmin=0, cmax=0.2, colorbar=dict(title='Accuracy gain   <br>   (zero-shot)<br> <br>',
                                                           tickfont=dict(size=ticksize+4), tickvals=tickvals, ticktext=ticktext,
                                                           x=1.05, y=0.5, yanchor='middle', len=1, thickness=20)
                       ))
    fig.update_xaxes(tickangle=90)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@savedata
def feature_importances_shap(weather='all', common_origin_distance=False, fold_idx=None, use_kfold=False, test_size=0.1, avg_trajs=True, predicted_only=None, **kwargs):
    clf = models.XGB(weather=weather, common_origin_distance=common_origin_distance, **kwargs)
    data, specs = clf.compute_shap(use_kfold=use_kfold, n_splits=5, fold_idx=fold_idx, test_size=test_size, avg_trajs=avg_trajs, predicted_only=predicted_only)
    return data, specs

def feature_importances_shap_all_folds(fold_idxs=5, **kwargs):
    """
    Computes feature importances for all folds and concatenates them.

    NOTE: All specs, including specs['label'] (category_to_label dict), are the same for all folds. Thus, they are computed only once.
    """
    if 'use_kfold' in kwargs:
        del kwargs['use_kfold']
    data1, specs = feature_importances_shap(fold_idx=1, use_kfold=True, **kwargs)
    data_list = [data1]
    for fold_idx in tqdm(range(2, fold_idxs+1)):
        data_list.append(feature_importances_shap(fold_idx=fold_idx, use_kfold=True,  **kwargs)[0])
    data = _helper.merge_dict_list(*data_list, concat_arr_axis=0)
    return data, specs

def feature_importances_shap_prunner(shap_class='predicted', **kwargs):
    """
    Returns the feature importances (SHAP values) for the dataset.

    Parameters
    ----------
    shap_class[str] : 'predicted' or 'real'. If 'predicted', the feature importance is computed for the predicted class. If 'real', the feature importance is computed for the real class.
    **kwargs : Keyword arguments for `analysis.feature_importances_shap_all_folds`.
    """
    data, specs = feature_importances_shap_all_folds(avg_trajs=True, **kwargs)

    @savedata
    def _feature_importances_shap_prunner(shap_class='predicted', **kwargs):
        avg_trajs = kwargs.pop('avg_trajs', True)
        if not avg_trajs:
            raise ValueError("avg_trajs=False not implemented yet")
        N = data['shap_values'].shape[0]
        if shap_class == 'real':
            shap_class_idxs = data['y_short'].copy()
        elif shap_class == 'predicted':
            shap_class_idxs = data['preds'].copy()
        else:
            raise ValueError(f"shap_class must be one of 'real', 'predicted', got {shap_class}")
        shap_values = data['shap_values'].squeeze()[np.arange(N), shap_class_idxs]
        return shap_values

    data['shap_values'] = _feature_importances_shap_prunner(shap_class=shap_class, **kwargs)
    return data, specs

def join_feature_effects(data, specs):
    """
    Join effects of features that are part of the same group.

    Location: x, y, z or WE, SN
    Time: cos t, sin t
    """
    print("Joining feature effects:\n Location: x, y, z or WE, SN\n Time: cos t, sin t")
    shap_values = data['shap_values']
    features = specs['features']
    feature_to_idx = {feature: i for i, feature in enumerate(features)}
    if 'x' in features:
        location_remap = dict(Location=['x', 'y', 'z'])
    else:
        location_remap = dict(Location=['WE', 'SN'])
    time_remap = dict(Time=['cos t', 'sin t'])
    idxs_location = [feature_to_idx[feature] for feature in location_remap['Location']]
    idxs_time = [feature_to_idx[feature] for feature in time_remap['Time']]
    shap_location = shap_values[:, idxs_location].sum(axis=1)
    shap_time = shap_values[:, idxs_time].sum(axis=1)
    idxs_delete = idxs_location + idxs_time
    shap_values = np.delete(shap_values, idxs_delete, axis=1)
    features = np.delete(features, idxs_delete)
    shap_values = np.concatenate([shap_location[:, None], shap_time[:, None], shap_values], axis=1)
    features = ['Location', 'Time'] + features.tolist()
    data['shap_values'] = shap_values
    specs['features'] = features
    return data, specs

def feature_effects_join_diff(plot=False):
    """
    Difference between joining the feature effects before taking absolute values and after.

    Results show that the difference is negligible.
    """
    data, specs = feature_importances_shap_prunner()
    data_cp, specs_cp = deepcopy(data), deepcopy(specs)
    d2, s2 = join_feature_effects(data_cp, specs_cp)
    df = pd.DataFrame(data['shap_values'], columns=specs['features']).abs()
    df2 = pd.DataFrame(d2['shap_values'], columns=s2['features']).abs()
    df['Location'] = df[['x','y','z']].sum(axis=1).values
    diff = df['Location'] - df2['Location']
    df['Time'] = df[['cos t', 'sin t']].sum(axis=1).values
    diff_time = df['Time'] - df2['Time']
    if plot:
        plt.plot(diff, diff_time, '.')
        plt.xlabel('Location')
        plt.ylabel('Time')
        plt.axvline(diff.mean(), color='k')
        plt.axhline(diff_time.mean(), color='k')

    print(pd.concat([diff, diff_time], axis=1).describe())
    return

@savefig
def feature_importances_shap_abs_plot(shap_class='predicted', engine='plotly', max_display=10, **kwargs):
    """
    Plot the feature importance (mean absolute SHAP value) for the dataset.

    Parameters
    ----------
    shap_class[str] : 'predicted', 'real' or None. If 'predicted', the feature importance is computed for the predicted class. If 'real', the feature importance is computed for the real class. If None, the feature importance is computed for all classes.
    engine[str] : 'plotly' or 'matplotlib'. The engine to use for plotting.
    max_display[int] : The maximum number of features to display.
    **kwargs : Keyword arguments for `analysis.feature_importances_shap_all_folds`.
    """
    # TODO: micro average shap_values (a) over species and (b) over species followed by over taxa. Right now it is averaged over trajectories.
    # DONE. Check feature_importance_setting_comparison
    avg_trajs = kwargs.pop('avg_trajs', True)
    if not avg_trajs:
        raise ValueError("avg_trajs=False not implemented yet")

    data, specs = feature_importances_shap_prunner(shap_class=shap_class, **kwargs)
    shap_values = data['shap_values']
    if engine == 'matplotlib':
        X_list = data['X_orig']
        df = pd.DataFrame(data['X'], columns=specs['features'])
        L = pd.Series(X_list).apply(lambda x: x.shape[1])
        start_idx = L.cumsum().shift(1).fillna(0).astype(int)
        end_idx = L.cumsum().astype(int)
        df_fig = df.copy()
        df_fig = pd.DataFrame(np.concatenate([df.iloc[start:end].mean(axis=0).to_frame().T for start, end in zip(start_idx, end_idx)], axis=0), columns=df.columns)
        df_fig.columns = [params.feature_map[f] for f in df.columns]
        fig = shap.summary_plot(shap_values, df_fig, plot_type="bar", show=False, max_display=max_display)
    elif engine == 'plotly':
        columns = [params.feature_map[f] for f in specs['features']]
        shap_values = pd.DataFrame(shap_values, columns=columns)
        shap_abs_mean = shap_values.abs().mean(axis=0).sort_values(ascending=False)
        shap_abs_mean = shap_abs_mean.iloc[:max_display]

        fig = get_figure(xaxis_title="Feature importance (⟨|SHAP|⟩)", yaxis_title="Feature")
        fig.add_trace(go.Bar(x=shap_abs_mean.values[::-1], y=shap_abs_mean.index[::-1], orientation='h'))
        fig.update_layout(plot_bgcolor='white',
                          yaxis=dict(showline=True, linecolor='black', linewidth=2.4),
                          xaxis=dict(showline=True, linecolor='black', linewidth=2.4))
    else:
        raise ValueError(f"engine must be one of 'plotly' or 'matplotlib', got {engine}")
    return fig

# TODO:https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
# bar plot with clustering to show redundant features

def feature_importances_taxa_all(plot_type='violin', max_display=10, join_groups=False, xlims=(-1.25,3.5), reference='dataset', ext='png', **kwargs):
    """
    # TODO: Implement join_groups=True. Since the violin plot uses
    """
    data, specs = feature_importances_shap_prunner(**kwargs)
    if join_groups:
        raise NotImplementedError("join_groups=True not implemented yet. It is not clear how to plot the joined features.")
        data, specs = join_feature_effects(data, specs)

    shap_values = data['shap_values']
    y_pred = data['preds']
    idx_to_species = specs['labels']
    X = data['X_orig']
    X_mean = pd.Series(X).apply(lambda x: x.mean(axis=1))
    columns = [params.feature_map[c] for c in specs['features']]

    species_to_taxa = preprocessing.get_species_to_taxa()
    idx_to_taxa = lambda x: species_to_taxa[idx_to_species[x]]

    taxas = pd.Series(y_pred).apply(idx_to_taxa)
    taxa_to_idx = taxas.groupby(taxas).apply(lambda x: x.index.values)

    if reference is None:
        df_reference = None
    else:
        df_reference = pd.DataFrame(np.vstack(X_mean.values), columns=columns)
        if reference == 'sample':
            df_reference['taxa'] = taxas
            def get_samples(df):
                return df.apply(lambda x: x.sort_values().values[np.linspace(0, len(x)-1, 200).astype(int)], axis=0)

            df_reference = df_reference.groupby('taxa').apply(get_samples).drop(columns='taxa')
        elif reference != 'dataset':
            raise ValueError(f"reference must be one of 'dataset', 'sample' or None, got {reference}")

    @savefig
    def feature_importances_taxa(taxa='', plot_type='', max_display=1, xlims=xlims, **params):
        """
        Wrapped in savefig to save the figure in the correct directory.
        """
        shap_taxa = shap_values[taxa_to_idx[taxa]]
        X_taxa = X_mean.loc[taxa_to_idx[taxa]]
        df_taxa = pd.DataFrame(np.vstack(X_taxa.values), columns=columns)

        if plot_type == 'violin':
            shap_plots_mod.violin(shap_taxa, features=df_taxa, features_reference=df_reference, max_display=max_display, show=False)
        elif plot_type == 'bar':
            shap.plots.bar(shap_taxa, features=df_taxa, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_taxa, df_taxa, max_display=max_display, show=False)
        if xlims is not None:
            plt.xlim(*xlims)
        return plt.gcf()

    for taxa in tqdm(taxas.unique()):
        feature_importances_taxa(taxa=taxa, plot_type=plot_type, max_display=max_display, xlims=xlims, reference=reference, ext=ext, **kwargs)
    return

def feature_importances_diff_taxa_all(max_display=10, join_groups=True, xlims=None, ext='png', **kwargs):
    data, specs = feature_importances_shap_prunner(**kwargs)
    if join_groups:
        data, specs = join_feature_effects(data, specs)
    shap_values = data['shap_values']
    if 'shap_class' in kwargs and kwargs['shap_class'] == 'real':
        y = data['y_short']
    else:
        y = data['preds']
    idx_to_species = specs['labels']
    species_to_taxa = preprocessing.get_species_to_taxa()
    idx_to_taxa = lambda x: species_to_taxa[idx_to_species[x]]
    taxas = pd.Series(y).apply(idx_to_taxa)
    taxa_categories = taxas.unique()

    columns = [params.feature_map[c] for c in specs['features']]

    @savefig
    def feature_importances_diff_taxa(taxa='', max_display=10, xlims=None, **params):
        mask_taxa = (taxas == taxa).values
        shap.plots.group_difference(shap_values, mask_taxa, max_display=max_display, feature_names=columns, xlabel=f'⟨|SHAP|⟩ difference ({taxa} - rest)', show=False)
        if xlims is not None:
            plt.xlim(*xlims)
        plt.yticks(fontsize=26)
        plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=24)
        plt.xlabel(f'⟨|SHAP|⟩ difference ({taxa} - rest)', fontsize=24)
        plt.subplots_adjust(right=1.15, top=0.85)
        plt.ylim(-0.8, max_display-0.5)
        return plt.gcf()

    @savefig
    def feature_importances_diff_taxa_pair(taxa1='', taxa2='', max_display=10, xlims=None, **params):
        mask_pair = (taxas == taxa1).values | (taxas == taxa2).values
        shap_pair = shap_values[mask_pair]
        mask_taxa = taxas[mask_pair] == taxa1
        shap.plots.group_difference(shap_pair, mask_taxa, max_display=max_display, feature_names=columns, xlabel=f'⟨|SHAP|⟩ difference ({taxa1} - {taxa2})', show=False)
        if xlims is not None:
            plt.xlim(*xlims)
        plt.yticks(fontsize=26)
        plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=24)
        plt.xlabel(f'⟨|SHAP|⟩ difference ({taxa1} - {taxa2})', fontsize=24)
        plt.subplots_adjust(right=1.15, top=0.85)
        plt.ylim(-0.8, max_display-0.5)
        return plt.gcf()

    print("Difference between taxa and the rest")
    for taxa in tqdm(taxa_categories):
        feature_importances_diff_taxa(taxa=taxa, max_display=max_display, xlims=xlims, ext=ext, **kwargs)

    print("Difference between taxa pairs")
    for taxa1 in tqdm(taxa_categories):
        for taxa2 in taxa_categories:
            if taxa1 == taxa2:
                continue
            feature_importances_diff_taxa_pair(taxa1=taxa1, taxa2=taxa2, max_display=max_display, xlims=xlims, ext=ext, **kwargs)
    return

@savedata
def shap_abs_avg_prunned(**kwargs):
    """
    Computes the averages and discards the unused data

    Join effects of features that are part of the same group.
        Location: x, y, z or WE, SN
        Time: cos t, sin t
    """
    data, specs = feature_importances_shap_prunner(**kwargs)
    data, specs = join_feature_effects(data, specs)
    for micro in [None, 'species', 'taxa']:
        data, specs = shap_abs_avg(data, specs, micro=micro)
    data_keep = ['shap_values_abs_mean', 'shap_values_abs_species_mean', 'shap_values_abs_taxa_mean',
                 'shap_values_abs_species', 'shap_values_abs_taxa']
    data = {k: data[k] for k in data_keep}
    return data, specs

def shap_abs_avg(data, specs, micro=None):
    """
    Compute the average of the absolute SHAP values.
    micro: ['species', 'taxa', None]
    - None -> compute the average over all the features.
    - 'species' or 'taxa' -> compute the micro average over the species or taxa.
    """
    if micro is None:
        return _shap_avg_by_trajectories(data, specs)
    elif micro == 'species':
        return _shap_micro_avg_by_species(data, specs)
    elif micro == 'taxa':
        return _shap_micro_avg_by_taxa(data, specs)
    else:
        raise ValueError(f"micro must be one of ['species', 'taxa', None], got {micro} instead.")

def _shap_avg_by_trajectories(data, specs):
    """
    Compute the average of the absolute SHAP values for each trajectory.
    Stores the result in data['shap_values_abs_mean'].
    """
    data['shap_values_abs_mean'] = pd.DataFrame(data['shap_values'], columns=specs['features']).abs().mean(axis=0).sort_values(ascending=False)
    return data, specs

def _shap_micro_avg_by_species(data, specs):
    """
    Compute the micro average of the absolute SHAP values for each species.
    Stores the result in data['shap_values_abs_mean_species'].
    """
    shap_values_abs = pd.DataFrame(data['shap_values'], columns=specs['features']).abs()
    cat = pd.Series(data['y_short'])
    cat_to_species = specs['labels']
    shap_values_abs['species'] = cat.apply(lambda x: cat_to_species[x]).values
    shap_values_species = shap_values_abs.groupby('species').mean()
    data['shap_values_abs_species'] = shap_values_species
    data['shap_values_abs_species_mean'] = shap_values_species.mean(axis=0).sort_values(ascending=False)
    return data, specs

def _shap_micro_avg_by_taxa(data, specs):
    """
    Compute the micro average of the absolute SHAP values for each taxa.
    Stores the result in data['shap_values_abs_mean_taxa'].
    """
    if 'shap_values_abs_species' not in data:
        data, specs = _shap_micro_avg_by_species(data, specs)
    shap_values_species = data['shap_values_abs_species']
    species_to_taxa = preprocessing.get_species_to_taxa()
    shap_values_species['taxa'] = shap_values_species.index.map(species_to_taxa)
    shap_values_taxa = shap_values_species.groupby('taxa').mean()
    data['shap_values_abs_taxa'] = shap_values_taxa
    data['shap_values_abs_taxa_mean'] = shap_values_taxa.mean(axis=0).sort_values(ascending=False)
    return data, specs

@savedata
def feature_importances_CI(alpha=0.05, R=10000, common_origin_distance=False, weather='all', resample_species=True, seed=0, **kwargs):
    """
    Computes CI of feature importances using bootstrap. The CI is computed using the percentile method.
    Statistic: for each species, compute the median feature importances.
               The statistic of interest is the mean of the medians.
    """
    np.random.seed(seed)

    data, specs = feature_importances_shap_prunner(common_origin_distance=common_origin_distance, weather=weather, **kwargs)
    data, specs = join_feature_effects(data, specs)

    X = np.abs(data['shap_values'])
    features = specs['features']
    label = data['y_short']
    label_categories = np.unique(label)
    base_stat = nb_funcs.avg_of_medians(X, label, label_categories)
    X_s = pd.Series(list(X), index=label)
    blocks = tuple(X_s.groupby(label).apply(lambda x: np.vstack(x.values)).values)

    result = np.empty((R, X.shape[1]))
    @njit
    def resample_single_block(block):
        return [b[np.random.randint(low=0, high=b.shape[0], size=b.shape[0])] for b in block]

    if resample_species:
        num_blocks = len(blocks)
        def iteration():
            resample_blocks_idxs = np.random.randint(low=0, high=num_blocks, size=num_blocks)
            block_species_resampling = [blocks[i] for i in resample_blocks_idxs]
            blocks_i = resample_single_block(block_species_resampling)
            L = np.array([b.shape[0] for b in blocks_i])
            labels_i = np.repeat(resample_blocks_idxs, L)
            X_i = np.concatenate(blocks_i)
            return nb_funcs.avg_of_medians(X_i, labels_i, np.unique(resample_blocks_idxs))
    else:
        def iteration():
            block_i = resample_single_block(blocks)
            L = np.array([b.shape[0] for b in block_i])
            labels_i = np.repeat(label_categories, L)
            X_i = np.concatenate(block_i)
            return nb_funcs.avg_of_medians(X_i, labels_i, label_categories)

    for i in tqdm(range(R)):
        result[i] = iteration()

    CI = pd.Series([bootstrap._compute_CI_percentile(result[:, i], alpha, 'two-sided') for i in range(X.shape[1])], index=features, name="CI")
    base_stat = pd.Series(base_stat, index=features, name='base_stat')
    df = pd.concat([base_stat, CI], axis=1).sort_values('base_stat', ascending=False)
    return df

@savefig('micro+CI')
def feature_importance_setting_comparison(micro='species', max_display=10, offset=2.5, bar_halfwidth=0.4, legend_sep=45, CI=True, **CI_kwargs):
    settings = params.feature_importance_settings
    shap_data = {}
    errors = {}
    for k, v in settings.items():
        if CI:
            data = feature_importances_CI(**v, **CI_kwargs)
            shap = data['base_stat']
            shap.name = None
            error = data['CI'].rename(index=params.feature_map)
        else:
            data, _ = shap_abs_avg_prunned(**v)
            if micro is None:
                shap = data['shap_values_abs_mean']
            elif micro == 'species':
                shap = data['shap_values_abs_species_mean']
            else:
                shap = data['shap_values_abs_taxa_mean']
            error = None
        # Format feature names
        shap = shap.rename(index=params.feature_map)
        shap_data[k] = shap
        errors[k] = error

    shap_common_origin = shap_data['common_origin']
    shap_env = shap_data['env']
# Sort them according to common origin and put the ranking in number for the two setups.
    df_common_origin = shap_common_origin.reset_index().rename(columns={'index': 'feature', 0: 'shap'})
    df_common_origin = df_common_origin.reset_index().rename(columns={'index': 'rank'})
    df_env = shap_env.reset_index().rename(columns={'index': 'feature', 0: 'shap'})
    rank_to_feature = df_env.sort_values('shap', ascending=False).reset_index(drop=True).feature.to_dict()
    feature_to_rank = {v: k for k, v in rank_to_feature.items()}
    df_env['rank'] = df_env.feature.apply(lambda x: feature_to_rank[x])
    order = df_common_origin.feature.to_list()[:max_display]
    df_common_origin = df_common_origin.iloc[:max_display]
    df_env = df_env.set_index('feature').loc[order].reset_index()

    max_shap_value = max(df_common_origin.shap.max(), df_env.shap.max())
    max_shap = max_shap_value + offset
    fig = get_figure(xaxis_title="Feature importance (⟨|SHAP|⟩)", xaxis_range=[-max_shap-0.1, max_shap+0.1])
    fig.update_layout(plot_bgcolor='white',
                      yaxis=dict(visible=False),
                      xaxis=dict(showline=True, linecolor='black', linewidth=2.4))
# reverse the ordering for plotting
    df_common_origin = df_common_origin.iloc[::-1]
    df_env = df_env.iloc[::-1]
# first plot feature names in the center
    num_features = df_common_origin.shape[0]
    fig.add_trace(go.Scatter(x=np.zeros((num_features)), y=np.arange(num_features), text=df_common_origin.feature, mode='text', textposition='middle center', textfont=dict(color='black', size=30), showlegend=False))
    colors = {k:c for k, c in zip(settings.keys(), plotly_default_colors())}
    for i, shap in enumerate(df_common_origin.shap):
        fig.add_shape(type='rect',
                      x0=offset,
                      x1=offset + shap,
                      y0=i-bar_halfwidth,
                      y1=i+bar_halfwidth,
                      fillcolor=colors['common_origin']
                      )
    for i, shap in enumerate(df_env.shap):
        fig.add_shape(type='rect',
                      x0=-offset,
                      x1=-offset - shap,
                      y0=i-bar_halfwidth,
                      y1=i+bar_halfwidth,
                      fillcolor=colors['env']
                      )
    # plot rank # on the right for common_origin and left for env
    for i, rank in enumerate(df_common_origin['rank']+1):
        fig.add_trace(go.Scatter(x=[offset-0.05], y=[i], text=[rank], mode='text', textposition='middle left', textfont=dict(color=colors['common_origin'], size=24), showlegend=False))
    for i, rank in enumerate(df_env['rank']+1):
        fig.add_trace(go.Scatter(x=[-offset + 0.05], y=[i], text=[rank], mode='text', textposition='middle right', textfont=dict(color=colors['env'], size=24), showlegend=False))

    # Add horizontal legend on top
    legend_items = [
        dict(name='Real location', traceindex=1, marker=dict(color=colors['env']), mode='markers'),
        dict(name=' '*legend_sep, traceindex=None, mode='none'),  # Blank space
        dict(name='Common origin', traceindex=0, marker=dict(color=colors['common_origin']), mode='markers'),
    ]
    for item in legend_items:
        if item['mode'] != 'none':
            item['marker']['size'] = 30
            item['marker']['symbol'] = 'square'
            fig.add_trace(go.Scatter(x=[None], y=[None], mode=item['mode'], name=item['name'], marker=item.get('marker')))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode=item['mode'], name=item['name']))
    fig.update_layout(legend=dict(x=0.5, y=1.05, orientation='h', xanchor='center', yanchor='bottom', font=dict(size=28)))

    # update xaxis ticks
    xrange = np.arange(0, max_shap_value+1, 1)
    xrange_vals = xrange + offset
    tickvals = np.concatenate([-xrange_vals[::-1], xrange_vals])
    ticktext = np.concatenate([xrange[::-1], xrange])
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=24))

    if CI:
        # plot error bars
        def plot_errorbar(fig, x, y, color='black', lw=1.5, error_halfwidth=0.07):
            fig.add_shape(type='line',
                          x0=x,
                          x1=x,
                          y0=y-error_halfwidth,
                          y1=y+error_halfwidth,
                          line=dict(color=color, width=lw)
                          )
            return
        for i, ci in enumerate(errors['common_origin'][order].values[::-1]):
            ci = ci.squeeze()
            fig.add_shape(type='line',
                          x0=offset + ci[0],
                          x1=offset + ci[1],
                          y0=i,
                          y1=i,
                          line=dict(color='black', width=2)
                          )
            plot_errorbar(fig, offset + ci[0], i)
            plot_errorbar(fig, offset + ci[1], i)

        for i, ci in enumerate(errors['env'][order].values[::-1]):
            ci = ci.squeeze()
            fig.add_shape(type='line',
                          x0=-offset - ci[0],
                          x1=-offset - ci[1],
                          y0=i,
                          y1=i,
                          line=dict(color='black', width=2)
                          )
            plot_errorbar(fig, -offset - ci[0], i)
            plot_errorbar(fig, -offset - ci[1], i)
        fig.update_layout(xaxis_range=[-max_shap-0.2, max_shap+0.2])
    return fig

def corr_acc_vs_feature_pairwise_corr(feature='mean effort'):
    pos_cols = ['median latitude', 'median longitude']
    mag_cols = ['mean effort',  'mean SST anomaly']
    acc_cols = ['acc_deviation']
    @njit
    def great_circle_distance(lat_0, lon_0, lat_f=None, lon_f=None):
        sigma = 2*np.arcsin(np.sqrt(np.sin(0.5*(lat_f-lat_0))**2 + np.cos(lat_f)*np.cos(lat_0)*np.sin(0.5*(lon_f - lon_0))**2))
        return sigma

    def _compute_all_diffs(pos1, mag1, acc1, pos2, mag2, acc2):
        n1, n2 = pos1.shape[0], pos2.shape[0]
        num_magnitudes = mag1.shape[1]
        diffs = np.empty((n1, n2, num_magnitudes + 1))
        eff = np.empty((n1, n2))
        for i in range(n1):
            diffs[i,:,0] = great_circle_distance(pos1[i, 0], pos1[i, 1], pos2[:, 0], pos2[:, 1])
            effort = mag1[i, 0] - mag2[:, 0] # shape n2
            swap = effort > 1
            effort[swap] = 1 / effort[swap]
            eff[i] = np.fmax(mag1[i, 0], mag2[:, 0])
            diffs[i,:, 1] = 1 - effort
            diffs[i,:,2] = mag1[i, 1] - mag2[:, 1] # sst
        distance, effort_ratio, sst = diffs.T
        # expand acc1 and acc2 to be of shape (n1, n2)
        acc1 = np.repeat(acc1, n2).reshape((n1, n2))
        acc2 = np.tile(acc2, n1).reshape((n1, n2))
        return distance.ravel(), effort_ratio.ravel(), sst.ravel(), acc1.ravel(), acc2.ravel(), eff.ravel()
    def compute_diffs(df1, df2):
        pos1 = df1[pos_cols].values
        pos2 = df2[pos_cols].values
        mag1 = df1[mag_cols].values
        mag2 = df2[mag_cols].values
        acc1 = df1[acc_cols].values
        acc2 = df2[acc_cols].values
        distance, effort_ratio, sst, acc1, acc2, effort = _compute_all_diffs(pos1, mag1, acc1, pos2, mag2, acc2)
        return pd.DataFrame(dict(distance=distance, effort_ratio=effort_ratio, sst=sst, acc1=acc1, acc2=acc2, effort=effort))
    c = corr_with_accuracy_micro()
    cf = c[feature].sort_values().dropna()

    dataset, target = error_analysis_dataset(v2=True, weather='all', common_origin_distance=False, random_states=range(1,6), species_level=False, exclude_cols=['sex'])
    df = pd.concat([dataset, target], axis=1)
    df['species'] = preprocessing.map_ID_to_species(df.index.values)

    # TODO: modify it for other features
    df = df[['species', 'mean effort', 'Accuracy', 'median latitude', 'median longitude', 'mean SST anomaly']]
    df['median latitude'] *= np.pi / 180
    df['median longitude'] *= np.pi / 180

    acc_deviation = df.groupby('species').apply(lambda s: s.Accuracy - s.Accuracy.mean())
    acc_deviation.index = acc_deviation.index.droplevel(0)
    df['acc_deviation'] = acc_deviation

    species = cf.index
    N = len(species)
    diffs = {}
    for i, s1 in enumerate(tqdm(species)):
        df1 = df[df.species == s1]
        if i < N - 1:
            for s2 in species[i+1:]:
                df2 = df[df.species == s2]
                diffs[(s1, s2)] = compute_diffs(df1, df2)
    diffs = pd.Series(diffs)

    def prune_by_effort(df, min_effort=500, threshold=0.1, distance_threshold=0.35):
        """
        Selects regions with same effort and ensures they correspond to the same location (distance < distance_threshold).
        This way any correlation between effort and accuracy is not due to the fact that low difference in effort occurred in different regions.

        Distance threshold is in radians. To convert it to km multiply by the radius of the Earth.
        Example: 0.35 * 6370 = 2200 km
        """
        # TODO: calculate deviations in magnitudes w.r.t. the mean for each species.
        # TODO: then calculate the distances between pairs of species.
        # TODO: Verify if in regions with same effort, one species tend to increase its acc and the other to decrease it. This is, the differences in the deviations in accuracy.
        return df[(df.effort_ratio > threshold).values & (df.distance < distance_threshold).values & (df.effort > min_effort)]

    diffs_prunned = diffs.apply(prune_by_effort)
    diffs_prunned_len = diffs_prunned.apply(lambda df: df.shape[0])
    diffs_prunned = diffs_prunned[diffs_prunned_len > 1]
    corrs = diffs_prunned.apply(lambda df: corr_matrix_pruned(df, method='spearman').loc['acc1', 'acc2'])
    cf_up = cf > 0
    return corrs.dropna().unstack(), cf_up

@savefig
def corr_acc_vs_feature_pairwise_corr_plot(**kwargs):
    df, cf_up = corr_acc_vs_feature_pairwise_corr(**kwargs)
    species_to_color = cf_up.replace({True: 'red', False: 'blue'})
    def color_HTML(color, text):
        # color: hexadecimal
        s = "<span style='color:" + str(color) + "'>" + str(text) + "</span>"
        return s
    fig = px.imshow(df)
    ticksize = 24
    fig.update_layout(
        autosize=False,
        width=1050,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        font_size=32,
        xaxis=dict(title='Species j', tickfont=dict(size=ticksize)),
        yaxis=dict(title='Species i', tickfont=dict(size=ticksize)),
        coloraxis=dict(cmin=-1, cmax=1, colorbar=dict(title='Corr(acc<sub>i</sub>, acc<sub>j</sub>)<br>', tickfont=dict(size=ticksize+4),
                                                       x=1.0, y=0.5, yanchor='middle', len=0.5
                                                       ),
                       colorscale='RdBu_r'),
    )
    fig.update_xaxes(tickangle=90)
    # write ticks as species names with color
    xaxes_text = [color_HTML(c, t) for (t, c) in species_to_color.loc[df.columns].items()]
    yaxes_text = [color_HTML(c, t) for (t, c) in species_to_color.loc[df.index].items()]
    fig.update_xaxes(ticktext=xaxes_text, tickvals=np.arange(len(xaxes_text)), tickfont_size=ticksize)
    fig.update_yaxes(ticktext=yaxes_text, tickvals=np.arange(len(yaxes_text)), tickfont_size=ticksize)
    return fig

def tracking_device_effect_in_acc(alpha=0.1):
    dataset, target = error_analysis_dataset(v2=True, weather='all', common_origin_distance=False, random_states=range(1,6), species_level=True)

    df = pd.concat([dataset, target], axis=1)
    tracking_cols = ['Argos', 'Fastloc GPS', 'GLS', 'GPS']
    df_tracking = df[tracking_cols]
    tracking_rest = ['Fastloc GPS', 'GLS', 'GPS']
    shares_ARGOS = df_tracking.Argos > 0 & (df_tracking[tracking_rest].sum(axis=1) > 0)
    c_ARGOS = corr_matrix_pruned(df[shares_ARGOS], method='spearman', alpha=alpha)
    c_significant = c_ARGOS.loc['Accuracy', 'Argos']
    c_not_pruned = df[shares_ARGOS].corr('spearman')['Accuracy'].dropna().sort_values()
    c = c_not_pruned.loc['Argos']
    return dict(c=c, c_significant=c_significant)

@savedata
def fine_tuning_stage_learning(weather='coast-d+bathymetry', common_origin_distance=True, pad_day_rate=6, species_train='Black-browed albatross', invert_lat=False, delete_features=[], model='inception', nf=32, stage_remap='binary', species_transfer_acc='all', remapped=False):
    """
    nf = nb_filters for inception or n_feature_maps for resnet.
    The goal is to measure the accuracy when trained in species_transfer_acc and fine tuned in species_train.
    """
    if stage_remap == 'binary':
        func = preprocessing.relabel_stages_binary

    if species_transfer_acc == 'all':
        species_transfer_acc = [s for s in params.stage_species_v2 if s != species_train]
    elif isinstance(species_transfer_acc, str):
        species_transfer_acc = [species_transfer_acc]
    elif isinstance(species_transfer_acc, float):
        min_transfer_acc = deepcopy(species_transfer_acc)
        if remapped:
            df = stage_transfer_learning_matrix(model=model, pad_day_rate=pad_day_rate, weather=weather, common_origin_distance=common_origin_distance, stage_remap=stage_remap, negative_to=None, cluster=False, delete_features=delete_features, invert_lat=invert_lat)
        else:
            df = stage_transfer_learning_binary_matrix_acc(pad_day_rate=pad_day_rate, weather=weather, common_origin_distance=common_origin_distance, delete_features=delete_features, invert_lat=invert_lat)
            assert model == 'inception', "Only inception model is supported for remapped=False."
        species_transfer_acc = df[species_train].dropna() # species that have transfer_acc towards species_train
        species_transfer_acc = species_transfer_acc[species_transfer_acc.abs() > min_transfer_acc]
        switched_labels = species_transfer_acc < 0
        species_switched = species_transfer_acc[switched_labels].index.to_list()
        species_normal = species_transfer_acc[~switched_labels].index.to_list()
        if stage_remap == 'binary':
            species_transfer_acc = species_normal + species_switched
            if not species_transfer_acc:
                raise RuntimeError(f"No species with transfer accuracy > {min_transfer_acc} for test species {species_train}.")
            if species_switched:
                warnings.warn(f"Species {species_switched} have switched labels. Relabeling them.", RuntimeWarning)
                def func(df):
                    df = preprocessing.relabel_stages_binary(df)
                    switch_map = {'non-breeding': 'breeding', 'breeding': 'non-breeding'}
                    for s in species_switched:
                        df.loc[df.COMMON_NAME == s, 'Stage'] = df.loc[df.COMMON_NAME == s, 'Stage'].map(switch_map)
                    return df
        elif stage_remap == 'not-in-train':
            stages_transfer_acc = species_normal

    elif not isinstance(species_transfer_acc, (list, tuple)):
        raise TypeError(f"species_transfer_acc must be a list, tuple, float or 'all'. Got {type(species_transfer_acc)}.")

    species_stage = species_transfer_acc + [species_train]
    if stage_remap == 'not-in-train':
        # TODO: relabel_stages with biological knowledge
        warnings.warn("Relabels not down with biological knowledge.", UserWarning)
        func = lambda df: preprocessing.relabel_stages(df, species_stage, remap_breeding=False)
    elif stage_remap != 'binary':
        raise NotImplementedError(f"stage_remap {stage_remap} not implemented.")

    split_by = dict(column=['COMMON_NAME'], colvalue=[species_train])
    prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, common_origin_distance=common_origin_distance, species_stage=species_stage, delete_features=delete_features, pad_day_rate=pad_day_rate, fill_pad_day_rate=False, scale_by_velocity=True, split_by=split_by)

    if pad_day_rate in params.pad_day_rate_to_maxlen:
        maxlen = params.pad_day_rate_to_maxlen[pad_day_rate]
    else:
        tree = models.DecisionTree(**kwargs)
        maxlen_data = tree.labels_side.groupby('COMMON_NAME').apply(lambda S: np.percentile(S.Length.values, 90))
        maxlen_data[species_train] = np.percentile(tree.labels.Length.values, 90)
        maxlen = int(maxlen_data.median())
        warnings.warn(f"maxlen not in params.pad_day_rate_to_maxlen. Using median of 90th percentile of lengths: {maxlen}", RuntimeWarning)
    if model == 'tree':
        clf = models.DecisionTree(**kwargs)
    elif model == 'inception':
        clf = models.InceptionTime(**kwargs, nb_filters=nf, maxlen=maxlen, get_input_len_from='maxlen')
    else:
        raise NotImplementedError(f"model {model} not implemented.")

    results = defaultdict(list)
    print(f"Training with {species_transfer_acc}")
    _ = clf.train(epochs=80, test_size=0.1, verbose=0)

    acc = clf.evaluator()[-1]
    print(f"Accuracy in species_transfer_acc: {acc}")
    clf.get_cat_to_label()
    print(f"Categories and label: {clf.cat_to_label}")
    df = clf.classification_report(partition='side')
    results['zero-shot-acc'] = (df.Stage == df.Predicted).mean()
    results['zero-shot-loss'] = clf.loss_fn(clf.y_side, clf.model(clf.X_side)).numpy()
    results['base-acc'] = pd.Series(clf.y_side_short).value_counts(normalize=True).iloc[0]

    print("Fine tuning")
    n_splits_side = 5
    kfold = StratifiedKFold(n_splits=n_splits_side, shuffle=True, random_state=1)
    model_cp = tf.keras.models.clone_model(clf.model)
    compiler_kwargs_cp = deepcopy(clf.compiler_kwargs)
    clf_report = []
    for fold_idx, (train, test) in enumerate(kfold.split(clf.X_side, clf.y_side_short), start=1):
        print(f"Fold {fold_idx}")
        clf.split_side_set(train, test)

        save_best_model = tf.keras.callbacks.ModelCheckpoint(clf.weights_path, save_best_only=True, save_weights_only=True, monitor=clf.performance_monitor)
        base_callbacks = [save_best_model]#, tensorboard_callback]
        callbacks = clf.callbacks + base_callbacks
        clf.model = tf.keras.models.clone_model(model_cp)
        compiler_kwargs = deepcopy(compiler_kwargs_cp)
        clf.model.compile(**compiler_kwargs)
        history = clf.model.fit(
                    clf.X_side_train,
                    clf.y_side_train,
                    batch_size=64,
                    epochs=80,
                    verbose=0,
                    validation_data = (clf.X_side_test, clf.y_side_test),
                    callbacks = callbacks,
                    class_weight = clf.class_weights_computer() if hasattr(clf, "class_weights_computer") else clf.class_weights
                )
        clf.model.load_weights(clf.weights_path) # retain best model
        if clf.delete_weights:
            os.remove(clf.weights_path)
            shutil.rmtree(clf.log_dir, ignore_errors=True)
        else:
            clf.weights_path = nn_utils.get_weight_path(clf.weights_dir)

        loss = history.history['val_loss']
        acc = history.history[f'val_{clf.metrics[0]}']
        lowest_loss = np.argmin(loss)
        results['fine-tuned-acc'].append(acc[lowest_loss])
        results['fine-tuned-loss'].append(loss[lowest_loss])
        clf_report.append(clf.classification_report(partition='side_test', recompute_pred=True))

    clf_report = pd.concat([df.assign(fold=fold_idx) for fold_idx, df in enumerate(clf_report, start=1)], axis=0)

    return pd.Series(results), clf_report

@savedata
def stage_clf_performance_kfold(weather='coast-d+bathymetry', common_origin_distance=True, pad_day_rate=6, species_train='Black-browed albatross', invert_lat=False, delete_features=[], model='inception', nf=32, stage_remap='binary'):
    """
    nf = nb_filters for inception or n_feature_maps for resnet.
    The goal is to measure the accuracy when trained in species_train (kfold).
    """
    if stage_remap == 'binary':
        func = preprocessing.relabel_stages_binary
    else:
        # TODO: relabel_stages with biological knowledge
        warnings.warn("Relabels not down with biological knowledge.", UserWarning)
        func = lambda df: preprocessing.relabel_stages(df, species_train, remap_breeding=False)

    species_stage = [species_train]
    prunning_function=preprocessing.get_prunning_function(label='Stage', func=func)
    kwargs = dict(v2=True, weather=weather, prunning_function=prunning_function, common_origin_distance=common_origin_distance, species_stage=species_stage, delete_features=delete_features, pad_day_rate=pad_day_rate, fill_pad_day_rate=False, scale_by_velocity=True)

    if pad_day_rate in params.pad_day_rate_to_maxlen:
        maxlen = params.pad_day_rate_to_maxlen[pad_day_rate]
    else:
        tree = models.DecisionTree(**kwargs)
        maxlen_data = tree.labels_side.groupby('COMMON_NAME').apply(lambda S: np.percentile(S.Length.values, 90))
        maxlen_data[species_train] = np.percentile(tree.labels.Length.values, 90)
        maxlen = int(maxlen_data.median())
        warnings.warn(f"maxlen not in params.pad_day_rate_to_maxlen. Using median of 90th percentile of lengths: {maxlen}", RuntimeWarning)
    if model == 'tree':
        clf = models.DecisionTree(**kwargs)
    elif model == 'inception':
        clf = models.InceptionTime(**kwargs, nb_filters=nf, maxlen=maxlen, get_input_len_from='maxlen')
    else:
        raise NotImplementedError(f"model {model} not implemented.")

    results = defaultdict(list)
    print(f"Training with {species_train}")
    train_history, test_history = clf.train(epochs=80, use_kfold=True, verbose=0)

    acc = clf.evaluator()[-1]
    print(f"Accuracy: {acc}")
    clf_report = clf.classification_report(fold_idx=range(5), save=False)
    return test_history, clf_report

def stage_binary_baseline_loss():
    *_, df_side, df_train = stage_clf_report_binary(species_train='Black-browed albatross')
    baseline_loss = df_side.groupby('COMMON_NAME').apply(compute_binary_baseline_loss)
    baseline_loss['Black-browed albatross'] = compute_binary_baseline_loss(df_train)
    return baseline_loss

def stage_binary_baseline_acc():
    *_ , df_side, df_train = stage_clf_report_binary(species_train='Black-browed albatross')
    baseline_acc = df_side.groupby('COMMON_NAME').apply(compute_binary_baseline_acc)
    baseline_acc['Black-browed albatross'] = compute_binary_baseline_acc(df_train)
    return baseline_acc

def compute_binary_baseline_loss(df):
    p_majority = df.Stage.value_counts(normalize=True).max()
    loss = - (p_majority * np.log(p_majority) + (1 - p_majority) * np.log(1 - p_majority))
    return loss

def compute_binary_baseline_acc(df):
    p_majority = df.Stage.value_counts(normalize=True).max()
    return p_majority

def stage_transfer_learning_summary(fine_tuning_transfer_acc='all', max_base_acc=0.95, pad_day_rate=None, weather=None, common_origin_distance=True, stage_remap='binary', **kwargs):
    specs = dict(pad_day_rate=pad_day_rate, weather=weather, common_origin_distance=common_origin_distance, **kwargs)
    acc = {}
    loss = {}
    if stage_remap == 'binary':
        zero_shot_acc, zero_shot_loss, zero_shot_is_switched, zero_shot_loss_std, zero_shot_acc_std = stage_transfer_learning_binary_matrix(transfer=False, **specs)
        species_iter = params.stage_species_v2
    elif stage_remap == 'not-in-train':
        zero_shot_acc, zero_shot_loss, zero_shot_loss_std, zero_shot_acc_std, zero_shot_baseline_acc, zero_shot_baseline_loss, zero_shot_num_categories = stage_transfer_learning_multi_matrix(**specs)
        species_iter = params.stage_species_v2_multilabel
    else:
        raise NotImplementedError(f"stage_remap {stage_remap} not implemented.")

    for species_train in species_iter:
        # fine tuning
        result_fine_tuning = fine_tuning_stage_learning(species_train=species_train, skip_computation=True, species_transfer_acc=fine_tuning_transfer_acc, stage_remap=stage_remap, **specs)
        if not isinstance(result_fine_tuning, SavedataSkippedComputation):
            fine_tuning = result_fine_tuning[0]
            acc[(species_train, 'fine-tuned')] = np.mean(fine_tuning['fine-tuned-acc'])
            acc[(species_train, 'fine-tuned-std')] = np.std(fine_tuning['fine-tuned-acc'])
            loss[(species_train, 'fine-tuned')] = np.mean(fine_tuning['fine-tuned-loss'])
            loss[(species_train, 'fine-tuned-std')] = np.std(fine_tuning['fine-tuned-loss'])

        # species only
        result_species_only = stage_clf_performance_kfold(species_train=species_train, skip_computation=True, stage_remap=stage_remap, **specs)
        if not isinstance(result_species_only, SavedataSkippedComputation):
            species_only = result_species_only[0]
            acc[(species_train, 'species-only')] = np.mean(species_only['acc-lowest-loss'])
            acc[(species_train, 'species-only-std')] = np.std(species_only['acc-lowest-loss'])
            loss[(species_train, 'species-only')] = np.mean(species_only['loss'])
            loss[(species_train, 'species-only-std')] = np.std(species_only['loss'])
            if stage_remap == 'not-in-train':
                df_report = result_species_only[1]
                acc[(species_train, 'species-only-baseline')] = compute_binary_baseline_acc(df_report)
                loss[(species_train, 'species-only-baseline')] = compute_binary_baseline_loss(df_report)

        # zero-shot learning (train in another species and then test it in species_train)
        if stage_remap == 'not-in-train':
            best_transfer_acc = (zero_shot_acc[species_train] - zero_shot_baseline_acc[species_train]).idxmax()
            best_transfer_loss = (zero_shot_loss[species_train] - zero_shot_baseline_loss[species_train]).idxmin()
            acc[(species_train, 'zero-shot-baseline')] = zero_shot_baseline_acc[species_train][best_transfer_acc]
        else:
            best_transfer_acc = zero_shot_acc[species_train].idxmax()
            best_transfer_loss = zero_shot_loss[species_train].idxmin()
        if isinstance(best_transfer_acc, str):
            acc[(species_train, 'zero-shot')] = zero_shot_acc[species_train][best_transfer_acc]
            acc[(species_train, 'zero-shot-std')] = zero_shot_acc_std[species_train][best_transfer_acc]
        else:
            acc[(species_train, 'zero-shot')] = np.nan
            acc[(species_train, 'zero-shot-std')] = np.nan
        if isinstance(best_transfer_loss, str):
            loss[(species_train, 'zero-shot')] = zero_shot_loss[species_train][best_transfer_loss]
            loss[(species_train, 'zero-shot-std')] = zero_shot_loss_std[species_train][best_transfer_loss]
        else:
            loss[(species_train, 'zero-shot')] = np.nan
            loss[(species_train, 'zero-shot-std')] = np.nan

    acc = pd.Series(acc).unstack()
    loss = pd.Series(loss).unstack()
    if stage_remap == 'binary':
        acc['base'] = stage_binary_baseline_acc()
        loss['base'] = stage_binary_baseline_loss()
        valid = acc['base'] < max_base_acc
        acc = acc[valid]
        loss = loss[valid]
    return acc, loss

@savefig
def stage_transfer_learning_summary_plot(mode='accuracy', plot_type='bar', weather=None, delete_features=[], fine_tuning_transfer_acc='all', common_origin_distance=True, pad_day_rate=None, **kwargs):
    acc, loss = stage_transfer_learning_summary(weather=weather, delete_features=delete_features, fine_tuning_transfer_acc=fine_tuning_transfer_acc, common_origin_distance=common_origin_distance, pad_day_rate=pad_day_rate, **kwargs)

    if plot_type == 'bar':
        plotter = go.Bar
    elif plot_type == 'scatter':
        plotter = go.Scatter
    if mode == 'accuracy':
        magnitude = acc
        upper_bound = 1
        yaxis_range = [0.45, 1.05]
        # ascending_sort = True
    elif mode == 'loss':
        magnitude = loss
        upper_bound = np.inf
        yaxis_range = None
        # ascending_sort = False
    else:
        raise ValueError(f"mode {mode} not recognized.")
    lower_bound = 0

    # sort by acc
    idx = acc.sort_values(by='species-only', ascending=True).index.to_list()
    magnitude = magnitude.loc[idx]

    categories = ['Base', 'Zero-shot', 'Same-Species', 'Fine-tuned']
    colors = {k: c for k, c in zip(categories, plotly_default_colors())}
    symbols = {k: i for i, k in enumerate(categories)}
    cat_to_col = {
                  'Zero-shot': dict(col='zero-shot', error=True, ms=18),
                  'Same-Species': dict(col='species-only', error=True, ms=18),
                  # 'Fine-tuned': dict(col='fine-tuned', error=True, ms=18),
                  'Base': dict(col='base', error=False, ms=14),
                  }

    def plot_single_species(species, showlegend=False):
        data_species = magnitude.loc[species]
        cat_to_val = pd.Series({k: data_species[col_data['col']] for k, col_data in cat_to_col.items()})
        order = cat_to_val.sort_values(ascending=False).index.to_list()
        for cat in order:
            col_data = cat_to_col[cat]
            col = col_data['col']
            if plot_type == 'scatter':
                kwargs = dict(mode='markers', marker_size=col_data['ms'], marker_symbol=symbols[cat])
            else:
                kwargs = dict(width=0.4)
            if col_data['error']:
                error = data_species[f"{col}-std"]
                val = data_species[col]
                error_up = np.clip(error, 0, upper_bound - val)
                error_down = np.clip(error, 0, val - lower_bound)
                error_y = dict(type='data', array=[error_up], arrayminus=[error_down], visible=True, color=color_gradient(colors[cat], 'black', 6)[1])
                fig.add_trace(plotter(x=[species], y=[data_species[col]], name=cat, marker=dict(color=colors[cat]), showlegend=showlegend, error_y=error_y, **kwargs))
            else:
                fig.add_trace(plotter(x=[species], y=[data_species[col]], name=cat, marker=dict(color=colors[cat]), showlegend=showlegend, **kwargs))
        return

    fig = get_figure(xaxis_title='Species', yaxis_title=mode.capitalize())
    species = magnitude.index.to_list()
    plot_single_species(species[0], showlegend=True)
    for s in species[1:]:
        plot_single_species(s)
    fig.update_layout(yaxis_range=yaxis_range,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    if plot_type == 'bar':
        fig.update_layout(barmode='overlay')
    return fig

@savefig
def stage_transfer_learning_summary_plot_new(mode='accuracy', weather=None, delete_features=[], fine_tuning_transfer_acc='all', common_origin_distance=True, pad_day_rate=None, stage_remap='binary'):
    """
    Difference: plot magnitudes one next to the other, and baseline as a grey bar.
    """
    acc, loss = stage_transfer_learning_summary(weather=weather, delete_features=delete_features, fine_tuning_transfer_acc=fine_tuning_transfer_acc, common_origin_distance=common_origin_distance, pad_day_rate=pad_day_rate, stage_remap=stage_remap)

    if mode == 'accuracy':
        magnitude = acc
        upper_bound = 1
        if stage_remap == 'binary':
            yaxis_range = [0.45, 1.05]
        elif stage_remap == 'not-in-train':
            yaxis_range = [0, 1.05]
        else:
            raise NotImplementedError(f"stage_remap {stage_remap} not implemented.")
        # ascending_sort = True
    elif mode == 'loss':
        magnitude = loss
        upper_bound = np.inf
        yaxis_range = None
        # ascending_sort = False
    else:
        raise ValueError(f"mode {mode} not recognized.")
    lower_bound = 0

    # sort by acc
    idx = acc.sort_values(by='species-only', ascending=True).index.to_list()
    magnitude = magnitude.loc[idx]

    categories = ['Base', 'Zero-shot', 'Same-Species', 'Fine-tuned']
    colors = {k: c for k, c in zip(categories, plotly_default_colors())}
    cat_to_col = {
                  'Same-Species': dict(col='species-only', error=True, ms=18),
                  'Zero-shot': dict(col='zero-shot', error=True, ms=18),
                  # 'Fine-tuned': dict(col='fine-tuned', error=True, ms=18),
                  # 'Base': dict(col='base', error=False, ms=14),
                  }
    idx_to_int = {idx: i for i, idx in enumerate(idx)}
    width = 0.4
    x = np.arange(len(idx))
    x0 = x - width/2
    x1 = x + width/2

    magnitude['species-only-x'] = x0
    magnitude['zero-shot-x'] = x1
    fig = get_figure(xaxis_title='Species', yaxis_title=mode.capitalize())
    species = magnitude.index.to_list()

    for cat, specs in cat_to_col.items():
        col = specs['col']
        magnitude[f'{col}-error-up'] = np.clip(magnitude[f'{col}-std'], 0, upper_bound - magnitude[col])
        magnitude[f'{col}-error-down'] = np.clip(magnitude[f'{col}-std'], 0, magnitude[col] - lower_bound)
        fig.add_trace(go.Bar(x=magnitude[f'{col}-x'], y=magnitude[col], name=cat, marker=dict(color=colors[cat]), width=width))
        if stage_remap == 'binary':
            base_col = 'base'
        elif stage_remap == 'not-in-train':
            base_col = f'{col}-baseline'
        fig.add_trace(go.Bar(x=magnitude[f'{col}-x'], y=magnitude[base_col], name=cat, marker=dict(color='grey'), width=width, showlegend=False))

    def plot_error(species):
        data_species = magnitude.loc[species]
        x = idx_to_int[species]
        fig.add_trace(go.Scatter(x=[x-width/2], y=[data_species['species-only']], error_y=dict(type='data', array=[data_species['species-only-error-up']], arrayminus=[data_species['species-only-error-down']], color=color_gradient(colors['Same-Species'], 'black', 6)[1], width=4, thickness=3), mode='markers', marker=dict(color=colors['Same-Species'], size=0.5), showlegend=False))
        fig.add_trace(go.Scatter(x=[x+width/2], y=[data_species['zero-shot']], error_y=dict(type='data', array=[data_species['zero-shot-error-up']], arrayminus=[data_species['zero-shot-error-down']], color=color_gradient(colors['Zero-shot'], 'black', 6)[1], width=4, thickness=3), mode='markers', marker=dict(color=colors['Zero-shot'], size=0.5), showlegend=False))
        return

    for species in magnitude.index:
        plot_error(species)

    fig.update_layout(yaxis_range=yaxis_range, barmode='overlay',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      xaxis=dict(tickvals=x, ticktext=idx, tickangle=90))
    return fig

@savefig
def feature_multicollinearity(method='ward', corr_method='spearman'):
    from scipy.cluster.hierarchy import dendrogram, linkage
    corr = preprocessing.feature_corr(method=corr_method)
    corr_linkage = linkage(corr, method=method)
    order = dendrogram(corr_linkage, no_plot=True)['leaves']
    df_ordered = corr.iloc[order, order]
    fig = px.imshow(df_ordered, zmin=-1, zmax=1)
    fig.update_layout(height=1000, width=1300, xaxis_tickfont_size=15, yaxis_tickfont_size=15,
                      xaxis_tickangle=-90,
                      margin=dict(l=100, r=60, t=0, b=100),
                      coloraxis_colorbar=dict(len=0.5, tickfont_size=20, title=dict(text='{} correlation'.format(corr_method.capitalize()), font=dict(size=26))),
                      )
    return fig

def stage_vs_species(weather='all', common_origin_distance=False, switch=True, binary=True):
    """
    Returns a dataframe with the confusion of the species and the stage transfer learning.
    """
    # stage
    cm_acc, _, is_switched, *_2 = stage_transfer_learning_binary_matrix(pad_day_rate=None, delete_features=[], common_origin_distance=True, weather=None)
    cm_acc.fillna(0, inplace=True)
    cm_abs = cm_acc.copy()
    if switch:
        cm_abs[is_switched] *= -1

    # species
    df = classification_report_random_states(weather=weather, common_origin_distance=common_origin_distance, random_states=range(1, 6))
    cm_species, _ = compute_confusion_matrix(df=df, artificial_trajectory=[])
    np.fill_diagonal(cm_species.values, 0)
    cm_species = cm_species.loc[cm_acc.index, cm_acc.columns]

    df = pd.concat([cm_species.stack(), cm_abs.stack()], axis=1)
    df.columns = ['Species confusion', 'Stage transfer']
    if binary:
        df[df > 0] = 1
        df[df < 0] = 0
    return df

def _stage_vs_species_metric(metric, **kwargs):
    df = stage_vs_species(**kwargs)
    base = metric(df.values)
    CI = bootstrap.CI_bca(df.values, metric, R=int(1e5))
    return base, CI

@savedata
def stage_vs_species_lift(binary=True, **kwargs):
    @njit
    def excess_lift(X):
        """
        X: binary matrix
        excess_lift = lift - 1
        The lift is the ratio of the probability of two events occurring together to the probability of them occuring together if they were independent.
        """
        A, B = X.T
        p_A = A.mean()
        p_A_given_B = (A[B > 0] > 0).mean()
        excess = (p_A_given_B / p_A) - 1

        return excess

    return _stage_vs_species_metric(excess_lift, binary=binary, **kwargs)

@savedata
def stage_vs_species_spearman(binary=False, **kwargs):
    @njit
    def rank_data(data):
        """
        Assign ranks to data, handling ties by assigning the mean of the ranks
        that would have been assigned to all the tied values.
        """
        # Sort data and get sorted indices
        sorted_indices = np.argsort(data)
        ranks = np.empty(len(data), dtype=np.float64)

        # Iterate over sorted data, handling ties
        i = 0
        while i < len(data):
            # Detect ties: find the extent of tied values
            tie_extent = 1
            while i + tie_extent < len(data) and data[sorted_indices[i]] == data[sorted_indices[i + tie_extent]]:
                tie_extent += 1

            # Average rank for ties
            average_rank = i + 0.5 * (tie_extent - 1) + 1
            for j in range(tie_extent):
                ranks[sorted_indices[i + j]] = average_rank

            i += tie_extent

        return ranks

    @njit
    def spearman_correlation(X):
        """
        Calculates the Spearman rank correlation coefficient between two arrays.
        """
        x, y = X.T
        if len(x) != len(y):
            raise ValueError("Both arrays must be of the same length.")

        rank_x = rank_data(x)
        rank_y = rank_data(y)

        # Compute the covariance of ranks and the standard deviations of ranks
        covariance = np.mean((rank_x - np.mean(rank_x)) * (rank_y - np.mean(rank_y)))
        std_x = np.sqrt(np.mean((rank_x - np.mean(rank_x))**2))
        std_y = np.sqrt(np.mean((rank_y - np.mean(rank_y))**2))

        # Spearman correlation is the covariance of ranks divided by the product of their standard deviations
        return covariance / (std_x * std_y)

    return _stage_vs_species_metric(spearman_correlation, binary=binary, **kwargs)

@savedata
def stage_vs_species_indep_test(**kwargs):
    from scipy.stats import chi2_contingency
    df = stage_vs_species(binary=True, **kwargs)
    contingency = pd.crosstab(df['Species confusion'], df['Stage transfer'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return p

def stage_vs_species_summary(label=None, **kwargs):
    def format(value, CI, ndigits=2):
        value_str = str(round(value, ndigits))
        CI_str = str(CI.squeeze().round(2).tolist())
        return f"{value_str} ({CI_str})"

    p = stage_vs_species_indep_test(**kwargs)
    lift, lift_CI = stage_vs_species_lift(**kwargs)
    lift += 1
    lift_CI += 1
    spearman = stage_vs_species_spearman(**kwargs)
    S = pd.Series({'$\chi^{2}$ p-value': p, 'Lift (CI)': format(lift, lift_CI), 'Spearman (CI)': format(*spearman)})
    if label is not None:
        S.name = label
    return S

def stage_vs_species_summary_table():
    table = pd.concat([stage_vs_species_summary('Geo+env'),
                       stage_vs_species_summary('Common origin', weather=None, common_origin_distance=True)],
                      axis=1).T
    table.index.name = 'Setting'
    table = table.reset_index()
    pd_utils.latex_table(table, index=False)
    return

def dataset_specs():
    """
    Returns a dictionary with the specs of the dataset
    """
    df, labels, _ = preprocessing.load_all_data(v2=True, weather=None, return_labels=True)
    df_exp = preprocessing.load_all_data(v2=True, weather=None, return_labels=False, expand_df=True)

    L = df.apply(lambda x: x.shape[1])
    labels =labels.reset_index()
    num_animals = labels.groupby('COMMON_NAME').size()
    num_points = df_exp.shape[0]
    num_trajectories = labels.shape[0]
    num_taxa = labels['Taxa'].nunique()

    totals  = {'Taxas': num_taxa,
               'Species': num_animals.shape[0],
               'Trajectories': num_trajectories,
               'Observations': num_points,
               'Days': labels['Days in trajectory (all)'].sum(),
               }
    average_by_species = {'Trajectories': num_animals.mean(),
                          'Observations': L.groupby(level=0).mean().mean(),
                          'Days': labels.groupby('COMMON_NAME')['Days in trajectory (all)'].mean().mean(),
                         }
    average_trajectories = {'Observations': L.median(),
                            'Days': labels['Days in trajectory (all)'].median(),
                        }
    summary = {'Total': totals, 'Average (across species)': average_by_species, 'Average (across trajectories)': average_trajectories}
    return summary
