import numpy as np
import pandas as pd
from pathlib import Path
import os
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

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
sys.path.append(os.path.join(RootDir, "utils"))
fullPath = lambda path: os.path.join(RootDir, path)

from utils import file_management, other_utils, data_visualization, preprocessing, nn_utils, models
from models import *
import getopt

# Params
size = 0.3
size_str = other_utils.encoder(size)
add_dt = True
add_dt_str = "_add-dt"

for velocity in ["arch-segment"]: #, "x-y-z"]:
    for to_origin in [None, "all"]: #[None]:#, "all"]: # "space"
        to_origin_str = "" if to_origin is None else to_origin     
        shap_values = []
        expected_value = []
        preds = []    
        X = []
        for fold in tqdm(range(1,4)):
            shap_data = file_management.load_lzma(
                fullPath(f"utils/data/shap_values/XGB/shap_size-{size_str}_20-trees_depth-30_v-{velocity}_{to_origin_str}_size-{size_str}{add_dt_str}fold-{fold}.lzma"
                        )
            )
            shap_values.append(np.array(shap_data["values"]))
            expected_value.append(np.array(shap_data["expected_value"]))
            preds.append(shap_data["preds"])
            X.append(shap_data["X"])
        
        shap_values = np.concatenate(shap_values, axis=1)
        X = np.vstack(X)       

        labels = file_management.load_lzma(fullPath("utils/data/labels_split-by-day_groupby-ID_default.lzma"))
        label_to_cat = {v:k for k,v in shap_data["labels"].items()}
        taxa_labels = {}
        for taxa, df in labels.groupby("Taxa"):
            taxa_labels[taxa] = [label_to_cat[species] for species in set(df["COMMON_NAME"]) if species in label_to_cat.keys()]
            if len(taxa_labels[taxa]) == 0:
                del taxa_labels[taxa]
            
        # SHAP shape: (species, #data, features)
        print("Computing probabilities ...")
        expected_value = expected_value[0]
        num_labels, data_size, num_features = shap_values.shape
        phi_0 = expected_value[:,None, None] * np.ones((expected_value.size, data_size, 1))
        phi = np.concatenate([shap_values, phi_0], axis=-1)
        N = np.exp(phi.sum(axis=-1)).sum(axis=0)[None,:,None]
        delta_log_p = phi[:,:,:-1] - ((np.log(N)-phi_0) / num_features)
        
        # Average and std in the probability displacement caused by each feature
        print("Computing impact of each feature on the probability...")        
        features = XGB(add_dt=add_dt, velocity=velocity, weather="all").features
        taxa_probability_impact ={}
        for taxa, taxa_idxs in tqdm(taxa_labels.items()):
            delta_log_p_taxa = delta_log_p[taxa_idxs]
            delta_log_p_mean = delta_log_p_taxa.mean(axis=1).mean(axis=0)
            delta_log_p_std = delta_log_p_taxa.std(axis=1).mean(axis=0)
            for feature, mean, std in zip(features, delta_log_p_mean, delta_log_p_std):
                taxa_probability_impact[(taxa, feature)] = np.array([mean, std])
        
        print("Saving...")
        file_management.save_lzma(taxa_probability_impact, f"shap_probability_impact_to-origin-{to_origin_str}{add_dt_str}", fullPath("utils/data/shap_values"))

print("Done!")