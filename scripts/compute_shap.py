#Packages
import numpy as np
import pandas as pd 
from pathlib import Path
from time import time
import os

# custom 
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullpath = lambda path: os.path.join(RootDir, path)
sys.path.append(RootDir)
sys.path.append(os.path.join(RootDir, "utils"))
from utils import file_management, other_utils, data_visualization, preprocessing, nn_utils, models
from models import *
from other_utils import dark_figure, timer
import getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], "v:o:t:T:", 
                               ["velocity=", "origin=", "add-dt=", "taxa="])
except getopt.GetoptError:
    print('check args')

taxa = None
for opt, arg in opts:
    if opt in ('-v', '--velocity'):
        velocity = arg
    elif opt in ('-o', '--origin'):
        if arg.lower() == "none":
            to_origin = None
            to_origin_str = ""
        else:
            to_origin = arg
            to_origin_str = arg
    elif opt in ('-t', '--add-dt'):
        add_dt = arg.lower() in ["yes", "true", "y", "t"]
        add_dt_str = "_add-dt" if add_dt else ""
    elif opt in ('-T', '--taxa'):
        taxa = arg
        
print(other_utils.getopt_printer(opts))

parentDir=fullpath("utils/data/shap_values")
if taxa is not None:
    prunning_function = preprocessing.get_binary_prunning_function(colvalue=taxa)
    parentDir = os.path.join(parentDir, taxa)
else:
    prunning_function = None    

xgb_clf = XGB(num_round=10, max_depth=25, weather="all", velocity=velocity, to_origin=to_origin, add_dt=add_dt, prunning_function=prunning_function)
print("Training...")
xgb_clf.train()
p, c, acc = xgb_clf.evaluator()
print("Accuracy: {:.2f}".format(acc))

print("Computing SHAP values...")
# First we try to compute it using 30% of the data, just in case there is not enough memory in the GPU.
for size in [0.3]: #, 1]: # [0.3, 0.5, 1]:
    print(f"Size: {size}")
    size_str = other_utils.encoder(size)
    filename = f"20-trees_depth-30_v-{velocity}_{to_origin_str}_size-{size_str}{add_dt_str}"
    shap_values, X = xgb_clf.compute_shap(size=size, save=True, dataname=filename, full_set=True, parentDir=parentDir)
    
    #print("Figures...")
    #xgb_clf.shap_plot(shap_values=shap_values, data=X, show_classes=False, figname=filename)
    #xgb_clf.shap_plot(shap_values=shap_values, data=X, show_classes=True, figname=filename)

print("Done!")





