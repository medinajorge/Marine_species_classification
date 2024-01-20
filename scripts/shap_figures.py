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

add_dt = True
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

data_visualization.plot_shap(taxa=taxa, velocity=velocity, add_dt=add_dt, to_origin=to_origin)

print("Done!")





