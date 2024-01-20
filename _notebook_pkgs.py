import numpy as np
import pandas as pd
import math
from numba import njit
import itertools
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from time import time
import random
import os
import matplotlib.gridspec as gridspec
from inspect import signature, getsource
from copy import deepcopy
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
#import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# custom
# import sys
# sys.path.append('utils')
from utils import file_management, other_utils, data_visualization, preprocessing, nn_utils, models, params, acc_by_complexity, acc_by_sample_size, nc_preprocess, geometry
from utils.models import * #DenseNN, DecisionTree, RandomForest, SVM, FCN, ResNet, LSTM
from utils.other_utils import dark_figure, timer, latex_table
from utils import shap_cluster_mod, nb_funcs, error_analysis
import tidypath, phdu
from tidypath import storage

try:
    import tensorflow as tf
    import tensorflow.keras as keras
    num_cores = 0
    tf.config.threading.set_inter_op_parallelism_threads(2) #num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.set_soft_device_placement(True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
except:
    print('tensorflow not installed')
    pass
try:
    import networkx as nx
except:
    pass

import phdu
import tidypath
from tidypath import storage
from phdu.stats import bootstrap, conf_interval
from phdu import savedata, savefig, SavedataSkippedComputation, clustering
from phdu.plots.base import *
from phdu.plots.plotly_utils import *
