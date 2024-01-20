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

from utils import file_management, other_utils, data_visualization, preprocessing, nn_utils, models, analysis
from utils.models import *
import getopt
from tidypath import fmt
import phdu
from phdu import savedata

# Params
use_kfold = True
n_splits = 5
artificial_trajectory_type = None
artificial_percentage = None
individual_label = False
artificial_trajectory_identifier = ""
min_animals = 5
minlen = 5
random_state = 1
overwrite = False
scale_by_velocity = False
as_image = False
as_image_density = False
as_image_indiv_scaling = False
remove_outliers = True
common_origin_distance = None
weather = None
species = [None]
taxa = None
percentage_delete = None
N = None
mode = None
to_origin = None
v2 = True
stratify = True

classifier_params = dict(
    tree = dict(clf=DecisionTree, model=dict(), train=dict()),
    forest = dict(clf = RandomForest,
                  model = dict(forest_kwargs=dict(n_estimators = 60,
                                                  max_depth = 25)),
                  train = dict()
                 ),
    svm = dict(clf=SVM,
               model=dict(svm_kwargs=dict(C=1)),
               train=dict()
              ),
    xgb = dict(clf=XGB,
               model=dict(max_depth=20), # 25
               train=dict()
              ),
    dense = dict(clf=DenseNN,
                 model=dict(),
                 train=dict(epochs=15, batch_size=2048, verbose=0)),
    fcn = dict(clf=FCN,
               model=dict(maxlen=512),
               train=dict(epochs=80, batch_size=128, verbose=0)
              ),
    resnet = dict(clf=ResNet,
                  model=dict(maxlen=512, n_feature_maps=128),
                  train=dict(epochs=70, batch_size=64, verbose=0)
              ),
    inception = dict(clf=InceptionTime,
                     model=dict(maxlen=512, nb_filters=128),
                     train=dict(epochs=80, batch_size=128, verbose=0)
              ),
    lstm = dict(clf=LSTM,
                model=dict(num_lstm=2, bidirectional=True, maxlen=1000),
                train=dict(epochs=80, batch_size=128, verbose=0)
               ),
    conv_lstm = dict(clf=ConvLSTM,
                     model=dict(maxlen=500, num_lstm=2, bidirectional=True),
                     train=dict(epochs=80, batch_size=128, verbose=0)
                    ),
    deep_conv_lstm = dict(clf=DeepConvLSTM,
                          model=dict(maxlen=500),
                          train=dict(epochs=80, batch_size=128, verbose=0)
                    ),

)

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "c:v:O:r:i:d:I:R:C:w:a:L:S:T:P:N:M:o:V:s:",
                               ["classifier=", "velocity=", "overwrite=", "split-random-state=", "as-image=", "as-image-density=", "as-image-indiv-scaling=", "remove-outliers=", "common-origin-distance=", "weather=", "min-animals=", "minlen=", "species=", "taxa=", "percentage-delete=", "N=", "mode=", "to-origin=", "v2=", "stratify="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-c", "--classifier"):
        classifier = arg
        params = classifier_params[arg]
    elif opt in ("-v", "--velocity"):
        scale_by_velocity = fmt.decoder(arg)
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-r", "--split-random-state"):
        random_state = fmt.decoder(arg)
    elif opt in ("-i", "--as-image"):
        as_image = fmt.decoder(arg)
    elif opt in ("-d", "--as-image-density"):
        as_image_density = fmt.decoder(arg)
    elif opt in ("-I", "--as-image-indiv-scaling"):
        as_image_indiv_scaling = fmt.decoder(arg)
    elif opt in ("-R", "--remove-outliers"):
        remove_outliers = fmt.decoder(arg)
    elif opt in ("-C", "--common-origin-distance"):
        common_origin_distance = fmt.decoder(arg)
    elif opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-a", "--min-animals"):
        min_animals = fmt.decoder(arg)
    elif opt in ("-L", "--minlen"):
        minlen = fmt.decoder(arg)
    elif opt in ("-S", "--species"):
        species = arg.replace('"', '').split(",")
    elif opt in ("-T", "--taxa"):
        taxa = arg
    elif opt in ("-P", "--percentage-delete"):
        percentage_delete = fmt.decoder(arg)
    elif opt in ("-N", "--N"):
        N = fmt.decoder(arg)
    elif opt in ("-M", "--mode"):
        mode = arg
    elif opt in ("-o", "--to-origin"):
        to_origin = fmt.decoder(arg)
    elif opt in ("-V", "--v2"):
        v2 = fmt.decoder(arg)
    elif opt in ("-s", "--stratify"):
        stratify = fmt.decoder(arg)

phdu.getopt_printer(opts)

data = analysis.error_analysis_shap(v2=v2, weather=weather, common_origin_distance=common_origin_distance, random_states=range(1, random_state+1), stratify=stratify, overwrite=overwrite)

print("Done!")
