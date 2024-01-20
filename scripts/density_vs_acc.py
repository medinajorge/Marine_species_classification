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
use_kfold = True
n_splits = 5
artificial_trajectory_type = None
artificial_percentage = None
artificial_trajectory_identifier = ""

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
               model=dict(max_depth=25),
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
                  train=dict(epochs=80, batch_size=64, verbose=0)
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
    opts, args = getopt.getopt(sys.argv[1:], "c:v:o:w:a:p:", 
                               ["classifier=", "velocity=", "origin=", "weather=", "artificial-trajectory-type=", "artificial-percentage="]) # argv[0] is the name of the script
except getopt.GetoptError:
    print('check errors')

def check_none(arg):
    if arg.lower() == "none":
        magnitude = None
        magnitude_str = ""
    else:
        magnitude, magnitude_str = arg, arg
    return magnitude, magnitude_str
    
for opt, arg in opts:
    if opt in ("-c", "--classifier"):
        classifier_str = arg
        params = classifier_params[arg]
    elif opt in ('-v', '--velocity'):
        velocity, velocity_str = check_none(arg)
    elif opt in ('-o', '--origin'):
        to_origin, to_origin_str = check_none(arg)
    elif opt in ('-w', '--weather'):
        weather, weather_str = check_none(arg)
    elif opt in ('-a', '--artificial-trajectory-type'):
        if "," in arg:
            artificial_trajectory_type = arg.split(",")
            artificial_trajectory_type_str = "--".join(artificial_trajectory_type)            
        else:
            artificial_trajectory_type, artificial_trajectory_type_str = check_none(arg)
        artificial_trajectory_identifier += f"_{artificial_trajectory_type_str}"
    elif opt in ('-p', '--artificial-percentage'):
        artificial_percentage = float(arg)
        artificial_percentage_str = f'p_{other_utils.encoder(artificial_percentage)}'
        artificial_trajectory_identifier += f"_{artificial_percentage_str}"

print(other_utils.getopt_printer(opts))
identifier = f'velocity-{velocity_str}_to-origin-{to_origin_str}_weather-{weather_str}{artificial_trajectory_identifier}'
print(f"Identifier: {identifier}")

model = params["clf"](velocity=velocity, to_origin=to_origin, weather=weather, artificial_trajectory_type=artificial_trajectory_type, artificial_percentage=artificial_percentage,
                      **params["model"])

print("Training ...")
_ = model.train(use_kfold=use_kfold, n_splits=n_splits, **params["train"])
print("Acc: {:.3f}".format(model.evaluator()[-1]))

print("Storing results ...")
if len(model.input_shape) == 1:
    _ = model.density_accuracy_df(dataname=f"density-acc_kfold-5_{identifier}", save=True, mode="point", del_kfold_data=False)
_ = model.density_accuracy_df(dataname=f"density-acc_kfold-5_{identifier}", save=True, mode="trajectory", del_kfold_data=True)

    
print("Writing log file...")
logDir = fullPath("nuredduna_programmes/log/density_acc")
Path(logDir).mkdir(exist_ok=True, parents=True)
with open(os.path.join(logDir, f"density-acc_{artificial_trajectory_identifier}.txt"), "a") as f:
    f.write(f'{classifier_str}: {identifier} \n')
    f.close()

print("Done!")