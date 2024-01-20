import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
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

from utils import file_management, other_utils, data_visualization, preprocessing, nn_utils, models, params
from models import *
import getopt 

try:
    opts, args = getopt.getopt(sys.argv[1:], "b:d:v:l:W:w:m:s:B:E:D:", 
                               ["add-batyhmetry=", "delete-features=", "velocity=", "maxlen=", "weights=", "weather=", "min-days=", "scale=", "batch-size=", "epochs=",
                                "diff="
                               ]) # argv[0] is the name of the script
except getopt.GetoptError:
    print('check errors')
    
# Defaults
add_bathymetry = False
delete_features = ["x", "y", "z"]
velocity = "x-y-z"
maxlen = 60
class_weights = "linear"
weather = "all"
min_days = 1
scale = ["col", "COMMON_NAME"]
batch_size = 256
epochs = 120
diff = False
n_splits = 10

for opt, arg in opts:
    if opt in ("-b", "--bathymetry"):
        add_bathymetry = other_utils.decoder(arg)
    elif opt in ("-d", "--delete-features"):
        if arg.lower() == "none":
            delete_features = []
        else:
            delete_features = [s.replace("-", " ") for s in arg.split(",")]
    elif opt in ("-v", "--velocity="):
        velocity = other_utils.decoder(arg)
    elif opt in ("-l", "--maxlen"):
        maxlen = int(arg)
    elif opt in ("-W", "--weights"):
        class_weights = other_utils.decoder(arg)
    elif opt in ("-w", "--weather"):
        weather = other_utils.decoder(arg)
    elif opt in ("-m", "--min-days"):
        min_days = int(arg)
    elif opt in ("-s", "--scale"):
        scale = other_utils.decoder(arg)
    elif opt in ("-B", "--batch-size"):
        batch_size = int(arg)
    elif opt in ("-E", "--epochs"):
        epochs = int(arg)
    elif opt in ("-D", "--diff"):
        diff = other_utils.decoder(arg)
        
if scale is None:
    scale = ["col", "COMMON_NAME"]
        
other_utils.getopt_printer(opts)

inception = InceptionTime(label="Stage", velocity=velocity, delete_features=delete_features, maxlen=maxlen, scale=scale, class_weights=class_weights, weather=weather, diff=diff,
                          prunning_function=preprocessing.get_stage_prunner_all(mapping=params.stage_mapping, NaN_value=None, minlen=30, min_days=min_days, min_animals=10),
                          split_by=dict(column="Stage", colvalue=["breeding: unknown", "non-breeding: unknown", "unknown"]),
                          nb_filters=32, use_residual=True, use_bottleneck=False, depth=4, step=1, #depth 4
                          kernel_size_s=[33,19,9,5], strides=[5,3,1,1],
                          reduce_lr=False, inception_activation="linear",
                          bottleneck_size=16,
                         )

N = len(inception.X)
n = inception.y.max() + 1
print("Dataset size: {} \nclasses: {}".format(N, n))


hist = inception.train(epochs=epochs, batch_size=batch_size, use_kfold=True, n_splits=n_splits)
acc = int(100 * np.mean(hist[1]["highest acc"]))
loss = inception.compute_loss(fold_idx=[*range(n_splits)])

scale_ID = dict(scale=scale) if isinstance(scale, bool) else dict()
num_delete_features = len([i for i in delete_features if i != ""])

identifier = other_utils.dict_to_id(bat=add_bathymetry, del_ftr=num_delete_features, v=velocity, acc=acc, loss=loss, B=batch_size,
                                    L=maxlen, W=class_weights, weather=weather, days=min_days, diff=diff, **scale_ID, N=N, n=n,
                                    ndigits=2
                                   )

parentDir = fullPath("utils/data/weights/Stage/")
Path(parentDir).mkdir(exist_ok=True, parents=True)
inception.model.save_weights(os.path.join(parentDir, f"{identifier}.h5"))
df = inception.classification_report(save=True, dataname=f"stage-clf_{identifier}", fold_idx=[*range(n_splits)])

print("Done!")