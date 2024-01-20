import numpy as np
import pandas as pd
from pathlib import Path
import os
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

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
sys.path.append(os.path.join(RootDir, "utils"))
fullPath = lambda path: os.path.join(RootDir, path)

from utils import analysis
import getopt
from tidypath import fmt
import phdu

# Default params
weather = 'all'
common_origin_distance=False
fold_idx = None
test_size = 0.1
overwrite = False

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "w:C:i:s:O:",
                               ["weather=", "common_origin_distance=", "fold_idx=", "test_size=", "overwrite="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-C", "--common_origin_distance"):
        common_origin_distance = fmt.decoder(arg)
    elif opt in ("-i", "--fold_idx"):
        fold_idx = fmt.decoder(arg)
    elif opt in ("-s", "--test_size"):
        test_size = fmt.decoder(arg)
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)

phdu.getopt_printer(opts)

use_kfold = fold_idx is not None
if use_kfold:
    test_size = 0.1 # default
avg_trajs = use_kfold

kws = dict(weather=weather, common_origin_distance=common_origin_distance, fold_idx=fold_idx, test_size=test_size, use_kfold=use_kfold, avg_trajs=avg_trajs, overwrite=overwrite)
print(f"Params:\n{kws}")

data, specs = analysis.feature_importances_shap(**kws)

print("Done!")
