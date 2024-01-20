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

from utils import analysis, params
import getopt
from tidypath import fmt
import phdu

# Default params
weather = None
pad_day_rate=None
common_origin_distance=False
delete_features=[]
model='inception'
nf=32
invert_lat=False
overwrite = False
stage_remap = 'binary'
# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "w:p:C:D:m:n:O:i:S:R:",
                               ["weather=", "pad_day_rate=", "common_origin_distance=", "delete_features=", "model=", "nf=", "overwrite=", "invert_lat=", "species_train=", "stage_remap="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-p", "--pad_day_rate"):
        pad_day_rate = fmt.decoder(arg)
    elif opt in ("-C", "--common_origin_distance"):
        common_origin_distance = fmt.decoder(arg)
    elif opt in ("-D", "--delete_features"):
        delete_features = arg.replace('"', '').split(",")
    elif opt in ("-m", "--model"):
        model = arg
    elif opt in ("-n", "--nf"):
        nf = int(arg)
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-i", "--invert_lat"):
        invert_lat = fmt.decoder(arg)
    elif opt in ("-S", "--species_train"):
        species_train = arg.replace('"', '')
    elif opt in ("-R", "--stage_remap"):
        stage_remap = arg

phdu.getopt_printer(opts)

kws = dict(weather=weather, pad_day_rate=pad_day_rate, common_origin_distance=common_origin_distance, delete_features=delete_features, model=model, nf=nf, overwrite=overwrite, invert_lat=invert_lat, species_train=species_train, stage_remap=stage_remap)
print(f"Params:\n{kws}")

results, df = analysis.stage_clf_performance_kfold(**kws)

print("Done!")
