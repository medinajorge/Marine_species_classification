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

from utils import preprocessing
import getopt
from tidypath import fmt
import phdu

# Default params
feature = 'x'
maxlen = 300
n_neighbors = 10

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:m:n:",
                               ["feature=", "maxlen=", "n_neighbors="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-f", "--feature"):
        feature = arg.replace('"', '')
    elif opt in ("-m", "--maxlen"):
        maxlen = int(arg)
    elif opt in ("-n", "--n_neighbors"):
        n_neighbors = int(arg)

phdu.getopt_printer(opts)

kws = dict(feature=feature, maxlen=maxlen, n_neighbors=n_neighbors)
print(f"Params:\n{kws}")
mi = preprocessing.features_redundancy(**kws)

print(mi)

print("Done!")
