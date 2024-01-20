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
reduce = 'mean'
lat_width = 1
lon_width = 1

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "r:l:L:",
                               ["reduce=", "lat_width=", "lon_width="])

except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-r", "--reduce"):
        reduce = fmt.decoder(arg)
    elif opt in ("-l", "--lat_width"):
        lat_width = float(arg)
    elif opt in ("-L", "--lon_width"):
        lon_width = float(arg)

phdu.getopt_printer(opts)

kwargs = dict(reduce=reduce, lat_width=lat_width, lon_width=lon_width)
print(f"Params:\n{kwargs}")

S = preprocessing.occurrences_count(**kwargs)

print("Done!")
