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
N_taxa = 20000
min_N_species = 200
magnitude = 'accuracy'
overwrite = False
# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "N:n:m:O:", ["N_taxa=", "min_N_species=", "magnitude=", "overwrite="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-N", "--N_taxa"):
        N_taxa = fmt.decoder(arg)
    elif opt in ("-n", "--min_N_species"):
        min_N_species = int(arg)
    elif opt in ("-m", "--magnitude"):
        magnitude = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)

phdu.getopt_printer(opts)

kws = dict(N_taxa=N_taxa, min_N_species=min_N_species, magnitude=magnitude, overwrite=overwrite)
print(f"Params:\n{kws}")

feature_clustering = analysis.feature_clustering(**kws, gpu=True)

print("Done!")
