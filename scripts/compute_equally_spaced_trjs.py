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

weather = None
pad_day_rate = 3
# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "w:p:",
                               ["weather=", "pad_day_rate="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-p", "--pad_day_rate"):
        pad_day_rate = fmt.decoder(arg)

phdu.getopt_printer(opts)

df = preprocessing.equally_spaced_trajectories_avg_v2(weather=weather, pad_day_rate=pad_day_rate)

print("Done!")
