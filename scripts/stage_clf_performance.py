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

weather = None
common_origin_distance = True
epochs = 140
filter_kwargs = {}
# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "c:s:w:C:f:e:",
                               ["classifier=", "species=", "weather=", "common_origin_distance=", "filters=", "epochs="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-c", "--classifier"):
        classifier = arg
    elif opt in ("-s", "--species"):
        species = arg.replace('"', '')
    elif opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-C", "--common_origin_distance"):
        common_origin_distance = fmt.decoder(arg)
    elif opt in ("-f", "--filters"):
        filters = fmt.decoder(arg)
        if classifier == 'inception':
            filter_kwargs = {'nb_filters': filters}
        elif classifier == 'resnet':
            filter_kwargs = {'n_feature_maps': filters}
        else:
            warnings.warn('No filter parameters for classifier {}'.format(classifier))
    elif opt in ("-e", "--epochs"):
        epochs = fmt.decoder(arg)

phdu.getopt_printer(opts)

acc = analysis.stage_clf_performance(species=species, model=classifier, weather=weather, common_origin_distance=common_origin_distance, epochs=epochs, **filter_kwargs)
print("Accuracy: {}".format(acc))


print("Done!")
