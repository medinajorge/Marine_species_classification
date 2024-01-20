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
    from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)
import getopt
import phdu

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
sys.path.append(os.path.join(RootDir, "utils"))
fullPath = lambda path: os.path.join(RootDir, path)

from utils import nc_preprocess
from tidypath import fmt

try:
    opts, args = getopt.getopt(sys.argv[1:],
                               "y:v:O:",
                               ["year=", "variable=", "overwrite="])
except getopt.GetoptError:
    print('nc_data.py -y <year> -v <variable> -O <overwrite>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-y", "--year"):
        years = [int(i) for i in arg.split(',')]
    elif opt in ("-v", "--variable"):
        variable = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)

phdu.getopt_printer(opts)

for year in tqdm(years):
    for month in tqdm(range(1, 13)):
        print('Processing year: {}, month: {}\n'.format(year, month))
        nc_preprocess.env_data(year, month, variable, overwrite=overwrite)

print('Done!')
