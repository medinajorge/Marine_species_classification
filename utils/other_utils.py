# Let's try to code a decorator for timing any function:
from functools import wraps
#import numba as nb
import numpy as np
import pandas as pd
from copy import deepcopy
import os
from pathlib import Path
from collections.abc import Iterable
from functools import reduce
import time


##############################################################################################################################
"""                                                   I. Decorators                                                        """
##############################################################################################################################

def timer(orig_func):
    """Wrapper for timing functions"""
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        t1 = time.time()
        result = orig_func(*args,**kwargs)
        dt = time.time() - t1
        print('{} ran in: {:.2f} s = {:.2f} min = {:.2f} h'.format(orig_func.__name__, dt,dt/60,dt/3600))
        return result
    return wrapper

class timer_class(object):
    """Wrapper for timing functions. Same as timer but in a class."""
    def __init__(self, orig_func):
        self.orig_func = orig_func

    def __call__(self, *args, **kwargs):
        t1 = time.time()
        result = self.orig_func(*args, **kwargs)
        dt = time.time() - t1
        print('{} ran in: {} sec'.format(self.orig_func.__name__, dt))
        return result


def dark_figure(orig_func):
    """Wrapping for showing figures in black in jupyter notebook, after having saved them in normal color."""
    import matplotlib.pyplot as plt
    plt.ion()
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        result = orig_func(*args,**kwargs)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            for ax in fig.get_axes():
                ax.set_facecolor('#161b1e')
                ax.xaxis.label.set_color('#a8a49d') 
                ax.yaxis.label.set_color('#a8a49d')
                ax.tick_params(axis='both', colors='#a8a49d')
                l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18, facecolor='#161b1e', edgecolor='#161b1e', framealpha=1)
                for text in l.get_texts():
                    text.set_color('#a8a49d')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#a8a49d')
                ax.set_title(ax.get_title(), color='#a8a49d')
            fig.patch.set_facecolor('#161b1e')
        plt.show()
        return result
    return wrapper

def identity(orig_func):
    """Wrapper that does not do anything."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        result = orig_func(*args, **kwargs)
        return result
    return wrapper



##############################################################################################################################
"""                                                 II. Getopt utils                                                       """
##############################################################################################################################

def encoder(x, ndigits=2, iterables=(list, tuple, np.ndarray)):
    """x -> string version of x"""
    if x is None:
        return "none"
    elif isinstance(x, str):
        return x
    elif isinstance(x, float):
        if x == int(x):
            return str(int(x))
        else:
            return str(round(x, ndigits=ndigits)).replace('.', '--')
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, iterables):
        return '-'.join([encoder(sub_x) for sub_x in x])
    else:
        return str(x)
    
def decoder(x, iterables=(list, tuple, np.ndarray)):
    """string version of x -> x"""
    if x.lower() == "none":
        return None
    elif x.lower() == "false":
        return False
    elif x.lower() == "true":
        return True
    elif "--" in x:
        return float(x.replace("--", "."))
    elif isinstance(x, iterables):
        return [decoder(sub_x) for sub_x in x]
    else:
        try:
            return int(x)
        except:
            return x

def getopt_printer(opts):
    """Prints getopt input in a readable way."""
    print('\n'.join(f'{opt} => {arg}' for opt, arg in (("Args", "Values"), *opts)))
    
def dict_to_id(*args, ndigits=2, **kwargs):
    """Generate ID of the form k1-v1_k2-v2... for k_i, v_i keys and values of the dictionary d or the kwargs."""
    key_formatter = lambda k: k.replace("_", "-")
    d = args[0] if len(args) > 0 else kwargs
    return "_".join([f"{key_formatter(k)}-{encoder(d[k], ndigits=ndigits)}" for k in sorted(d.keys())])

def id_to_dict(identifier):
    """Inverse of dict_to_id."""
    s = identifier.split("/")[-1] # retain filename only
    s = os.path.splitext(s)[0] # remove extension
    d = {}
    for split in s.split("_"):
        var_value = split.split("-")
        if len(var_value) > 1:
            if "" in var_value: # value is a float
                var_value_arr = np.array(var_value)
                idx_dot = np.argwhere(var_value_arr == "")[0, 0]
                key_idx = 0 if idx_dot == 2 else slice(0, idx_dot-2)
                d["-".join(var_value[key_idx])] = decoder(f"{var_value_arr[idx_dot-1]}--{var_value_arr[idx_dot+1]}")
            else:
                d["-".join(var_value[:-1])] = decoder(var_value[-1])
    return d

def id_updater(filename, update_dict, mode="add"):
    """
    Modifies filename by updating the underlying dict.
    Attrs:
        - filename:    id to be modified
        - update_dict: dict to use for updating the id. if update_dict={} => filename rearranged according to other_utils.dict_to_id.
        - mode:        - "add":    add update_dict to the id.
                       - "delete": delete update_dict from the id.
    Returns modified filename.
    """
    split_dirs = filename.split("/")
    parentDir = "/".join(split_dirs[:-1])
    file = split_dirs[-1]
    d = id_to_dict(file)
    if mode == "add":
        d.update(update_dict)
    elif mode == "delete":
        d = {k: v for k, v in d.items() if k not in update_dict.keys()}
    var_values = [part.split("-") for part in file.split("_")]
    init = "_".join([part[0] for part in var_values if len(part) == 1])
    ext = os.path.splitext(file)[1] 
    new_filename = os.path.join(parentDir, f"{init}_{dict_to_id(d)}{ext}")

    return new_filename

def id_renamer(update_dict, parentDir, key=None, mode="add"):
    """
    Modifies id of files in parentDir by updating the underlying dict.
    Attrs:
        - update_dict: dict to use for updating the id
        - parentDir: folder where files are located.
        - key: string contained in the file for it to be modified.
        - mode:   - "add": add update_dict to the id.
                  - "delete": delete update_dict from the id.
    Returns #modified files.
    NOTE: If update_dict={} => filenames will be rearranged according to other_utils.dict_to_id.
    """
    r = 0
    for file in os.listdir(parentDir):
        if key is None or key in file:
            old_filename = os.path.join(parentDir, file)
            new_filename = id_updater(old_filename, update_dict, mode=mode)
            os.rename(old_filename, new_filename)
            r += 1
    return r


##############################################################################################################################
"""                                                    III. Other                                                          """
##############################################################################################################################


def latex_table(df, index=False, **kwargs):
    """Pandas DataFrame -> Latex table."""
    col_format = "c" if isinstance(df, pd.core.series.Series) else "c"*len(df.columns)
    if index:
        col_format += "c"
    table_replacements = (("\\toprule", "\\toprule "*2),
                          ("\\bottomrule", "\\bottomrule "*2)
    )
    text_replacements = (("\\textbackslash ", "\\"),
                         ("\{", "{"), 
                         ("\}", "}"),
                         ("\$", "$"),
                         ("\_", "_"),
                         ("\\textasciicircum ", "^")
    )
    table_formatter = lambda x:  reduce(lambda a, kv: a.replace(*kv), table_replacements, x)
    text_formatter = lambda x: reduce(lambda a, kv: a.replace(*kv), text_replacements, x)
    formatter = lambda x: text_formatter(table_formatter(x))
    print(formatter(df.to_latex(index=index, column_format=col_format, **kwargs)))
    return

def fig_saver(filename):
    """Figure saver without overwriting."""
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))
