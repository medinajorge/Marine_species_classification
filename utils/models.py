"""
Models for classification of animal trajectories. Most relevant: InceptionTime, XGB.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc
import datetime
from functools import wraps
import re
from inspect import ismethod
from pathlib import Path
from copy import deepcopy
import os
import shutil
import sys
sys.stdout.flush()
import warnings
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

from collections import defaultdict
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold #KFold
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
try:
    import xgboost as xgb
except:
    pass
try:
    import shap
except:
    pass
from .nn_utils import class_counter
from . import preprocessing, analysis, data_visualization, file_management, nn_utils, other_utils
try:
    import tensorflow as tf
    #import tensorflow.keras.backend as K
    num_cores = 0
    tf.config.threading.set_inter_op_parallelism_threads(2) #num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.set_soft_device_placement(True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
except:
    print("tensorflow not available")

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)


##############################################################################################################################
"""                                                   0. Contents

          Sklearn classifiers:           decision tree, random forest, SVM

          Xgboost:                       xgboost

          Neural network:                dense, fully convolutional, ResNet, InceptionTime
                                         LSTM, Convolutional LSTM, Deep convolutional SLTM,

"""
##############################################################################################################################


##############################################################################################################################
"""                                                    I. Parameters                                                       """
##############################################################################################################################

interesting_cols = ["ID", "COMMON_NAME", "Taxa", "Class", "SEX", "DATABASE", "TAG", "NumberOfSatellites", "Length", "Mean_latitude", "Mean_longitude", "Mean_year"]
secondary_cols = ["Order", "Family", "SatelliteProg", "TAG_TYPE", "ResidualError", "Stage", "AGE", "BODY_LENGTH"]
totally_unimportant_cols = ["Colour"]
updated_cols = ['Cluster ID', 'Cluster ID confidence', 'Cluster ID confidence interval', 'Animals in dataset', 'Animals in dataset interval', 'Length interval', "Mean year interval"]
all_cols = interesting_cols + secondary_cols + totally_unimportant_cols + updated_cols
artificial_label_cols = ['COMMON_NAME', 'Taxa', 'Class', 'SEX', 'DATABASE', 'TAG', 'Order', 'Family', 'SatelliteProg', 'TAG_TYPE', 'Stage', 'AGE']

##############################################################################################################################
"""                                       II. General Classifier class and methods                                          """
##############################################################################################################################

class Classifier():
    """Class containing common methods for all classifiers."""

    def __init__(self, scale=True, delete_features=[], scale_padded=False, side_scaling="specific", scale_by_velocity=False, as_image=False, as_image_density=False, as_image_indiv_scaling=True, remove_outliers=True, fit_scale_side=True, N_subset=None, **preprocessing_kwargs):
        data, label_col, features, scale_idxs = preprocessing.preprocessing(delete_features=delete_features, scale_padded=scale_padded,
                                                                            as_image=as_image,
                                                                            **preprocessing_kwargs)
        for k, v in data.items():
            setattr(self, k, v)
        if N_subset is not None:
            self.subset(N=N_subset)

        self.side_scaling = side_scaling
        self.side_set_exists = hasattr(self, "X_side")
        self.fit_scale_side = fit_scale_side
        self.label_col = label_col
        self.features = features
        self.num_labels = len(set(self.y))
        self.num_signals = self.X[0].shape[0] #2 + (12 if categorical == 'months' else 53 if categorical == 'weeks' else 1)
        self.scale = scale
        if isinstance(scale, bool):
            self.scaler = StandardScaler() if scale else None
        self.base_scaler = StandardScaler()
        self.scale_by_velocity = scale_by_velocity
        self.remove_outliers = remove_outliers
        self.as_image = as_image
        self.as_image_density = as_image_density
        self.as_image_indiv_scaling = as_image_indiv_scaling
        self.idx_to_scaler_train = None
        self.idx_to_scaler_test = None
        self.idx_to_scaler_side = None
        self.scale_padded = scale_padded
        self.scale_idxs = scale_idxs
        self.preprocessing_func_kwargs = {}
        self.interesting_cols = interesting_cols
        self.secondary_cols = secondary_cols
        self.totally_unimportant_cols = totally_unimportant_cols
        self.all_cols = all_cols
        return

    def subset(self, N=1000, min_obs=5):
        """
        N: number of observations to keep.
        min_obs: minimum number of observations per target category.
        """
        print(f"Taking a subset with {N} observations and {min_obs} observations per target category.")
        self.X = self.X[:N]
        self.X_original = self.X_original[:N]
        self.Year = self.Year[:N]
        self.labels = self.labels.iloc[:N]
        self.y = self.y[:N]

        num_obs = pd.Series(self.y).value_counts()
        discard = num_obs[num_obs < min_obs].index
        idx_discard = np.array([i for i, y in enumerate(self.y) if y in discard])
        select_idxs = lambda X: [x for i, x in enumerate(X) if i not in idx_discard]

        self.X = select_idxs(self.X)
        self.X_original = select_idxs(self.X_original)
        self.Year = select_idxs(self.Year)
        idx_preserve = np.array([i for i in range(len(self.y)) if i not in idx_discard])
        self.labels = self.labels.iloc[idx_preserve]
        self.y = self.y[idx_preserve]
        self.y = preprocessing.remap_labels(self.y)
        return

    def get_cat_to_label(self):
        if not hasattr(self, 'cat_to_label'):
            cat_to_label = {cat: l for cat, l in zip(self.y_train_short, self.labels_train[self.label_col].values)}
            self.cat_to_label = {k: cat_to_label[k] for k in sorted(cat_to_label.keys())}
            if self.side_set_exists:
                self.cat_to_label.update({cat: l for cat, l in zip(self.y_side_short, self.labels_side[self.label_col].values)})
            self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}
            return

    def train(self, test_size=0.2, random_state=1, stratify=True, use_kfold=False, n_splits=5, shuffle=True, evaluator_kwargs={}, **training_kwargs):

        def train_and_store():
            training_results = self.training_step(**training_kwargs)
            take_pred = lambda x: x if len(x.shape) == 1 else x.argmax(axis=1)
            self.y_train_pred = take_pred(self.model.predict(self.X_train))
            self.y_test_pred = take_pred(self.model.predict(self.X_test))
            y_train_pred_short, confidence_train, acc_train = self.evaluator(partition='train', **evaluator_kwargs)
            y_test_pred_short, confidence_test, acc_test = self.evaluator(partition='test', **evaluator_kwargs)
            self.y_train_pred_short = y_train_pred_short
            self.y_test_pred_short = y_test_pred_short
            return training_results, confidence_train, confidence_test, acc_train, acc_test

        if use_kfold:

            attrs = ['X_train_list', 'X_train', 'X_test_list', 'X_test', 'y_train_short', 'y_train', 'y_test_short', 'y_test', "labels_train", "labels_test",
                     "y_train_pred", "y_test_pred", "y_train_pred_short", "y_test_pred_short",
                     'Year_train', 'Year_test'
                    ]
            kfold_attrs = [f'{attr}_kfold' for attr in attrs]
            for attr in kfold_attrs:
                setattr(self, attr, [])

            kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            train_history = defaultdict(list)
            test_history = defaultdict(list)
            def store_results(history, confidence, acc):
                history['confidence correct'].append(np.mean(confidence['correct']))
                history['confidence correct std'].append(np.std(confidence['correct']))
                history['confidence wrong'].append(np.mean(confidence['wrong']))
                history['confidence wrong std'].append(np.std(confidence['wrong']))
                history['highest acc'].append(acc)
                return

            def store_val_loss(kfold_history, training_results):
                val_loss = np.array(training_results.history['val_loss'])
                val_acc = np.array(training_results.history[f'val_{self.metrics[0]}'])
                lowest_loss_idx = val_loss.argmin()
                kfold_history['loss'].append(val_loss[lowest_loss_idx])
                kfold_history['acc-lowest-loss'].append(val_acc[lowest_loss_idx])
                return

            select_indices = lambda x, idxs: [x[i] for i in idxs]

            breakline = "-" * 100 + "\n"
            print(breakline)

            for fold_number, kfold_split in enumerate(kfold.split(self.X, self.y), start=1):
                self.preprocess(kfold_split=kfold_split, random_state=random_state)

                print(f'Training for fold {fold_number} ...')
                training_results, confidence_train, confidence_test, acc_train, acc_test = train_and_store()

                # store results
                for kfold_attr, attr in zip(kfold_attrs, attrs):
                    getattr(self, kfold_attr).append(getattr(self, attr))
                if training_results is not None:
                    log = pd.DataFrame(training_results.history)
                    metric = self.metrics[0]
                    acc_train = log[metric].max()
                    acc_test = log[f"val_{metric}"].max()

                store_results(train_history, confidence_train, acc_train)
                store_results(test_history, confidence_test, acc_test)
                if training_results is not None:
                    store_val_loss(train_history, training_results)

            print(f'{breakline} Stats:')
            for data, history in zip(['Train', 'Test'], [train_history, test_history]):
                print('\n {}:'.format(data))
                for k,v in history.items():
                    print('{}: {:.3f}, std: {:.3f}'.format(k, np.mean(v), np.std(v)))
            return train_history, test_history

        else:
            self.preprocess(test_size=test_size, random_state=random_state, stratify=stratify)
            training_results, confidence_train, confidence_test, acc_train, acc_test = train_and_store()

            return training_results

    def scaler_by_col(self, col="COMMON_NAME", scaler_constructor=StandardScaler, **kwargs):
        idxs_train = self.labels_train.groupby(col).indices
        idxs_test = self.labels_test.groupby(col).indices
        scaler_dict = {}
        if self.side_set_exists:
            idxs_side = self.labels_side.groupby(col).indices
            for species, idxs in idxs_train.items():
                scaler = scaler_constructor(**kwargs)
                scaler_dict[species] = (scaler, idxs,
                                        idxs_test[species] if species in idxs_test.keys() else [],
                                        idxs_side[species] if species in idxs_side.keys() else []
                                       )
            non_common_species = set(idxs_side.keys()) - set(idxs_train.keys())
            if self.side_scaling == "specific":
                for species in non_common_species:
                    scaler_dict[species] = (scaler_constructor(**kwargs), [], [], idxs_side[species])
            else:
                if len(self.scale_idxs) > 0:
                    Z = np.concatenate(tuple(self.X+self.X_side), axis=1).T
                    scale_pts = np.ones((Z.shape[0]), dtype=np.bool) if self.scale_padded else  ~ ((np.abs(Z) < 1e-8).all(axis=1))
                    self.base_scaler.fit(Z[np.ix_(scale_pts, self.scale_idxs)])
                for species in non_common_species:
                    scaler_dict[species] = (self.base_scaler, [], [], idxs_side[species])
        else:
            for species, idxs in idxs_train.items():
                scaler = scaler_constructor(**kwargs)
                scaler_dict[species] = (scaler, idxs,
                                        idxs_test[species] if species in idxs_test.keys() else [],
                                        None
                                       )
        return scaler_dict

    def preprocess(self, kfold_split=None, test_size=0.2, random_state=1, stratify=True, side_set=True):
        """Apply preprocessing."""
        if kfold_split is None:
            stratify_by = self.y if stratify else None
            get_label_df = lambda values: pd.DataFrame({col: arr for col, arr in zip(self.labels.columns, values.T)})
            idxs = np.arange(len(self.X))
            train, test, X_train_list, X_test_list, X_original_train, X_original_test, y_train_short, y_test_short, labels_train, labels_test, Year_train, Year_test = train_test_split(idxs, self.X, self.X_original, self.y, self.labels.values, self.Year, test_size=test_size, random_state=random_state, stratify=stratify_by)

            self.labels_train, self.labels_test = get_label_df(labels_train), get_label_df(labels_test)
            self.Year_train, self.Year_test = Year_train, Year_test
        else:
            train, test = kfold_split
            select_indices = lambda x, idxs: [x[i] for i in idxs]
            X_train_list, X_test_list = select_indices(self.X, train), select_indices(self.X, test)
            X_original_train, X_original_test = select_indices(self.X_original, train), select_indices(self.X_original, test)
            self.Year_train, self.Year_test = select_indices(self.Year, train), select_indices(self.Year, test)
            y_train_short, y_test_short = self.y[train], self.y[test]
            self.labels_train, self.labels_test = self.labels.iloc[train] , self.labels.iloc[test]

        if hasattr(self, 'v'):
            self.v_train = self.v.iloc[train]
            self.v_test = self.v.iloc[test]
            self.v_norm_train = self.v_norm.iloc[train]
            self.v_norm_test = self.v_norm.iloc[test]
            self.dt_train = self.dt.iloc[train]
            self.dt_test = self.dt.iloc[test]

        if isinstance(self.scale, Iterable):
            if self.scale[0] == "col":
                self.scaler = self.scaler_by_col(col=self.scale[1])
            self.idx_to_scaler_train = {u: s for s, i_train, _, _2 in self.scaler.values() for u in i_train}
            self.idx_to_scaler_test = {u: s for s, _, i_test, _2 in self.scaler.values() for u in i_test}
            if self.side_set_exists:
                self.idx_to_scaler_side = {u: s for s, _, _2, i_side in self.scaler.values() for u in i_side}
        if self.scale_by_velocity:
            if hasattr(self, 'v'):
                self.scaler_v = self.scaler_by_col(col="COMMON_NAME", scaler_constructor=MinMaxScaler, clip=True)
                self.idx_to_scaler_train_v = {u: s for s, i_train, _, _2 in self.scaler_v.values() for u in i_train}
                self.idx_to_scaler_test_v = {u: s for s, _, i_test, _2 in self.scaler_v.values() for u in i_test}
                X_train_list = preprocessing.rescale_by_velocity(X_train_list, self.v_train, self.v_norm_train, self.dt_train, self.scaler_v, fit=True, remove_outliers=self.remove_outliers)
                X_test_list = preprocessing.rescale_by_velocity(X_test_list, self.v_test, self.v_norm_test, self.dt_test, self.scaler_v, fit=False, remove_outliers=self.remove_outliers)
                if self.side_set_exists and side_set:
                    self.idx_to_scaler_side_v = {u: s for s, _, _2, i_side in self.scaler_v.values() for u in i_side}
                    self.X_side = preprocessing.rescale_by_velocity(self.X_side, self.v_side, self.v_norm_side, self.dt_side, self.scaler_v, fit=self.fit_scale_side, remove_outliers=self.remove_outliers, side=True)
            else:
                warnings.warn('Velocity must be computed for scaling by velocity', RuntimeWarning)
        if self.as_image:
            # TODO: implement without scaling
            self.scaler_D = self.scaler_by_col(col="COMMON_NAME", scaler_constructor=MinMaxScaler, clip=False)
            self.idx_to_scaler_train_D = {u: s for s, i_train, _, _2 in self.scaler_D.values() for u in i_train}
            self.idx_to_scaler_test_D = {u: s for s, _, i_test, _2 in self.scaler_D.values() for u in i_test}
            remove_outliers_img = not self.scale_by_velocity and self.remove_outliers
            X_train_list = preprocessing.rescale_to_img(X_train_list, self.v_train, self.v_norm_train, self.dt_train, self.scaler_D, fit=True, density=self.as_image_density, indiv_scaling=self.as_image_indiv_scaling, remove_outliers=remove_outliers_img)
            X_test_list = preprocessing.rescale_to_img(X_test_list, self.v_test, self.v_norm_test, self.dt_test, self.scaler_D, fit=False, density=self.as_image_density, indiv_scaling=self.as_image_indiv_scaling, remove_outliers=remove_outliers_img)
            if self.side_set_exists and side_set:
                self.idx_to_scaler_side_D = {u: s for s, _, _2, i_side in self.scaler_D.values() for u in i_side}
                self.X_side = preprocessing.rescale_to_img(self.X_side, self.v_side, self.v_norm_side, self.dt_side, self.scaler_D, fit=self.fit_scale_side, density=self.as_image_density, indiv_scaling=self.as_image_indiv_scaling, remove_outliers=remove_outliers_img, side=True)



        data =  self.preprocessing_func(X_train_list, X_test_list,  X_original_train, X_original_test, y_train_short, y_test_short,
                                        self.scaler, scale_idxs=self.scale_idxs, idx_to_scaler_train=self.idx_to_scaler_train, idx_to_scaler_test=self.idx_to_scaler_test,
                                        base_scaler=self.base_scaler, **self.preprocessing_func_kwargs)
        for k, v in data.items():
            setattr(self, k, v)

        if self.side_set_exists and side_set:
            data =  self.preprocessing_func(None, self.X_side, None, self.X_original_side, None, self.y_side,
                                            self.scaler, scale_idxs=self.scale_idxs, idx_to_scaler_train=None, idx_to_scaler_test=self.idx_to_scaler_side, side=True,
                                            base_scaler=self.base_scaler, **self.preprocessing_func_kwargs)
            for k, v in data.items():
                if 'test' in k:
                    setattr(self, k.replace("test", "side"), v)
        return

    def set_kfold_attrs(self, fold_idx):
        k_fold_attrs = [attr for attr in dir(self) if attr.endswith("kfold")]
        attrs = [attr.replace("_kfold", "") for attr in k_fold_attrs]
        for k_fold_attr, attr in zip(k_fold_attrs, attrs):
            fold_idx_attr = getattr(self, k_fold_attr)[fold_idx]
            setattr(self, attr, fold_idx_attr)
        return

    def kfold_iter(concat_df=False):
        """
        Decorator for iteration of a function across Kfold indices.
        If fold_idx is an iterable, returns an iterable containing the results for each element of fold_idx.
        Else, the result for the specified kfold is returned.
        """
        def _kfold_iter(orig_func):
            @wraps(orig_func)
            def wrapper(*args, fold_idx=None, **kwargs):
                self = args[0]

                label_kfold = lambda x, fold_idx: x.assign(Kfold = fold_idx) if isinstance(x, pd.core.frame.DataFrame) else x
                is_iterable = lambda x: isinstance(x, Iterable) and not isinstance(x, pd.core.frame.DataFrame)

                def set_fold_and_eval(fold_idx):
                    if fold_idx is not None:
                        self.set_kfold_attrs(fold_idx)
                    result = orig_func(*args, **kwargs)
                    if is_iterable(result):
                        result = [label_kfold(r, fold_idx) for r in result]
                    else:
                        result = label_kfold(result, fold_idx)
                    return result

                if isinstance(fold_idx, Iterable):
                    results = [set_fold_and_eval(idx) for idx in fold_idx]
                    if concat_df:
                        if is_iterable(results[0]):
                            result = [pd.concat(r, axis=0, ignore_index=True) for r in zip(*results)]
                        else:
                            result = pd.concat(results, axis=0, ignore_index=True)
                    else:
                        result = zip(*results) if isinstance(results[0], Iterable) else results
                else:
                    result = set_fold_and_eval(fold_idx)
                return result
            return wrapper
        return _kfold_iter

    @kfold_iter(concat_df=True)
    def split_labels(self, partition="test", recompute_pred=False, evaluator_kwargs={}):
        """
        Splits the labels containing all the info of the animal in correct and wrong groups (correctly/incorrectly classified).
        fold_idx of the kfold training can be specified to obtain the results for a given kfold.

        recompute_pred should be false if fold_idx is specified.
        """
        if partition == "all":
            return [pd.concat(dfs, axis=0) for dfs in zip(*[self.split_labels(partition=p, evaluator_kwargs=evaluator_kwargs) for p in ["train", "test"]])]
        else:
            labels = getattr(self, f'labels_{partition}')
            y = getattr(self, f'y_{partition}_short')
            if recompute_pred or not hasattr(self, f'y_{partition}_pred_short'):
                y_pred = self.evaluator(partition=partition, **evaluator_kwargs)[0]
                setattr(self, f'y_{partition}_pred_short', y_pred)
            else:
                y_pred = getattr(self, f'y_{partition}_pred_short')
            self.get_cat_to_label()
            labels["Predicted"] = [self.cat_to_label[p] for p in y_pred]

            correct = np.argwhere(y == y_pred)[:, 0]
            wrong = np.argwhere(y != y_pred)[:, 0]
            labels_correct = labels.iloc[correct]
            labels_wrong = labels.iloc[wrong]

            return labels_correct, labels_wrong

    def get_target_idxs(self, label_col="COMMON_NAME", target="Sooty tern", label_type="correct", **split_kwargs):
        """Get idxs of label_col belonging to the target"""
        lc, lw = self.split_labels(**split_kwargs)
        if label_type == "all":
            label_group = pd.concat([lc, lw], axis=0)
        elif label_type == "correct":
            label_group = lc
        else:
            label_group = lw
        label_target = label_group[label_group[label_col] == target]
        idxs = label_target.index.values
        return idxs

    def relative_accuracy(self, col_type="all", partition="test", **evaluator_kwargs):
        """Returns dict of dataframes containing the relative proportions of correct and wrong predictions for each level of a category."""
        cols = getattr(self, f'{col_type}_cols')
        label_types = ["Correct", "Wrong"]
        splitted_labels = self.split_labels(**evaluator_kwargs)
        labels_full = getattr(self, f'labels_{partition}')[cols]

        data = defaultdict(dict)
        proportions = lambda df, col: df[col].value_counts(normalize=True, sort=False)
        support = lambda df, col: df[col].value_counts(sort=False)
        for labels, label_type in zip(splitted_labels, label_types):
            for col in cols:
                ratio = proportions(labels, col) / proportions(labels_full, col)
                s = support(labels_full, col)
                df = pd.concat([ratio, s], axis=1)
                df.columns = pd.MultiIndex.from_tuples([(col, f'{label_type} / Total'), (col, "Support")])
                # concat and df.columns = pd.MULTIINDEX
                data[label_type][col] = df

        data["Relative"] = pd.DataFrame({col: {"Wrong / Correct": (data["Wrong"] / data["Correct"]).iloc[:,0].values, "Support": data["Wrong"]["Support"]} for col in cols})

        return data

    @kfold_iter(concat_df=True)
    def accuracy_df(self, mode="point", partition="test",  parentDir=fullPath("utils/data/accuracy_df"),  dataname="", save=False):
        """Returns df with columns: Latitude, Longitude, Accuracy"""
        mode = "trajectory" if len(self.input_shape) > 1 else mode
        #to_numpy = lambda x: x if type(x) == np.ndarray else x.numpy() # if x is a tf.tensor

        X_list = getattr(self, 'X_{}_list'.format(partition))
        Year = getattr(self, 'Year_{}'.format(partition))
        non_periodic_X = getattr(self, 'X_original_{}'.format(partition))

        if mode == "point":
            #y = to_numpy(getattr(self, 'y_{}'.format(partition)))
            y = np.asarray(getattr(self, 'y_{}'.format(partition)))
            preds = getattr(self, 'y_{}_pred'.format(partition))
            accs = (preds == y.squeeze()).astype(np.int32)
        elif mode == "trajectory":
            y = getattr(self, 'y_{}_short'.format(partition))
            preds = getattr(self, 'y_{}_pred_short'.format(partition))
            accs = (y == preds).astype(np.int32)
            trajectory_lengths = [x.shape[1] for x in X_list]
            accs = np.repeat(accs, trajectory_lengths)
            #accs = np.array([acc for (acc, x) in zip(accs, X_list) for _ in range(x.shape[1])], dtype=np.int32)
        else:
            raise ValueError(f'mode {mode} not valid. Available: "point", "trajectory"')

        #X_2D = np.concatenate(tuple(X_list), axis=1)
        #Year_arr = np.concatenate(Year)
        #non_periodic_X = preprocessing.undo_periodic(X_2D, Year_arr, scaler=None, scale_idxs=self.scale_idxs, scale_padded=self.scale_padded)

        df = pd.DataFrame(dict(Latitude=non_periodic_X[0], Longitude=non_periodic_X[1], Time=non_periodic_X[2], Accuracy=accs))

        if save:
            Path(parentDir).mkdir(exist_ok=True, parents=True)
            df.to_csv(os.path.join(parentDir, f"acc-df_{self.__class__.__name__}_{dataname}.csv"), index=False)

        return df

    def density_accuracy_df(self, n_lat=125, n_lon=500, mode="point", parentDir=fullPath("utils/data/density_accuracy"), save=False, del_kfold_data=False, dataname=""):
        """Returns DataFrame including relation between #individuals, #species and accuracy in the lat-lon grid."""
        mode = "trajectory" if len(self.input_shape) > 1 else mode
        acc_bins = np.arange(3) - 0.5
        lat, lon = preprocessing.lat_lon_bins(n_lat=n_lat, n_lon=n_lon)
        bins = [lat, lon, acc_bins]

        num_individuals = np.zeros((lat.size-1, lon.size-1), dtype=np.int32)
        accuracy = np.NAN * np.empty((self.y.size, *num_individuals.shape), dtype=np.float32)
        species = np.zeros((np.unique(self.y).size, *num_individuals.shape), dtype=np.int32)

        i = 0
        to_numpy = lambda x: x if isinstance(x, np.ndarray) else x.numpy().squeeze()
        for fold_idx in tqdm(range(5)):
            self.set_kfold_attrs(fold_idx)
            #X = [preprocessing.undo_periodic(periodic_z, year) for periodic_z, year in zip(self.X_test_list, self.Year_test)]
            X = deepcopy(self.X_original_test)
            if mode == "point":
                l = np.cumsum([x.shape[1] for x in X])[:-1]
                y_test = np.split(to_numpy(self.y_test), l)
                y_test_pred = np.split(to_numpy(self.y_test_pred), l)
            else:
                y_test = deepcopy(self.y_test_short)
                y_test_pred = deepcopy(self.y_test_pred_short)
            for x, y, target in zip(X, y_test_pred, y_test):
                if mode == "point":
                    acc = (y == target).astype(np.int32)
                    target = target[0]
                elif mode == "trajectory":
                    acc = int((y == target)) * np.ones((x.shape[1]), dtype=np.int32)
                H, edges = np.histogramdd((*x[:2], acc), bins=bins)
                H = H.astype(np.int32)
                density = H.sum(axis=-1)
                non_zero = np.where(density > 0)
                density = density[non_zero]

                num_individuals[non_zero] += 1
                species[(target, *non_zero)] += 1
                accuracy[(i, *non_zero)] = H.T[1].T[(*non_zero,)] / density
                i += 1

        non_empty = np.where(num_individuals > 0)
        indiv_species = species.copy()
        indiv_species[indiv_species > 1] = 1
        indiv_species = indiv_species.sum(axis=0)[non_empty]
        accuracy = np.nanmean(accuracy, axis=0)[non_empty]
        num_individuals = num_individuals[non_empty]
        del species, H, density, non_zero, X, y_test, y_test_pred
        if del_kfold_data:
            attr_list = [attr for attr in dir(self) if "kfold" in attr and not ismethod(getattr(self, attr))] + ["X_train", "X_test", "y_train", "y_test", "X"]
            for attr in attr_list:
                delattr(self, attr)
        gc.collect()

        col_labels = ["Latitude", "Longitude"]
        mid = lambda x: 0.5*(x[1:] + x[:-1])
        bin_centers = [mid(e) for e in edges]
        coordinates = {col: c[non_empty] for col, c in zip(col_labels, np.meshgrid(*bin_centers[:-1], indexing="ij"))}

        df = pd.DataFrame({"Individuals": num_individuals, "Species": indiv_species,
                           "Accuracy": accuracy,
                           **coordinates})
        if save:
            Path(parentDir).mkdir(exist_ok=True, parents=True)
            df.to_csv(os.path.join(parentDir, f"density-acc-df_{self.__class__.__name__}_mode-{mode}_{dataname}.csv"), index=False)
        return df

    def classification_report(self, fold_idx=None, parentDir=fullPath("utils/data/classification_report"), dataname="", save=False, **kwargs):
        """Returns and stores dataframe containing all the labeling information"""
        # cols = getattr(self, f'{col_set}_cols')
        cols = self.labels.columns.tolist()
        if "Predicted" not in cols:
            cols.append("Predicted")
        if self.label_col not in cols:
            cols.append(self.label_col)
        if f"{self.label_col}-original" in self.labels.columns:
            cols.append(f"{self.label_col}-original")

        def df_computer(cols, fold_idx):
            labels_correct, labels_wrong = self.split_labels(fold_idx=fold_idx, **kwargs)
            labels_correct = labels_correct[cols]
            labels_wrong = labels_wrong[cols]
            labels_correct = labels_correct.assign(Prediction = "Correct")
            labels_wrong = labels_wrong.assign(Prediction = "Wrong")
            df = pd.concat([labels_correct, labels_wrong], axis=0, ignore_index=True)
            df = df.assign(Classifier = self.__class__.__name__)
            return df

        if fold_idx is None:
            kfold_str = ""
            df = df_computer(cols, None)
        else:
            cols.append("Kfold")
            kfold_str = "{}-kfold".format(len(fold_idx))
            df = df_computer(cols, fold_idx=fold_idx)

        if save:
            parentDir_full = os.path.join(parentDir, self.__class__.__name__)
            Path(parentDir_full).mkdir(exist_ok=True, parents=True)
            df.to_csv(os.path.join(parentDir_full, f"report_{kfold_str}_{dataname}.csv"), index=False)
        return df

    def spatiotemporal_df(self, df=None, lat_boxes=125, lon_boxes=500, t_boxes=366, mode="trajectory", partition="test", **kwargs):
        """Returns two dataframes, containing the relation between accuracy and space, and between accuracy and space-time."""
        lat, lon = preprocessing.lat_lon_bins(n_lat=lat_boxes, n_lon=lon_boxes)
        t = np.arange(t_boxes)

        df = self.accuracy_df(mode=mode, partition=partition, **kwargs) if df is None else df
        num_points = df.shape[0]
        get_values = lambda x: [i.values for i in x]
        H, edges = np.histogramdd(get_values([df.Latitude, df.Longitude, df.Time, df.Accuracy]), bins=[lat, lon, t, 2])

        def density_df(H, edges, col_labels):
            """Returns the density of the histogram and the accuracies, assuming accuracies corresponds to the last dimension."""
            density = H.sum(axis=-1)
            non_empty =  np.argwhere(density > 0).T
            density = density[(*non_empty,)]
            acc = H.T[1].T[(*non_empty,)] / density
            density /= num_points

            mid = lambda x: 0.5*(x[1:] + x[:-1])
            bin_centers = [mid(e) for e in edges]
            coordinates = {col: c[(*non_empty,)] for col, c in zip(col_labels, np.meshgrid(*bin_centers, indexing="ij"))}

            df = pd.DataFrame({"Density": density, "Accuracy": acc, **coordinates})
            return df

        space_time_df = density_df(H, edges[:3], ["Latitude", "Longitude", "Time"])
        space_df = density_df(H.sum(axis=2), edges[:2], ["Latitude", "Longitude"])

        return space_time_df, space_df

    def accuracy_matrix_corr(self, coordinates="spacetime", method="spearman", figDir="figs/accuracy_density_corr", show=False,  mode="trajectory", partition="test", **kwargs):
        """Correlation matrix between the accuracy and the spatial(or spatiotemporal) density."""
        space_time_df, space_df = self.spatiotemporal_df(mode=mode, partition=partition, **kwargs)
        if coordinates == "spacetime":
            df = space_time_df
        elif coordinates == "space":
            df = space_df
        else:
            raise ValueError(f'Coordinates {coordinates} not valid. Available: "spacetime", "space".')
        corrs = df.corr(method=method)

        fig = px.imshow(corrs)
        fig.update_layout(coloraxis=dict(cmin=-1, cmax=1, colorbar=dict(title_text=f'{coeff.capitalize()} correlation', tickfont_size=22)), height=800, width=1200, font_size=28,
                          margin=dict(t=5, b=5, l=5, r=5, pad=1)
        )
        figDir_full = fullPath(os.path.join(figDir, self.__class__.__name__))
        Path(figDir_full).mkdir(exist_ok=True, parents=True)
        fig.write_image(os.path.join(figDir_full, '{}_{}_{}_{}.png'.format(mode, partition, coordinates, method)))
        if show:
            fig.update_layout(template="plotly_dark")
            fig.show()
        return

    def accuracy_geo(self, df=None, mode="trajectory", partition="test", lon_boxes=500, lat_boxes=250, figname="", figDir="figs/density_geo", **fig_kwargs):
        _, space_df = self.spatiotemporal_df(lon_boxes=lon_boxes, lat_boxes=lat_boxes, df=df)
        figname_full = "{}_{}_{}".format(mode, partition, figname)
        figDir_full = fullPath(os.path.join(figDir, self.__class__.__name__))
        data_visualization.density_geo(space_df, "Accuracy", figname=figname_full, figDir=figDir_full, **fig_kwargs)
        return


##############################################################################################################################
"""                                               III. Sklearn classifiers                                                 """
##############################################################################################################################

class SklearnClassifier(Classifier):
    """Base class for all classifiers imported from sklearn"""
    def __init__(self, **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.preprocessing_func = preprocessing.ravel_data
        self.input_shape = (self.num_signals,)
        return

    def training_step(self):
        self.model = self.model.fit(self.X_train, self.y_train)
        training_results = None
        return training_results

    def evaluator(self, partition='test'):
        X = getattr(self, 'X_{}_list'.format(partition))
        y = getattr(self, 'y_{}_short'.format(partition))

        y_pred = np.empty((len(y)), dtype=np.int32)
        confidence = defaultdict(list)

        for i, (x, target) in enumerate(zip(X, y)):
            xt = x.T.copy()
            if self.scaler is not None and self.scale_idxs.size > 0:
                if hasattr(self, "idx_to_scaler"):
                    xt[:, self.scale_idxs] = self.idx_to_scaler[i].transform(xt[:, self.scale_idxs])
                else:
                    xt[:, self.scale_idxs] = self.scaler.transform(xt[:, self.scale_idxs])
            pred_time_steps = np.bincount(self.model.predict(xt).astype(np.int32))
            prediction = pred_time_steps.argmax()
            y_pred[i] = prediction
            p = pred_time_steps[prediction] / xt.shape[0]
            if target == prediction:
                confidence['correct'].append(p)
            else:
                confidence['wrong'].append(p)

        acc = (y_pred == y).mean()

        return y_pred, confidence, acc

    def split_side_set(self, train):
        X = self.X_side
        y = self.y_side
        labels = self.labels_side
        L = labels.Length
        start = L.cumsum() - L
        end = start + L
        X_train =  []
        X_test = []
        y_train = []
        y_test = []
        min_idx = labels.index[0]
        for i in labels.index:
            if i - min_idx in train:
                X_train.append(X[start[i]:end[i]])
                y_train.append(y[start[i]:end[i]])
            else:
                X_test.append(X[start[i]:end[i]])
                y_test.append(y[start[i]:end[i]])
        self.labels_side_train = labels.iloc[train]
        self.labels_side_test = labels.iloc[test]
        self.X_side_train = np.concatenate(X_train)
        self.X_side_test = np.concatenate(X_test)
        self.y_side_train = np.concatenate(y_train)
        self.y_side_test = np.concatenate(y_test)
        self.y_side_test_short = self.y_side_short[test]
        self.y_side_train_short = self.y_side_short[train]
        return


class DecisionTree(SklearnClassifier):

    def __init__(self, tree_kwargs={}, **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.model = DecisionTreeClassifier(**tree_kwargs)
        return


class RandomForest(SklearnClassifier):

    def __init__(self, n_jobs=-1, forest_kwargs={}, **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.model = RandomForestClassifier(n_jobs=n_jobs, **forest_kwargs)
        return


class SVM(SklearnClassifier):

    def __init__(self,n_jobs=-1, svm_kwargs={}, **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.model = svm.SVC(n_jobs=n_jobs, **svm_kwargs)
        return


##############################################################################################################################
"""                                                     IV. XGBoost                                                        """
##############################################################################################################################

class XGB(Classifier):
    """XGBoost classifier"""
    def __init__(self, tree_method="gpu_hist", max_depth=20, eta=0.05, num_round=10, other_xgb_params={}, objective="softprob", **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.tree_method = tree_method
        self.max_depth = max_depth
        self.eta = eta
        self.num_round = num_round
        self.other_xgb_params = other_xgb_params
        self.preprocessing_func = preprocessing.DMatrix
        self.input_shape = (self.num_signals,)
        self.objective = objective

        if self.num_labels > 2:
            self.eval_metric = "mlogloss"
            self.objective = "multi:softprob" if objective == "probability" else "multi:softmax"
            classification_params = dict(eval_metric=self.eval_metric, objective=self.objective, num_class=self.num_labels)
        else:
            self.eval_metric = "logloss"
            self.objective = "binary:logistic"
            classification_params = dict(eval_metric=self.eval_metric, objective=self.objective)

        self.params = {"tree_method": tree_method, "max_depth": max_depth, "eta": eta,
                       **classification_params,
                       **other_xgb_params}
        return

    def training_step(self):
        self.model = xgb.train(self.params, self.X_train, self.num_round)
        training_results = None
        return training_results

    def evaluator(self, partition='test'):
        X = getattr(self, 'X_{}_list'.format(partition))
        y = getattr(self, 'y_{}_short'.format(partition))

        y_pred = np.empty((len(y)), dtype=np.int32)
        confidence = defaultdict(list)

        def prediction_probs(z):
            if self.num_labels > 2:
                ps = self.model.predict(z).sum(axis=0) # sum of probabilities for all datapoints belonging to the same individual.
                prediction = ps.argmax()
                c = ps[prediction] / ps.sum()
            else:
                ps = self.model.predict(z)
                prediction = np.zeros((ps.size), dtype=np.int32)
                prediction[ps >= 0.5] = 1
                c = np.abs(ps - 0.5) / 0.5
            return prediction, c
        def prediction_softmax(z):
            pred_time_steps = np.bincount(self.model.predict(z).astype(np.int32))
            prediction = pred_time_steps.argmax()
            c = pred_time_steps[prediction] / pred_time_steps.sum()
            return prediction, c

        predictor = prediction_probs if self.objective == "probability" else prediction_softmax

        for i, (x, target) in enumerate(zip(X, y)):
            xt = x.T.copy()
            y_arr = np.repeat(target, xt.shape[0])
            if self.scaler is not None and self.scale_idxs.size > 0:
                if hasattr(self, "idx_to_scaler"):
                    xt[:, self.scale_idxs] = self.idx_to_scaler[i].transform(xt[:, self.scale_idxs])
                else:
                    xt[:, self.scale_idxs] = self.scaler.transform(xt[:, self.scale_idxs])
            dxt = xgb.DMatrix(xt, label=y_arr)
            prediction, c = predictor(dxt)
            y_pred[i] = prediction
            if target == prediction:
                confidence['correct'].append(c)
            else:
                confidence['wrong'].append(c)

        acc = (y_pred == y).mean()

        return y_pred, confidence, acc

    def compute_shap(self, partition='test', n_splits=5, use_kfold=True, test_size=0.2, random_state=1, avg_trajs=True, fold_idx=None, predicted_only=None):
        """
        Computes the shap values for the specified partition.
        If predicted_only is True, only the shap values for predicted categories are stored.
        """
        if predicted_only is None:
            predicted_only = not avg_trajs

        data = defaultdict(list)
        def aux(kfold_split):
            self.preprocess(kfold_split=kfold_split, random_state=random_state, test_size=test_size) # test_size is only used if use_kfold is False.
            print("Training model")
            model = xgb.train(self.params, self.X_train, self.num_round)
            print("Computing shap values")
            explainer = shap.TreeExplainer(model)
            X_partition_orig = getattr(self, 'X_{}_list'.format(partition))
            X_partition = np.concatenate(tuple(X_partition_orig), axis=1).T
            y_partition = getattr(self, 'y_{}'.format(partition))
            y_short = getattr(self, 'y_{}_short'.format(partition))
            self.get_cat_to_label()

            shap_values = []
            preds = []
            for x, y in zip(tqdm(X_partition_orig), y_short):
                xt = x.T
                y_trj = np.repeat(y, xt.shape[0])
                if self.scaler is not None and self.scale_idxs.size > 0:
                    if hasattr(self, "idx_to_scaler"):
                        xt[:, self.scale_idxs] = self.idx_to_scaler[i].transform(xt[:, self.scale_idxs])
                    else:
                        xt[:, self.scale_idxs] = self.scaler.transform(xt[:, self.scale_idxs])
                dMat = xgb.DMatrix(xt, label=y_trj)
                shap_values_trj = np.array(explainer.shap_values(dMat)) # shape (num_classes, num_time_steps, num_features)
                prediction = model.predict(dMat).astype(np.int32)
                prediction_trj = np.bincount(prediction).argmax()
                if predicted_only:
                    shap_values_trj = shap_values_trj[prediction_trj][None] # shape (1, num_time_steps, num_features)

                if avg_trajs:
                    shap_values_trj = shap_values_trj.mean(axis=1) # shape (num_classes, num_features)
                    preds.append(prediction_trj)
                else:
                    shap_values_trj = shap_values_trj.transpose(1, 0, 2) # shape (num_time_steps, num_classes, num_features)
                    preds.append(prediction)


                shap_values.append(shap_values_trj)

            if avg_trajs:
                shap_values = np.array(shap_values)
                preds = np.array(preds)
            else:
                shap_values = np.concatenate(shap_values, axis=0)
                preds = np.concatenate(preds, axis=0)

            data['shap_values'].append(shap_values)
            data['X'].append(X_partition)
            data['X_orig'].append(X_partition_orig)
            data['y_short'].append(y_short)
            data['y'].append(y_partition)
            data['expected_value'].append(explainer.expected_value)
            data['preds'].append(preds)

        if use_kfold:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for fold_number, kfold_split in enumerate(kfold.split(self.X, self.y), start=1):
                if fold_idx is None or fold_idx == fold_number:
                    print(f"fold {fold_number}")
                    aux(kfold_split)
                else:
                    print(f"fold {fold_number} skipped")
        else:
            aux(None)

        specs = {'labels': self.cat_to_label, 'features': self.features}
        return data, specs

    def _compute_shap(self, partition="test", size=0.5, n_splits=5, full_set=False, random_state=1, save=False, dataname="",
                     parentDir=fullPath("utils/data/shap_values"), label_col="COMMON_NAME",
                     return_all_folds=False):
        """
        Wrong implementation. See compute_shap.

        Attributes:
            - partition: "train" or "test"
            - size: Percentage of the partition for which the shap values are computed.
        Returns: shap values and the data used for their computation.
        """
        X = getattr(self, 'X_{}_list'.format(partition))
        y = getattr(self, 'y_{}'.format(partition))
        X = np.concatenate(tuple(X), axis=1).T
        self.get_cat_to_label()

        def aux(data, label, kfold_str):
            model = deepcopy(self.model)
            dMat = xgb.DMatrix(data, label=label)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dMat)

            if save:
                parentDir_full = os.path.join(parentDir, self.__class__.__name__)
                Path(parentDir_full).mkdir(exist_ok=True, parents=True)
                shap_data = dict(values=shap_values, X=data, labels=self.cat_to_label, expected_value=explainer.expected_value, preds=self.model.predict(dMat))
                size_str = other_utils.encoder(size)
                file_management.save_lzma(shap_data, f"shap_size-{size_str}_{dataname}{kfold_str}", parent_dir=parentDir_full)
            return shap_values

        if full_set:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if return_all_folds:
                shap_values = []
            for fold_number, (_, test) in enumerate(kfold.split(X, y), start=1):
                print(f"fold {fold_number}")
                data = X[test]
                label = y[test]
                if return_all_folds:
                    shap_values.append(aux(data, label, f"fold-{fold_number}"))
                else:
                    shap_values = aux(data, label, f"fold-{fold_number}")

        else:
            if size < 1:
                data, _, label, _ = train_test_split(X, y, test_size=1-size, random_state=random_state, stratify=y)
            else:
                data, label = X, y
            shap_values = aux(data, label, "")

        return shap_values, data

    def shap_plot(self, shap_values=None, data=None, cat_to_label=None, show_classes=False, plot_type="bar", figDir="figs/shap", figname="", show=False, **shap_kwargs):
        if shap_values is None or data is None:
            shap_values, data = self.compute_shap(**shap_kwargs)
        if show_classes:
            self.get_cat_to_label()
            shap.summary_plot(shap_values, data, plot_type=plot_type, feature_names=self.features, class_names=[*self.cat_to_label.values()], class_inds=[*self.cat_to_label.keys()], max_display=len(self.features), plot_size=(15,8), show=False)
        else:
            shap_class_sum = np.abs(np.array(shap_values)).sum(axis=0)
            shap.summary_plot(shap_class_sum, data, plot_type=plot_type, feature_names=self.features, max_display=len(self.features), plot_size=(15,8), show=False)

        show_classes_str = "show_classes" if show_classes else ""
        fig = plt.gcf()
        figDir_full = fullPath(os.path.join(figDir, self.__class__.__name__))
        Path(figDir_full).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(figDir_full, f"shap_{len(self.features)}-features_{show_classes_str}_{plot_type}_{figname}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        return

##############################################################################################################################
"""                                                 V. Neural networks                                                     """
##############################################################################################################################


class NN(Classifier):
    """Neural network."""

    def __init__(self, categorical=None, dropout=0, pad_sequence=False, pad_kwargs={}, raw_sequences=False, callbacks=[], compiler_kwargs=None, kernel_initializer=None,
                 initializer_seed=0, delete_weights=True, use_bias=True, lr=1e-3, activation="relu", np_seed=0, tf_seed=100, class_weights=None,
                 performance_monitor="loss", binary_auto=False,
                 **preprocessing_kwargs):
        np.random.seed(np_seed)
        tf.random.set_seed(tf_seed)
        self.np_seed = np_seed
        self.tf_seed = tf_seed
        super().__init__(**preprocessing_kwargs)
        self.categorical = categorical # categorical features to define in preprocessing
        self.dropout = dropout
        self.use_bias = use_bias
        self.activation = activation
        self.pad_sequence = pad_sequence
        self.pad_kwargs = pad_kwargs
        self.raw_sequences = raw_sequences
        self.preprocessing_func = preprocessing.preprocessing_NN
        self.preprocessing_func_kwargs = {k: getattr(self, k) for k in ["categorical", "pad_sequence", "pad_kwargs", "raw_sequences"]}

        num_GPU = len(tf.config.list_physical_devices('GPU'))
        self.device = '/{}:0'.format("GPU" if num_GPU > 0 else "CPU")
        self.kernel_initializer = tf.keras.initializers.GlorotUniform if kernel_initializer is None else kernel_initializer
        self.initializer_seed = initializer_seed
        self.initializer = self.kernel_initializer(self.initializer_seed)

        self.delete_weights = delete_weights
        self.weights_dir = fullPath("utils/data/weights/{}".format(self.__class__.__name__))
        Path(self.weights_dir).mkdir(exist_ok=True, parents=True)
        self.weights_path = nn_utils.get_weight_path(self.weights_dir)
        self.log_dir = fullPath("logs/fit/{}".format(self.__class__.__name__))
        self.callbacks = callbacks if isinstance(callbacks, Iterable) else [callbacks]
        self.lr = lr

        if compiler_kwargs is None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=10000, decay_rate=0.95, staircase=True)
            # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
            if self.num_labels > 2 or self.num_labels == 2 and binary_auto:
                self.loss = "sparse_categorical_crossentropy"
                self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                self.metrics = ["sparse_categorical_accuracy"]
                self.output_size = self.num_labels
                self.output_activation = "softmax"
            elif self.num_labels == 2:
                self.loss = "binary_crossentropy"
                self.loss_fn = tf.keras.losses.BinaryCrossentropy()
                self.metrics = ["binary_accuracy"]
                self.output_size = 1
                self.output_activation = "sigmoid"
            else:
                raise ValueError("num_labels must be greater than 1.")
            compiler_kwargs = dict(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        self.compiler_kwargs = compiler_kwargs

        self.class_weights = class_weights
        if isinstance(class_weights, str):
            if class_weights == "linear":
                self.class_weights_computer = self.class_weights_linear

        if performance_monitor == "loss":
            self.performance_monitor = "val_loss"
        elif performance_monitor == "accuracy":
            self.performance_monitor = "val_{}".format(compiler_kwargs["metrics"][0])
        return

    def class_weights_linear(self):
        n_classes = self.y.max() + 1
        weights = self.y.size / (n_classes * np.bincount(self.y))
        return {c: w for c, w in enumerate(weights)}

    def compute_loss(self, fold_idx=[]):
        if len(fold_idx) > 0:
            n_splits = len(fold_idx)
            L = np.empty((n_splits))
            for fold_idx in range(n_splits):
                self.set_kfold_attrs(fold_idx)
                L[fold_idx] = log_loss(self.y_test, self.model.predict(self.X_test))
            return L.mean()
        else:
            return log_loss(self.y_test, self.model.predict(self.X_test))

    def compute_gradient_attribution(self, partition='side'):
        """
        Computes the gradient attribution of the input instance.

        Instances with a positive gradient attribution contribute positively to the prediction of the target species.
        """
        X = getattr(self, f'X_{partition}')
        y = getattr(self, f'y_{partition}')

        loss = np.empty((X.shape[0], self.output_size))
        gradient = np.empty(tf.squeeze(X).shape)
        for i, x in enumerate(tqdm(X)):
            with tf.GradientTape() as tape:
                x = tf.expand_dims(x, axis=0)
                tape.watch(x)
                prediction = self.model(x, training=False)
                loss_x = self.loss_fn(y[i], prediction)
                loss[i] = loss_x.numpy()
                gradient[i] = tape.gradient(loss_x, x).numpy().squeeze()
        return gradient, loss

    def split_side_set(self, train, test):
        """
        Splits the side set into a training and a test set.
        """
        self.labels_side_train = self.labels_side.iloc[train]
        self.labels_side_test = self.labels_side.iloc[test]
        train_tf = tf.convert_to_tensor(train)
        test_tf = tf.convert_to_tensor(test)
        self.X_side_train = tf.gather(self.X_side, train_tf)
        self.X_side_test = tf.gather(self.X_side, test_tf)
        self.y_side_train = tf.gather(self.y_side, train_tf)
        self.y_side_test = tf.gather(self.y_side, test_tf)
        self.y_side_test_short = self.y_side_short[test]
        self.y_side_train_short = self.y_side_short[train]
        return

    def training_step(self, verbose=1, batch_size=64, epochs=5):
        self.build_model()

        # initial view of weights
        w = tf.summary.create_file_writer(os.path.join(self.log_dir, "initial_weights"))
        with w.as_default():
            for layer in self.model.layers:
                for weights in layer.weights:
                    tf.summary.histogram(weights.name, weights, step=0)

        #self.weights_path = nn_utils.get_weight_path(self.weights_dir)
        save_best_model = tf.keras.callbacks.ModelCheckpoint(self.weights_path, save_best_only=True, save_weights_only=True, monitor=self.performance_monitor)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_images=True)
        base_callbacks = [save_best_model]#, tensorboard_callback]
        #[] #[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)]
        callbacks = self.callbacks + base_callbacks

        self.model.compile(**self.compiler_kwargs)

        if self.raw_sequences:     # Train using a generator. 1 sequence at a time.
            history = self.model.fit(
                self.X_train(),
                epochs=epochs,
                verbose=verbose,
                validation_data = self.X_test(),
                callbacks = callbacks,
                class_weight = self.class_weights_computer() if hasattr(self, "class_weights_computer") else self.class_weights
            )
        else:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data = (self.X_test, self.y_test),
                callbacks = callbacks,
                class_weight = self.class_weights_computer() if hasattr(self, "class_weights_computer") else self.class_weights
        )

        self.model.load_weights(self.weights_path) # retain best model
        if self.delete_weights:
            os.remove(self.weights_path)
            shutil.rmtree(self.log_dir, ignore_errors=True)
        else:
            self.weights_path = nn_utils.get_weight_path(self.weights_dir)

        self.initializer_seed += 1
        self.initializer = self.kernel_initializer(self.initializer_seed)
        return history

    def evaluator(self, partition='test', mode='mean'):
        """
        mode: max (prediction of point x = most likely class of point x, and prediction of trajectory is the most frequent prediction)
              mean (prediction of trajectory is the most likely class after adding all the probabilities for the points x)
        """
        X = getattr(self, 'X_{}'.format(partition))
        y = getattr(self, 'y_{}_short'.format(partition))

        probs_all = self.model.predict(X)

        def point_calculation():

            def prediction_max(probs_x):
                ps = np.bincount(probs_x.argmax(axis=1), minlength=self.num_labels) # Prediction of point x = most likely class of point x
                prediction = ps.argmax()
                return prediction, ps
            def prediction_mean(probs_x):
                ps = probs_x.sum(axis=0) # sum of probabilities for all datapoints belonging to the same individual.
                prediction = ps.argmax()
                return prediction, ps
            predictor = prediction_max if mode == 'max' else prediction_mean if mode == 'mean' else None

            X_list = getattr(self, 'X_{}_list'.format(partition))
            pointer = 0
            y_pred = np.empty((len(y)), dtype=np.int32)
            confidence = defaultdict(list)
            for i, (x, target) in enumerate(zip(X_list, y)):
                probs_x = probs_all[pointer:pointer+x.shape[1]]
                pointer += x.shape[1]
                prediction, ps = predictor(probs_x)
                y_pred[i] = prediction
                c = ps[prediction] / ps.sum()
                if target == prediction:
                    confidence['correct'].append(c)
                else:
                    confidence['wrong'].append(c)

            confidence = {k: np.array(v) for k, v in confidence.items()}
            return y_pred, confidence

        def trajectory_calculation():
            if self.output_size > 1:
                iterator_x = np.arange(probs_all.shape[0])
                y_pred = probs_all.argmax(axis=1)
                confidence_in_predicted = probs_all[iterator_x, y_pred]
            else:
                y_pred = np.round(probs_all).astype(np.int32).squeeze()
                confidence_in_predicted = np.empty((y_pred.size))
                is_zero = y_pred == 0
                confidence_in_predicted[is_zero] = 1 - probs_all[is_zero, 0]
                confidence_in_predicted[~is_zero] = probs_all[~is_zero, 0]
            correct = y == y_pred
            confidence = {'correct': confidence_in_predicted[correct],
                          'wrong': confidence_in_predicted[~correct]
            }
            return y_pred, confidence

        y_pred, confidence = trajectory_calculation() if len(self.input_shape) > 1 else point_calculation()
        acc = (y == y_pred).mean()

        return y_pred, confidence, acc

    def scheme(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        summ_string = "\n".join(stringlist)
        table = [line for line in stringlist[1:-4][1:-1] if not line.startswith("_") and not line.startswith("  ")]
        new_table = []
        for entry in table:
            entry = re.split(r'\s{2,}', entry)[:-1] # remove whitespace
            new_table.append(entry)

        df = pd.DataFrame(new_table[1:], columns=[c if not c.startswith(" ") else c[1:] for c in new_table[0]]).iloc[1:]
        df.loc[:, "Layer (type)"] = df["Layer (type)"].apply(lambda x: x[1:] if x.startswith(" ") else x)
        return df

    def last_conv_layer_name(self, keyword="add"):
        df = self.scheme()
        conv_layers = []
        for layer in df["Layer (type)"]:
            layer_name = str(layer).split(" ")[0]
            if layer_name.startswith(keyword):
                conv_layers.append(layer_name)
        name = conv_layers[-1] if len(conv_layers) > 0 else None
        return name

    def draw_model(self, rankdir='TB', figname=""):
        figDir = fullPath("figs/model_schemes")
        Path(figDir).mkdir(exist_ok=True, parents=True)
        filename = os.path.join(figDir, f'{self.__class__.__name__}_{figname}.png')
        tf.keras.utils.plot_model(
            self.model, to_file=filename, show_shapes=False, show_dtype=False,
            show_layer_names=False, rankdir=rankdir, expand_nested=True, dpi=96
        )
        return


    def gradcam_trajectories(self, target="Sooty tern", keyword="add", heatmap="real", num_trajs=None, eq_spacing=True, size="original", norm=False, clip=False,
                             partition="test", **kwargs):
        """
        heatmap:     real: returns heatmap for the gradients corresponding to the actual label of the training example.
                     predicted:  ''                                                     label of the prediction.
        """
        if heatmap == "real":
            self.get_cat_to_label()
            pred_index = self.label_to_cat[target]
        elif heatmap == "predicted":
            pred_index = None
        else:
            raise ValueError(f"heatmap {heatmap} not valid. Available: 'real', 'predicted'")

        last_conv_layer_name = self.last_conv_layer_name(keyword=keyword)
        idxs = self.get_target_idxs(target=target, partition=partition, **kwargs)
        idxs = idxs if num_trajs is None else idxs[:num_trajs]
        if partition != "all":
            X = tf.gather(getattr(self, f"X_{partition}"), idxs)
            X_original = [x for i, x in enumerate(getattr(self, f"X_original_{partition}")) if i in idxs]
        else:
            X = tf.gather(tf.concat([self.X_train, self.X_test], axis=0), idxs)
            X_original = [x for i, x in enumerate(self.X_original_train + self.X_original_test) if i in idxs]

        #years = self.Year_test
        lengths = self.X_test.shape[1] * np.ones((idxs.size), dtype=np.int32) if eq_spacing else [self.X_test_list[i].shape[1] for i in idxs]
        heatmaps = []
        original_trajectories = []
        trajectories_train = []

        for trajectory, original_trajectory, l in zip(X, X_original, lengths):
            #original_trajectory = preprocessing.undo_periodic(trajectory.numpy().squeeze().T, year,
            #                                                  scaler=self.scaler, scale_idxs=self.scale_idxs, scale_padded=self.scale_padded)
            heatmap = analysis.make_gradcam_heatmap(tf.expand_dims(trajectory, axis=0), self.model, last_conv_layer_name, pred_index=pred_index, norm=norm, clip=clip)
            trajectory_train = trajectory.numpy().squeeze().T
            if size == "original":
                idxs = ~(np.abs(trajectory_train) < 1e-5).all(axis=0)
                original_trajectories.append(original_trajectory)
                trajectories_train.append(trajectory_train[:, idxs])
                heatmaps.append(heatmap[idxs])
            else:
                original_trajectories.append(original_trajectory[:, :size])
                heatmaps.append(heatmap[:size])
                trajectories_train.append(trajectory_train[:, :size])

        return  trajectories_train, original_trajectories, heatmaps

    def gradcam_plot(self, target="Sooty tern", label_type="correct", scale=3, markersize=5, num_trajs=None, geo=True, subplots=False, eq_spacing=True,
                     figname="", figDir="figs/gradcam", **kwargs):
        num_trajs_str = "all-trajs" if num_trajs is None else f"{other_utils.encoder(num_trajs)}-trajs"
        figname_full = f"grad-cam_{target}_{num_trajs_str}_{label_type}_{figname}"

        _, original_trajectories, heatmaps = self.gradcam_trajectories(target=target, num_trajs=num_trajs, eq_spacing=eq_spacing, label_type=label_type,
                                                                    **kwargs)
        fig = data_visualization.get_figure()
        if geo:
            figname_full += "_geo"
            for trajectory, heatmap in zip(original_trajectories, heatmaps):
                lat, lon, t = trajectory[:3]
                fig.add_trace(go.Scattergeo(
                            lat = lat,
                            lon = lon,
                            mode = 'lines+markers',
                            line = dict(width=1),
                            marker = dict(size=markersize, color=heatmap.numpy(), cmin=0, cmax=1),
                            opacity = 1,
                            hoverinfo='skip',
                            showlegend = False
                            #name = f'{cluster} ({percentage_str})',
                            )
                    )
            mean_lat = np.mean([t[0].mean() for t in original_trajectories])
            mean_lon = np.mean([t[1].mean() for t in original_trajectories])

            fig.update_layout(
                geo = dict(
                     projection_scale=scale, #this is kind of like zoom
                     center=dict(lat=mean_lat, lon=mean_lon)
                )
            )
        else:
            colors = data_visualization.plotly_default_colors() + data_visualization.plotly_colors()
            if subplots:
                figname_full += "_subplots"
                fig = data_visualization.get_subplots(cols=3, shared_xaxes=True, width=1500, height=500, subplot_titles=["Longitude", "Latitude", "Day"])
                for trajectory, heatmap, line_color in zip(original_trajectories, heatmaps, colors):
                    l = np.arange(trajectory.shape[1])
                    lat, lon, t = trajectory[:3]
                    for col, y in enumerate([lon, lat, t], start=1):
                        fig.add_trace(go.Scatter(
                            x = t if eq_spacing else l,
                            y = y,
                            mode = 'lines+markers',
                            line = dict(width=1, color=line_color),
                            marker = dict(size=markersize, color=heatmap.numpy(), cmin=0, cmax=1),
                            opacity = 1,
                            hoverinfo='skip',
                            showlegend = False
                            #name = f'{cluster} ({percentage_str})',
                        ),
                        row=1, col=col
                        )

            else:
                fig.update_layout(xaxis_title_text="Longitude", yaxis_title_text="Latitude")
                for trajectory, heatmap, line_color in zip(original_trajectories, heatmaps, colors):
                    lat, lon, t = trajectory[:3]
                    fig.add_trace(go.Scatter(
                                x = lon,
                                y = lat,
                                mode = 'lines+markers',
                                line = dict(width=1, color=line_color),
                                marker = dict(size=markersize, color=heatmap.numpy(), cmin=0, cmax=1),
                                opacity = 1,
                                hoverinfo='skip',
                                showlegend = False
                                #name = f'{cluster} ({percentage_str})',
                                )
                        )
        Path(figDir).mkdir(exist_ok=True, parents=True)
        fig.write_image(os.path.join(figDir, f"{figname_full}.png"))
        fig.show()
        return

class DenseNN(NN):
    """N-layer, fully connected neural network."""
    def __init__(self, N=3, n=None, **preprocessing_kwargs):
        super().__init__(**preprocessing_kwargs)
        self.N = N # number of hidden layers
        self.n = n # list of sizes of the hidden layers
        self.input_shape = (self.num_signals,)
        return

    def build_model(self, dropout=None, N=None):
        if dropout is not None:
            self.dropout = dropout
        if N is not None:
            self.N = N
        if self.n is None:
            self.n = [self.num_labels+2]*self.N + [self.output_size]
        hidden_layers = [tf.keras.layers.Dense(n, activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias) for n in self.n[1:-1]]
        layers = ([tf.keras.layers.Dense(self.n[0], activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias, input_shape=self.input_shape)]
                                                + hidden_layers
                                                + [tf.keras.layers.Dense(self.n[-1], activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias)]
                 )
        if self.dropout > 0:
            added_dropout = [[l, tf.keras.layers.Dropout(self.dropout)] for l in layers[:-1]]
            layers = [layer for layer_pair in added_dropout for layer in layer_pair] + [layers[-1]]
        with tf.device(self.device):
            model = tf.keras.models.Sequential(layers)
        self.model = model
        return


class FCN(NN):
    """Fully convolutional neural network."""
    def __init__(self, n_filters=[128, 256, 256, 256, 128], kernel_sizes=[128, 64, 32, 8, 3], strides=[16,5,3,1,1], maxlen=400, pad_day_rate=None, step=1, **preprocessing_kwargs):
        # Previous defaults: n_filters=[128, 256, 128], kernel_sizes=[8, 5, 3], strides=[1,1,1]
        input_len = maxlen if pad_day_rate is None else 1 + pad_day_rate*365

        super().__init__(pad_sequence = True,
                         pad_kwargs = dict(maxlen=input_len, step=step),
                         pad_day_rate = pad_day_rate,
                         **preprocessing_kwargs
        )
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.input_shape = (input_len, self.num_signals)
        return

    def build_model(self):
        """Fully Convolutional Network as described in Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline."""
        x = tf.keras.layers.Input(shape=self.input_shape)
        conv_x = tf.keras.layers.BatchNormalization()(x)
        for n_filter, f, s in zip(self.n_filters, self.kernel_sizes, self.strides):
            conv_x = tf.keras.layers.Conv1D(n_filter, f, s, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x)
            conv_x = tf.keras.layers.BatchNormalization()(conv_x)
            conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_x = tf.keras.layers.GlobalAveragePooling1D()(conv_x)
        out = tf.keras.layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x) #'softmax'

        with tf.device(self.device):
            model = tf.keras.models.Model(inputs=x, outputs=out)
        self.model = model
        return


class ResNet(NN):
    def __init__(self, n_feature_maps=64, maxlen=400, step=1, pad_day_rate=None, as_image=False, **preprocessing_kwargs):
        if as_image:
            input_len = 100
        else:
            input_len = maxlen if pad_day_rate is None else 1 + pad_day_rate*365
        super().__init__(pad_sequence = True,
                         pad_kwargs = dict(maxlen=input_len, step=step),
                         pad_day_rate = pad_day_rate,
                         as_image = as_image,
                         **preprocessing_kwargs
        )
        self.n_feature_maps = n_feature_maps
        if as_image:
            self.input_shape = (input_len, input_len, 1)
        else:
            self.input_shape = (input_len, self.num_signals, 1)
        return

    def build_model(self):
        print ('build conv_x')
        x = tf.keras.layers.Input(shape=self.input_shape)
        conv_x = tf.keras.layers.BatchNormalization()(x)
        conv_x = tf.keras.layers.Conv2D(self.n_feature_maps, 8, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = tf.keras.layers.Conv2D(self.n_feature_maps, 5, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = tf.keras.layers.Conv2D(self.n_feature_maps, 3, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == self.n_feature_maps)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv2D(self.n_feature_maps, 1, 1,padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(x)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x)
        print ('Merging skip connection')
        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)

        print ('build conv_x')
        x1 = y
        conv_x = tf.keras.layers.Conv2D(self.n_feature_maps*2, 8, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(x1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = tf.keras.layers.Conv2D(self.n_feature_maps*2, 5, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = tf.keras.layers.Conv2D(self.n_feature_maps*2, 3, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == self.n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv2D(self.n_feature_maps*2, 1, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(x1)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x1)
        print ('Merging skip connection')
        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)

        print ('build conv_x')
        x1 = y
        conv_x = tf.keras.layers.Conv2D(self.n_feature_maps*2, 8, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(x1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = tf.keras.layers.Conv2D(self.n_feature_maps*2, 5, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = tf.keras.layers.Conv2D(self.n_feature_maps*2, 3, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == self.n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv2D(self.n_feature_maps*2, 1, 1, padding='same', kernel_initializer=self.initializer, use_bias=self.use_bias)(x1)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x1)
        print ('Merging skip connection')
        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)

        full = tf.keras.layers.GlobalAveragePooling2D()(y)
        out = tf.keras.layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias)(full)
        print ('        -- model was built.')

        with tf.device(self.device):
            model = tf.keras.models.Model(inputs=x, outputs=out)
        self.model = model
        return

class InceptionTime(NN):
    """
    Citation:
    @article{IsmailFawaz2020inceptionTime,
             Title   = {InceptionTime: Finding AlexNet for Time Series Classification},
             Author  = {Ismail Fawaz, Hassan and Lucas, Benjamin and Forestier, Germain and Pelletier, Charlotte and Schmidt, Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and Idoumghar, Lhassane and Muller, Pierre-Alain and Petitjean, François},
             journal = {Data Mining and Knowledge Discovery},
             Year    = {2020}
    }
    """
    def __init__(self, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, maxlen=400, step=1,
                 as_image=False,
                 kernel_size_s=[64,32,8], strides=[5,3,1],
                 reduce_lr=False, inception_activation="linear",
                 bottleneck_size=32, conv_dropout=0, dense_sizes=[], pad_day_rate=None, use_bias_conv=False, get_input_len_from="pad-day-rate",
                 **preprocessing_kwargs):
        if as_image:
            input_len = 100
        else:
            input_len = maxlen if pad_day_rate is None or get_input_len_from == "maxlen" else 1 + pad_day_rate*365
        super().__init__(pad_sequence = True,
                         pad_kwargs = dict(maxlen=input_len, step=step),
                         pad_day_rate = pad_day_rate,
                         as_image = as_image,
                         **preprocessing_kwargs
        )
        self.use_bias_conv = use_bias_conv
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1 # not used
        self.bottleneck_size = bottleneck_size
        self.kernel_size_s = kernel_size_s
        self.strides = strides
        self.dense_sizes = dense_sizes
        if as_image:
            self.input_shape = (input_len, input_len)
        else:
            self.input_shape = (input_len, self.num_signals)
        self.inception_activation = inception_activation
        self.conv_dropout = conv_dropout

        if reduce_lr:
            self.callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001))
        return

    def _inception_module(self, input_tensor, stride=1):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=self.inception_activation, use_bias=self.use_bias_conv)(input_tensor)
            # This layer has too few parameters to use dropout (~ 500).
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        conv_list = []
        #kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        #for i in range(len(kernel_size_s)):
        #    conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
        #                                         strides=stride, padding='same', activation=self.inception_activation, use_bias=self.use_bias)(
        #        input_inception))
        #kernel_size_s = [64, 32, 8]
        #strides=[5,3,1]
        for s, k in zip(self.strides, self.kernel_size_s):
            conv_step = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=k,
                                                 strides=s, padding='same', activation=self.inception_activation, use_bias=self.use_bias_conv)(
                input_inception)
            #resize_step = tf.keras.layers.Lambda(lambda x: tf.image.resize_with_pad(x, x.shape[0], self.input_shape[0]))(conv_step)
            resize_step = tf.keras.layers.ZeroPadding1D(padding=(0, int(self.input_shape[0] * (1-1/s))))(conv_step)
            conv_list.append(resize_step)

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=self.inception_activation, use_bias=self.use_bias_conv)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=self.activation)(x)
        if self.conv_dropout > 0:
            x = tf.keras.layers.Dropout(self.conv_dropout)(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=self.use_bias_conv)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation(self.activation)(x)
        # Too few params to use dropout here.
        #if self.dropout > 0:
        #    x = tf.keras.layers.Dropout(self.dropout)(x)
        return x

    def build_model(self):
        input_layer = tf.keras.layers.Input(self.input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        x = tf.keras.layers.GlobalAveragePooling1D()(x) # GAP layer

        for dense_size in self.dense_sizes: #[512, 256]
            x = tf.keras.layers.Dense(dense_size, activation=self.activation, use_bias=self.use_bias)(x)
            if self.dropout > 0:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        output_layer = tf.keras.layers.Dense(self.output_size, activation=self.output_activation, use_bias=self.use_bias)(x)

        with tf.device(self.device):
            model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        self.model = model
        return

    @staticmethod
    def gradcam_weights_path(parentDir=fullPath("utils/data/weights/gradcam/inception/"), **kwargs):
        identifier = other_utils.dict_to_id(**kwargs)
        for file in os.listdir(parentDir):
            if identifier in file:
                break
        return os.path.join(parentDir, file)

    def load_weights(self, **kwargs):
        self.build_model()
        self.model.load_weights(self.gradcam_weights_path(**kwargs))
        return


class LSTM(NN):
    """Basic LSTM network."""
    def __init__(self, num_lstm=1, lstm_sizes=[128], dense_size=128, maxlen=400, step=1, bidirectional=False, pad_sequences=True, pad_day_rate=None, **preprocessing_kwargs):
        input_len = maxlen if pad_day_rate is None else 1 + pad_day_rate*365
        super().__init__(pad_sequence = pad_sequences,
                         pad_kwargs = dict(maxlen=input_len, step=step, expand_dim=False),
                         raw_sequences = not pad_sequences,
                         pad_day_rate = pad_day_rate,
                         **preprocessing_kwargs
        )
        self.bidirectional = bidirectional
        self.num_lstm = num_lstm
        self.lstm_sizes = lstm_sizes if len(lstm_sizes) == num_lstm else list(lstm_sizes) * num_lstm
        self.dense_size = dense_size
        self.input_shape = (input_len, self.num_signals) if pad_sequences else (None, self.num_signals)
        return

    def build_model(self):
        """Basic LSTM network"""

        with tf.device(self.device):
            model = tf.keras.models.Sequential()
            add_dropout = lambda: model.add(tf.keras.layers.Dropout(self.dropout)) if self.dropout > 0 else None

            for i, lstm_size in enumerate(self.lstm_sizes):
                if i == 0:
                    if self.bidirectional:
                        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias, input_shape=self.input_shape)))
                    else:
                        model.add(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias, input_shape=self.input_shape))
                else:
                    if self.bidirectional:
                        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias)))
                    else:
                        model.add(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias))
                add_dropout()

            model.add(tf.keras.layers.Dense(self.dense_size, activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias))
            model.add(tf.keras.layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias))

        self.model = model
        return

class ConvLSTM(NN):
    """
    Convolutional layers for feature extraction and LSTM for sequence learning.
    Source: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
    Paper: Convolutional, Long Short-Term Memory, fully connected Deep Neural Networks, Tara N. Sainath et. al (2015)
    """
    def __init__(self, num_lstm=1, lstm_sizes=[128], dense_size=128, n_filters=[128, 128], kernel_sizes=[3, 3], maxlen=400, step=1, n_steps=5, bidirectional=False, pad_day_rate=None,
                 **preprocessing_kwargs):
        input_len = maxlen if pad_day_rate is None else 1 + pad_day_rate*365
        super().__init__(pad_sequence=True,
                         pad_kwargs = dict(maxlen=input_len, step=step, expand_dim=False, n_steps=n_steps),
                         pad_day_rate=pad_day_rate,
                         **preprocessing_kwargs
        )
        self.bidirectional = bidirectional
        self.num_lstm = num_lstm
        self.lstm_sizes = lstm_sizes if len(lstm_sizes) == num_lstm else list(lstm_sizes) * num_lstm
        self.dense_size = dense_size
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.input_shape = (None, input_len // n_steps, self.num_signals)
        return

    def build_model(self):
        """Basic LSTM network"""
        with tf.device(self.device):
            model = tf.keras.models.Sequential()
            add_dropout = lambda: model.add(tf.keras.layers.Dropout(self.dropout)) if self.dropout > 0 else None
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=self.n_filters[0], kernel_size=self.kernel_sizes[0], activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias), input_shape=self.input_shape))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=self.n_filters[1], kernel_size=self.kernel_sizes[1], activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias)))
            add_dropout()
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

            for i, lstm_size in enumerate(self.lstm_sizes):
                if self.bidirectional:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias)))
                else:
                    model.add(tf.keras.layers.LSTM(lstm_size, kernel_initializer=self.initializer, return_sequences=i < self.num_lstm - 1, use_bias=self.use_bias))
                add_dropout()

            model.add(tf.keras.layers.Dense(self.dense_size, activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias))
            model.add(tf.keras.layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias))

        self.model = model
        return

class DeepConvLSTM(NN):
    """
    Deep Convolutional LSTM network.
    Unlike an LSTM that reads the data in directly in order to calculate internal state and state transitions, and unlike the CNN LSTM that is interpreting the output from CNN models,
    the ConvLSTM is using convolutions directly as part of reading input into the LSTM units themselves.

    Source: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
    Paper: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, Xingjian Shi et. al (2015).
    """
    def __init__(self, num_ConvLSTM=1, dense_size=128, n_filters=[128], kernel_sizes=[3], maxlen=400, step=1, n_steps=5, bidirectional=False, pad_day_rate=None,
                 **preprocessing_kwargs):
        input_len = maxlen if pad_day_rate is None else 1 + pad_day_rate*365
        super().__init__(pad_sequence = True,
                         pad_kwargs = dict(maxlen=input_len, step=step, expand_dim=False, n_steps=n_steps, insert_row=True),
                         pad_day_rate = pad_day_rate,
                         **preprocessing_kwargs
        )
        fill = lambda x: x if len(x) == num_ConvLSTM else list(x) * num_ConvLSTM
        self.bidirectional = bidirectional
        self.num_ConvLSTM = num_ConvLSTM
        self.dense_size = dense_size
        self.n_filters = fill(n_filters)
        self.kernel_sizes = fill(kernel_sizes)
        self.input_shape = (n_steps, 1,  input_len // n_steps, self.num_signals)
        return

    def build_model(self):
        """Basic LSTM network"""
        with tf.device(self.device):
            model = tf.keras.models.Sequential()
            add_dropout = lambda: model.add(tf.keras.layers.Dropout(self.dropout)) if self.dropout > 0 else None

            for i in range(self.num_ConvLSTM):
                if self.bidirectional:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.ConvLSTM2D(filters=self.n_filters[i], kernel_size=(1,self.kernel_sizes[i]), activation='relu',
                                                                                       input_shape=self.input_shape, kernel_initializer=self.initializer,
                                                                                       return_sequences = i < self.num_ConvLSTM - 1, use_bias=self.use_bias)))
                else:
                    model.add(tf.keras.layers.ConvLSTM2D(filters=self.n_filters[i], kernel_size=(1,self.kernel_sizes[i]), activation='relu', input_shape=self.input_shape,
                                                         kernel_initializer=self.initializer, return_sequences = i < self.num_ConvLSTM - 1, use_bias=self.use_bias))
                add_dropout()

            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(self.dense_size, activation='relu', kernel_initializer=self.initializer, use_bias=self.use_bias))
            model.add(tf.keras.layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer, use_bias=self.use_bias))

        self.model = model
        return



classifier_params = dict(
    tree = dict(clf=DecisionTree, model=dict(), train=dict()),
    forest = dict(clf = RandomForest,
                  model = dict(forest_kwargs=dict(n_estimators = 60,
                                                  max_depth = 25)),
                  train = dict()
                 ),
    svm = dict(clf=SVM,
               model=dict(svm_kwargs=dict(C=1)),
               train=dict()
              ),
    xgb = dict(clf=XGB,
               model=dict(max_depth=20), # 25
               train=dict()
              ),
    dense = dict(clf=DenseNN,
                 model=dict(),
                 train=dict(epochs=15, batch_size=2048, verbose=0)),
    fcn = dict(clf=FCN,
               model=dict(maxlen=512),
               train=dict(epochs=80, batch_size=128, verbose=0)
              ),
    resnet = dict(clf=ResNet,
                  model=dict(maxlen=512, n_feature_maps=128),
                  train=dict(epochs=70, batch_size=64, verbose=0)
              ),
    inception = dict(clf=InceptionTime,
                     model=dict(maxlen=512, nb_filters=128),
                     train=dict(epochs=80, batch_size=128, verbose=0)
              ),
    lstm = dict(clf=LSTM,
                model=dict(num_lstm=2, bidirectional=True, maxlen=1000),
                train=dict(epochs=80, batch_size=128, verbose=0)
               ),
    conv_lstm = dict(clf=ConvLSTM,
                     model=dict(maxlen=500, num_lstm=2, bidirectional=True),
                     train=dict(epochs=80, batch_size=128, verbose=0)
                    ),
    deep_conv_lstm = dict(clf=DeepConvLSTM,
                          model=dict(maxlen=500),
                          train=dict(epochs=80, batch_size=128, verbose=0)
                    ),

)
