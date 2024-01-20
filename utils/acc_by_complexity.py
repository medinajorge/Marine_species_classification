"""
Analyze accuracy as a function of the complexity of the data passed to the model.

In order of increasing complexity:
random < blind < common_origin_2D_image < common_origin_2D_occupancy < common_origin_point_estimation < common_origin (sequential data) < geo (sequential data + real location) < geo+env(sequential data + real location + environmental variables)
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import os
from collections import defaultdict
import plotly.graph_objects as go
from phdu import savedata, savefig
from phdu.plots.plotly_utils import get_figure
from phdu.plots.base import plotly_default_colors
from tidypath import fmt
from . import models, analysis

@savedata
def random_choice_expected_value(v2=True):
    """
    Expected accuracy if species are predicted randomly.
    """
    if v2:
        df = models.DecisionTree(common_origin_distance=True, v2=True, min_animals=5, minlen=5).labels
    else:
        df = pd.read_csv("utils/data/classification_report/report_InceptionTime_5-kfold_species_clf_velocity-arch-segment_to-origin-_weather-.csv")
    N = df.shape[0]
    pi = df.value_counts("COMMON_NAME").values / N
    acc_random_macro = np.sum(pi**2)
    acc_random_micro = np.mean(pi**2)
    return acc_random_macro, acc_random_micro

@savedata
def blind_accuracy(v2=True):
    """
    Accuracy if classifier always predicts the majority class.
    """
    if v2:
        df = models.DecisionTree(common_origin_distance=True, v2=True, min_animals=5, minlen=5).labels
    else:
        df = pd.read_csv("utils/data/classification_report/report_InceptionTime_5-kfold_species_clf_velocity-arch-segment_to-origin-_weather-.csv")
    N = df.shape[0]
    pi = df.value_counts("COMMON_NAME").values / N
    acc_blind_macro = np.max(pi)
    acc_blind_micro = 1 / pi.size
    return acc_blind_macro, acc_blind_micro

def load_clf_report_v1(clf='InceptionTime', scale_by_velocity=False, random_state=1, min_animals=10, minlen=20, as_image=False, as_image_density=False, as_image_indiv_scaling=False, remove_outliers=True, num_iter=None):
    parentDir = f"utils/data/classification_report/{clf}"
    def loader(identifier):
        fullname = f"report_5-kfold_species_clf_{identifier}.csv"
        fullpath = os.path.join(parentDir, fullname)
        df = pd.read_csv(fullpath)
        return df
    def get_identifier(num_iter):
        if num_iter == 3:
            return fmt.dict_to_id(scale_by_velocity=scale_by_velocity, random_state=random_state, min_animals=min_animals, minlen=minlen, image=as_image, density=as_image_density, indiv_scaling=as_image_indiv_scaling, rem_out=remove_outliers)
        elif num_iter == 2:
            return fmt.dict_to_id(scale_by_velocity=scale_by_velocity, random_state=random_state, min_animals=min_animals, minlen=minlen, image=as_image, density=as_image_density, indiv_scaling=as_image_indiv_scaling)
        elif num_iter == 1:
            return fmt.dict_to_id(scale_by_velocity=scale_by_velocity, random_state=random_state, min_animals=min_animals, minlen=minlen)
        else:
            raise ValueError("num_iter must be 1, 2 or 3")
    if num_iter is None:
        try:
            df = loader(get_identifier(3))
            num_iter = 3
        except:
            try:
                df = loader(get_identifier(2))
                num_iter = 2
            except:
                try:
                    df = loader(get_identifier(1))
                    num_iter = 1
                except:
                    raise ValueError("No report found for any value of num_iter")
    else:
        try:
            df = loader(get_identifier(num_iter))
        except:
            raise ValueError(f"No report found for the given value of num_iter = {num_iter}")
    return df, num_iter

def common_origin_acc_v1(**kwargs):
    """
    num_iter: 1 => common origin, scale or not by velocity
    num_iter: 2 => the rest except remove outliers
    num_iter: 3 => full kwargs.
    """
    df = load_clf_report_v1(**kwargs)[0]
    acc_macro = (df["Prediction"] == "Correct").mean()
    acc_micro = df.groupby("COMMON_NAME").apply(lambda S: (S["Prediction"] == "Correct").mean()).mean()
    return acc_macro, acc_micro

def common_origin_acc_v2(scale_by_velocity=False, as_image=False, common_origin_distance=None, as_image_density=False, random_states=range(1,6), classifier='inception'):
    kwargs = analysis.get_classification_report_default_kwargs()
    kwargs["classifier"] = classifier
    kwargs["scale_by_velocity"] = scale_by_velocity
    kwargs["as_image"] = as_image
    kwargs["as_image_density"] = as_image_density
    kwargs["common_origin_distance"] = common_origin_distance if common_origin_distance is not None else not as_image

    acc = defaultdict(list)
    for random_state in random_states:
        kwargs["random_state"] = random_state
        df = analysis.classification_report(**kwargs)
        acc["macro"].append((df["Prediction"] == "Correct").mean())
        acc["micro"].append(df.groupby("COMMON_NAME").apply(lambda S: (S["Prediction"] == "Correct").mean()).mean())
    return pd.DataFrame(acc)

def acc_v2(weather=None, random_states=range(1, 6), **specs):
    kwargs = analysis.get_classification_report_default_kwargs()
    kwargs["weather"] = weather
    kwargs.update(specs)

    acc = defaultdict(list)
    for random_state in random_states:
        kwargs["random_state"] = random_state
        df = analysis.classification_report(**kwargs)
        acc["macro"].append((df["Prediction"] == "Correct").mean())
        acc["micro"].append(df.groupby("COMMON_NAME").apply(lambda S: (S["Prediction"] == "Correct").mean()).mean())
    return pd.DataFrame(acc)

def get_complexity_specs():
    complexity_specs = dict(random = (random_choice_expected_value, dict()),
                            blind = (blind_accuracy, dict()),
                            common_origin_2D_image = (common_origin_acc_v2, dict(scale_by_velocity=True, as_image=True)),
                            common_origin_2D_occupancy = (common_origin_acc_v2, dict(scale_by_velocity=True, as_image=True, as_image_density=True)),
                            common_origin_point_estimation = (common_origin_acc_v2, dict(scale_by_velocity=False, as_image=False, as_image_density=False, classifier='xgb')),
                            common_origin = (common_origin_acc_v2, dict(scale_by_velocity=False, as_image=False, as_image_density=False)),
                            geo = (acc_v2, dict()),
                            )
    complexity_specs['geo+env'] = (acc_v2, dict(weather='all'))
    return complexity_specs

@savefig('all-simple_axes')
def acc_by_complexity_plot(simple_axes=True):
    complexity_specs = get_complexity_specs()
    fig = get_figure(yaxis_title="Complexity", xaxis_title="Accuracy", xaxis_range=[-0.05, 1.05], width=1400, simple_axes=simple_axes)
    color_macro, color_micro = ['yellowgreen', 'slategray']
    color_macro_error = 'green'
    color_micro_error = 'darkslategray'
    for name, (func, kwargs) in complexity_specs.items():
        acc = func(**kwargs)
        if isinstance(acc, tuple):
            acc_macro, acc_micro = acc
            acc_macro_std, acc_micro_std = 0, 0
        elif isinstance(acc, pd.core.frame.DataFrame):
            acc_macro, acc_micro = acc.mean()
            acc_macro_std, acc_micro_std = acc.std()
        else:
            raise ValueError(f"Unknown type for acc: {type(acc)}")
        label = name.replace('_', ' ').capitalize()
        fig.add_trace(go.Scatter(y=[label], x=[acc_macro], error_x=dict(type='data', array=[acc_macro_std], color=color_macro_error, thickness=3, width=8), marker=dict(size=26, color=color_macro, line=dict(color='dimgray', width=3), opacity=1), name='Macro', showlegend=name=='random'))
        fig.add_trace(go.Scatter(y=[label], x=[acc_micro], error_x=dict(type='data', array=[acc_micro_std], color=color_micro_error, thickness=3, width=8), marker=dict(size=26, color=color_micro,  line=dict(color='dimgray', width=3), opacity=1), name='Micro', showlegend=name=='random'))

    fig.update_layout(legend=dict(x=0.7, y=0.2))
    return fig


def acc_mean_std(**kwargs):
    df = acc_v2(**kwargs)
    result = pd.concat([df.mean(), df.std()], axis=1)
    result.columns = ['mean', 'std']
    return result

@savefig
def feature_selection_performance():
    settings = {'Geo': dict(common_origin_distance=False, weather=None),
                'MRMR': dict(common_origin_distance=False, weather='mrmr'),
                'VIF': dict(common_origin_distance=False, weather='vif'),
                'Env': dict(common_origin_distance=True, weather='all'),
                'Collinearity-filtered': dict(common_origin_distance=False, weather='pruned'),
                # 'mrmr+collinear': dict(common_origin_distance=False, weather='mrmr+collinear'),
                'Geo+env': dict(common_origin_distance=False, weather='all'),

                # 'mrmr+vif': dict(common_origin_distance=False, weather='mrmr+vif'),
                # 'mrmrloop+vif': dict(common_origin_distance=False, weather='mrmrloop+vif'),
                }
    fig = get_figure(yaxis_title="Setting", xaxis_title="Accuracy", width=1400, simple_axes=True)
    color_macro, color_micro = ['yellowgreen', 'slategray']
    color_macro_error = 'green'
    color_micro_error = 'darkslategray'
    for label, kwargs in settings.items():
        acc = acc_mean_std(**kwargs)
        acc_macro = acc.loc['macro', 'mean']
        acc_micro = acc.loc['micro', 'mean']
        acc_macro_std = acc.loc['macro', 'std']
        acc_micro_std = acc.loc['micro', 'std']
        fig.add_trace(go.Scatter(y=[label], x=[acc_macro], error_x=dict(type='data', array=[acc_macro_std], color=color_macro_error, thickness=3, width=8), marker=dict(size=26, color=color_macro, line=dict(color='dimgray', width=3), opacity=1), name='Macro', showlegend=label=='Geo'))
        fig.add_trace(go.Scatter(y=[label], x=[acc_micro], error_x=dict(type='data', array=[acc_micro_std], color=color_micro_error, thickness=3, width=8), marker=dict(size=26, color=color_micro,  line=dict(color='dimgray', width=3), opacity=1), name='Micro', showlegend=label=='Geo'))

    fig.update_layout(legend=dict(x=0.8, y=0.1))
    return fig

def get_complexity_kwargs():
    kwargs = dict(common_origin_2D_image = dict(scale_by_velocity=True, as_image=True, common_origin_distance=False),
        common_origin_2D_occupancy = dict(scale_by_velocity=True, as_image=True, as_image_density=True, common_origin_distance=False),
        common_origin_point_estimation = dict(scale_by_velocity=False, as_image=False, as_image_density=False, classifier='xgb', common_origin_distance=True),
        common_origin = dict(scale_by_velocity=False, as_image=False, as_image_density=False, common_origin_distance=True),
        geo = dict(common_origin_distance=False, weather=None)
    )
    kwargs['geo+env'] = dict(common_origin_distance=False, weather='all')
    return kwargs

def accuracy_mantained_with_complexity(random_states=range(1, 6), c0='common_origin', cf='geo+env'):
    kws = get_complexity_kwargs()
    kwargs_0 = kws[c0]
    kwargs_f = kws[cf]
    df1 = analysis.classification_report_random_states(random_states=random_states, **kwargs_0)
    df2 = analysis.classification_report_random_states(random_states=random_states, **kwargs_f)
    p_obs = np.empty((len(random_states)))
    p_exp = np.empty((len(random_states)))
    p_value = np.empty((len(random_states)))
    for i, random_state in enumerate(random_states):
        r1 = df1.query(f"random_state == {random_state}")
        r2 = df2.query(f"random_state == {random_state}")
        A = r1[r1.Prediction == 'Correct'].index
        p_B = (r2.Prediction == 'Correct').mean()
        p_B_given_A = (r2.loc[A, 'Prediction'] == 'Correct').mean()
        p_obs[i] = p_B_given_A
        p_exp[i] = p_B

        n_obs = A.size
        n_exp = r2.shape[0]
        success_obs = int(p_B_given_A * n_obs)
        success_exp = int(p_B * n_exp)
        # contingency table
        table = np.array([[success_obs, n_obs - success_obs], [success_exp, n_exp - success_exp]])
        p_value[i] = ss.chi2_contingency(table, correction=False)[1]

    # quantify effect
    phi_2 = 2 * np.arcsin(np.sqrt(p_obs))
    phi_1 = 2 * np.arcsin(np.sqrt(p_exp))
    h = phi_2 - phi_1
    return p_obs, p_exp, p_value, h

def accuracy_mantained_with_complexity_summary(**kwargs):
    settings = [*get_complexity_kwargs().keys()]
    num_settings = len(settings)
    H = {}
    P = {}
    P_OBS = {}
    P_EXP = {}
    for i, c0 in enumerate(settings):
        H[(c0, c0)] = np.NaN
        P[(c0, c0)] = np.NaN
        P_OBS[(c0, c0)] = np.NaN
        P_EXP[(c0, c0)] = np.NaN
        if i < num_settings:
            for cf in settings[i+1:]:
                p_obs, p_exp, p_value, h = accuracy_mantained_with_complexity(c0=c0, cf=cf, **kwargs)
                H[(c0, cf)] = h.mean()
                P[(c0, cf)] = p_value.mean()
                P_OBS[(c0, cf)] = p_obs.mean()
                P_EXP[(c0, cf)] = p_exp.mean()

                H[(cf, c0)] = np.NaN
                P[(cf, c0)] = np.NaN
                P_OBS[(cf, c0)] = np.NaN
                P_EXP[(cf, c0)] = np.NaN
    to_df = lambda d: pd.Series(d).unstack(1).loc[settings[:-1], settings]
    return to_df(H), to_df(P), to_df(P_OBS), to_df(P_EXP)
