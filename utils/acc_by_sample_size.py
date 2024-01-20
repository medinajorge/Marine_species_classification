"""
Accuracy in the classification of a species or taxa, as a function of the number of trajectories (sample size).
"""

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import warnings
import signal
import scipy.stats as ss
from collections import defaultdict
from copy import deepcopy
from phdu import savedata, savefig, bootstrap, SavedataSkippedComputation
from phdu.plots.plotly_utils import get_figure, CI_plot, get_subplots, color_std, plotly_default_colors
from phdu.plots.base import color_gradient
import plotly.graph_objects as go
from . import analysis, preprocessing, params
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

def update_kwargs(**kwargs_new):
    kwargs_cp = analysis.get_classification_report_default_kwargs()
    kwargs_cp.update(kwargs_new)
    return kwargs_cp

def load_clf_reports_species(kwargs_dict, time_limit=1, verbose=0):
    def handler(signum, frame):
        raise TimeoutError(f"Time limit of {time_limit} seconds exceeded")

    kwargs = deepcopy(kwargs_dict)
    species = kwargs.pop('species')
    taxa = kwargs.pop('taxa', None)
    percentage_delete = kwargs.pop('percentage_delete', None)
    N = kwargs.pop('N', None)
    mode = kwargs.pop('mode', None)
    if 'overwrite' in kwargs:
        del kwargs['overwrite']
    prunning_kws = dict(species=species, taxa=taxa, percentage=percentage_delete, N=N, random_state=kwargs["random_state"], mode='specific')
    get_prunning_function = preprocessing.get_trajectory_deleter
    if verbose:
        print(f"kwars:\n\n {kwargs}")
        print(f"prunning_kws:\n\n {prunning_kws}")
    def loader():
        df = analysis.classification_report(**kwargs,
                                        get_prunning_function=get_prunning_function, skip_computation=True,
                                        prunning_kws=prunning_kws)
        return df
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    try:
        df = loader()
        if isinstance(df, SavedataSkippedComputation):
            df = pd.DataFrame()
        signal.alarm(0)
        return df
    except TimeoutError as e:
        print(e)
        return pd.DataFrame()
    except Exception as e:
        print(e)
        return pd.DataFrame()

@savedata
def acc_by_sample_size(random_states = range(1, 6), mode='same', acc_type='species', **specs):
    species = ['Audouins gull',
               'Corys shearwater',
               'Northern gannet',
               'Little penguin',
               'Black-browed albatross',
               'Southern elephant seal',
               'Bullers albatross',
               'Grey-headed albatross',
               'Australian sea lion',
               'Macaroni penguin',
               'Loggerhead turtle',
               'Chinstrap penguin',
               'Scopolis shearwater',
               'Blue shark',
               'Shortfin mako shark',
               'Whale shark',
               'Tiger shark',
               'Leatherback turtle',
               'Wandering albatross',
               'Northern elephant seal',
               'Humpback whale',
               'Salmon shark',
               'Short-finned pilot whale',
               'Adelie penguin',
               'King eider',
               'Hawksbill turtle',
               'White shark',
               'California sea lion',
               'Masked booby',
               'Green turtle',
               'Long-nosed fur seal',
               'Blue whale',
               'Black-footed albatross',
               'Trindade petrel',
               'Laysan albatross',
               'Ringed seal']
    Ns = [5, 10, 22, 46, 96, 200]

    if acc_type == 'species':
        compute_acc = lambda df, species: (df.query(f"COMMON_NAME == '{species}'").Prediction == 'Correct').mean()
    elif acc_type == 'taxa':
        def compute_acc(df, species):
            species_to_taxa = {s: df.Taxa[df.COMMON_NAME == s].iloc[0] for s in df.COMMON_NAME.unique()}
            df['Predicted-taxa']= df.Predicted.apply(lambda x: species_to_taxa[x]).values
            df_species = df.query(f"COMMON_NAME == '{species}'")
            return (df_species['Taxa'] == df_species['Predicted-taxa']).mean()

    accuracies = defaultdict(list)
    for s in tqdm(species):
        for N in Ns:
            for random_state in random_states:
                kwargs = update_kwargs(species=s, N=N, random_state=random_state, classifier="inception", **specs)
                df = load_clf_reports_species(kwargs)
                if len(df) > 0:
                    if mode == 'same':
                        accuracies[(s, N)].append(compute_acc(df, s))
                    elif mode == 'others':
                        for s2 in species:
                            if s2 != s:
                                accuracies[(s, s2, N)].append(compute_acc(df, s2))
                    elif mode == 'all':
                        for s2 in species:
                            accuracies[(s, s2, N)].append(compute_acc(df, s2))
                    else:
                        raise ValueError(f"Invalid value for mode: {mode}")
    S = pd.Series(accuracies).unstack(0)
    return S

def transfer_accuracy_corr(S=None, corr='spearman'):
    """
    Correlation between the accuracy in species B and the sample size of species A.
    This intends to measure the transferability of the knowledge from species A to species B.
    """
    compute_corr = getattr(ss, f"{corr}r")
    if S is None:
        S = acc_by_sample_size(mode='others', acc_type='taxa')
    mu = S.applymap(np.mean, na_action="ignore")
    species = mu.columns
    Ns = mu.index.levels[1]
    R = {}
    for s in species:
        for s2 in species:
            if s != s2:
                acc_data = mu.loc[s, s2].values
                valid = ~np.isnan(acc_data)
                R[(s, s2)] = compute_corr(Ns[valid], acc_data[valid])[0]
    return pd.Series(R).unstack(0)

def transfer_accuracy_regression(S=None):
    """
    Linear regression coefficient between the accuracy in species B and the sample size of species A.
    This intends to measure the transferability of the knowledge from species A to species B.
    """
    compute_p = lambda x, y: np.polyfit(x, y, 1)
    if S is None:
        S = acc_by_sample_size(mode='others', acc_type='taxa')
    mu = S.applymap(np.mean, na_action="ignore")
    species = mu.columns
    Ns = mu.index.levels[1]
    R = {}
    for s in species:
        for s2 in species:
            if s != s2:
                acc_data = mu.loc[s, s2].values
                valid = ~np.isnan(acc_data)
                R[(s, s2)] = compute_p(Ns[valid], acc_data[valid])[0]
    return pd.Series(R).unstack(0)

def transfer_accuracy_diff(S=None, Nf=100):
    """
    Difference in the accuracy in species B when the sample size of species A increases to Nf.
    This intends to measure the transferability of the knowledge from species A to species B.
    """
    if S is None:
        S = acc_by_sample_size(mode='others', acc_type='taxa')
    mu = S.applymap(np.mean, na_action="ignore")
    species = mu.columns
    Ns = mu.index.levels[1]
    R = {}
    for s in species:
        for s2 in species:
            if s != s2:
                acc_data = mu.loc[s, s2]
                acc_0 = acc_data.iloc[0]
                acc_f = acc_data.loc[Nf]
                R[(s, s2)] = acc_f - acc_0
    return pd.Series(R).unstack(0)

def acc_by_sample_size_filtered(taxa=None, **kwargs):
    S = acc_by_sample_size(**kwargs, mode='same')
    if taxa is not None:
        taxa_to_species = preprocessing.get_taxa_to_species(v2=True)
        species_in_taxa = taxa_to_species[taxa]
        species = [s for s in S.columns if s in species_in_taxa]
        S = S[species]
    return S


@savefig
def acc_by_sample_size_plot(xaxis_type='log', taxa=None, **kwargs):
    S = acc_by_sample_size_filtered(taxa=taxa, **kwargs)
    species = S.columns
    acc_mean = S.applymap(np.mean, na_action="ignore")
    acc_std = S.applymap(np.std, na_action="ignore")
    fig = get_figure(xaxis_title="Sample size", yaxis_title="Mean Accuracy", xaxis_type=xaxis_type, yaxis_range=[0, 1])
    for s in species:
        fig.add_trace(go.Scatter(x=acc_mean.index, y=acc_mean[s], name=s, error_y=dict(type='data', array=acc_std[s], width=8), marker=dict(size=16, symbol='square'), line_width=4))
    # fig.update_layout(legend=dict(x=0.5 if xaxis_type is None else 0.52, y=0.07))
    return fig

def resample_mean(X):
    N, R = X.shape
    idxs = np.random.randint(low=0, high=R, size=N*R).reshape(N, R)
    return np.array([x[idx].mean() for x, idx in zip(X, idxs)])

def resample_mean_uneven_shape(X):
    return np.array([np.random.choice(x, size=x.size, replace=True).mean() for x in X])

@njit
def logistic_function(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

@njit
def st_exp(x,c,tau,beta):
        return c*(np.exp(-(x/tau)**beta))
@njit
def exponential(x, a, b):
    """
    1 - acc(n) = exponential(n, a, b)
    """
    return a * np.exp(b * x)

@njit
def linear(x, a, b):
    return a + b * x

@njit
def power_law(x, a, b):
    return a * x**b

def logistic_fit(sample_sizes, accuracies):
    # Fit the logistic regression model to the data
    popt, _ = curve_fit(logistic_function, sample_sizes, accuracies, bounds=([0, 0], [np.inf, np.inf]))
    return popt

def linear_fit(sample_sizes, accuracies):
    """
    acc(n) = linear(n, a, b)
    """
    p0 = np.polyfit(sample_sizes, accuracies, 1)[::-1]
    popt, _ = curve_fit(linear, sample_sizes, accuracies, p0=p0)
    return popt

def exponential_fit(sample_sizes, accuracies):
    """
    1 - acc(n) = exponential(n, a, b)
    """
    p0 = np.polyfit(np.log(sample_sizes), 1 - accuracies, 1)[::-1]
    popt, _ = curve_fit(exponential, sample_sizes, 1 - accuracies, bounds=[[-0.1, -np.inf], [np.inf, 0.1]], p0=p0)
    return popt

@savefig('all')
def logistic_fit_plot(species, fix_zero=False, n_range=(0, 300), xaxis_type=None, **kwargs):
    S = acc_by_sample_size(**kwargs).applymap(np.mean)
    s = S[species].dropna()
    if fix_zero:
        s.loc[0] = 0
    accuracies = s.values
    sample_sizes = s.index.values
    popt = logistic_fit(sample_sizes, accuracies)
    n = np.linspace(*n_range, 100)
    fit = logistic_function(n, *popt)
    fig = get_figure(xaxis_title='Sample size', yaxis_title='⟨Accuracy⟩', yaxis_range=[0, 1], xaxis_type=xaxis_type)
    fig.add_trace(go.Scatter(x=n, y=fit, mode='lines', name='Logistic fit'))
    fig.add_trace(go.Scatter(x=sample_sizes, y=accuracies, mode='markers', name='Data', marker=dict(size=16, line=dict(color='black', width=1))))
    fig.update_layout(legend=dict(x=0.65, y=0.1))
    return fig

@savefig('all')
def complementary_acc_plot(species, stretched=False, exp_only=True, **kwargs):
    S = acc_by_sample_size(**kwargs).applymap(np.mean)
    s = S[species].dropna()
    accuracies = s.values
    sample_sizes = s.index.values
    if stretched:
        fig = get_subplots(shared_xaxes=False, shared_yaxes=False, subplot_titles=['log (exponential)', 'log-log (power-law)', 'loglog-log (stretched exp)'], x_title='Sample size', y_title='1 - ⟨Accuracy⟩', rows=1, cols=3)
        fig.update_layout(yaxis_type='log', yaxis2_type='log', xaxis2_type='log', xaxis3_type='log', yaxis3_type='log')
        for col in [1,2]:
            fig.add_trace(go.Scatter(x=sample_sizes, y=1 - accuracies, mode='lines+markers', marker=dict(size=16, line=dict(color='black', width=1)), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=sample_sizes, y=np.log(1 - accuracies), mode='lines+markers', marker=dict(size=16, line=dict(color='black', width=1)), showlegend=False), row=1, col=3)
    # y(n) = 1 - acc(n) = exp(-n)
    elif exp_only:
        fig = get_figure(xaxis_title='Sample size', yaxis_title='1 - ⟨Accuracy⟩', yaxis_type='log', simple_axes=True)
        fig.add_trace(go.Scatter(x=sample_sizes, y=1 - accuracies, mode='lines+markers', marker=dict(size=16, line=dict(color='black', width=1)), showlegend=False))
        # fit
        popt = exponential_fit(sample_sizes, accuracies)
        n = np.linspace(0, sample_sizes[-1]*1.2, 100)
        fit = exponential(n, *popt)
        fig.add_trace(go.Scatter(x=n, y=fit, mode='lines', showlegend=False))
        fig.update_layout(yaxis_nticks=5)
        return fig

    else:
        fig = get_subplots(shared_xaxes=False, shared_yaxes=False, subplot_titles=['log (exponential)', 'log-log (power-law)'], x_title='Sample size', y_title='1 - ⟨Accuracy⟩', rows=1, cols=2)
        fig.update_layout(yaxis_type='log', yaxis2_type='log', xaxis2_type='log')
        for col in [1,2]:
            fig.add_trace(go.Scatter(x=sample_sizes, y=1 - accuracies, mode='lines+markers', marker=dict(size=16, line=dict(color='black', width=1)), showlegend=False), row=1, col=col)
    return fig

def estimate_sample_size(sample_sizes, accuracies, target_accuracy, method='exponential'):
    def base_solution(func, yf):
        y = func(sample_sizes)
        closest = np.argmin(np.abs(y - yf))
        return sample_sizes[closest]

    if method == 'logistic':
        popt = logistic_fit(sample_sizes, accuracies)

        # Solve for the required sample size by setting the logistic function to the desired accuracy
        func = lambda x: logistic_function(x, *popt) - target_accuracy
        required_sample_size = int(fsolve(func, base_solution(func, target_accuracy))[0])
    elif method == 'linear':
        popt = linear_fit(sample_sizes, accuracies)
        func = lambda x: linear(x, *popt) - target_accuracy
        required_sample_size = int(fsolve(func, base_solution(func, target_accuracy))[0])
    elif method == 'exponential':
        popt = exponential_fit(sample_sizes, accuracies) # 1 - acc(n) = exponential(n, a, b)
        func = lambda x: exponential(x, *popt) - (1 - target_accuracy)
        required_sample_size = int(fsolve(func, base_solution(func, 1 - target_accuracy))[0])
    else:
        raise ValueError(f"Invalid method: {method}")
    return max(required_sample_size, 1)

def estimate_dn_slope(sample_sizes, accuracies):
    """
    Estimate the slope of the dn curve for the given sample sizes and accuracies.


    dn(y, dy) = 1/b * log [1 - (dy / (1-y))]

    returns 1/b
    """
    popt = exponential_fit(sample_sizes, accuracies) # 1 - acc(n) = exponential(n, a, b)
    return 1 / popt[1]

def compute_estimates(sample_sizes, accuracies, target_accuracy, n_resamples=10000, seed=0, use_tqdm=False, method='exponential', store_data=False):
    np.random.seed(seed)

    if method == 'exponential-slope':
        estimator = lambda acc: estimate_dn_slope(sample_sizes, acc)
    else:
        estimator = lambda acc: estimate_sample_size(sample_sizes, acc, target_accuracy, method=method)

    if isinstance(accuracies, list):
        resampler = resample_mean_uneven_shape
        acc = np.array([x.mean() for x in accuracies])
        base_stat = estimator(acc)
    else:
        resampler = resample_mean
        acc = accuracies.mean(axis=1)
        base_stat = estimator(acc)
    estimates = np.empty((n_resamples))
    data = [(sample_sizes, acc)]
    if use_tqdm:
        iterator = tqdm(range(n_resamples))
    else:
        iterator = range(n_resamples)
    for i in iterator:
        acc = resampler(accuracies)
        estimates[i] = estimator(acc)
        if store_data:
            data.append((sample_sizes, acc))
    return base_stat, estimates, data

def CI_sample_size_for_accuracy(sample_sizes, accuracies, target_accuracy=0.6, alpha=0.05, alternative='two-sided', n_resamples=1000, seed=0, use_tqdm=False, method='exponential', boot='percentile'):
    base_stat, estimates, data = compute_estimates(sample_sizes, accuracies, target_accuracy, n_resamples=n_resamples, seed=seed, use_tqdm=use_tqdm, method=method)
    if boot == 'percentile':
        CI = bootstrap._compute_CI_percentile(estimates, alpha, alternative)
    elif boot == 'bca':
        # TODO: implement bca
        def _bca_interval(data, data2, statistic, probs, theta_hat_b, account_equal, use_numba):
            """Bias-corrected and accelerated interval."""
            # calculate z0_hat
            theta_hat = statistic(data[0])
            percentile = bootstrap._percentile_of_score(theta_hat_b, theta_hat, axis=-1, account_equal=account_equal)
            z0_hat = bootstrap.ndtri(percentile)

            # calculate a_hat
            if data2 is None:
                jackknife_computer = jackknife_stat_nb if use_numba else jackknife_stat
                theta_hat_jk = jackknife_computer(data, statistic)  # jackknife resample
            else:
                theta_hat_jk = jackknife_stat_two_samples(data, data2, statistic)
            n = theta_hat_jk.shape[0]
            theta_hat_jk_dot = theta_hat_jk.mean(axis=0)

            U = (n - 1) * (theta_hat_jk_dot - theta_hat_jk)
            num = (U**3).sum(axis=0) / n**3
            den = (U**2).sum(axis=0) / n**2
            a_hat = 1/6 * num / (den**(3/2))

            # calculate alpha_1, alpha_2
            def compute_alpha(p):
                z_alpha = ndtri(p)
                num = z0_hat + z_alpha
                return ndtr(z0_hat + num/(1 - a_hat*num))
            alpha_bca = compute_alpha(probs[(probs != 0) & (probs != 1)])
            if (alpha_bca > 1).any() or (alpha_bca < 0).any():
                warnings.warn('percentiles must be in [0, 1]. bca percentiles: {}\nForcing percentiles in [0,1]...'.format(alpha_bca), RuntimeWarning)
                alpha_bca = np.clip(alpha_bca, 0, 1)
            return alpha_bca, a_hat  # return a_hat for testing

        probs = np.array([alpha/2, 1 - alpha/2])
        account_equal = False
        use_numba = False
        theta_hat_b = estimates
        data2 = None
        def statistic(data):
            sample_sizes, accuracies = data
            return estimate_sample_size(sample_sizes, accuracies, target_accuracy, method=method)

        alpha_bca = bootstrap._bca_interval(data, data2, statistic, probs, theta_hat_b, account_equal, use_numba)[0]
        if np.isnan(alpha_bca).all():
            warnings.warn('CI shows there is only one value. Check data.', RuntimeWarning)
            if data2 is None:
                sample_stat = statistic(data)
            else:
                sample_stat = statistic(data, data2)
            CI = np.array([sample_stat, sample_stat])
        else:
            CI = bootstrap._compute_CI_percentile(theta_hat_b, alpha_bca, alternative)
    else:
        raise ValueError(f"Invalid boot method: {boot}")
    return base_stat, CI

def _update_sample_size_for_accuracy(s, n, label, target_accuracy, alpha, n_resamples, seed, method='exponential', boot='percentile'):
    try:
        accuracies = np.vstack(s.values)
        if np.isnan(accuracies).any():
            raise ValueError('NaNs in accuracies')
    except:
        def remove_nans(x):
            x = np.array(x)
            return x[~np.isnan(x)]
        accuracies = [remove_nans(x) for x in s.values]
    sample_sizes = np.array(s.index)
    if method == 'exponential':
        n_min = params.species_to_nmin_fit[label]
        n_max = params.species_to_nmax_fit[label]
        s = pd.Series(list(accuracies), index=sample_sizes)
        s_mean = s.apply(np.mean)
        if n_min is None or n_max is None:
            if n_min is None:
                jump_threshold = 0.2
                dacc = s_mean.diff().iloc[:4]
                dacc_sorted = dacc.sort_values(ascending=False)
                if dacc_sorted.iloc[0] > jump_threshold and not dacc_sorted.iloc[1] > jump_threshold: # only one jump
                    n_min = dacc_sorted.index[0]
                else:
                    n_min = 5
            if n_max is None:
                is_acc_1 = s_mean > 0.98
                if is_acc_1.sum() >= 2:
                    n_max = is_acc_1.index[np.where(is_acc_1)[0][0]]
                else:
                    n_max = s.index[-1]
        if s.index.duplicated().any():
            target_acc_smaller = target_accuracy < s_mean.loc[n_min].mean()
        else:
            target_acc_smaller = target_accuracy < s_mean.loc[n_min]
        if n_min > 5 and target_acc_smaller:
            warnings.warn(f"Using linear regime to estimate sample size. The accuracy has a jump at the beginning and the {method} method is not valid.")
            method = 'linear'
            s = s[s.index <= n_min]
        else:
            s = s.loc[(s.index >= n_min) & (s.index <= n_max)]
        sample_sizes = np.array(s.index)
        try:
            accuracies = np.vstack(s.values)
        except:
            accuracies = list(s.values)
    base_stat, CI = CI_sample_size_for_accuracy(sample_sizes, accuracies, target_accuracy=target_accuracy, alpha=alpha, alternative='two-sided', n_resamples=n_resamples, seed=seed, method=method, boot=boot)
    n[(label, 'CI')] = CI
    n[(label, 'base')] = base_stat
    return

@savedata
def sample_size_for_accuracy_species(target_accuracy=0.6, n_resamples=1000, alpha=0.05, seed=0, taxa=None, method='exponential', boot='percentile', **kwargs):
    S = acc_by_sample_size_filtered(taxa=taxa, **kwargs)

    n = {}
    for species in tqdm(S.columns):
        s = S[species].dropna()
        _update_sample_size_for_accuracy(s, n, species, target_accuracy, alpha, n_resamples, seed, method=method, boot=boot)
    n = pd.Series(n).unstack(0)
    return n

@savedata
def dn_slope_species(**kwargs):
    assert 'method' not in kwargs or kwargs['method'] == 'exponential-slope', 'method must be exponential-slope'
    return sample_size_for_accuracy_species(method='exponential-slope', **kwargs)

@savedata
def sample_size_for_accuracy_taxas(target_accuracy=0.6, n_resamples=1000, alpha=0.05, seed=0, method='exponential', boot='percentile', **kwargs):
    n = {}
    for taxa in tqdm(preprocessing.get_taxa_to_species().keys()):
        S = acc_by_sample_size_filtered(taxa=taxa, **kwargs)
        s = S.melt(ignore_index=False)['value'].dropna()
        s = s.groupby(s.index, group_keys=False).apply(lambda x: np.hstack(x.values)) # group by sample size
        if s.size:
            _update_sample_size_for_accuracy(s, n, taxa, target_accuracy, alpha, n_resamples, seed, method=method, boot=boot)
    n = pd.Series(n).unstack(0)
    return n

@savedata
def dn_slope_taxas(**kwargs):
    assert 'method' not in kwargs or kwargs['method'] == 'exponential-slope', 'method must be exponential-slope'
    return sample_size_for_accuracy_taxas(method='exponential-slope', **kwargs)

@savefig('all-yrange-ciwidth')
def sample_size_for_accuracy_species_plot(yrange=[0, 400], target_accuracy=0.6, ciwidth=0.2, **kwargs):
    n = sample_size_for_accuracy_species(target_accuracy=target_accuracy, **kwargs)
    fig = CI_plot(n.columns, n.loc['base'].values, np.vstack(n.loc['CI'].values), x_title='Species', y_title='n for {}% accuracy'.format(int(100*target_accuracy)), width=ciwidth)
    fig.update_layout(yaxis_range=yrange)
    fig.update_layout(xaxis_tickangle=-90)
    fig.update_layout(plot_bgcolor='white', yaxis=dict(showline=True, linecolor='black', linewidth=2.4),
                      xaxis=dict(showline=True, linecolor='black', linewidth=2.4))
    return fig

@savefig('all-yrange-ciwidth')
def sample_size_for_accuracy_taxas_plot(yrange=[0, 400], target_accuracy=0.6, ciwidth=0.2, **kwargs):
    n = sample_size_for_accuracy_taxas(target_accuracy=target_accuracy, **kwargs)
    fig = CI_plot(n.columns, n.loc['base'].values, np.vstack(n.loc['CI'].values), x_title='Taxa', y_title='n for {}% accuracy'.format(int(100*target_accuracy)), width=ciwidth)
    fig.update_layout(yaxis_range=yrange)
    fig.update_layout(xaxis_tickangle=-90)
    fig.update_layout(plot_bgcolor='white', yaxis=dict(showline=True, linecolor='black', linewidth=2.4),
                      xaxis=dict(showline=True, linecolor='black', linewidth=2.4))
    return fig

@savefig('target_accuracy')
def sample_size_for_accuracy_taxas_comp(ciwidth=0.1, target_accuracy=0.6):
    specs = params.sample_size_for_acc_specs
    n_env = sample_size_for_accuracy_taxas(v2=True, target_accuracy=target_accuracy, **specs['env'])
    n_co = sample_size_for_accuracy_taxas(v2=True, target_accuracy=target_accuracy, **specs['common_origin'])
    order = ['Penguins', 'Birds', 'Seals', 'Cetaceans', 'Turtles', 'Fishes']
    n_env = n_env[order]
    n_co = n_co[order]

    colors = plotly_default_colors(2)
    colors = {'env': colors[1], 'common_origin': colors[0]}

    fig = CI_plot(n_env.columns, n_env.loc['base'].values, np.vstack(n_env.loc['CI'].values), x_title='Taxa', y_title='n for {}% accuracy'.format(int(100*target_accuracy)), simple_axes=True, width=ciwidth, color=color_std(colors['env']), color_sample_stat=colors['env'], label='Geo+env', color_legend=color_std(colors['env'], opacity=0.6))
    fig = CI_plot(n_co.columns, n_co.loc['base'].values, np.vstack(n_co.loc['CI'].values), fig=fig, width=ciwidth, color=color_std(colors['common_origin']), color_sample_stat=colors['common_origin'], label='Common origin', color_legend=color_std(colors['common_origin'], opacity=0.6))
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    return fig

@savefig('acc_1+acc_2')
def sample_size_for_accuracy_taxas_comp_2(ci_width=0.1, acc_1=0.6, acc_2=0.8, yrange=[0, 1250], yaxis_type='log'):
    """
    Equivalent of sample_size_for_accuracy_taxas_comp but with two target accuracies.
    """

    if yaxis_type == 'log':
        yrange = np.log10(yrange)

    specs = params.sample_size_for_acc_specs

    colors = plotly_default_colors(2)
    colors = {'env': colors[1], 'common_origin': colors[0]}
    order = ['Penguins', 'Birds', 'Seals', 'Cetaceans', 'Turtles', 'Fishes']
    fig = None
    def plot(fig, color, acc, label, kwargs):
        n = sample_size_for_accuracy_taxas(v2=True, target_accuracy=acc, **kwargs)
        n = n[order]
        if fig is None:
            fig_kwargs = dict(x_title='Taxa', y_title='Sample size', simple_axes=True)
        else:
            fig_kwargs = dict()
        fig = CI_plot(n.columns, n.loc['base'].values, np.vstack(n.loc['CI'].values), fig=fig, width=ci_width, color=color_std(color), color_sample_stat=color, label=label, color_legend=color_std(color, opacity=0.7), **fig_kwargs)
        return fig

    # estimate sample size for reaching acc_1 and acc_2
    fig = plot(fig, colors['env'], acc_1, 'Geo+env', specs['env'])
    fig = plot(fig, colors['common_origin'], acc_1, 'Common origin', specs['common_origin'])
    fig = plot(fig, color_gradient(colors['env'], 'black', 6)[2], acc_2, None, specs['env'])
    fig = plot(fig, 'navy', acc_2, None, specs['common_origin'])

    # adding sample size present in the dataset
    df = analysis.classification_report_random_states(common_origin_distance=False, weather='all', v2=True, random_states=range(1, 6))
    unique = df.loc[~df.ID.duplicated()]
    n = unique.groupby('COMMON_NAME').size().to_frame('n')
    species_to_taxa = preprocessing.get_species_to_taxa()
    n['Taxa'] = n.index.map(species_to_taxa)
    n_mean = n.groupby('Taxa').mean()
    fig.add_trace(go.Scatter(x=np.arange(len(order), dtype=int), y=n_mean.loc[order].values.flatten(), mode='markers',
                             marker=dict(size=20, symbol='star', color='gold', line=dict(color='black', width=2), opacity=1), showlegend=False))

    # legend
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                      yaxis=dict(range=yrange, type=yaxis_type),
                      )
    if yaxis_type == 'log':
        fig.update_layout(yaxis_tickvals=[10, 100, 1000], yaxis_ticktext=[10, 100, 1000])
    return fig

@savefig
def dn_slope_taxas_comp(ci_width=0.1, yrange=[10,1000]):
    colors = plotly_default_colors(2)
    colors = {'env': colors[1], 'common_origin': colors[0]}
    specs = params.sample_size_for_acc_specs
    order = ['Penguins', 'Birds', 'Seals', 'Cetaceans', 'Turtles', 'Fishes']
    fig = None
    def plot(fig, color, label, kwargs):
        kwargs.pop('method', None)
        n = dn_slope_taxas(v2=True, **kwargs)
        n = n[order] * (-1)
        n.loc['CI'] = n.loc['CI'].apply(lambda x: x[:, ::-1])
        if fig is None:
            fig_kwargs = dict(x_title='Taxa', y_title='-b<sup>-1</sup>', simple_axes=True)
        else:
            fig_kwargs = dict()
        fig = CI_plot(n.columns, n.loc['base'].values, np.vstack(n.loc['CI'].values), fig=fig, width=ci_width, color=color_std(color), color_sample_stat=color, label=label, color_legend=color_std(color, opacity=0.7), **fig_kwargs)
        return fig

    fig = plot(fig, colors['env'], 'Geo+env', specs['env'])
    fig = plot(fig, colors['common_origin'], 'Common origin', specs['common_origin'])
    fig.update_layout(xaxis_tickangle=-90,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    fig.update_layout(yaxis=dict(type='log', range=np.log10(yrange), tickvals=[10, 100, 1000], ticktext=['10', '100', '1000']))
    return fig

@savefig
def dn_dacc_slope_taxas_comp(ci_width=0.1, acc_base=0.7, acc_diff=0.1):
    colors = plotly_default_colors(2)
    colors = {'env': colors[1], 'common_origin': colors[0]}
    specs = params.sample_size_for_acc_specs
    order = ['Penguins', 'Birds', 'Seals', 'Cetaceans', 'Turtles', 'Fishes']
    fig = None
    def plot(fig, color, label, kwargs):
        kwargs.pop('method', None)
        n = dn_slope_taxas(v2=True, **kwargs)
        n = n[order] * np.log(1 - (acc_diff / (1 - acc_base)))
        n.loc['CI'] = n.loc['CI'].apply(lambda x: x[:, ::-1])
        if fig is None:
            # y_title is dn (acc_base -> acc_base + acc_diff)
            y_title = 'dn<sub>acc: {:.1f}→{:.1f}</sub>'.format(acc_base, acc_base + acc_diff)
            fig_kwargs = dict(x_title='Taxa', y_title=y_title, simple_axes=True)
        else:
            fig_kwargs = dict()
        fig = CI_plot(n.columns, n.loc['base'].values, np.vstack(n.loc['CI'].values), fig=fig, width=ci_width, color=color_std(color), color_sample_stat=color, label=label, color_legend=color_std(color, opacity=0.7), **fig_kwargs)
        return fig

    fig = plot(fig, colors['env'], 'Geo+env', specs['env'])
    fig = plot(fig, colors['common_origin'], 'Common origin', specs['common_origin'])
    fig.update_layout(xaxis_tickangle=-90,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    fig.update_layout(yaxis=dict(type='log', range=np.log10([1, 1000]), tickvals=[1, 10, 100, 1000], ticktext=['1', '10', '100', '1000']))
    return fig
