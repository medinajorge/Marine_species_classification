import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
try:
    from umap import UMAP
except:
    pass
from . import other_utils
from pathlib import Path
import os
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    pass

def distance_matrix(X):
    """X: array of coordinates. shape: (N, d)"""
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        j = i+1
        D[i, j:] = np.square(X[i][None] - X[j:]).sum(axis=1)
    D = np.sqrt(D)
    D += D.T
    return D

def stress(D, D_trans, normalize=True):
    from scipy.spatial.distance import squareform
    """
    Returns stress when computing MultiDimensional Scaling.

    Attributes:
        - D: Original distance matrix. Admits condensed version.
        - D_trans:    Distance matrix reconstructed from the MDS coordinates. Admits condensed version.
        - normalize:  True  => computes normalized stress (stress-1).
                               Value 0 indicates "perfect" fit, 0.025 excellent, 0.05 good, 0.1 fair, and 0.2 poor.
                               source: “Nonmetric multidimensional scaling: a numerical method” Kruskal, J. Psychometrika, 29 (1964), page 3.
                      False => Raw stress.
    Returns: stress value for the MDS fit.
    """
    to_condensed = lambda X: X.copy() if X.shape[0] != X.shape[1] else squareform(X)
    d = to_condensed(D)
    d_trans = to_condensed(D_trans)
    if normalize:
        sigma = np.sqrt(np.square(np.square(d_trans - d).sum() / np.square(d).sum()))
    else:
        sigma = np.square(d - d_trans).sum()
    return sigma

def stress_by_dimensionality(X, n_components=[*range(1, 11)], plot=True, figDir="figs/dimensionality_reduction/error/by_dimensionaly", figname="default"):
    """X: distance matrix"""
    results = defaultdict(list)
    for n in tqdm(n_components):
        model = MDS(n_components=n, metric=True, n_init=4,max_iter=1000,verbose=0, eps=0.001, n_jobs=4, random_state=42, dissimilarity='precomputed')
        X_trans = model.fit_transform(X.copy())
        D_trans = distance_matrix(X_trans)
        results["stress"].append(stress(X, D_trans))
        results["iterations"].append(model.n_iter_)

    if plot:
        fig = data_visualization.get_subplots(2, subplot_titles=["Stress", "Iterations"], shared_xaxes=True, width=1200, height=400, x_title="Number of components",
                                     horizontal_spacing=0.06)
        for col, y in enumerate(results.values(), start=1):
            fig.add_trace(go.Scatter(x=n_components, y=y, mode="lines", showlegend=False, name=None), row=1, col=col)
        fig.update_layout(margin=dict(l=20, r=20, b=80, t=50, pad=0))

        Path(figDir).mkdir(exist_ok=True, parents=True)
        fig.write_image(os.path.join(figDir, f"{figname}.png"))

    return n_components, results


def to_df(X, identifier_cols={}, mode=None, sort=False):
    df = pd.DataFrame(X, columns=[f"Component {i}" for i in range(1, X.shape[1]+1)])
    if len(identifier_cols) > 0:
        for k, v in identifier_cols.items():
            df[k] = v
        if sort:
            df = df.sort_values([*identifier_cols.keys()])
    df = df.assign(Algorithm=mode.upper())
    return df

def reduce_dim(X, identifier_cols={}, mode="tsne", sort=False, **kwargs):
    default = dict(pca = dict(func=PCA, attrs=dict(n_components=2)),
                   tsne = dict(func=TSNE, attrs=dict(n_components=2, perplexity=5, learning_rate=10.0, n_iter=1000, random_state=0)),
                   umap = dict(func=UMAP, attrs=dict(random_state=999, n_neighbors=15, min_dist=.01, metric="manhattan")),
                   mds = dict(func=MDS, attrs=dict(n_components=3, metric=True, n_init=4, max_iter=1000, verbose=0, eps=0.001, n_jobs=4, random_state=42, dissimilarity='precomputed'))
                  )
    specs = default[mode]
    specs["attrs"].update(kwargs)

    X_embedded = specs["func"](**specs["attrs"]).fit_transform(X)
    df = to_df(X_embedded, identifier_cols, mode, sort)
    return df



def plot(df, plot="lm", hue=None, figDir="figs/dimensionality_reduction", figdict={}, figname="default", **kwargs):
    Path(figDir).mkdir(exist_ok=True, parents=True)
    specs = dict(s=100, alpha=0.7)
    specs.update(kwargs)

    if hue is None:
        hue = [c for c in df.columns.to_list() if not c.startswith("Component")][0]

    if plot == "scatter":
        fig = sns.scatterplot(x='Component 1', y='Component 2', data=df, hue=hue,
                                   linewidth=0, **specs
                                  )
        fig.legend(loc='center left', bbox_to_anchor=(1, .5))
    elif plot == "lm":
        g = sns.lmplot(x='Component 1',
               y='Component 2',
               data=df,
               fit_reg=False,
               legend=False,
               size=9,
               hue=hue,
               palette="flare",
               scatter_kws=specs)

        plt.title(f'{df["Algorithm"].iloc[0]} clustering', weight='bold').set_fontsize('26')
        plt.xlabel('Component 1', weight='bold').set_fontsize('28')
        plt.ylabel('Component 2', weight='bold').set_fontsize('28')
        plt.legend(fontsize=16)
        g.fig.set_size_inches(18.5, 10.5)
        fig = g.fig
    else:
        raise ValueError(f"plot {plot} not valid. Available: 'scatter', 'lm'.")
    figdict.update(dict(plot=plot, hue=hue))
    fig.savefig(os.path.join(figDir, f"{other_utils.dict_to_id(figdict)}_{figname}.png"))
    return fig
