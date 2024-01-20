import numpy as np
import pandas as pd
import signal
import warnings
try:
    import shap
except:
    pass
try:
    import xgboost as xgb
except:
    pass
import os
from . import analysis, preprocessing
import matplotlib.pyplot as plt
try:
    import networkx as nx
    from infomap import Infomap
    from umap import UMAP
except:
    pass
try:
    import dtw
except:
    pass
import sys
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

import plotly.graph_objects as go
import plotly.express as px
from phdu import savedata, savefig, SavedataSkippedComputation
from phdu.plots.plotly_utils import get_figure
from phdu.plots.base import plotly_default_colors, color_gradient
from phdu import clustering, _helper, bootstrap

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)

def create_graph(cm, edge_threshold=0.):
    G = nx.DiGraph()
    G.add_nodes_from(cm.columns)
    for i, row in cm.iterrows():
        for j, value in row.items():
            if j != i and value > edge_threshold:
                G.add_edge(i, j, weight=value)
    return G

@savedata(save=False, keys='weight_threshold+seeds')
def infomap_communities(G, int_to_cat, weight_threshold=None, seeds=range(1, 201), return_infomap_data=True):
    cat_to_int = {v:k for k, v in int_to_cat.items()}
    time_limit = 5
    def handler(signum, frame):
        raise TimeoutError(f"Time limit of {time_limit} seconds exceeded")
    def compute_infomap(seed):
        infomap = Infomap(directed=True, weight_threshold=weight_threshold, verbosity_level=0)
        for u, v, w in G.edges.data('weight'):
            infomap.add_link(cat_to_int[u], cat_to_int[v], w)
        infomap.run(seed=seed)
        return infomap
    best = compute_infomap(1000)
    best_codelength = best.codelength
    best_seed = 1000
    record = {1000: best_codelength}
    for seed in seeds:
        signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(time_limit)
            infomap = compute_infomap(seed)
            signal.alarm(0)
        except TimeoutError:
            warnings.warn(f"Timeout for seed {seed}")
            record[seed] = np.nan
            continue
        codelength = infomap.codelength
        record[seed] = codelength
        if codelength < best_codelength:
            best_codelength = codelength
            best = infomap
            best_seed = seed

    clusters = pd.Series(best.get_modules())
    clusters.index = [int_to_cat[i] for i in clusters.index]
    record = pd.Series(record).sort_values()

    infomap_data = dict(best=best, best_codelength=best_codelength, best_seed=best_seed, record=record)
    if return_infomap_data:
        return clusters, infomap_data
    else:
        return clusters


def modularity(G, clusters):
    community_dict = clusters.to_dict()
    Q = 0
    m = G.size(weight='weight')

    for node_i, node_j, data in G.edges(data=True):
        if node_i in community_dict and node_j in community_dict:
            if community_dict[node_i] == community_dict[node_j]:
                Q += data['weight'] - G.degree(node_i, weight='weight') * G.degree(node_j, weight='weight') / (2 * m)

    modularity =  Q / (2 * m)
    return modularity

def cluster_data(G, clusters):
    cluster_to_nodes = clusters.groupby(clusters).groups
    community_sizes = {cluster_id: len(nodes) for cluster_id, nodes in cluster_to_nodes.items()}
    intra_community_weights = {
        cluster_id: sum(G[u][v]['weight'] for u in nodes for v in nodes if G.has_edge(u, v))
        for cluster_id, nodes in cluster_to_nodes.items()
    }
    average_intra_community_weights = {
        cluster_id: total_weight / community_sizes[cluster_id]
        for cluster_id, total_weight in intra_community_weights.items()
    }
    inter_community_weights = {
        (cluster_id1, cluster_id2): sum(G[u][v]['weight'] for u in cluster_to_nodes[cluster_id1] for v in cluster_to_nodes[cluster_id2] if G.has_edge(u, v))
        for cluster_id1 in cluster_to_nodes for cluster_id2 in cluster_to_nodes if cluster_id1 != cluster_id2
    }
    communities_data = pd.DataFrame([community_sizes, average_intra_community_weights, intra_community_weights]).T
    communities_data.columns = ['community_sizes', 'average_intra_community_weights', 'intra_community_weights']
    inter_community_weights = pd.Series(inter_community_weights).sort_values(ascending=False)
    total_intercommunity_weights = inter_community_weights.unstack().sum(axis=1)
    communities_data['inter_community_weights'] = total_intercommunity_weights
    return communities_data, inter_community_weights, cluster_to_nodes

def describe_community_data(G, clusters):
    results = {}
    communities_data, *_ = cluster_data(G, clusters)
    results['total_confusion'] = nx.adjacency_matrix(G, weight='weight').toarray().sum()
    results['intra_community_percentage_total'] = communities_data['intra_community_weights'].sum() / results['total_confusion']
    results['intra_community_percentage'] = communities_data['intra_community_weights'] / results['total_confusion']
    results['intra_community_percentage_cum'] = communities_data['intra_community_weights'].cumsum() / results['total_confusion']
    results['intra_community_intra_percentage_cum'] = communities_data['intra_community_weights'].cumsum() / (results['total_confusion']*results['intra_community_percentage_total'])
    results['intra_community_node_percentage'] = communities_data['intra_community_weights'] / (communities_data['intra_community_weights'] + communities_data['inter_community_weights'])
    return pd.Series(results)

def sensitivity_analyis(cm, weight_thresholds=np.linspace(0, 0.2, 40), create_G=True, **infomap_kwargs):
    """
    Returns a dataframe with the following columns:
    - weight_threshold: the weight threshold used
    - codelength: the codelength of the best infomap run
    - modularity: the modularity of the best infomap run
    - intra_community_percentage_total: the percentage of the total confusion (edge weight) that is intra-community
    - intra_community_node_percentage: Average of the percentage of confusion (edge weight) that is intra-community per node
    - size: Average size of the communities
    - n_clusters: Number of communities

    cm: confusion matrix
    weight_thresholds: list of thresholds to use for the edge weights
    create_G: whether to create the graph or not. If not, weight thresholds are passed to infomap
    """
    species = cm.columns.to_list()
    int_to_species = {i: s for i, s in enumerate(species)}
    if not create_G:
        G = create_graph(cm, edge_threshold=0)
    sensitivity_data = {}
    for weight_threshold in tqdm(weight_thresholds):
        if create_G:
            G = create_graph(cm, edge_threshold=weight_threshold)
            clusters, infomap_data = infomap_communities(G, int_to_cat=int_to_species, weight_threshold=None, **infomap_kwargs)
        else:
            clusters, infomap_data = infomap_communities(G, int_to_cat=int_to_species, weight_threshold=weight_threshold, **infomap_kwargs)
        data = describe_community_data(G, clusters)
        sensitivity_data[(weight_threshold, 'codelength')] = infomap_data['best_codelength']
        sensitivity_data[(weight_threshold, 'modularity')] = modularity(G, clusters)
        sensitivity_data[(weight_threshold, 'intra_community_percentage_total')] = data['intra_community_percentage_total']
        sensitivity_data[(weight_threshold, 'intra_community_node_percentage')] = data['intra_community_node_percentage'].mean()
        sensitivity_data[(weight_threshold, 'size')] = clusters.value_counts().mean()
        sensitivity_data[(weight_threshold, 'n_clusters')] = clusters.nunique()
    sensitivity_data = pd.Series(sensitivity_data).unstack()
    sensitivity_data.index.name = 'weight_threshold'
    return sensitivity_data

@savedata
def dissimilarity_graph_sensitivity_analysis(weather='all', common_origin_distance=False, random_states=range(1,6), **kwargs):
    df = analysis.classification_report_random_states(weather=weather, common_origin_distance=common_origin_distance, random_states=random_states)
    cm, _ = analysis.compute_confusion_matrix(df=df, artificial_trajectory=[])
    return sensitivity_analyis(cm, **kwargs)

def visualize_clusters_with_taxa(G, clusters, node_scaling='in-degree', lw=40, min_node_size=1000, layout='umap', threshold_edge=0, max_node_size=10000, arrowsize=30, taxa_legend_out=False, avoid_overlap=False, **layout_kwargs):
    # Mock species to taxa dictionary for illustration
    species_to_taxa = preprocessing.get_species_to_taxa()

    # Unique shapes for each taxa
    taxa_shapes = {'Fishes': 'o', 'Cetaceans': 's', 'Turtles': '^', 'Sirenians': 'v', 'Seals': 'p', 'Polar bears': '8',  'Birds': '>', 'Penguins': '<'}
    # Colors for clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters.unique())))
    cluster_to_color = {cluster: colors[i] for i, cluster in enumerate(sorted(clusters.unique()))}
    node_colors = np.array([cluster_to_color[clusters[node]] for node in G.nodes()])

    # Node shapes based on taxa
    node_shapes = [taxa_shapes[species_to_taxa[node]] for node in G.nodes()]

    # Adjusted node sizes
    if node_scaling == 'cc':
        coeff = nx.clustering(G)
        weighted_coeffs = [sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)]) * coeff[node] for node in G.nodes()]
        node_sizes = np.array([coeff * max_node_size for coeff in weighted_coeffs])  # multiplier for visibility
    elif node_scaling == 'out-degree':
        outer_edges_weights = [sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)]) for node in G.nodes()]
        node_sizes = np.array(outer_edges_weights) * max_node_size
    elif node_scaling == 'in-degree':
        indegree_weights = [sum(d['weight'] for u, v, d in G.in_edges(node, data=True)) for node in G.nodes()]
        node_sizes = np.array(indegree_weights) * max_node_size
    else:
        raise ValueError("node_scaling must be one of 'cc', 'out-degree', 'in-degree'")
    node_sizes = np.clip(node_sizes, min_node_size, max_node_size)

    if layout == 'shell':
        shells = []
        for cluster in np.unique(clusters)[::-1]:
            shells.append([node for node in G.nodes() if clusters[node] == cluster])
        pos = nx.shell_layout(G, shells, **layout_kwargs)
    elif layout == 'spring':
        pos = nx.spring_layout(G, **layout_kwargs)
    elif layout == 'kamada-kawai':
        pos = nx.kamada_kawai_layout(G, weight='weight', **layout_kwargs)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G, weight='weight', **layout_kwargs)
    elif layout == 'umap':
        defaults = dict(n_neighbors=3, metric='precomputed', random_state=42)
        layout_kwargs = {**defaults, **layout_kwargs}
        A = nx.adjacency_matrix(G, weight='weight')
        A = A.todense()
        D = A.max() - A # dissimilarity matrix (distance matrix)
        reducer = UMAP(n_components=2, **layout_kwargs)
        D_reduced = reducer.fit_transform(D)
        pos = {node: tuple(D_reduced[i]) for i, node in enumerate(G.nodes())}
    if avoid_overlap:
        X = np.array([*pos.values()])[:, 0]
        distances_X = np.abs(X[:, None] - X[None, :])
        for i, (s, (x, y)) in enumerate(pos.items()):
            if distances_X[i].min() < 0.1:
                pos[s] = (x + 0.1, y)
                distances_X[:, i] = np.abs(X - x - 0.1)

    # Draw the graph
    plt.figure(figsize=(25, 20))
    nodes = np.array([*G.nodes()])
    for taxa, shape in taxa_shapes.items():
        is_taxa = np.array([species_to_taxa[node] == taxa for node in nodes])
        nx.draw_networkx_nodes(G, pos, node_shape=shape, node_size=node_sizes[is_taxa], node_color=node_colors[is_taxa], alpha=0.7, nodelist=nodes[is_taxa])
    # Only label nodes with large sizes (indicating frequent misclassification or high local clustering)
    # large_nodes = [node for node, size in zip(G.nodes(), node_sizes) if size > 300]  # adjust the size threshold
    # labels = {node: node for node in large_nodes}
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    edge_widths = [G[u][v]['weight']*lw for u, v in G.edges()]  # multiply by a constant to make it more visible
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold_edge]
    # intra-cluster edge colors are the same as the node colors
    edge_colors = []
    style = []
    for u, v in strong_edges:
        if clusters[u] == clusters[v]:
            edge_colors.append(cluster_to_color[clusters[u]])
            style.append('solid')
        else:
            edge_colors.append('black')
            style.append((0, (5, 10)))
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, width=edge_widths, alpha=0.5, edge_color=edge_colors, style=style, arrowsize=arrowsize)

    # Legend for clusters
    cluster_to_taxa = {cluster: [species_to_taxa[node] for node in nodes if clusters[node] == cluster] for cluster in clusters.unique()}
    # sort taxa by frequency
    cluster_to_taxa = {cluster: sorted(taxas, key=lambda x: taxas.count(x), reverse=True) for cluster, taxas in cluster_to_taxa.items()}
    # unique respecting the order of appearance
    cluster_to_taxa = {cluster: pd.Series(taxas).unique() for cluster, taxas in cluster_to_taxa.items()}
    cluster_to_symbol = {cluster: [taxa_shapes[taxa] for taxa in taxas] for cluster, taxas in cluster_to_taxa.items()}

    symbol_to_unicode = {'s': '■',
                         'o': '●',
                         '<': '◀',
                         '>': '▶',
                         '^': '▲',
                         'v': '▼',
                         'p': '⬟',
                         '8': '⬢'}
    cluster_to_symbol = {cluster: "".join([symbol_to_unicode[symbol] for symbol in symbols]) for cluster, symbols in cluster_to_symbol.items()}

    legend_markers_clusters = [plt.Line2D([0], [0], marker='s', color='w', label='{}: {}'.format(cluster + 1, cluster_to_symbol[cluster]), markersize=20, markerfacecolor=color) for cluster, color in cluster_to_color.items()]
    leg1 = plt.legend(handles=legend_markers_clusters, loc='upper left', title="Community", title_fontsize=36, labelspacing=0.4, handletextpad=0.5)
    plt.gca().add_artist(leg1)  # To ensure that the first legend is not overridden by the second one

    # Add legend for taxa
    unique_shapes = np.unique(node_shapes)
    taxa_shapes_prunned = {taxa: shape for taxa, shape in taxa_shapes.items() if shape in unique_shapes}
    legend_markers = [plt.Line2D([0], [0], marker=shape, color='w', label=taxa, markersize=25, markerfacecolor='black') for taxa, shape in taxa_shapes_prunned.items()]
    if taxa_legend_out:
        plt.legend(handles=legend_markers, loc='center right', bbox_to_anchor=(1.22, 0.8), title="Taxa", title_fontsize=36, labelspacing=0.45)
    else:
        plt.legend(handles=legend_markers, loc='upper right', title="Taxa", title_fontsize=36, labelspacing=0.45)

    plt.title("Misclassification Communities", fontsize=44)
    # leave space for the legend
    min_x, max_x = min(x for x, y in pos.values()), max(x for x, y in pos.values())
    min_y, max_y = min(y for x, y in pos.values()), max(y for x, y in pos.values())
    # plt.xlim(min_x - 1.3, max_x + 0.3)
    plt.ylim(min_y - 0.2, max_y + 0.8)
    return plt.gcf()

@savefig("all-min_intra_community_avg_weight-seeds")
def community_visualization(weather='all', common_origin_distance=False, random_states=range(1,6), edge_threshold=0.05, min_intra_community_avg_weight=0, seeds=range(1, 201), **fig_kwargs):
    df = analysis.classification_report_random_states(weather=weather, common_origin_distance=common_origin_distance, random_states=random_states)
    cm, _ = analysis.compute_confusion_matrix(df=df, artificial_trajectory=[])
    int_to_species = {i: s for i, s in enumerate(cm.columns)}
    G = create_graph(cm, edge_threshold=edge_threshold)
    clusters, _ = infomap_communities(G, int_to_cat=int_to_species, weight_threshold=None, seeds=seeds)
    # name clusters by number of species
    cluster_to_nodes = clusters.groupby(clusters).groups
    cluster_to_size = {cluster_id: len(nodes) for cluster_id, nodes in cluster_to_nodes.items()}
    cluster_to_size = pd.Series(cluster_to_size).sort_values(ascending=False)
    cluster_to_size = cluster_to_size.reset_index().rename(columns={'index': 'cluster_id', 0: 'size'})
    new_to_old = cluster_to_size['cluster_id'].to_dict()
    old_to_new = {v: k for k, v in new_to_old.items()}
    clusters = clusters.replace(old_to_new)

    communities_data, _, cluster_to_nodes = cluster_data(G, clusters)
    valid_clusters = communities_data[communities_data['average_intra_community_weights'] > min_intra_community_avg_weight].index
    valid_nodes = [node for c in valid_clusters for node in cluster_to_nodes[c]]
    # prune G
    G_prunned = G.subgraph(valid_nodes)
    clusters_prunned = clusters[clusters.isin(valid_clusters)]
    return visualize_clusters_with_taxa(G_prunned, clusters_prunned, **fig_kwargs)

def community_trajectory_visualization_iter(cluster=range(1, 12), hub_to_gold=[True, False], **kwargs):
    df = preprocessing.load_all_data(v2=True, weather=None, return_labels=False)
    df.index = df.index.droplevel(0)
    for c in cluster:
        for h in hub_to_gold:
            community_trajectory_visualization(cluster=c, hub_to_gold=h, df=df, **kwargs)
    return

@savefig('all-df-height-width-min_size')
def community_trajectory_visualization(df=None, cluster=1, hub_to_gold=False, height=1500, width=2000, num_points=20, min_size=7):
    clf_report = analysis.classification_report_random_states(weather='all', common_origin_distance=False, random_states=range(1, 6))
    cm, _ = analysis.compute_confusion_matrix(df=clf_report, artificial_trajectory=[])
    edge_threshold = 0.06666666666666666666666667
    G = create_graph(cm, edge_threshold=edge_threshold)
    int_to_species = {i: s for i, s in enumerate(cm.columns)}
    in_degree = pd.Series(dict(G.in_degree(weight='weight')))

    clusters = infomap_communities(G, int_to_cat=int_to_species, weight_threshold=None, seeds=[12], return_infomap_data=False, save=True)
    clusters_to_n = clusters.value_counts()
    clusters = clusters[clusters.map(clusters_to_n) > 1]
    clusters_to_n = clusters_to_n[clusters_to_n > 1]
    remap = pd.Series(range(1, clusters_to_n.size+1), index=clusters_to_n.index)
    clusters = clusters.map(remap)
    # cluster_groups = clusters.groupby(clusters).apply(lambda x: x.index.tolist())
    # species_to_taxa = preprocessing.get_species_to_taxa()
    # cluster_taxa = cluster_groups.apply(lambda x: [species_to_taxa[species] for species in x])

    species = clusters[clusters == cluster].index
    wrong = clf_report[clf_report.Prediction == 'Wrong']
    wrong_cluster = wrong[(wrong.COMMON_NAME.isin(species)) & (wrong.Predicted.isin(species))]
    wrong_cluster = wrong_cluster[['COMMON_NAME', 'ID', 'Taxa']].drop_duplicates()

    taxas = ['Fishes', 'Cetaceans', 'Turtles', 'Sirenians', 'Seals', 'Birds']
    taxa_to_color = {taxa: c for taxa, c in zip(taxas, plotly_default_colors())}
    wrong_cluster['color'] = wrong_cluster.Taxa.map(taxa_to_color)

    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'hexagon', 'star', 'diamond-tall', 'diamond-wide']
    species_to_symbol = wrong_cluster.groupby('Taxa').apply(lambda df: pd.Series({s: marker_symbols[i] for i, s in enumerate(df.COMMON_NAME.unique())}))
    if isinstance(species_to_symbol, pd.core.frame.DataFrame):
        species_to_symbol = species_to_symbol.stack()
    species_to_symbol.index = species_to_symbol.index.droplevel(0)
    species_to_symbol = species_to_symbol.to_dict()
    wrong_cluster['symbol'] = wrong_cluster.COMMON_NAME.map(species_to_symbol)
    wrong_cluster = wrong_cluster.set_index('ID')
    wrong_cluster['in_degree'] = in_degree[wrong_cluster.COMMON_NAME].values
    wrong_cluster = wrong_cluster.sort_values(['Taxa', 'in_degree'], ascending=[True, False])

    if df is None:
        df = preprocessing.load_all_data(v2=True, weather=None, return_labels=False)
        df.index = df.index.droplevel(0)

    if hub_to_gold:
        s = wrong_cluster.sort_values('in_degree', ascending=False)['COMMON_NAME'].iloc[0]
        wrong_cluster.loc[wrong_cluster.COMMON_NAME == s, 'color'] = 'gold'

    fig = get_figure(xaxis_title='Longitude', yaxis_title='Latitude', height=height, width=width)

    for ID, row in tqdm(wrong_cluster.iterrows()):
        lat, lon, _ = df.loc[ID]
        idxs = np.linspace(0, lat.size-1, num_points).astype(int)
        lat = lat[idxs]
        lon = lon[idxs]
        has_appeared = any(d.name == row.COMMON_NAME for d in fig.data)
        size = min_size + row.in_degree * 10
        fig.add_trace(go.Scattergeo(lat=lat,
                                    lon=lon,
                                    mode='markers+lines',
                                    marker=dict(color=row.color, symbol=row.symbol, size=size, opacity=0.5),
                                    line=dict(color=row.color, width=1),
                                    legendgroup=row.COMMON_NAME,
                                    # legendgroup_title_text=row.Taxa,
                                    name=row.COMMON_NAME,
                                    showlegend=not has_appeared,
                                    hovertemplate=f"ID: {ID}<br>Species: {row.COMMON_NAME}<br>Taxa: {row.Taxa}<extra></extra>"
                                    )
                      )

    fig.update_layout(legend=dict(font=dict(size=35), tracegroupgap=1, groupclick='togglegroup', y=0.5),
                      margin=dict(t=0,b=0,l=0,r=0)
                      )
    fig.update_geos(scope='world',
                    showcountries=False,
                    showocean=True, oceancolor='lightblue',
                    showland=True, landcolor='tan',
                    showlakes=True, lakecolor='lightblue',
                    showrivers=True, rivercolor='lightblue',
                    showcoastlines=True, coastlinecolor='black', coastlinewidth=1,
                    )
    return fig
