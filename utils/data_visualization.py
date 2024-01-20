import numpy as np
import pandas as pd
import warnings
#import shap
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import os
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from colour import Color
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

from phdu.plots.plotly_utils import get_figure, mod_simple_axes, fig_base_layout
from phdu.plots.base import plotly_default_colors
from phdu import savefig

from . import other_utils, file_management, analysis, preprocessing, models, params, geometry

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)

##############################################################################################################################
"""                                                    I. Helper funcs                                                     """
##############################################################################################################################

def rows_cols(nrows, ncols):
    rows, cols = (np.vstack(np.divmod(range(nrows*ncols), ncols)) + 1)
    return rows, cols

def get_common_range(subplots, axes=["x", "y"]):
    data = defaultdict(list)
    for plot in subplots.data:
        for ax in axes:
            if hasattr(plot, f"error_{ax}") and getattr(plot, f"error_{ax}").array is not None:
                additions = [plot[f"error_{ax}"]["array"], -plot[f"error_{ax}"]["array"]]
            else:
                additions = [0]
            for addition in additions:
                arr = (plot[ax] + addition)[~np.isnan(plot[ax])]
                data[f"{ax}-min"].append(arr.min())
                data[f"{ax}-max"].append(arr.max())
    for k, v in data.items():
        func = min if "min" in k else max
        data[k] = func(v)
    ranges = {ax: [data[f"{ax}-min"], data[f"{ax}-max"]] for ax in axes}
    return ranges

get_tickdata = lambda subplots, axes=["x","y"], size=24: {"{}axis{}_tickfont_size".format(ax, i): size for ax in axes for i in [""] + [*range(1, subplots + 1)]}
get_logaxes = lambda subplots, axes=["y"]: {"{}axis{}_type".format(ax, i): "log" for ax in axes for i in [""] + [*range(1, subplots + 1)]}
get_exponent_format = lambda subplots, axes=["y"]: {"{}axis{}_exponentformat".format(ax, i): "power" for ax in axes for i in [""] + [*range(1, subplots + 1)]}
get_logaxes_expformat = lambda subplots, axes=["y"]: {**get_logaxes(subplots, axes), **get_exponent_format(subplots,axes)}
get_range_data = lambda subplots, axes, ranges: {"{}axis{}_range".format(ax, i): r for (ax, r) in zip(axes, ranges) for i in [""] + [*range(1, subplots + 1)]}
format_data = lambda subplots, key, val, axes=["x","y"]: {"{}axis{}_{}".format(ax, i, key): val for ax in axes for i in [""] + [*range(1, subplots + 1)]}

def get_subplots(cols, rows=1, horizontal_spacing=0.03, height=None, width=2500, ticksize=32, font_size=40, font_family="sans-serif", hovermode=False,
                 **kwargs):
    height = 800*rows if height is None else height
    fig = make_subplots(figure=go.Figure(layout=dict(margin=dict(l=100, r=20, b=80, t=80, pad=1), height=height, width=width)),
                        horizontal_spacing=horizontal_spacing, rows=rows, cols=cols, **kwargs
                       )

    fig.for_each_annotation(lambda a: a.update(font={'size':font_size, "family":font_family}))
    ticks_data = {'{}axis{}_tickfont_size'.format(ax, i): ticksize for ax in ["x", "y"] for i in [""] + [*range(1, cols + 1)]}
    fig.update_layout(**ticks_data, legend_font_size=font_size, hovermode=hovermode)
    return fig

# def get_figure(height=800, width=1200, ticksize=32, font_size=40, margin=None, font_family="sans-serif", hovermode=False, **kwargs):
#     fig = go.Figure(layout=dict(margin=dict(l=100, r=20, b=80, t=20, pad=1) if margin is None else margin,
#                                 height=height, width=width, yaxis=dict(tickfont_size=ticksize),
#                                 xaxis=dict(tickfont_size=ticksize), font_size=font_size, legend_font_size=font_size,
#                                 font_family=font_family, hovermode=hovermode,
#                                 **kwargs))
#     return fig

def break_str(x, separator="<br>"):
    x_list = x.split(" ")
    mid = len(x_list) // 2
    return "{} {} {}".format(" ".join(x_list[:mid]), separator, " ".join(x_list[mid:]))

def plotly_default_colors(mpl=10):
    cs = ['#1f77b4', '#ff7f0e',  '#2ca02c',  '#d62728',  '#9467bd',  '#8c564b',  '#e377c2',  '#7f7f7f',  '#bcbd22',  '#17becf'] * mpl
    return cs

def plotly_colors():
    cs = """aliceblue, antiquewhite, aqua, aquamarine, azure,
                beige, bisque, black, blanchedalmond, blue,
                blueviolet, brown, burlywood, cadetblue,
                chartreuse, chocolate, coral, cornflowerblue,
                cornsilk, crimson, cyan, darkblue, darkcyan,
                darkgoldenrod, darkgray, darkgrey, darkgreen,
                darkkhaki, darkmagenta, darkolivegreen, darkorange,
                darkorchid, darkred, darksalmon, darkseagreen,
                darkslateblue, darkslategray, darkslategrey,
                darkturquoise, darkviolet, deeppink, deepskyblue,
                dimgray, dimgrey, dodgerblue, firebrick,
                floralwhite, forestgreen, fuchsia, gainsboro,
                ghostwhite, gold, goldenrod, gray, grey, green,
                greenyellow, honeydew, hotpink, indianred, indigo,
                ivory, khaki, lavender, lavenderblush, lawngreen,
                lemonchiffon, lightblue, lightcoral, lightcyan,
                lightgoldenrodyellow, lightgray, lightgrey,
                lightgreen, lightpink, lightsalmon, lightseagreen,
                lightskyblue, lightslategray, lightslategrey,
                lightsteelblue, lightyellow, lime, limegreen,
                linen, magenta, maroon, mediumaquamarine,
                mediumblue, mediumorchid, mediumpurple,
                mediumseagreen, mediumslateblue, mediumspringgreen,
                mediumturquoise, mediumvioletred, midnightblue,
                mintcream, mistyrose, moccasin, navajowhite, navy,
                oldlace, olive, olivedrab, orange, orangered,
                orchid, palegoldenrod, palegreen, paleturquoise,
                palevioletred, papayawhip, peachpuff, peru, pink,
                plum, powderblue, purple, red, rosybrown,
                royalblue, saddlebrown, salmon, sandybrown,
                seagreen, seashell, sienna, silver, skyblue,
                slateblue, slategray, slategrey, snow, springgreen,
                steelblue, tan, teal, thistle, tomato, turquoise,
                violet, wheat, white, whitesmoke, yellow,
                yellowgreen"""
    li = cs.split(',')
    li = [l.replace('\n','') for l in li]
    li = [l.replace(' ','') for l in li]
    return li

def get_alternate_dashes():
    return ["solid", "dash", "dot"] * 10

def transparent_colorscale(fig, threshold=1e-10, upper=False, low_lim=0, upper_lim=1):
    """Values below threshold are invisible."""
    colorscale = fig.layout["coloraxis"]["colorscale"]
    if upper:
        high_limit = colorscale[-1]
        new_high_limit = (upper_lim - threshold, high_limit[1])
        new_colorscale = (*colorscale[:-1], new_high_limit, (upper_lim, 'rgba(0,0,0,0)'))
    else:
        low_limit = colorscale[0]
        new_low_limit = (low_lim + threshold, low_limit[1])
        new_colorscale = ((low_lim, 'rgba(0,0,0,0)'), new_low_limit, *colorscale[1:])
    return new_colorscale


def color_gradient(start, end, n):
    return [*Color(start).range_to(Color(end), n)]



##############################################################################################################################
"""                                                II. Dataset visualization                                               """
##############################################################################################################################

def nan_by_column(df, figDir="figs/missing_data", figname="", width=1200, height=800, xaxis_tickangle=None):
    nan_percentage = 100 * df.isna().sum(axis=0).values / df.shape[0]
    x = [col if not col.startswith("Significant") else "Height of waves and swell" for col in df.columns]
    fig = go.Figure(data=[go.Bar(x=x, y=nan_percentage, showlegend=False)],
                    layout=dict(xaxis=dict(title_text='Feature', tickfont_size=20), yaxis=dict(title_text='% NaN values', tickfont_size=20, range=[-2, 102]),
                     width=width, height=height, xaxis_tickangle=xaxis_tickangle, font_size=26, margin=dict(l=50,r=10,b=80,t=10, pad=1))
                   )
    Path(figDir).mkdir(exist_ok=True, parents=True)

    num_vars = len(df.columns)
    fig.write_image(os.path.join(figDir, f'missing_data_{num_vars}-vars_{figname}.png'))
    fig.update_layout(template='plotly_dark')
    fig.show()
    return

def nan_by_year(df, maxcols=5, figDir="figs/metadata/", figname="", mode="bar"):
    nan_values = analysis.nan_by_year(df)
    nan_cols = [col for col in df.columns if nan_values[col][1].sum() > 0]

    if mode == "bar":
        rows, cols =  divmod(len(nan_cols), maxcols)
        rows += 1
        fig = get_subplots(x_title="Year", y_title="% NaN values", cols=cols, rows=rows, subplot_titles=[col if len(col) < 30 else break_str(col) for col in nan_cols], shared_yaxes=True)
        fig.update_layout(yaxis_range=[-0.05, 1.05])
        for i, col in enumerate(nan_cols):
            vals = nan_values[col]
            row, col = np.array(divmod(i, maxcols)) + 1
            fig.add_trace(go.Bar(x=vals[0], y=vals[1], showlegend=False), row=row, col=col) #marker_color='#1f77b4'
    elif mode == "scatter":
        fig = get_figure(xaxis_title_text="Year", yaxis_title_text="% NaN values", yaxis_range=[-0.05, 1.05])
        fig.update_layout(font_size=18)
        num_cols = len(nan_cols)
        colorlist = plotly_default_colors()
        for i, (col, color) in enumerate(zip(nan_cols, colorlist)):
            x, y = nan_values[col]
            x_width = [*x] + [*x[::-1]]
            y_width = [*(y + (0.05/(i+1)))] + [*(y - (0.05/(i+1)))][::-1]
            #fig.add_trace(go.Scatter(x=x_width, y=y_width, fill='toself', fillcolor=color, showlegend=False, opacity=0.2))
            fig.add_trace(go.Scatter(x=x, y=y, name=break_str(col), mode="lines+markers", marker_color=color, marker_symbol=i, marker_size=8 + 8*((i+1)%2),
                                     line_dash="dot" if (i+1) % 2 else None)
                         ) #opacity=0.5, line_width=3*num_cols-3*i, marker_size=2*i+3))
    else:
        raise ValueError(f'mode {mode} not valid. Available: "bar", "scatter".')

    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'nan_by_year_{mode}{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def trajectories_common_maxlen(L=[*range(10, 1100, 10)]):
    fig = get_figure(width=1000, xaxis_title="Common time length [8h]", yaxis_title="# trajectories", xaxis_range=[L[0]-10, L[-1]+10], title="Trajectories with data at equal times",
                     margin=dict(l=100, t=100, r=10, b=80)
                    )
    trajectories_maxlen = preprocessing.trajectories_common_time_max(L)
    fig.add_trace(go.Scatter(x=L, y=[x.size for _, x in trajectories_maxlen.values()], showlegend=False))
    fig.write_image("figs/dataset_size/trajectories_data_equal_times.png")
    return

def percentages_visualization(df, col, fig_folder='figs/percentages/', figname='', sort=True, **layout_kwargs):
    percentages = df[col].value_counts(normalize=True, sort=sort) * 100
    cumsum = percentages.cumsum()
    colvals = percentages.index
    fig = go.Figure([go.Bar(x=colvals, y=percentages, name='Percentage'),
                     go.Scatter(x=colvals, y=cumsum, mode="lines", line_width=5, name='Cumulative')])
    fig.update_layout(xaxis_title_text=col, width=1500, height=800, xaxis_tickangle=-45, margin=dict(l=50,r=50,b=5,t=10), font=dict(size=32), xaxis={'tickfont': {'size':16}}, yaxis={'tickfont':{'size':20}},
                     **layout_kwargs)
    Path(fig_folder).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(fig_folder, "percentages_{}{}.png".format(col, figname)))
    fig.update_layout(template='plotly_dark', height=500) # for notebook display
    fig.show()
    return

def col_hist(df=None, col="Length", Nbins_plot=50, Nbins_hist=200, figDir="figs/percentages/", figname="", log=False):
    if df is None:
        df = file_management.load_lzma("utils/data/labels_split-by-day_groupby-ID_default.lzma")
    l = df[col]
    h, edges = np.histogram(l, bins=Nbins_hist, density=True)
    mid_edges = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    p = dx * h
    mid_edges = np.hstack([l.min(), mid_edges])
    p = np.hstack([0, p])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    logaxis = get_logaxes_expformat(1) if log else {}
    fig.update_layout(**logaxis, xaxis_title_text="Trajectory length (log)", margin=dict(l=100, r=20, b=80, t=20, pad=1), height=800, width=1200, yaxis=dict(tickfont_size=20),
                      xaxis=dict(tickfont_size=20), font_size=26)

    fig.add_trace(go.Histogram(x=np.log10(l), histnorm="probability density", nbinsx=Nbins_plot, name="Probability density"))
    fig.add_trace(go.Scatter(x=np.log10(mid_edges), y=p.cumsum(), mode="lines", line_width=4, name="Cumulative probability"), secondary_y=True)

    fig.write_image(os.path.join(figDir, f'{col}_hist_{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def col_hist_taxa(df=None, hist_col="Length", log=True, x_range=None, shared_y=True, Nbins_plot=50, Nbins_hist=200, figDir="figs/percentages/", figname=""):
    if df is None:
        df = file_management.load_lzma("utils/data/labels_split-by-day_groupby-ID_default.lzma")
    data = df[[hist_col, "COMMON_NAME", "Taxa"]]
    taxas = set(data["Taxa"])
    nrows, ncols = 3, 4
    specs = [[{"secondary_y": True} for col in range(ncols)] for row in range(nrows)]

    log_str = "log" if log else ""
    shared_y_str = "_shared_y" if shared_y else ""
    taxas = [taxa for taxa, _ in data.groupby("Taxa")]
    fig = make_subplots(subplot_titles=taxas, rows=nrows, cols=ncols, specs=specs, x_title=f"Trajectory {hist_col} {log_str}", y_title="Probability density",
                        shared_yaxes=shared_y, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.03,
                        figure=go.Figure(layout=dict(margin=dict(l=100, r=30, t=80, b=80, pad=1), width=2200, height=1800)))
    fig.for_each_annotation(lambda t: t.update(font_size=30))
    tickdata = get_tickdata(nrows*ncols)
    l_range = np.log10(data[hist_col]) if log else data[hist_col]
    x_range = [l_range.min() - 0.5, l_range.max() + 0.5] if x_range is None else x_range
    xaxes_ranges = get_range_data(nrows*ncols, ["x"], [x_range])
    secondary_tickdata = {"yaxis{}_tickfont_size".format(i): 20 for i in range(nrows*ncols + 1, 2*nrows*ncols + 1)}
    fig.update_layout(**tickdata, **secondary_tickdata, **xaxes_ranges)

    colors = plotly_default_colors()
    for i, (taxa, df) in enumerate(data.groupby("Taxa")):
        row, col = divmod(i, ncols)
        row += 1
        col += 1
        l = df[hist_col]
        h, edges = np.histogram(l, bins=Nbins_hist, density=True)
        mid_edges = 0.5 * (edges[1:] + edges[:-1])
        dx = edges[1] - edges[0]
        p = dx * h
        mid_edges = np.hstack([l.min(), mid_edges])
        p = np.hstack([0, p])
        x = np.log10(l) if log else l
        x_cum = np.log10(mid_edges) if log else mid_edges
        fig.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=Nbins_plot, showlegend=False, marker_color=colors[0]), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_cum, y=p.cumsum(), mode="lines", line_width=4, marker_color=colors[1], name="Cumulative probability", showlegend=False),
                      secondary_y=True, row=row, col=col)

    fig.write_image(os.path.join(figDir, f'{hist_col}_hist_taxa_{figname}{shared_y_str}{log_str}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def stage_sunburst_visualization(path=['Stage-type', 'Stage', 'Taxa'], html=False, font_size=24):
    inception = models.InceptionTime(label="Stage", velocity=None, delete_features=["x", "y", "z"], maxlen=60,
                                     nb_filters=32, use_bottleneck=False, depth=4, kernel_size_s=[33,19,9,5], strides=[5,3,1,1], bottleneck_size=16,
                                     scale=["col", "COMMON_NAME"], class_weights="linear", weather=None, diff=False,
                                     prunning_function=preprocessing.get_stage_prunner_all(mapping=params.stage_mapping, NaN_value=None, minlen=30, min_days=1, min_animals=10)
                                    )
    df = inception.labels.value_counts(["Taxa", "COMMON_NAME", "Stage"], sort=False)
    df.name = "Count"
    df = df.reset_index()
    df["Stage-type"] = df["Stage"].map(lambda c: c.split(":")[0].capitalize())
    df["Stage"] = df["Stage"].map(lambda c: c.split(": ")[-1].capitalize())

    fig = px.sunburst(df, path=path, values='Count')
    fig.update_layout(height=1000, width=1000, margin=dict(l=5, r=5, t=5, b=5), font_size=font_size)#, uniformtext=dict(minsize=10, mode="hide"))

    parentDir = fullPath("figs/stage_classification")
    Path(parentDir).mkdir(exist_ok=True)
    if html:
        fig.write_html(os.path.join(parentDir, "dataset.html"))
    ID = other_utils.dict_to_id(path=path)
    fig.write_image(os.path.join(parentDir, f"dataset_{ID}.png"))
    return

def MultiIndex_to_edges(index):
    levshape = index.levshape
    num_levels = len(levshape)
    edges = [(0, i+1) for i in index.levels[0].values] # 0 == root level. For visualization purposes
    for l in range(num_levels-1):
        edges += [*zip(index.get_level_values(l) + 1, index.get_level_values(l+1) + 1)]
    return edges

def index_to_numeric(index, levels_unique=[]):
    """levels_unique: Treat levels in levels_unique as if all the categories are different."""
    levshape = index.levshape
    n0 = 0
    arrays = []
    labels = []
    for l in range(len(levshape)):
        if l in levels_unique:
            nmax = n0 + index.shape[0]
            arrays.append(np.arange(n0, nmax))
            labels.append(index.get_level_values(l))
        else:
            nmax = n0 + levshape[l]
            arrays.append(index.set_levels(np.arange(n0, nmax), level=l).get_level_values(l))
            labels.append(index.levels[l].values)
        n0 = nmax
    index_numeric = pd.MultiIndex.from_arrays(arrays, names=index.names)
    labels = np.hstack(labels)
    return index_numeric, labels

def tree_layout(edges):
    G = Graph(edges)
    lay = G.layout("rt", root=0)
    num_vertices = G.vcount()
    position = {k: lay[k] for k in range(num_vertices)}
    Y = [lay[k][1] for k in range(num_vertices)]
    M = max(Y)

    Xn = [position[k][0] for k in range(num_vertices)]
    Yn = [2*M - position[k][1] for k in range(num_vertices)]
    Xe = []
    Ye = []
    for edge in edges:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2*M - position[edge[0]][1], 2*M - position[edge[1]][1], None]
    return Xe, Ye, Xn, Yn, position

def make_annotations(df, font_size_mpl=0.8, font_color='rgb(250,250,250)'):
    annotations = []
    for label, x, y, fs in zip(df.index, df.X, df.Y, df["size"]*font_size_mpl):
        annotations.append(dict(text=label, x=x, y=y, xref='x1', yref='y1', font=dict(color=font_color, size=fs, family="sans-serif"),
                                showarrow=False))
    return annotations

def node_fmt(df, levels_unique=[2], base_color="#6175c1", base_symbol="square-dot", base_size=100, size_mpl=8,
             symbol_func = lambda x: "diamond-dot" if "non-breeding" in x else "circle-dot"
            ):
    index = df.index
    palette = plotly_colors()[::5] #data_visualization.plotly_default_colors(mpl=1)[1:] +
    _, labels = index_to_numeric(index, levels_unique=levels_unique)
    colors = {}
    fmt = defaultdict(list)
    labels_unique = np.hstack([index.levels[l] for l in levels_unique])
    i = 0
    for label in np.unique(labels):
        if label in labels_unique:
            colors[label] = palette[i]
            i += 1
        else:
            colors[label] = base_color
    start_color_idx = np.cumsum(index.levshape)[levels_unique[0]-1]
    fmt["symbol"] += [base_symbol] * start_color_idx
    fmt["size"] += [base_size] * index.levshape[0] + [base_size/2] * index.levshape[1]
    fmt["color"] += [base_color] * start_color_idx
    sizes = size_mpl * (np.log10(df.values) + 1)
    for label, size in zip(labels[start_color_idx:], sizes):
        fmt["symbol"].append(symbol_func(label))
        fmt["color"].append(colors[label])
        fmt["size"].append(size)
    return pd.DataFrame(fmt, index=labels)

def tree_plotting_specs(df, levels_unique=[2], **kwargs):
    index = df.index
    levshape = index.levshape
    root_edges = 3*levshape[0]
    index_numeric, labels = index_to_numeric(index, levels_unique=levels_unique)
    edges = MultiIndex_to_edges(index_numeric)
    Xe, Ye, Xn, Yn, pos = tree_layout(edges)
    pos = pd.Series([*pos.values()][1:], index=labels)
    edge_pos = pd.DataFrame(dict(X=Xe[root_edges:], Y=Ye[root_edges:]))
    fmt = node_fmt(df, levels_unique=levels_unique, **kwargs)
    node_pos = pd.DataFrame(dict(X=Xn[1:], Y=Yn[1:]), index=labels) # removed root
    node_specs = pd.concat([node_pos, fmt], axis=1)
    return node_specs, edge_pos, pos

def stage_dataset_tree_visualization(base_color="#6175c1", base_size=75, size_mpl=9, font_size_mpl=0.2, legend_font_size=16):
    inception = models.InceptionTime(label="Stage", velocity=None, delete_features=["x", "y", "z"], maxlen=60,
                                     nb_filters=32, use_bottleneck=False, depth=4, kernel_size_s=[33,19,9,5], strides=[5,3,1,1], bottleneck_size=16,
                                     scale=["col", "COMMON_NAME"], class_weights="linear", weather=None, diff=False,
                                     prunning_function=preprocessing.get_stage_prunner_all(mapping=params.stage_mapping, NaN_value=None, minlen=30, min_days=1, min_animals=10),
                                    )
    df = inception.labels.value_counts(["Taxa", "COMMON_NAME", "Stage"], sort=False)
    node_specs, edge_pos, pos = tree_plotting_specs(df, base_size=base_size, size_mpl=size_mpl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_pos.X, y=edge_pos.Y, mode='lines', line=dict(color='rgb(210,210,210)', width=1), hoverinfo='none', showlegend=False))
    base = node_specs[node_specs["color"] == base_color]
    rest = node_specs[node_specs["color"] != base_color]
    fig.add_trace(go.Scatter(x=base.X, y=base.Y, mode='markers', showlegend=False,
                             marker=dict(symbol=base.symbol, size=base["size"], color=base_color, line=dict(color='rgb(50,50,50)', width=1)),
                             text=base.index,
                             hoverinfo='text',
                             opacity=0.8
                            ))
    colors = color_gradient("red", "blue", rest.color.unique().size)
    for (l, data), c in zip(rest.reset_index().groupby("index"), colors):
        fig.add_trace(go.Scatter(x=data.X, y=data.Y, mode='markers', name=data.iloc[0,0], showlegend=True,
                          marker=dict(symbol=data.symbol, size=data["size"], color=[c.hex]*data.shape[0], line=dict(color='rgb(50,50,50)', width=1)),
                          text=data["index"],
                          hoverinfo='text',
                          opacity=0.8
                          ))
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    xrange, yrange = [*get_common_range(fig).values()]
    fig.update_layout(height=500, width=1400, margin=dict(l=5, r=5, b=10, t=5), plot_bgcolor='rgb(248,248,248)',
                      annotations=make_annotations(node_specs.iloc[:3], font_size_mpl=font_size_mpl),
                      legend=dict(orientation="h", font_family="sans-serif", font_size=legend_font_size),
                      xaxis=axis, yaxis=axis,
                      xaxis_range = [xrange[0]-0.3, xrange[1]+0.8],
                      yaxis_range = [yrange[0]-0.14, yrange[0]+2.3],
                     )
    fig.write_image("figs/stage_classification/dataset_tree.png")
    return


##############################################################################################################################
"""                                              III. Scatter and bar charts                                               """
##############################################################################################################################

def pca_variance(df, figDir="figs/PCA", figname=""):
    x = StandardScaler().fit_transform(df.values)
    pca = PCA()
    pca.fit(x)
    fig = go.Figure(data=[go.Bar(x=np.arange(len(pca.components_))+1, y=pca.explained_variance_ratio_, showlegend=False)],
                    layout=dict(margin=dict(t=10, l=100, r=10, b=80, pad=1), height=800, width=1200, yaxis=dict(title_text="Variance explained", tickfont_size=20),
                           xaxis=dict(title_text="PCA component", tickfont_size=20), font_size=26)
    )

    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'pca_variance_{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def pca_feature_importance(df, figDir="figs/PCA", figname=""):
    x = StandardScaler().fit_transform(df.values)
    pca = PCA()
    pca.fit(x)
    col_importance = []
    for pca_comp_var in pca.components_.T:
        col_importance.append(np.sum(np.abs(pca_comp_var) * pca.explained_variance_ratio_))
    col_importance = np.array(col_importance)
    # Alternatively: col_importance = (np.abs(pca.components_) * pca.explained_variance_ratio_[:, None]).sum(axis=0)
    col_importance /= col_importance.sum()
    cols = [col if not col.startswith("Significant") else "Height of waves and swell" for col in df.columns]
    fig = go.Figure(data=[go.Bar(x=cols, y=col_importance, showlegend=False)],
                    layout=dict(margin=dict(t=10, l=100, r=10, b=80, pad=1), height=800, width=1500, yaxis=dict(title_text="Importance", tickfont_size=20, range=[-0.05, 1.05]),
                           xaxis=dict(title_text="Variable", tickfont_size=20), font_size=26)
    )

    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'pca_feature_importance_{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def visualize_pca_components(pca, features, figDir="figs/PCA/components", figdict={}, dx=[-15, 10], dy=[-10, 10], **fig_kwargs):
    """Visualize pca components projections. dx, dy are offset values for the x and y axes."""
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    fig = get_figure(xaxis_title="PCA 1", yaxis_title="PCA 2", **fig_kwargs)
    for i, feature in enumerate(features):
        fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])
        fig.add_annotation(x=loadings[i, 0],
                           y=loadings[i, 1] if loadings[i,1] > 0 else loadings[i,1] - 4,
                           ax=0, ay=0, xanchor="center", yanchor="bottom", text=feature,
        )
    fig.update_layout(xaxis=dict(range=[dx[0] + loadings.min(axis=0)[0], dx[1]+loadings.max(axis=0)[0]],
                             showgrid=False, zeroline=False, visible=True, tickvals=[]),
                  yaxis=dict(range=[dy[0] + loadings.min(axis=0)[1], dy[1]+loadings.max(axis=0)[1]],
                             showgrid=False, zeroline=False, visible=True, tickvals=[]),
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                 )
    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, "{}.png".format(other_utils.dict_to_id(figdict))))
    return


##############################################################################################################################
"""                                                       IV. Geo                                                          """
##############################################################################################################################

def draw_square_scatter_geo(fig, vertices):
    a, b, c, d = vertices
    fig.add_trace(go.Scattergeo(
        lon=[a[0], b[0]],
        lat=[a[1], b[1]],
        mode='lines',
        line_color='black',
        showlegend=False
    ))

    fig.add_trace(go.Scattergeo(
        lon=[c[0], d[0]],
        lat=[c[1], d[1]],
        mode='lines',
        line_color='black',
        showlegend=False
    ))

    fig.add_trace(go.Scattergeo(
        lon=[a[0], d[0]],
        lat=[a[1], d[1]],
        mode='lines',
        line_color='black',
        showlegend=False
    ))

    fig.add_trace(go.Scattergeo(
        lon=[b[0], c[0]],
        lat=[b[1], c[1]],
        mode='lines',
        line_color='black',
        showlegend=False
    ))
    return

@savefig('species+i+full_world')
def trajectory(species, i, df=None, v2=True, height=1500, width=2000, marker_size=10, lw=3, lat_pad=1, lon_pad=1, full_world=True):

    if df is None:
        df = preprocessing.load_all_data(v2=v2, weather=None, return_labels=False)
    lat, lon, _ = df[species].iloc[i]


    fig = get_figure(xaxis_title='Longitude', yaxis_title='Latitude', height=height, width=width)
    fig.add_trace(go.Scattergeo(
        lon=lon,
        lat=lat,
        mode='lines+markers',
        marker_size=marker_size,
        marker_opacity=0.8,
        line_dash='solid',
        line=dict(width=lw),
        showlegend=False
    ))

    if full_world:
        lataxis_range = [-90, 90]
        lonaxis_range = [-180, 180]
    else:
        lataxis_range = [lat.min()-lat_pad, lat.max()+lat_pad]
        lonaxis_range = [lon.min()-lon_pad, lon.max()+lon_pad]

    fig.update_geos(landcolor='rgb(217, 217, 217)',
                    showocean=True, oceancolor='rgb(255, 255, 255)',
                    showlakes=True, lakecolor='rgb(255, 255, 255)',
                    showrivers=True, rivercolor='rgb(255, 255, 255)',
                    lataxis_range=lataxis_range, lonaxis_range=lonaxis_range,
                    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig

@savefig('species+ID+density')
def trajectory_img(df=None, species='Grey-headed albatross', ID='3531_BIRDLIFE', nbins_ns=60, nbins_we=60, bin_lims='trajectory', density=False):
    if df is None:
        df = preprocessing.compute_df_distance()[0]
    x = df[species][ID][:2]
    if bin_lims == 'dataset':
        lims_ns = [-1.91, 1.29]
        lims_we = [-2.22, 8.58]
    elif bin_lims == 'trajectory':
        lims_ns = [x[0].min() - 0.3, x[0].max() + 0.1]
        lims_we = [x[1].min() - 0.3, x[1].max() + 0.1]
    else:
        raise ValueError(f"bin_lims must be 'dataset' or 'trajectory', not {bin_lims}")
    bins_ns = np.linspace(*lims_ns, nbins_ns+1)
    bins_we = np.linspace(*lims_we, nbins_we+1)

    bins = [bins_ns, bins_we]
    x_hist = np.histogram2d(*x, bins=bins, density=True)[0]
    if density:
        title = 'Probability<br>density (log)<br><span style="color: white; opacity: 0; font-size: 0.1px">aa</span>'
        x_hist = np.log(x_hist)
        x_hist[x_hist == -np.inf] = np.nan
        colorbar_len = 0.8
    else:
        x_hist[x_hist > 0] = 1
        title = 'Occurrence'
        colorbar_len = 0.3
    clims = np.round([np.nanmin(x_hist), np.nanmax(x_hist)], 2)
    fig = px.imshow(x_hist, x=bins[1][:-1], y=bins[0][:-1][::-1], color_continuous_scale='greens')
    fig.update_layout(**fig_base_layout(),
                      xaxis_title='WE distance', yaxis_title='SN distance')
    fig.update_layout(**mod_simple_axes(fig), plot_bgcolor='white')
    colorscale = transparent_colorscale(fig, upper=False)
    fig.update_layout(width=1000, height=350, coloraxis=dict(colorscale=colorscale, colorbar=dict(len=colorbar_len, y=0.6, title=title, title_font_size=35, tickvals=clims, tickfont_size=28)),
                       margin=dict(l=0, r=0, b=0, t=0),
                      yaxis=dict(tickvals=[-0.5, 0], ticktext=['0', '-0.5'])
                      )
    return fig

@savefig('species+ID+lines')
def trajectory_common_origin(df=None, species='Grey-headed albatross', ID='3531_BIRDLIFE', lines=False, lims='trajectory'):
    if df is None:
        df = preprocessing.compute_df_distance()[0]
    x = df[species][ID][:2]
    fig = get_figure(xaxis_title='WE distance', yaxis_title='SN distance', height=350, width=1000, simple_axes=True)
    if lines:
        mode = 'lines+markers'
    else:
        mode = 'markers'
    fig.add_trace(go.Scatter(x=x[1], y=x[0], mode=mode, marker=dict(size=10, opacity=1,line=dict(color='gray', width=0.2)), line_dash='solid', line=dict(width=2), showlegend=False))
    if lims == 'dataset':
        lims_ns = [-1.91, 1.29]
        lims_we = [-2.22, 8.58]
    elif lims == 'trajectory':
        lims_ns = [x[0].min() - 0.3, x[0].max() + 0.1]
        lims_we = [x[1].min() - 0.3, x[1].max() + 0.1]
    fig.update_layout(xaxis=dict(range=lims_we), yaxis=dict(range=lims_ns))
    return fig

@savefig('species+v2+vertices')
def trajectories_by_stage(species, v2=True, vertices=[], height=1500, width=2000, marker_size=8, legend_itemsize=125, legend_y=0.5, include_lines=True):

    df, labels, _ = preprocessing.load_all_data(v2=v2, weather=None, return_labels=True, species_stage=species)
    stages = labels.loc[species]['Stage']
    df_species = df[species]
    if len(vertices) > 0:
        df_species = df_species.apply(lambda x: geometry.trajectory_inside_vertices(x, vertices)).dropna()
        stages = stages.loc[df_species.index]
    lat = np.hstack(df_species.apply(lambda x: x[0]).values)
    lon = np.hstack(df_species.apply(lambda x: x[1]).values)

    fig = get_figure(xaxis_title='Longitude', yaxis_title='Latitude', height=height, width=width)
    stage_types = stages.value_counts(ascending=False).index # plot first the most frequent stages
    stage_to_color = {s: c for s, c in zip(stage_types, plotly_default_colors())}
    if include_lines:
        mode = 'lines+markers'
    else:
        mode = 'markers'
    for s in stage_types:
        IDS = stages[stages == s].index
        df_stage = df_species.loc[IDS]
        for i, x in enumerate(df_stage.values):
            fig.add_trace(go.Scattergeo(
                lon=x[1],
                lat=x[0],
                mode=mode,
                marker_color=stage_to_color[s],
                marker_size=marker_size,
                marker_opacity=0.5,
                line_color=stage_to_color[s],
                line_dash='solid',
                line=dict(width=1),
                name = s,
                showlegend=i==0
            ))

    if vertices:
        draw_square_scatter_geo(fig, vertices)

    # zoom scattergeo to the locations where the points are
    fig.update_geos(lataxis_range=[lat.min()-1, lat.max()+1], lonaxis_range=[lon.min()-1, lon.max()+1])
    fig.update_layout(legend=dict(itemsizing='constant', itemwidth=legend_itemsize, y=legend_y))
    return fig

def trajectories_visualization(height=1500, trajectories=None, species=None, figname='all', filter_func=lambda x,y,z: True):
    trajectories = file_management.load_lzma('utils/data/trajectories_default.lzma') if trajectories is None else trajectories
    species = file_management.load_lzma('utils/data/labels_default.lzma')['COMMON_NAME'] if species is None else species
    species_counts = species.value_counts()
    species_names = species_counts.index
    colors = plotly_colors()[:len(species_names)]
    species_to_color = {s:c for s,c in zip(species_names, colors)}
    fig = go.Figure(go.Scattergeo())
    fig.update_geos(
        projection_type="orthographic",
        resolution=50,
        showcoastlines=True, coastlinecolor="RebeccaPurple",
        showland=True, landcolor="LightGreen",
        showocean=True, oceancolor="LightBlue"
    )
    fig.update_layout(height=height, margin={"r":0,"t":0,"l":0,"b":0})

    species_drawn = set()
    trajs_drawn = 0
    for t, s in zip(trajectories, tqdm(species)):
        if filter_func(t, s, species_counts):
            plot_idxs = np.linspace(0, t.shape[1]-1, 20, dtype=np.int32) if t.shape[1] > 20 else [*range(t.shape[1])]
            fig.add_trace(go.Scattergeo(
                    lat = t[0][plot_idxs],
                    lon = t[1][plot_idxs],
                    mode = 'lines',
                    line = dict(width = 1, color = species_to_color[s]),
                    opacity = 1,
                    name = s,
                    showlegend = s not in species_drawn
                    )
            )
            species_drawn.add(s)
            trajs_drawn += 1
        else:
            continue

    fig.update_layout(legend_title_text='Species', showlegend=True, legend_font_size=10)
    Path('figs/trajectory_geo').mkdir(exist_ok=True, parents=True)
    fig.write_html('figs/trajectory_geo/{}_{:.0f}-species_{:.0f}-trajectories.html'.format(figname, len(species_drawn), trajs_drawn))
    fig.update_layout(template='plotly_dark')
    fig.show()
    return

def density_geo(df, magnitude, figDir="figs/density_geo", figname="", log=False, radius=None, cmin=0, cmax=1, showscale=False, show=False):
    if radius is None:
        radius = 5 if magnitude == "Density" else 3
    z = np.log10(df[magnitude]) if log else df[magnitude]
    if showscale:
        width = 1250
        colorbar_title = f'{magnitude} (log)' if log else magnitude
        coloraxis = dict(cmin=cmin, cmax=cmax, colorbar=dict(title=dict(text=colorbar_title, font_size=22), tickfont_size=18, bgcolor='rgba(0,0,0,0)'))
        coloraxis_str = "coloraxis"
    else:
        width = 1050
        coloraxis = None
        coloraxis_str = None
    fig = go.Figure(data=[go.Densitymapbox(lat=df['Latitude'], lon=df['Longitude'], z=z, radius=radius, showlegend=False, showscale=showscale, coloraxis=coloraxis_str)],
                    layout=dict(width=width, height=900, margin={'l': 0, 'r': 0, 'b': 0, 't': 0, 'pad': 0}, mapbox=dict(), coloraxis=coloraxis,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                               )
                   ) #zoom=1.2,
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
    fig.for_each_annotation(lambda a: a.update(font=dict(size=25)))
    fig.update_layout() #,geo=dict(scope='world', projection=dict(type='natural earth'),
                                                                                            #lonaxis=dict(range=[-180,180]), lataxis=dict(range=[-50,90])))
    Path(figDir).mkdir(exist_ok=True, parents=True)
    log_str = "_log" if log else ""
    fig.write_image("{}.png".format(os.path.join(figDir, f'{magnitude}_{figname}{log_str}')))

    if show:
        fig.show()
    return

def draw_clusters(epsilon, min_size, figDir="figs/clustering/boxes", figname="", projection="mercator", show=False):
    epsilon_str = other_utils.encoder(epsilon)
    min_size_str = other_utils.encoder(min_size)
    X = file_management.load_lzma('utils/data/trajectories_split-by-day_groupby-ID_default.lzma')
    X = np.concatenate(tuple(X), axis=1)
    labels = file_management.load_lzma(f"utils/data/cluster_labels_hdbscan_eps-{epsilon_str}_cluster-size-{min_size_str}.lzma")
    df = pd.DataFrame(dict(lat=X[0], lon=X[1], cluster=labels))

    borders = defaultdict(lambda: np.empty((2,5)))
    for cluster, cdf in df.groupby("cluster"):
        borders[cluster][0] = [cdf.lon.min()]*2 + [cdf.lon.max()]*2 + [cdf.lon.min()]
        borders[cluster][1] = [cdf.lat.min()] + [cdf.lat.max()]*2 + [cdf.lat.min()]*2

    cluster_percentage = 100 * df["cluster"].value_counts(normalize=True, sort=False)
    norm = plt.Normalize(vmin=0.1, clip=True)
    colors = plt.cm.jet(norm(cluster_percentage.values))
    borders = {k: borders[k] for k in cluster_percentage.keys()}

    fig = get_figure()
    for color, (cluster, (lon, lat)) in zip(colors, borders.items()):
        percentage = cluster_percentage.loc[cluster]
        percentage_str = "{:.1f}".format(percentage) if percentage > 0.1 else "< 0.1"
        fig.add_trace(go.Scattergeo(
                    lat = lat,
                    lon = lon,
                    mode = 'lines',
                    line = dict(width = 1, color=f'rgb{tuple(color)}'),
                    opacity = 1,
                    showlegend = False
                    #name = f'{cluster} ({percentage_str})',
                    )
            )
        fig.add_trace(go.Scattergeo(lat = [0.5*(lat.max() + lat.min())],
                                    lon = [0.5*(lon.max() + lon.min())],
                                    mode = "text",
                                    text = f"{cluster}",
                                    textfont = dict(color = f'rgb{tuple(color)}', size=14),
                                    showlegend = False)
                     )

    #fig.update_layout(legend=dict(title="Cluster (% data)", orientation="h"))
    fig.update_layout(geo=dict(projection_type=projection))
    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'boxes_eps-{epsilon_str}_min-size-{min_size_str}_{figname}.png'))
    if show:
        fig.show()
    return


def trajectory_shift(lats, lons, origin=(0,0), save=False, figDir="figs/trajectory_shift", lw=1, dashes=["solid", "dash"], colors=["blue", "red"], figname="", eps=True,
                     opacity=0.7):
    """Plots the original trajectory (blue) and the version shifted to the origin (red). Starting point is the thickest, ending point the other one."""
    if isinstance(lats, np.ndarray):
        if len(lats.shape) == 1:
            lats = [lats]
            lons = [lons]

    fig = get_figure()

    for lat, lon, color in zip(lats, lons, colors):
        lats_shifted, lons_shifted = preprocessing.sphere_translation_riemann(lat, lon, origin=origin)

        marker_sizes = np.ones((lat.size))
        marker_sizes[0] = 20
        marker_sizes[-1] = 10
        fig.add_trace(go.Scattergeo(
                    lat = lat * 180/np.pi,
                    lon = lon * 180/np.pi,
                    mode = 'lines+markers',
                    line = dict(width = lw, color=color, dash=dashes[0]),
                    opacity = 1,
                    name = "Original",
                    marker_size = marker_sizes
                    )
            )
        fig.add_trace(go.Scattergeo(
                    lat = lat[[0, -1]] * 180/np.pi,
                    lon = lon[[0, -1]] * 180/np.pi,
                    mode = 'markers',
                    marker_color=color,
                    opacity = 1,
                    showlegend = False,
                    marker_size = marker_sizes[[0, -1]]
                    )
            )

        fig.add_trace(go.Scattergeo(lat = lats_shifted * 180/np.pi,
                                    lon = lons_shifted * 180/np.pi,
                                    mode = "lines+markers",
                                    line = dict(width=lw, color=color, dash=dashes[1]),
                                    opacity = opacity,
                                    name = "Shifted",
                                    marker_size = marker_sizes)
                     )
        fig.add_trace(go.Scattergeo(lat = lats_shifted[[0, -1]] * 180/np.pi,
                                    lon = lons_shifted[[0, -1]] * 180/np.pi,
                                    mode = "markers",
                                    marker_color=color,
                                    opacity = opacity,
                                    showlegend=False,
                                    marker_size = marker_sizes[[0, -1]])
                     )
    if save:
        Path(figDir).mkdir(exist_ok=True, parents=True)
        if eps:
            fig.write_image(os.path.join(figDir, f"trajectory-shift_{figname}.eps"), format="eps")
        else:
            fig.write_image(os.path.join(figDir, f"trajectory-shift_{figname}.png"))

    fig.show()
    return



##############################################################################################################################
"""                                                       V. Matrices                                                      """
##############################################################################################################################

def corr_matrix(df, font_size=12, height=800, width=1200, figDir="figs/corr_matrix", figname=""):
    df.columns = [col if not col.startswith("Significant") else "Height of waves and swell" for col in df.columns]
    fig = px.imshow(df.corr())
    fig.update_layout(coloraxis=dict(cmin=-1, cmax=1, colorbar=dict(title=dict(text="Correlation", font_size=22), tickfont_size=18)), height=height, width=width, font_size=font_size,
                          margin=dict(t=5, b=5, l=5, r=5, pad=1))
    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'corr_matrix_{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return

def confusion_matrix(df=None, cols=["Taxa", "COMMON_NAME"], order=[], diagonal_out=True, sort=True, cmax_diag=0.2, colorscale_diag='viridis', colorscale_matrix='Blues', return_fig='matrix', artificial_trajectory=["levy-flight", "random-walk"], classifier="InceptionTime", velocity="arch-segment", size=None, figname="",
                     figDir="figs/confusion_matrix", tickfont_size=8., show_OX_dividers=True, to_origin="", save=False, legend_tickfont_size=32, font_size=20, ylabel=None,
                     **layout_kwargs):
    """Confusion matrix. X axis predicted species, Y axis actual species. A(i,j) = odds of predicting i for class j."""
    confusion_matrix_df, categories = analysis.compute_confusion_matrix(df=df, cols=cols, artificial_trajectory=artificial_trajectory, classifier=classifier, velocity=velocity, size=size, to_origin=to_origin)

    if diagonal_out and not order:
        diagonal = pd.Series(confusion_matrix.values.diagonal().copy(), index=confusion_matrix_df.index.get_level_values(1))
        np.fill_diagonal(confusion_matrix.values, 0)
    fig = px.imshow(confusion_matrix_df, color_continuous_scale=colorscale_matrix)
    if len(cols) > 1:
        if order:
            num_by_cat = [len(categories[cat]) for cat in order]
            cat_sup = [cat for cat, num in zip(order, num_by_cat) for i in range(num)]
            cat_inf = [cs for cat in order for cs in categories[cat]]
            confusion_matrix_df = confusion_matrix_df.loc[cat_inf, cat_inf]
            if diagonal_out:
                diagonal = pd.Series(confusion_matrix_df.values.diagonal().copy(), index=cat_inf)
                np.fill_diagonal(confusion_matrix_df.values, 0)
                if sort:
                    diagonal.index = pd.MultiIndex.from_arrays([cat_sup, cat_inf])
                    diagonal = diagonal.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(ascending=False))
                    diagonal = diagonal[order]
                    cat_sup = diagonal.index.get_level_values(0)
                    cat_inf = diagonal.index.get_level_values(1)
                    diagonal.index = diagonal.index.droplevel(0)
                    confusion_matrix_df = confusion_matrix_df.loc[diagonal.index, diagonal.index]

            fig = px.imshow(confusion_matrix_df, color_continuous_scale=colorscale_matrix)
        else:
            cat_sup = [cs for cs, ci in categories.items() for i in range(len(ci))]
            cat_inf = [cs for cs_list in categories.values() for cs in cs_list]

        multilabel = [cat_sup, cat_inf]
        fig.data[0]["y"] = multilabel
        fig.data[0]["x"] = multilabel
        label_type = "multicategory"
    else:
        fig.data[0]["y"] = categories
        fig.data[0]["x"] = categories
        label_type = "category"

    if diagonal_out:
        colorscale = transparent_colorscale(fig, upper=False)
        fig.update_layout(
                          coloraxis = dict(cmin=0, cmax=cmax_diag, colorscale=colorscale, colorbar=dict(title=dict(text='',font=dict(family='sans-serif', size=40)), tickfont_size=legend_tickfont_size, x=1.01, len=0.6, y=0.45, thickness=40, xpad=80,
                                                                                 tickvals = [0, 0.04, 0.09, 0.14, 0.19],
                                                                                 ticktext = ["0", "0.05", "0.1", "0.15", "> 0.2"]
                                                                                 )),
                          # height=1550, width=1650,
                          height=1490, width=1700,
                          font_size=font_size,
                          margin = dict(t=0, b=80, l=250, r=100, pad=1),
                          xaxis = dict(title=dict(text="Predicted", font=dict(family='sans-serif', size=44)),  tickfont_size=tickfont_size, type=label_type, showdividers=show_OX_dividers), #, tickson='boundaries', showgrid=True, gridwidth=1, showdividers=False),
                          yaxis = dict(title=dict(text="Real" if ylabel is None else ylabel, font=dict(family='sans-serif', size=44)), tickfont_size=tickfont_size, type=label_type, showdividers=True)
                )
        fig.add_annotation(
            text="Confusion<br>Probability",
            x=1.02,  # Adjust this for horizontal positioning
            y=0.8,    # Adjust this for vertical positioning
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(family='sans-serif', size=40)
        )

        fig.update_layout(**layout_kwargs)
        diagonal = pd.Series(diagonal, index=cat_inf).to_frame()
        fig_diag = px.imshow(diagonal, color_continuous_scale=colorscale_diag)
        fig_diag.update_layout(coloraxis = dict(cmin=0, cmax=1, colorbar=dict(title=dict(text='', font=dict(family='sans-serif', size=40)), tickfont_size=legend_tickfont_size, x=1.02, len=0.6, y=0.45, thickness=40, xpad=70,
                                                                              tickvals=[0, 0.25, 0.5, 0.75, 1],
                                                                              ticktext=["0", "0.25", "0.5", "0.75", "1"]
                                                                                )),
                               height=1500, width=400,
                               font_size=font_size,
                               margin = dict(t=0, b=80, l=0, r=100, pad=1),
                               xaxis = dict(visible=False),
                               yaxis = dict(visible=False)
                               )
        fig_diag.add_annotation(
            text="Accuracy",
            x=1.23,  # Adjust this for horizontal positioning
            y=0.8,    # Adjust this for vertical positioning
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(family='sans-serif', size=40)
        )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_diag.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        if return_fig == 'matrix':
            return fig
        elif return_fig == 'diagonal':
            return fig_diag
        else:
            raise ValueError(f"Unknown return_fig value: {return_fig}")
        # TODO: join the two figures
        fig_grid = go.Figure()
        # Add the traces from the first figure
        for trace in fig.data:
            trace.x += 0.5  # to align the heatmaps
            fig_grid.add_trace(trace)

        # Add the traces from the second figure
        for trace in fig_diag.data:
            trace.x += confusion_matrix.shape[0] + 1.5  # to align the heatmaps
            fig_grid.add_trace(trace)

        # Layout settings
        fig_grid.update_layout(
            width=900,
            height=400,
            coloraxis_colorbar_title="Matrix",
            coloraxis2_colorbar_title="Diagonal"
        )
        return fig_grid

    else:
        colorscale = transparent_colorscale(fig)
        fig.update_layout(
                          coloraxis = dict(cmin=0, cmax=1, colorscale=colorscale, colorbar=dict(title_text='Probability', tickfont_size=legend_tickfont_size, x=1.02, len=0.7, y=0.55)),
                          # height=1550, width=1650,
                          height=1500, width=1700,
                          font_size=font_size,
                          margin = dict(t=0, b=80, l=240, r=100, pad=1),
                          xaxis = dict(title_text="Predicted", tickfont_size=tickfont_size, type=label_type, showdividers=show_OX_dividers), #, tickson='boundaries', showgrid=True, gridwidth=1, showdividers=False),
                          yaxis = dict(title_text="Real" if ylabel is None else ylabel, tickfont_size=tickfont_size, type=label_type, showdividers=True)
                )
        fig.update_layout(**layout_kwargs)
    if save:
        size_str = "" if size is None else f'_size-{other_utils.encoder(size)}'
        figDir_full = os.path.join(figDir, classifier)
        Path(figDir_full).mkdir(exist_ok=True, parents=True)
        fig.write_image(os.path.join(figDir_full, f"confusion_matrix_trajectory-{artificial_trajectory}_clf-{classifier}_velocity-{velocity}{size_str}_to-origin-{to_origin}{figname}.png"))

    return fig

def f1_plot(figDir, xaxis_title, *args, figdict={}, y_range=[0, 1], legend_x=0.58, legend_y=0.98, legend_title_fs=38, legend_fs=36, **kwargs):
    df = analysis.get_f1_scores(*args, **kwargs)
    fig = get_figure(yaxis_title="F1 score", xaxis_title=xaxis_title, width=1000, height=700, yaxis_range=y_range)
    f1_avg, f1_avg_base = df[["f1", "f1_base"]].mean(axis=0)
    fig.add_trace(go.Bar(x=df.index, y=df.f1_base, name="Base ({:.2f})".format(f1_avg_base)))
    fig.add_trace(go.Bar(x=df.index, y=df.f1, name="Classifier ({:.2f})".format(f1_avg)))
    fig.update_layout(legend=dict(title="Model (Macro F1)", title_font_size=legend_title_fs, font_size=legend_fs, x=legend_x, y=legend_y))
    Path(figDir).mkdir(exist_ok=True)
    ID = other_utils.dict_to_id(figdict)
    fig.write_image(os.path.join(figDir, f"{xaxis_title}_{ID}.png"))
    return fig

def f1_body_length(df, figdict={}, **kwargs):
    return f1_plot("figs/body_length/f1_score", "Body length", df, "BODY_LENGTH interval", figdict=figdict, **kwargs)

def f1_age(df, figdict={}, **kwargs):
    return f1_plot("figs/age/f1_score", "Age", df, "AGE", figdict=figdict, **kwargs)


##############################################################################################################################
"""                                                      VI. SHAP                                                          """
##############################################################################################################################

def plot_shap(clf="XGB", taxa=None, velocity="arch-segment", add_dt=True, to_origin="space", overwrite=False, alpha=0.3, xlims=None,
              max_display=None, ext="png"):
    kwds = {k:v for k, v in locals().items() if k not in ["clf", "overwrite", "alpha", "xlims", "max_display", "ext"]}
    figDir = fullPath(f"figs/shap/{clf}/") if taxa is None else fullPath(f"figs/shap/{clf}/{taxa}")
    figpath = os.path.join(figDir, f"{other_utils.dict_to_id(kwds)}.{ext}")
    if overwrite:
        figpath = figpath.replace(".{ext}", f"_{other_utils.dict_to_id(overwrite=overwrite, xlims=xlims, max_display=max_display)}.{ext}")

    if Path(figpath).exists() and not overwrite:
        warnings.warn("Already existed", RuntimeWarning)
        return
    else:
        taxa_str = "" if taxa is None else taxa
        clf = getattr(models, clf)(weather="all", **kwds)
        columns = [col if not col.startswith("Significant") else "Wave and swell height" for col in clf.features]
        if max_display is None:
            max_display = len(columns)
        print("Loading data ...")
        data = analysis.merge_shap_values(**kwds, use_tqdm=True)

        print("Plotting ...")
        shap.summary_plot(data["values"], pd.DataFrame(data["X"], columns=columns), max_display=max_display, show=False, alpha=alpha, sort=False)
        fig = plt.gcf()
        if taxa is not None:
            plt.title(taxa_str, fontdict=dict(family="serif", size=24))
        if xlims is not None:
            fig.axes[0].set_xlim(xlims)
        Path(figDir).mkdir(exist_ok=True, parents=True)
        plt.savefig(figpath)
        return fig

def shap_probabilty_impact():
    data = pd.DataFrame(file_management.load_lzma("utils/data/shap_values/shap_probability_impact_to-origin-_add-dt.lzma"))
    taxas = data.columns.levels[0]
    features = data[taxas[0]].columns
    features_renamed = [f.replace("metre", "m").replace("net ", "") if not f.startswith("Significant") else "Waves & swell height" for f in features]

    fig = get_subplots(cols=3, rows=4, subplot_titles=taxas, shared_xaxes=True, shared_yaxes=True, y_title="Increment in log probability", vertical_spacing=0.08,
                                          font_size=50,
                                         )
    rows, cols = rows_cols(4, 3)
    for row, col, (taxa, df) in zip(rows, cols, data.groupby(level=0, axis=1)):
        fig.add_trace(go.Bar(x=features_renamed, y=df.loc[0, (taxa, features)], error_y=dict(type="data", array=df.iloc[1], thickness=3, width=2), showlegend=False), col=col, row=row)
    fig.update_layout(**get_range_data(11, *[*zip(*[*get_common_range(fig, axes=["y"]).items()])]),
                      **get_tickdata(11, size=30), margin=dict(t=55, b=0, r=20, l=130), height=1600, width=2800
                     )
    fig.layout.annotations[-1]["xshift"] = -70
    fig.write_image("figs/shap/feature_importance_taxa_add-dt_common-range.png")
    return

##############################################################################################################################
"""                                                VII. Multicollinearity                                                  """
##############################################################################################################################

def hierarchy_dendrogram(X, labels=None, parentDir=fullPath("figs/"), fontsize=30, figname='default', save=False):
    """
    Notes on the linkage matrix:
    This matrix represents a dendrogram, where elements
        1, 2: two clusters merged at each step,
        3: distance between these clusters,
        4: size of the new cluster - the number of original data points included.
    """
    is_df = isinstance(X, pd.core.frame.DataFrame)
    if labels is None:
        if is_df:
            labels = X.columns.to_list()
        else:
            labels = [*range(X.shape[1])] if len(X.shape) > 1 else None #int(-1 + np.sqrt(1 + 8*X.size)/2))]

    Path(parentDir).mkdir(exist_ok=True, parents=True)
    fig = plt.figure(figsize=(8, 12))
    ax = plt.subplot(111)

    corr_linkage = hierarchy.ward(X.values if is_df else X)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=labels, ax=ax, leaf_rotation=90 #orientation="left"
    )
    if not is_df:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        fig.tight_layout()
        plt.savefig(os.path.join(parentDir, f"dendrogram_{figname}.png"))
    else:
        plt.close()
    return corr_linkage, dendro

def cluster_matrix(X, labels=None, title="Pearson correlation", parentDir=fullPath("figs/"), figname='default', hovermode=False, return_fig=False,
                   colorbar_x=0.9, ticksize=8, cmin=-1, cmax=1):
    _, dendro = hierarchy_dendrogram(X, labels=labels)
    order = dendro["leaves"]
    #corr = X.corr() if isinstance(X, pd.core.frame.DataFrame) else np.corrcoef(X)
    X_ordered = X[order, :][:, order] if labels is None else pd.DataFrame(X, index=labels, columns=labels).iloc[order,:].iloc[:, order]
    fig = px.imshow(X_ordered)
    fig.update_layout(margin=dict(l=0, b=30, r=60, t=10, pad=1), xaxis_tickfont_size=ticksize, yaxis_tickfont_size=ticksize,
                      coloraxis=dict(cmin=cmin, cmax=cmax, colorbar=dict(title_text=title, tickfont_size=16, title_font_size=20, x=colorbar_x)),
                      height=800, width=1200, font_size=20, hovermode=hovermode)
    if labels is not None and len(labels) == 2:
        multilabel = [labels[0][order], labels[1][order]]
        fig.data[0]["x"] = multilabel
        fig.data[0]["y"] = multilabel
        fig.update_layout(xaxis = dict(tickfont_size=ticksize, type="multicategory", showdividers=True),
                          yaxis = dict(tickfont_size=ticksize, type="multicategory", showdividers=True)
                         )
    Path(parentDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(parentDir, f"cluster_matrix-{figname}.png"))
    return fig if return_fig else None

def shapelet_dendrogram(d=0, minlen=3, zero_pad=False, compare_to="shapelets", distance="euclidean",
                        parentDir=fullPath("figs/shapelet/multicollinearity"), fontsize=30, save=False):
    kwds = dict(d=d, minlen=minlen, compare_to=compare_to, zero_pad=zero_pad, distance=distance)
    D = analysis.shapelet_distance(**kwds)
    #D = D[np.triu_indices_from(D, k=1)]
    if distance == "dtw":
        idxs = np.triu_indices_from(D, k=1)
        D_triang = 0.5 * (D[idxs] + D.T[idxs])
        D[idxs] = D_triang
        D[idxs[::-1]] = D_triang
    D = squareform(D)
    figname = f"shapelet_{other_utils.dict_to_id(kwds)}"
    return hierarchy_dendrogram(D, parentDir=parentDir, figname=figname, fontsize=fontsize, save=save)

def shapelet_cluster_matrix(d=0, minlen=3, zero_pad=False, compare_to="shapelets", parentDir=fullPath("figs/shapelet/multicollinearity")):
    kwds = dict(d=d, minlen=minlen, compare_to=compare_to, zero_pad=zero_pad)
    D = analysis.shapelet_distance(**kwds)
    figname = f"shapelet_{other_utils.dict_to_id(kwds)}"
    cluster_matrix(D, parentDir=parentDir, figname=figname)
    return



##############################################################################################################################
"""                                                     VIII. Forecast                                                     """
##############################################################################################################################

get_ID_dict = lambda data: dict(ID=data["ID"], length=data["length"], num_cds=data["x_valid"].shape[1], add_bathymetry=data["add_bathymetry"], add_dt=data["add_dt"], train_percentage=data["train_percentage"])

def cds_forecast(data, cds=["x", "y", "z", "sin t", "cos t"], figDir=fullPath("figs/forecast/coordinates")):
    num_cds = data["x_valid"].shape[1]
    fig = get_subplots(cols=1, rows=num_cds, width=1200, shared_xaxes=True, x_title="Day")
    colors = plotly_default_colors()
    for i in range(num_cds):
        fig.add_trace(go.Scatter(x=data["t_valid"], y=data["x_valid"].numpy().T[i], name="Original", showlegend=i==0, mode="lines+markers", line=dict(color=colors[0]), marker=dict(color=colors[0])), col=1, row=i+1)
        fig.add_trace(go.Scatter(x=data["t_valid"], y=data["rnn_forecast"].T[i], name="Forecasted", showlegend=i==0, mode="lines+markers", line=dict(color=colors[1]), marker=dict(color=colors[1])), col=1, row=i+1)
    tickdata = get_tickdata(num_cds)
    yaxis_data = {"yaxis{}_title".format(l): cd for l, cd in zip([*range(1, num_cds+1)], cds)}
    fig.update_layout(**tickdata, font_size=28, margin=dict(l=10, t=75), **yaxis_data, title=dict(text="Train: {:.0f} days      Test: {:.0f} days".format(data["dt_train"], data["dt_test"]), font_size=24))

    Path(figDir).mkdir(exist_ok=True, parents=True)
    ID_dict = get_ID_dict(data)
    fig.write_image(os.path.join(figDir, f"{other_utils.dict_to_id(ID_dict)}.png"))
    return

def forecast_lat_lon(data, geo=False, scale=20, figDir=fullPath("figs/forecast/lat-lon")):
    ID_dict = get_ID_dict(data)
    fig = get_figure()
    colors = plotly_default_colors()
    trajectories = dict(Training=(data["z_original"], colors[0]),
                        Original=(data["z_valid"], colors[1]),
                        Forecast=(data["z_forecast"], colors[2]))
    if geo:
        figDir = os.path.join(figDir, "geo")
        for label, (trajectory, color) in trajectories.items():
            marker_size = np.ones(trajectory.shape[1])
            marker_size[0] = 20
            marker_size[-1] = 12
            fig.add_trace(go.Scattergeo(lat = trajectory[0],
                                        lon = trajectory[1],
                                        mode = 'lines+markers',
                                        line = dict(width = 3, color=color, dash='dash'),
                                        opacity = 1,
                                        name = label,
                                        marker_size = marker_size
                                       )
                         )
        fig.update_layout(
                title=dict(text="Train: {:.0f} days      Test: {:.0f} days".format(data["dt_train"], data["dt_test"]), font_size=24, y=0.85),
                margin=dict(l=20, b=5, t=0),
                legend=dict(y=0.6),
                geo = dict(
                    projection_scale=scale, #this is kind of like zoom
                    center=dict(lat=data["z_valid"][0].mean(), lon=data["z_valid"][1].mean()), # this will center on the point
                ))
    else:
        for label, (trajectory, color) in trajectories.items():
            marker_size = np.ones(trajectory.shape[1])
            marker_size[0] = 20
            marker_size[-1] = 12
            fig.add_trace(go.Scatter(x=trajectory[1], y=trajectory[0], mode = 'lines+markers',
                                        line = dict(width = 3, color=color, dash='dash'),
                                        opacity = 1,
                                        name = label,
                                        marker_size = marker_size)
                         )
        fig.update_layout(
                title=dict(text="Train: {:.0f} days      Test: {:.0f} days".format(data["dt_train"], data["dt_test"]), font_size=24),
                margin=dict(l=60, b=80, t=80, r=80),
                xaxis_title_text = "Longitude",
                yaxis_title_text = "Latitude",
                font_size = 28
                )
    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f"{other_utils.dict_to_id(ID_dict)}.png"))
    return


##############################################################################################################################
"""                                                      IX. Other                                                         """
##############################################################################################################################

def spherical_grid(figDir="figs/grid", figname="", surface_color='#1f77b4', line_color='#101010', r=1, **bining_kwargs):
    """
    Visualization of spherical grid with cells of equal surface.
    """
    theta, phi = np.meshgrid(*preprocessing.lat_lon_bins(out="rad", **bining_kwargs)) # theta = latitude = pi/2 - polar angle.
    x = r*np.cos(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.cos(theta)
    z = r*np.sin(theta)

    colorscale = [[0, surface_color],
                  [1, surface_color]]
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, hidesurface=False, surfacecolor=np.zeros(z.shape), showlegend=False, showscale=False, cmin=0, cmax=1, colorscale=colorscale, opacity=1),
                          go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), mode='lines', line=dict(color=line_color, width=4), showlegend=False),
                          go.Scatter3d(x=x.T.flatten(), y=y.T.flatten(), z=z.T.flatten(), mode='lines', line=dict(color=line_color, width=4), showlegend=False)
                         ],
                    layout=dict(margin=dict(l=0, r=0, t=0, b=0, pad=0), width=800, height=800,
                               scene = dict(
                                            camera = dict(center=dict(x=0.5, y=0.5, z=0.35)),
                                            domain = dict(x=[0,1], y=[0,1]),
                                            xaxis = dict(showticklabels=False, title_text="", visible=False), #, visible=False),
                                            yaxis = dict(showticklabels=False, title_text="", visible=False), #, visible=False),
                                            zaxis = dict(showticklabels=False, title_text="")
                               ),
                                scene_aspectmode='data'
                               )
                    )
    Path(figDir).mkdir(exist_ok=True, parents=True)
    fig.write_image(os.path.join(figDir, f'spherical_grid_{figname}.png'))
    fig.update_layout(template="plotly_dark")
    fig.show()
    return
