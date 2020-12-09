#%%
import os
import pickle
import time
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from graspologic.plot import pairplot
from sparse_decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import calculate_explained_variance_ratio
from sparse_new_basis.data import load_scRNAseq
from sparse_new_basis.plot import savefig, set_theme
from sparse_new_basis.R import setup_R, sma_R_epca
import networkx as nx
from graspologic.utils import get_lcc, pass_to_ranks, to_laplace

set_theme()
epca = setup_R()


def sma_R(*args, **kwargs):
    return sma_R_epca(epca, *args, **kwargs)


fig_dir = Path("sparse_new_basis/results/maggot_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
data_dir = Path("sparse_new_basis/data/maggot")
g = nx.read_weighted_edgelist(
    data_dir / "G.edgelist", create_using=nx.DiGraph, nodetype=int
)
meta = pd.read_csv(data_dir / "meta_data.csv", index_col=0)
adj = nx.to_numpy_array(g, nodelist=meta.index)

adj, inds = get_lcc(adj, return_inds=True)
meta = meta.iloc[inds]

hemisphere = "left"
if hemisphere == "left":
    meta["inds"] = np.arange(len(meta))
    meta = meta[meta["left"]]
    inds = meta["inds"]
    adj = adj[np.ix_(inds, inds)]
# TODO just try with one hemisphere
#%%s
preprocessing = "ptr"
if preprocessing == "ptr":
    adj_to_embed = pass_to_ranks(adj)
elif preprocessing == "sqrt":
    pass  # TODO
else:
    adj_to_embed = adj
lap_to_embed = to_laplace(adj_to_embed, form="R-DAD")


#%%
currtime = time.time()
n_components = 20
gamma = 0.5 * np.sqrt(len(lap_to_embed) * n_components)
Z, B, Y, info = sma_R(
    lap_to_embed, k=n_components, gamma=gamma, epsilon=1e-5, return_all=True
)
print(f"{time.time() - currtime} elapsed to run SMA")

#%%
# from sparse_new_basis.plot import CLASS_COLOR_DICT
from src.visualization import CLASS_COLOR_DICT
from graspologic.plot import pairplot

left_latent = Z
print("Plotting pairplot...")
currtime = time.time()
columns = [f"Dimension {i+1}" for i in range(left_latent.shape[1])]
plot_df = pd.DataFrame(data=left_latent, columns=columns, index=meta.index)
plot_df = pd.concat((plot_df, meta), axis=1)
pg = sns.PairGrid(
    data=plot_df,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    vars=columns[:6],
    corner=True,
)
pg.map_lower(sns.scatterplot, s=10, linewidth=0, alpha=0.7)
pg.set(xticks=[], yticks=[])
stashfig("left_sma_R")
print(f"{time.time() - currtime} elapsed to plot pairplot")
#%%
right_latent = Y
print("Plotting pairplot...")
currtime = time.time()
columns = [f"Dimension {i+1}" for i in range(left_latent.shape[1])]
plot_df = pd.DataFrame(data=right_latent, columns=columns, index=meta.index)
plot_df = pd.concat((plot_df, meta), axis=1)
pg = sns.PairGrid(
    data=plot_df,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    vars=columns[:6],
    corner=True,
)
pg.map_lower(sns.scatterplot, s=10, linewidth=0, alpha=0.7)
pg.set(xticks=[], yticks=[])
stashfig("right_sma_R")
print(f"{time.time() - currtime} elapsed to plot pairplot")

#%%
embedding = Z
hue = "merge_class"
palette = CLASS_COLOR_DICT
p_nonzeros = []
all_component_neurons = []
for i, dim in enumerate(embedding.T[:10]):
    dim = dim.copy()
    # this just makes the biggest entries in abs value positive
    if dim[np.argmax(np.abs(dim))] < 0:
        dim = -dim
    sort_inds = np.argsort(dim)
    plot_df = pd.DataFrame()
    plot_df["dim"] = dim[sort_inds]
    plot_df["ind"] = range(len(plot_df))
    plot_df["labels"] = meta[hue].values[sort_inds]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.scatterplot(
        x="ind",
        y="dim",
        hue="labels",
        data=plot_df,
        ax=ax,
        palette=palette,
        legend=False,
        s=15,
        alpha=0.8,
        linewidth=0,
    )
    nonzero_inds = np.nonzero(dim)[0]
    p_nonzero = len(nonzero_inds) / len(dim)
    p_nonzeros.append(p_nonzero)

    zero_inds = np.nonzero(dim[sort_inds] == 0)[0]
    min_cutoff = min(zero_inds)
    max_cutoff = max(zero_inds)
    # min_cutoff = max(np.nonzero(dim[sort_inds][: len(dim) // 2])[0])
    # max_cutoff = min(np.nonzero(dim[sort_inds][len(dim) // 2 :])[0])

    line_kws = dict(color="grey", linewidth=1, linestyle="--")
    ax.axhline(0, **line_kws)
    ax.axvline(min_cutoff, **line_kws)
    ax.axvline(max_cutoff, **line_kws)

    ax.set(
        xticks=[],
        yticks=[],
        ylabel=f"Component {i+1}",
        xlabel="Index (sorted)",
        title=r"$p$ nonzero = " + f"{p_nonzero:.2}",
    )
    component_neurons = meta.iloc[nonzero_inds].index
    all_component_neurons.append(component_neurons)
    stashfig(f"Z_component_{i}")

#%%
from src.pymaid import start_instance
from src.visualization import plot_3view


def make_figure_axes():
    fig = plt.figure(figsize=(15, 5))
    # for the morphology plotting
    margin = 0.01
    # gap = 0.02
    n_col = 3
    morpho_gs = plt.GridSpec(
        1,
        3,
        figure=fig,
        wspace=0,
        hspace=0,
        left=margin,
        right=margin + 3 / n_col,
        top=1 - margin,
        bottom=margin,
    )
    morpho_axs = np.empty((1, 3), dtype="O")
    i = 0
    for j in range(3):
        ax = fig.add_subplot(morpho_gs[i, j], projection="3d")
        morpho_axs[i, j] = ax
        ax.axis("off")
    return fig, morpho_axs


skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)
start_instance()

start = 5
for i, component_neurons in enumerate(all_component_neurons[start:10]):
    i += start
    print(i)
    fig, axs = make_figure_axes()
    plot_3view(
        list(component_neurons), axs[0, :], palette=skeleton_color_dict, row_title="",
    )
    stashfig(f"Z_component_{i+1}_morphology")
