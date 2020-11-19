#%%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspologic.plot import pairplot
from graspologic.utils import get_lcc, pass_to_ranks, to_laplace
from sparse_matrix_analysis import SparseMatrixApproximation
from src.visualization import CLASS_COLOR_DICT
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

fig_dir = Path("sparse_new_basis/experiments/maggot/figs")


def stashfig(name, dpi=300, fmt="png", pad_inches=0.5, facecolor="w", **kws):
    plt.savefig(
        fig_dir / name,
        dpi=dpi,
        fmt=fmt,
        pad_inches=pad_inches,
        facecolor=facecolor,
        **kws,
    )


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
#%%
preprocessing = "ptr"
if preprocessing == "ptr":
    adj_to_embed = pass_to_ranks(adj)
elif preprocessing == "sqrt":
    pass  # TODO
else:
    adj_to_embed = adj
currtime = time.time()
sma = SparseMatrixApproximation(n_components=8, max_iter=0, gamma=None)
sma.fit_transform(adj_to_embed)
print(f"{time.time() - currtime} elapsed")

left_latent = sma.left_latent_
right_latent = sma.right_latent_

#%%

# print("Plotting...")
# currtime = time.time()
# columns = [f"Dimension {i+1}" for i in range(left_latent.shape[1])]
# plot_df = pd.DataFrame(data=left_latent, columns=columns, index=meta.index)
# plot_df = pd.concat((plot_df, meta), axis=1)
# pg = sns.PairGrid(
#     data=plot_df, hue="merge_class", palette=CLASS_COLOR_DICT, vars=columns[:4]
# )
# pg.map_upper(sns.scatterplot)
# plt.savefig(fig_dir / "left_sma", pad_inches=0.25, dpi=200)
# print(f"{time.time() - currtime} elapsed")

# pg = pairplot(left_latent, diag_kind=None)
# pg._legend.remove()
#
# pairplot(right_latent, labels=labels, palette=CLASS_COLOR_DICT, diag_kind=None)


# %%
def setup_R():
    import os

    # os.chdir("./")
    # print(os.getcwd())
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
    os.environ[
        "R_USER"
    ] = "/Users/bpedigo/miniconda3/envs/sparse/lib/python3.7/site-packages/rpy2"
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    epca = importr("epca")
    return epca


def run_sca_R(epca, *args):
    out = epca.sca(*args)
    loadings = out[0]
    scores = out[1]
    return np.asarray(scores), np.asarray(loadings)


def run_sma_R(epca, *args, **kwargs):
    out = epca.sma(*args, **kwargs)
    Z = np.asarray(out[0])
    B = np.asarray(out[1])
    Y = np.asarray(out[2])
    return Z, B, Y, out[3:]


epca = setup_R()
lap_to_embed = to_laplace(adj_to_embed, form="R-DAD")
#%%
currtime = time.time()
n_components = 20
gamma = 0.5 * np.sqrt(len(lap_to_embed) * n_components)
Z, B, Y, info = run_sma_R(epca, lap_to_embed, k=n_components, gamma=gamma, epsilon=1e-8)
print(f"{time.time() - currtime} elapsed to run SMA")

#%%
left_latent = Z
print("Plotting pairplot...")
currtime = time.time()
columns = [f"Dimension {i+1}" for i in range(left_latent.shape[1])]
plot_df = pd.DataFrame(data=left_latent, columns=columns, index=meta.index)
plot_df = pd.concat((plot_df, meta), axis=1)
pg = sns.PairGrid(
    data=plot_df, hue="merge_class", palette=CLASS_COLOR_DICT, vars=columns[:6]
)
pg.map_upper(sns.scatterplot, s=10, linewidth=0, alpha=0.7)
plt.savefig(fig_dir / "left_sma_R", pad_inches=0.25, dpi=200)
print(f"{time.time() - currtime} elapsed to plot pairplot")

#%%

embedding = Z
hue = "merge_class"
palette = CLASS_COLOR_DICT
p_nonzeros = []
all_component_neurons = []
for i, dim in enumerate(embedding.T[:20]):
    dim = dim.copy()
    # this just makes the biggest entries in abs value positive
    if dim[np.argmax(np.abs(dim))] < 0:
        dim = -dim
    sort_inds = np.argsort(dim)
    plot_df = pd.DataFrame()
    plot_df["dim"] = dim[sort_inds]
    plot_df["ind"] = range(len(plot_df))
    plot_df["labels"] = meta[hue].values[sort_inds]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.scatterplot(
        x="ind",
        y="dim",
        hue="labels",
        data=plot_df,
        ax=ax,
        palette=palette,
        legend=False,
        s=5,
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

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(p_nonzeros, kde=True, ax=ax)
sns.rugplot(p_nonzeros, ax=ax)

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


for i, component_neurons in enumerate(all_component_neurons):
    print(i)
    fig, axs = make_figure_axes()
    plot_3view(
        list(component_neurons), axs[0, :], palette=skeleton_color_dict, row_title="",
    )
    stashfig(f"component_{i+1}_morphology")

