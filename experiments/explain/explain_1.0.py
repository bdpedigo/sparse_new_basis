# %% [markdown]
# ##
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer import Rotator
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import unique_labels

from graspologic.cluster import AutoGMMCluster
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspologic.plot import pairplot
from graspologic.simulations import sbm
from graspologic.utils import remap_labels

from sparse_decomposition.decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import soft_threshold
from sparse_new_basis.plot import set_theme, savefig
from pathlib import Path

set_theme()


fig_dir = Path("sparse_new_basis/results/explain_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
n = 100
d = 16


def sample_data(n=100, d=16):
    rand_normal = np.random.normal(0, 1, size=(n, d))
    U, _ = np.linalg.qr(rand_normal)
    V = ortho_group.rvs(d)
    i_range = np.arange(1, d + 1)
    sigmas = 10 - np.sqrt(i_range)
    sigma = np.diag(sigmas)
    S = U @ sigma @ V.T

    rand_normal = np.random.normal(0, 1, size=(n, d))
    Y, _ = np.linalg.qr(rand_normal)
    Y = soft_threshold(Y, 20)

    X = S @ Y.T + np.random.normal(0, 0.03, size=(n, n))  # TODO variance right?
    return X


#%%
n_components = 4

from sparse_decomposition.decomposition.decomposition import (
    _varimax,
    _polar_rotate_shrink,
    _polar,
)

from sparse_decomposition.utils import soft_threshold
from graspologic.embed import selectSVD


def _initialize(X):
    U, D, Vt = selectSVD(X, n_components=n_components)
    return U, Vt.T


#%%
from pathlib import Path
import networkx as nx
from graspologic.utils import pass_to_ranks, get_lcc, symmetrize

data_dir = Path("sparse_new_basis/data/maggot")
g = nx.read_weighted_edgelist(
    data_dir / "G.edgelist", create_using=nx.DiGraph, nodetype=int
)
meta = pd.read_csv(data_dir / "meta_data.csv", index_col=0)
adj = nx.to_numpy_array(g, nodelist=meta.index)
adj = symmetrize(adj)
adj, inds = get_lcc(adj, return_inds=True)
meta = meta.iloc[inds]

hemisphere = "left"
if hemisphere == "left":
    meta["inds"] = np.arange(len(meta))
    meta = meta[meta["left"]]
    inds = meta["inds"]
    adj = adj[np.ix_(inds, inds)]
# TODO just try with one hemisphere

preprocessing = "ptr"
if preprocessing == "ptr":
    adj_to_embed = pass_to_ranks(adj)
elif preprocessing == "sqrt":
    pass  # TODO
else:
    adj_to_embed = adj
#%%
# X = sample_data()


def _initialize(X, n_components):
    U, D, Vt = selectSVD(X, n_components=n_components)
    return U, Vt.T


X = adj_to_embed
n_components = 10
Z_hat, Y_hat = _initialize(X, n_components)


def plot_components(U, axs, s=5, alpha=0.5, linewidth=0, **kwargs):
    for i in range(n_show):
        for j in range(n_show):
            ax = axs[i, j]
            if i != j:
                sns.scatterplot(
                    x=U[:, j],
                    y=U[:, i],
                    s=s,
                    alpha=alpha,
                    ax=ax,
                    linewidth=linewidth,
                    **kwargs,
                )
                ax.axvline(0, linewidth=1, linestyle=":", color="black")
                ax.axhline(0, linewidth=1, linestyle=":", color="black")
            else:
                p_nonzero = np.count_nonzero(U[:, i]) / len(U)
                ax.set(xlim=(0, 1), ylim=(0, 1))
                ax.text(
                    0.5,
                    0.5,
                    f"Loading {i+1}\n{p_nonzero:.3f}",
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                )
                ax.axis("off")
            # ax.axis("off")
            ax.set(xticks=[], yticks=[], ylabel="", xlabel="")


n_iter = 1
n_show = 5
fig, axs = plt.subplots(n_iter * n_show, 3 * n_show, figsize=(15, 5))
colors = sns.color_palette("deep", 3)

for i in range(n_iter):
    A = X.T @ Z_hat
    U, _, _ = selectSVD(A, n_components=A.shape[1], algorithm="full")
    plot_components(U, axs[i * n_show : (i + 1) * n_show, :n_show], color=colors[0])
    # pairplot(U, alpha=0.2)

    U_rot = _varimax(U)
    plot_components(
        U_rot, axs[i * n_show : (i + 1) * n_show, n_show : 2 * n_show], color=colors[1]
    )
    # pairplot(U_rot, alpha=0.2)

    U_thresh = soft_threshold(U_rot, gamma=5 * n_components)
    plot_components(
        U_thresh,
        axs[i * n_show : (i + 1) * n_show, 2 * n_show : 3 * n_show],
        color=colors[2],
    )
    # pairplot(U_thresh, alpha=0.2

    Y_hat = U_thresh
    Z_hat = _polar(X @ Y_hat)

    B_hat = Z_hat.T @ X @ Y_hat

    print(np.linalg.norm(B_hat))

for ax in axs.flat:
    [i.set_linewidth(0.3) for i in ax.spines.values()]

axs[0, 2].set_title(r"Polar ($\tilde{Y}$)", color=colors[0])
axs[0, 7].set_title(r"Rotate ($Y^*$)", color=colors[1])
axs[0, 12].set_title(r"Shrink ($\hat{Y}$)", color=colors[2])

stashfig("prs-explain")


# %%
