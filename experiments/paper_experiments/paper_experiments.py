#%% [markdown]
## A New Basis for Sparse PCA
# > Trying to replicate and extend results from the paper above by F. Chen and K. Rohe

# - toc: true
# - badges: true
# - categories: [pedigo, graspologic, sparse]
# - hide: false
# - search_exclude: false

#%% [markdown]
### Preliminaries

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

from sparse_matrix_analysis import sparse_component_analysis
from sparse_matrix_analysis.utils import soft_threshold


# plotting settings
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}


def set_theme():
    for key, val in rc_dict.items():
        mpl.rcParams[key] = val
    context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
    sns.set_context(context)


set_theme()


#%% [markdown]
### Trying to replicate Figure 4a - recovery for a simple low rank model
# %% [markdown]
# ##

X = np.random.normal(size=(10, 10))

Z, Y = sparse_component_analysis(X)


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


def proportion_variance_explained(X, Y):
    X = X.copy()
    # X -= X.mean(axis=0)[None, :]
    return (np.linalg.norm(X @ Y @ np.linalg.inv(Y.T @ Y) @ Y.T, ord="fro") ** 2) / (
        np.linalg.norm(X, ord="fro") ** 2
    )


def pca(X, n_components=2):
    pca_obj = PCA(n_components=n_components, svd_solver="full")
    Z_hat = pca_obj.fit_transform(X)
    Y_hat = pca_obj.components_.T
    return Z_hat, Y_hat


k_range = np.arange(2, d + 1, 2)
n_replicates = 10
rows = []
for i in range(n_replicates):
    X = sample_data()
    # X -= np.mean(X, axis=0) # TODO ?
    for k in k_range:
        gamma = k * 2.5
        Z_hat_sca, Y_hat_sca = sparse_component_analysis(
            X, n_components=k, gamma=gamma, max_iter=1
        )
        pve = proportion_variance_explained(X, Y_hat_sca)
        rows.append({"replicate": i, "k": k, "pve": pve, "method": "SCA"})

        Z_hat_pca, Y_hat_pca = pca(X, n_components=k)
        pve = proportion_variance_explained(X, Y_hat_pca)
        rows.append({"replicate": i, "k": k, "pve": pve, "method": "PCA"})

        # this just verifies that PVE calc seems reasonable
        # pca_obj = PCA(n_components=k, svd_solver="full")
        # Z_hat = pca_obj.fit_transform(X)
        # assert pca_obj.explained_variance_ratio_.sum() - pve < 1e-10

results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = sns.color_palette("deep", 10)

palette = {"SCA": colors[0], "PCA": colors[7]}
sns.lineplot(
    x="k",
    y="pve",
    data=results,
    ci=None,
    ax=ax,
    markers=["o", ","],
    hue="method",
    palette=palette,
    style="method",
)
dummy_palette = dict(
    zip(results["replicate"].unique(), n_replicates * [palette["SCA"]])
)
sns.lineplot(
    x="k",
    y="pve",
    data=results[results["method"] == "SCA"],
    ci=None,
    ax=ax,
    alpha=0.3,
    hue="replicate",
    palette=dummy_palette,
    legend=False,
    lw=1,
)
dummy_palette = dict(
    zip(results["replicate"].unique(), n_replicates * [palette["PCA"]])
)
sns.lineplot(
    x="k",
    y="pve",
    data=results[results["method"] == "PCA"],
    ci=None,
    ax=ax,
    alpha=0.3,
    hue="replicate",
    palette=dummy_palette,
    legend=False,
    lw=1,
)
ax.set(
    yticks=[0.25, 0.5, 0.75],
    xticks=[4, 8, 12, 16],
    ylabel="PVE",
    xlabel="# of PCs",
    ylim=(0.05, 0.95),
)
plt.savefig("PVE-by-rank", transparent=False, facecolor="w")


# %% [markdown]
# ##
X = sample_data()
k = 4
gamma = k * 2.5
Z_hat_sca, Y_hat_sca = sparse_component_analysis(
    X, n_components=k, gamma=gamma, max_iter=100, reorder_components=True
)  # the COLUMNS of Y_hat_sca are sparse

Z_hat_pca, Y_hat_pca = pca(X, n_components=k)


component_map = {"PCA": Y_hat_pca, "SCA": Y_hat_sca}

fig, axs = plt.subplots(4, 2, figsize=(16, 8))
for i in range(4):
    for j, method in enumerate(["PCA", "SCA"]):
        Y_hat = component_map[method]
        y = Y_hat[:, i]
        ax = axs[i, j]
        sns.scatterplot(x=range(len(y)), y=y, ax=ax, s=15)
        ax.set(xticks=[], yticks=[], ylim=(-0.3, 0.3))
        ax.axhline(0, color="black", lw=1.5, linestyle=":")
        ax.spines["bottom"].set_visible(False)
        if j == 0:
            ax.set_ylabel(f"PC {i + 1}")

axs[0, 0].set_title("PCA")
axs[0, 1].set_title("SCA")
axs[0, 0].set(yticks=[-0.3, 0, 0.3])


print("SCA")
for i in range(k):
    print(np.linalg.norm(X @ Y_hat_sca[:, i]))
print()
print("PCA")
for i in range(k):
    print(np.linalg.norm(X @ Y_hat_pca[:, i]))
plt.savefig("PC-sparsity", transparent=False, facecolor="w")

#%% [markdown]
### Recovery of blocks for an SBM
# Similar experiment to Figure 4 b, c
# A few notes:
# - have not been able to get MCR as low as it was in the paper for the 0.05 * B SBM

#%% [markdown]
#### The generative model
# Undirected, unweighted graph with no self-loops
#
# $n = 900$, number of nodes
#
# $k = 4$, number of blocks
#
# Nodes distributed equally among the blocks
#
# $B$, the block probability matrix as shown below
# %%
B = 0.05 * np.array(
    [
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.7, 0.05, 0.05],
        [0.1, 0.05, 0.6, 0.25],
        [0.1, 0.05, 0.25, 0.6],
    ]
)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(
    B,
    square=True,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    cbar=False,
    cbar_kws=dict(shrink=0.7),
    annot=True,
)
ax.set(title=r"$B$", xlabel="Block")
n_verts = 900
comm_sizes = np.repeat(n_verts / 4, 4,).astype(int)

#%% [markdown]
#### The experiment
# 1. Sample an SBM according to the model above
# 2. Estimate block assignments from the sampled graph each of 3 ways:
#    - GMM ($k=4$) on ASE embedding ($d=4$)
#    - GMM ($k=4$) on regularized LSE embedding ($d=4$)
#    - Use nonzeros from SCA as an indicator vector for each block, settling ties randomly
#         - SCA has a tuning parameter $\gamma$ to control the level of sparsity, here
#           that parameter is swept
# 3. Compute misclassification rate for each of the above
# 4. Repeat 1 - 3 30 times
#%%


def remap_labels(y_true, y_pred, return_map: bool = False,) -> np.ndarray:
    # REF: soon to be in graspologic
    """
    Remaps a categorical labeling (such as one predicted by a clustering algorithm) to
    match the labels used by another similar labeling.

    Given two $n$-length vectors describing a categorical labeling of $n$ samples, this
    method reorders the labels of the second vector (`y_pred`) so that as many samples
    as possible from the two label vectors are in the same category.


    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels, or, labels to map to.
    y_pred : array-like of shape (n_samples,)
        Labels to remap to match the categorical labeling of `y_true`. The categorical
        labeling of `y_pred` will be preserved exactly, but the labels used to
        denote the categories will be changed to best match the categories used in
        `y_true`.
    return_map : bool, optional
        Whether to return a dictionary where the keys are the original category labels
        from `y_pred` and the values are the new category labels that they were mapped
        to.

    Returns
    -------
    remapped_y_pred : np.ndarray of shape (n_samples,)
        Same categorical labeling as that of `y_pred`, but with the category labels
        permuted to best match those of `y_true`.
    label_map : dict
        Mapping from the original labels of `y_pred` to the new labels which best
        resemble those of `y_true`. Only returned if `return_map` was True.

    Examples
    --------
    >>> y_true = np.array([0,0,1,1,2,2])
    >>> y_pred = np.array([2,2,1,1,0,0])
    >>> remap_labels(y_true, y_pred)
    array([0, 0, 1, 1, 2, 2])

    Notes
    -----
    This method will work well when the label vectors describe a somewhat similar
    categorization of the data (as measured by metrics such as
    :func:`sklearn.metrics.adjusted_rand_score`, for example). When the categorizations
    are not similar, the remapping may not make sense (as such a remapping does not
    exist).

    For example, consider when one category in `y_true` is exactly split in half into
    two categories in `y_pred`. If this is the case, it is impossible to say which of
    the categories in `y_pred` match that original category from `y_true`.
    """

    if not isinstance(return_map, bool):
        raise TypeError("return_map must be of type bool.")

    labels = unique_labels(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred, labels=labels)
    row_inds, col_inds = linear_sum_assignment(confusion_mat, maximize=True)
    label_map = dict(zip(labels[col_inds], labels[row_inds]))

    remapped_y_pred = np.vectorize(label_map.get)(y_pred)
    if return_map:
        return remapped_y_pred, label_map
    else:
        return remapped_y_pred


def components_to_labels(Y):
    n_categories = Y.shape[1]
    inds = np.arange(n_categories)
    labels = np.empty(len(Y), dtype=int)
    for i in range(len(Y)):
        abs_row = np.abs(Y[i, :])
        max_val = np.max(abs_row)
        if max_val != 0:
            label = inds[abs_row == max_val]
        else:
            label = inds
        label = np.random.choice(label)
        labels[i] = label
    return labels


def spectral_clustering(adj, n_components=4, method="lse", return_embedding=False):
    if method == "ase":
        embedder = AdjacencySpectralEmbed(n_components=n_components)
    elif method == "lse":
        embedder = LaplacianSpectralEmbed(n_components=n_components, form="R-DAD")
    latent = embedder.fit_transform(adj)
    gc = AutoGMMCluster(min_components=4, max_components=4)
    pred_labels = gc.fit_predict(latent)
    if return_embedding:
        return pred_labels, latent
    else:
        return pred_labels


def compute_mcr(true_labels, pred_labels):
    confusion = confusion_matrix(labels, pred_labels)
    row_inds, col_inds = linear_sum_assignment(confusion, maximize=True)
    mcr = 1 - (np.trace(confusion[row_inds][:, col_inds]) / np.sum(confusion))
    return mcr


n_replicates = 30
gammas = [24, 36, 48, 60, 64]
rows = []
for replicate in range(n_replicates):
    # sample data
    adj, labels = sbm(comm_sizes, B, directed=False, return_labels=True)

    # GMMoASE
    ase_pred_labels, ase_embedding = spectral_clustering(
        adj, method="ase", return_embedding=True
    )
    ase_pred_labels = remap_labels(labels, ase_pred_labels)
    mcr = compute_mcr(labels, ase_pred_labels)
    for gamma in gammas:
        rows.append(
            {"mcr": mcr, "method": "GMMoASE", "gamma": gamma, "replicate": replicate}
        )

    # GMMoLSE
    lse_pred_labels, lse_embedding = spectral_clustering(
        adj, method="ase", return_embedding=True
    )
    lse_pred_labels = remap_labels(labels, lse_pred_labels)
    mcr = compute_mcr(labels, lse_pred_labels)
    for gamma in gammas:
        rows.append(
            {"mcr": mcr, "method": "GMMoLSE", "gamma": gamma, "replicate": replicate}
        )

    # SCA
    for gamma in gammas:
        Z_hat, Y_hat = sparse_component_analysis(
            adj, n_components=4, gamma=gamma, max_iter=5
        )
        sca_pred_labels = components_to_labels(Y_hat)
        sca_pred_labels = remap_labels(labels, sca_pred_labels)
        mcr = compute_mcr(labels, sca_pred_labels)
        rows.append(
            {"mcr": mcr, "method": "SCA", "gamma": gamma, "replicate": replicate}
        )

sca_embedding = Y_hat
results = pd.DataFrame(rows)
results

#%%
palette = dict(zip(["SCA", "GMMoASE", "GMMoLSE"], colors))
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=results, x="gamma", y="mcr", hue="method", ax=ax, palette=palette)
handles, labels = ax.get_legend_handles_labels()
labels[0] = "Method"
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.savefig("sbm-mcr", transparent=False, facecolor="w")

#%%
palette = dict(zip(range(len(B)), colors))
pairplot(ase_embedding, labels=ase_pred_labels, palette=palette, title="GMMoASE")
plt.savefig("ase-pair", transparent=False, facecolor="w")
pairplot(lse_embedding, labels=lse_pred_labels, palette=palette, title="GMMoLSE")
plt.savefig("lse-pair", transparent=False, facecolor="w")
pairplot(sca_embedding, labels=sca_pred_labels, palette=palette, title="SCA")
plt.savefig("sca-pair", transparent=False, facecolor="w")

#%%
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
sns.heatmap(np.abs(Y_hat), center=0, cmap="RdBu_r", ax=axs[0])
sns.heatmap(Y_hat != 0, center=0, cmap="RdBu_r", ax=axs[1])
