#%%
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

set_theme()


fig_dir = Path("sparse_new_basis/results/gene_sca__explain_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
sequencing_df, annotation_df = load_scRNAseq(fillna=True)


#%% throw out some genes with low variance
X = sequencing_df.values.copy()
var_thresh = VarianceThreshold(threshold=0.01)
X = var_thresh.fit_transform(X)
gene_index = sequencing_df.columns
gene_index = gene_index[var_thresh.get_support()]


#%%
neuron_index = sequencing_df.index
y = sequencing_df.index.get_level_values(level="Neuron_type").values

# stratify=y will try to set the distribution of class labels the same for train/test
X_train, X_test, index_train, index_test = train_test_split(
    X, neuron_index, stratify=y, train_size=2 ** 13
)

#%% center and scale training data
currtime = time.time()
scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
X_train = scaler.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to scale and center data.")

#%%
n_components = 4

from sparse_decomposition.sparse.sparse_decomposition import (
    _varimax,
    _polar_rotate_shrink,
    _polar,
)

from sparse_decomposition.utils import soft_threshold
from graspologic.embed import selectSVD


def _initialize(X):
    U, D, Vt = selectSVD(X, n_components=n_components)
    return U, Vt.T


X = X_train
Z_hat, Y_hat = _initialize(X_train)
A = X.T @ Z_hat
U, _, _ = selectSVD(A, n_components=A.shape[1], algorithm="full")
pairplot(U, alpha=0.2)
#%%
U_rot = _varimax(U)
pairplot(U_rot, alpha=0.2)

#%%
U_thresh = soft_threshold(U_rot, gamma=100)
pairplot(U_thresh, alpha=0.2)

#%%


#%%
currtime = time.time()
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to fit PCA model.")

#%%
currtime = time.time()
sca = SparseComponentAnalysis(n_components=n_components, max_iter=max_iter, gamma=gamma)
X_sca = sca.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to fit SCA model.")

#%%
max_iter = 5
gammas = [n_components, 100, 500, np.sqrt(X_train.shape[1]) * n_components, np.inf]
models_by_gamma = {}
Xs_by_gamma = {}
for i, gamma in enumerate(gammas):
    print(f"Gamma = {gamma}...")
    if gamma == np.inf:
        _max_iter = 0
    else:
        _max_iter = max_iter
    currtime = time.time()
    sca = SparseComponentAnalysis(
        n_components=n_components, max_iter=_max_iter, gamma=gamma
    )
    X_sca = sca.fit_transform(X_train)
    print(f"{time.time() - currtime:.3f} elapsed.")
    models_by_gamma[gamma] = sca
    Xs_by_gamma[gamma] = X_sca
    print()
#%%

rows = []
for gamma, model in models_by_gamma.items():
    explained_variance_ratio = model.explained_variance_ratio_
    for k, ev in enumerate(explained_variance_ratio):
        n_nonzero = np.count_nonzero(model.components_[: k + 1])
        rows.append(
            {
                "gamma": gamma,
                "explained_variance": ev,
                "n_components": k + 1,
                "n_nonzero": n_nonzero,
            }
        )
scree_df = pd.DataFrame(rows)

#%% screeplots, basically

# n_models = len(models_by_gamma)
# scree_df = pd.DataFrame(index=range(n_components))

# scree_df["n_components"] = np.tile(np.arange(1, n_components + 1), 2)
# scree_df["explained_variance"] = np.concatenate(
#     (np.cumsum(pca.explained_variance_ratio_), sca.explained_variance_ratio_)
# )
# scree_df["method"] = n_components * ["PCA"] + n_components * ["SCA"]
palette = dict(zip(gammas, sns.color_palette("deep", 10)))
gammas[:-1]

#%%
blue_shades = sns.color_palette("Blues", n_colors=len(gammas))[1:]
palette = dict(zip(gammas[:-1], blue_shades))
red_shades = sns.color_palette("Reds", n_colors=len(gammas))[1:]
palette[np.inf] = red_shades[-1]
# sns.color_palette("ch:start=.2,rot=-.3", as_cmap=False, n_colors=len(gammas) - 1)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

sns.lineplot(
    data=scree_df,
    x="n_components",
    y="explained_variance",
    hue="gamma",
    ax=ax,
    marker="o",
    palette=palette,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Gamma")
# ax.legend().set_title("Gamma")
ax.set(ylabel="Cumulative explained variance", xlabel="# of PCs")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.IndexLocator(base=5, offset=-1))
stashfig("screeplot")

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

sns.lineplot(
    data=scree_df,
    x="n_nonzero",
    y="explained_variance",
    hue="gamma",
    ax=ax,
    marker="o",
    palette=palette,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Gamma")
# ax.legend().set_title("Gamma")
ax.set(ylabel="Cumulative explained variance", xlabel="# nonzero elements")
plt.xscale("log")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
# ax.xaxis.set_major_locator(plt.IndexLocator(base=5, offset=-1))
stashfig("screeplot-by-params")

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
rows = []
for method, model in zip(["PCA", "SCA"], [pca, sca]):
    for i in range(1, n_components + 1):
        components = model.components_[:i]
        n_nonzero = np.count_nonzero(components)
        rows.append({"n_components": i, "n_nonzero": n_nonzero, "method": method})
nonzero_df = pd.DataFrame(rows)
sns.lineplot(
    data=nonzero_df, x="n_components", y="n_nonzero", hue="method", ax=ax, marker="o",
)
ax.set(ylabel="# nonzero parameters", xlabel="# of PCs")
ax.legend().set_title("Method")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.IndexLocator(base=4, offset=1))
stashfig("paramplot")

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
scree_df["n_nonzero"] = nonzero_df["n_nonzero"]
sns.lineplot(
    data=scree_df, x="n_nonzero", y="explained_variance", hue="method", marker="o"
)
ax.set(xlabel="# nonzero parameters", ylabel="Cumulative explained variance")
ax.legend().set_title("Method")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
stashfig("param-vs-variance-plot")
