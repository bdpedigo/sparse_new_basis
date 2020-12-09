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
from umap import UMAP

from graspologic.plot import pairplot
from sparse_decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import calculate_explained_variance_ratio
from sparse_new_basis.data import load_scRNAseq
from sparse_new_basis.plot import savefig, set_theme

set_theme()


fig_dir = Path("sparse_new_basis/results/gene_sca_umap_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
output_dir = Path("sparse_new_basis/experiments/genes/outputs")

var_thresh = 0.005
train_size = 2 ** 14
n_components = 125
max_iter = 20
with_mean = True
with_std = True
seed = 8888

global_params = (
    f"var_thresh={var_thresh}-train_size={train_size}-n_components={n_components}"
    f"-max_iter={max_iter}-with_std={with_std}-seed={seed}"
)
output_dir = output_dir / global_params


#%%
sequencing_df, annotation_df = load_scRNAseq(fillna=True)

#%% throw out some genes with low variance
X = sequencing_df.values.copy()
var_thresh = VarianceThreshold(threshold=var_thresh)
X = var_thresh.fit_transform(X)
gene_index = sequencing_df.columns
original_n_genes = len(gene_index)
gene_index = gene_index[var_thresh.get_support()]
sequencing_df = sequencing_df[gene_index]
new_n_genes = len(gene_index)
print(
    f"Number of genes removed: {original_n_genes - new_n_genes} "
    f"out of {original_n_genes}"
)

#%%
np.random.seed(seed)

neuron_index = sequencing_df.index
y = sequencing_df.index.get_level_values(level="Neuron_type").values

# stratify=y will try to set the distribution of class labels the same for train/test
X_train, X_test, index_train, index_test = train_test_split(
    X, neuron_index, stratify=y, train_size=train_size
)

#%% center and scale training data
currtime = time.time()
scaler = StandardScaler(with_mean=with_mean, with_std=with_std, copy=False)
X_train = scaler.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to scale and center data.")

#%%
np.random.seed(seed)
currtime = time.time()
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to fit PCA model.")

#%%
np.random.seed(seed)
gammas = [
    4 * n_components,
    # 100,
    # 250,
    # 500,
    # int(np.sqrt(X_train.shape[1]) * n_components),
    np.inf,
]
gammas = [float(g) for g in gammas]
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
        n_components=n_components, max_iter=_max_iter, gamma=gamma, verbose=10
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

#%% palette

palette = dict(zip(gammas, sns.color_palette("deep", 10)))
blue_shades = sns.color_palette("Blues", n_colors=len(gammas))[1:]
palette = dict(zip(gammas[:-1], blue_shades))
red_shades = sns.color_palette("Reds", n_colors=len(gammas))[1:]
palette[np.inf] = red_shades[-1]

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

currtime = time.time()
umap_pca = UMAP(min_dist=0.3, n_neighbors=75, metric="cosine")
X_umap_pca = umap_pca.fit_transform(Xs_by_gamma[np.inf])
print(f"{time.time() - currtime:.3f} elapsed for UMAP.")

currtime = time.time()
umap_sca = UMAP(min_dist=0.3, n_neighbors=75, metric="cosine")
X_umap_sca = umap_sca.fit_transform(Xs_by_gamma[gammas[0]])
print(f"{time.time() - currtime:.3f} elapsed for UMAP.")


#%%
neuron_types = index_train.get_level_values("Neuron_type").values
neuron_type_palette = dict(zip(np.unique(neuron_types), cc.glasbey_light))


sca = models_by_gamma[gammas[-1]]
components = sca.components_
prop_genes_used = np.count_nonzero(components.max(axis=0)) / components.shape[1]
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
ax = axs[0]
ax.axis("off")
sns.scatterplot(
    x=X_umap_pca[:, 0],
    y=X_umap_pca[:, 1],
    hue=neuron_types,
    palette=neuron_type_palette,
    alpha=0.2,
    s=10,
    linewidth=0,
    ax=ax,
)
ax.get_legend().remove()
ax.set(
    title=r"UMAP $\circ$ PCA - " + f"Proportion of genes used: {prop_genes_used:.2f}"
)

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sca = models_by_gamma[gammas[0]]
components = sca.components_
prop_genes_used = np.count_nonzero(components.max(axis=0)) / components.shape[1]

ax = axs[1]
ax.axis("off")
sns.scatterplot(
    x=X_umap_sca[:, 0],
    y=X_umap_sca[:, 1],
    hue=neuron_types,
    palette=neuron_type_palette,
    alpha=0.2,
    s=10,
    ax=ax,
    linewidth=0,
)
ax.get_legend().remove()
ax.set(
    title=r"UMAP $\circ$ SCA - " + f"Proportion of genes used: {prop_genes_used:.2f}"
)

stashfig(f"umap-n_components={n_components}")

#%%
from hyppo.discrim import DiscrimTwoSample, DiscrimOneSample

discrim = DiscrimOneSample(is_dist=False)
y = index_train.get_level_values("Neuron_type").values.copy()
uni_y = np.unique(y)
name_map = dict(zip(uni_y, range(len(uni_y))))
y = np.vectorize(name_map.get)(y)
currtime = time.time()
output = discrim.test(X_pca, y, reps=0)

print(f"{time.time() - currtime:.3f} elapsed.")

output

#%%

from sklearn.metrics import pairwise_distances

currtime = time.time()
dist_X_pca = pairwise_distances(X_pca[:2000], metric="cosine")
print(f"{time.time() - currtime:.3f} elapsed.")

currtime = time.time()
discrim = DiscrimOneSample(is_dist=True)
tstat, _ = discrim.test(dist_X_pca, y[:2000], reps=0)
print(f"{time.time() - currtime:.3f} elapsed.")
tstat

#%%
X_sca = Xs_by_gamma[gammas[0]]
currtime = time.time()
dist_X_sca = pairwise_distances(X_sca[:3000], metric="cosine")
print(f"{time.time() - currtime:.3f} elapsed.")

currtime = time.time()
discrim = DiscrimOneSample(is_dist=True)
tstat, _ = discrim.test(dist_X_sca, y[:3000], reps=0)
print(f"{time.time() - currtime:.3f} elapsed.")
tstat

#%%


def compute_metrics(model):
    final_pve = model.explained_variance_ratio_[-1]
    n_nonzero = np.count_nonzero(model.components_)
    p_nonzero = n_nonzero / model.components_.size
    n_nonzero_cols = np.count_nonzero(model.components_.max(axis=0))
    p_nonzero_cols = n_nonzero_cols / model.components_.shape[1]
    output = {
        "explained_variance": final_pve,
        "n_nonzero": n_nonzero,
        "p_nonzero": p_nonzero,
        "n_nonzero_cols": n_nonzero_cols,
        "p_nonzero_cols": p_nonzero_cols,
    }
    return output


from sparse_decomposition import SparseComponentAnalysis

max_iter = 15
tol = 1e-4
n_components_range = [30]  # 60, 120]
params = []
S_train_by_params = {}
models_by_params = {}
metric_rows = []
for n_components in n_components_range:
    gammas = [
        2 * n_components,
        0.25 * np.sqrt(n_components * X_train.shape[1]),
        0.5 * np.sqrt(n_components * X_train.shape[1]),
        np.sqrt(n_components * X_train.shape[1]),
        0.5 * n_components * np.sqrt(X_train.shape[1]),
    ]
    gammas = [float(int(g)) for g in gammas]
    gammas.append(np.inf)
    for gamma in gammas:
        print(f"n_components = {n_components}, gamma = {gamma}")
        print()
        curr_params = (n_components, gamma)
        params.append(curr_params)

        # fit model
        currtime = time.time()
        sca = SparseComponentAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            gamma=gamma,
            verbose=10,
            tol=tol,
        )
        S_train = sca.fit_transform(X_train)
        print(f"{time.time() - currtime:.3f} elapsed to train SCA model.")

        S_test = sca.transform(X_test)

        # save model fit
        models_by_params[curr_params] = sca
        S_train_by_params[curr_params] = S_train

        # save metrics
        metrics = compute_metrics(sca)
        metrics["gamma"] = gamma
        metrics["n_components"] = n_components
        metric_rows.append(metrics)

        print("\n\n\n")

#%%
n_subsamples = 5
n_per_subsample = 4096
metric = "cosine"

discrim_result_rows = []
for curr_params in params:
    X_transformed = S_train_by_params[curr_params]
    for i in range(n_subsamples):
        subsample_inds = np.random.choice(
            len(X_transformed), size=n_per_subsample, replace=False
        )
        dist_X_transformed = pairwise_distances(
            X_transformed[subsample_inds], metric=metric
        )
        currtime = time.time()
        discrim = DiscrimOneSample(is_dist=True)
        tstat, _ = discrim.test(dist_X_transformed, y[subsample_inds], reps=0)
        print(f"{time.time() - currtime:.3f} elapsed for discriminability.")

        result = {
            "tstat": tstat,
            "n_components": curr_params[0],
            "gamma": curr_params[1],
            "discrim_resample": i,
        }
        discrim_result_rows.append(result)

discrim_results = pd.DataFrame(discrim_result_rows)
discrim_results
#%%
# palette = dict(zip(gammas, sns.color_palette("deep", 10)))
# blue_shades = sns.color_palette("Blues", n_colors=len(gammas))[1:]
# palette = dict(zip(gammas[:-1], blue_shades))
# red_shades = sns.color_palette("Reds", n_colors=len(gammas))[1:]
# palette[np.inf] = red_shades[-1]

#%%
discrim_results["params"] = list(
    zip(discrim_results["n_components"], discrim_results["gamma"])
)
discrim_results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.stripplot(data=discrim_results, x="params", y="tstat", hue="n_components", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")

stashfig("discrim-by-params")

#%%
metrics = pd.DataFrame(metric_rows)
metrics["params"] = list(zip(metrics["n_components"], metrics["gamma"]))

discrim_results["p_nonzero_cols"] = discrim_results["params"].map(
    metrics.set_index("params")["p_nonzero_cols"]
)

discrim_results

#%%

plot_results = discrim_results[discrim_results["n_components"] == 30]


gammas = np.unique(plot_results["gamma"])
palette = dict(zip(gammas, sns.color_palette("deep", 10)))
blue_shades = sns.color_palette("Blues", n_colors=len(gammas))[1:]
palette = dict(zip(gammas[:-1], blue_shades))
red_shades = sns.color_palette("Reds", n_colors=len(gammas))[1:]
palette[np.inf] = red_shades[-1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=plot_results,
    x="p_nonzero_cols",
    y="tstat",
    hue="gamma",
    palette=palette,
    ax=ax,
)

#%%
dist_X_transformed = pairwise_distances(X_transformed, metric=metric)
currtime = time.time()
discrim = DiscrimOneSample(is_dist=True)
tstat, _ = discrim.test(dist_X_transformed, y, reps=0)
print(f"{time.time() - currtime:.3f} elapsed for discriminability.")
