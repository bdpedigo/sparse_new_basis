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
from hyppo.discrim import DiscrimOneSample, DiscrimTwoSample
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA, SparsePCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from graspologic.plot import pairplot
from sparse_decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import (
    calculate_explained_variance_ratio,
    l1_norm,
    proportion_variance_explained,
)
from sparse_new_basis.data import load_scRNAseq
from sparse_new_basis.plot import savefig, set_theme

set_theme()


fig_dir = Path("sparse_new_basis/results/gene_sca_discrim_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
var_thresh = 0.005
train_size = 2 ** 14
max_iter = 20
with_mean = True
with_std = True
seed = 8888


#%%
sequencing_df, annotation_df = load_scRNAseq(fillna=True)

#%% throw out some genes with low variance
X = sequencing_df.values
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
    X, neuron_index, stratify=y, train_size=train_size, shuffle=True
)

#%% center and scale training data
currtime = time.time()
scaler = StandardScaler(with_mean=with_mean, with_std=with_std, copy=False)
X_train = scaler.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to scale and center data.")
X_test = scaler.transform(X_test)


#%%


def compute_metrics(model, X_train, X_test):
    train_pve = proportion_variance_explained(X_train, model.components_.T)
    test_pve = proportion_variance_explained(X_test, model.components_.T)
    n_nonzero = np.count_nonzero(model.components_)
    p_nonzero = n_nonzero / model.components_.size
    n_nonzero_cols = np.count_nonzero(model.components_.max(axis=0))
    p_nonzero_cols = n_nonzero_cols / model.components_.shape[1]
    component_l1 = l1_norm(model.components_)
    output = {
        "train_pve": train_pve,
        "test_pve": test_pve,
        "n_nonzero": n_nonzero,
        "p_nonzero": p_nonzero,
        "n_nonzero_cols": n_nonzero_cols,
        "p_nonzero_cols": p_nonzero_cols,
        "component_l1": component_l1,
    }
    return output


params = []
S_train_by_params = {}
S_test_by_params = {}
models_by_params = {}
metric_rows = []

#%%
# Sparse Component Analysis and PCA
method = "SCA"
max_iter = 15
tol = 1e-4
n_components_range = [30]
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
        if gamma == np.inf:
            method = "PCA"
        else:
            method = "SCA"
        print(f"method = {method}, n_components = {n_components}, gamma = {gamma}")
        print()
        curr_params = (method, n_components, gamma)
        params.append(curr_params)

        # fit model
        currtime = time.time()
        model = SparseComponentAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            gamma=gamma,
            verbose=10,
            tol=tol,
        )
        S_train = model.fit_transform(X_train)
        train_time = time.time() - currtime
        print(f"{train_time:.3f} elapsed to train model.")

        S_test = model.transform(X_test)

        # save model fit
        models_by_params[curr_params] = model
        S_train_by_params[curr_params] = S_train
        S_test_by_params[curr_params] = S_test

        # save metrics
        metrics = compute_metrics(model, X_train, X_test)
        metrics["sparsity_param"] = gamma
        metrics["sparsity_level"] = gamma
        metrics["n_components"] = n_components
        metrics["train_time"] = train_time
        metrics["method"] = method
        metric_rows.append(metrics)
        print(f"Component L0 ratio: {metrics['p_nonzero']}")
        print(f"Component L0 columns: {metrics['p_nonzero_cols']}")

        print("\n\n\n")

#%%
# SparsePCA (Online Dictionary Learning)
method = "SparsePCA"
max_iter = 10
for n_components in n_components_range:
    alphas = [1, 5, 15, 30, 40]
    alphas = [float(int(a)) for a in alphas]
    for alpha in alphas:
        print(f"method = {method}, n_components = {n_components}, alpha = {alpha}")
        print()
        curr_params = (method, n_components, alpha)
        params.append(curr_params)

        # fit model
        currtime = time.time()
        model = SparsePCA(
            n_components=n_components,
            max_iter=max_iter,
            alpha=alpha,
            verbose=0,
            tol=tol,
            n_jobs=1,
        )
        S_train = model.fit_transform(X_train)
        train_time = time.time() - currtime
        print(f"{train_time:.3f} elapsed to train model.")

        S_test = model.transform(X_test)

        # save model fit
        models_by_params[curr_params] = model
        S_train_by_params[curr_params] = S_train
        S_test_by_params[curr_params] = S_test

        # save metrics
        metrics = compute_metrics(model, X_train, X_test)
        metrics["sparsity_param"] = alpha
        metrics["sparsity_level"] = -alpha
        metrics["n_components"] = n_components
        metrics["train_time"] = train_time
        metrics["method"] = method
        metric_rows.append(metrics)
        print(f"Component L0 ratio: {metrics['p_nonzero']}")
        print(f"Component L0 columns: {metrics['p_nonzero_cols']}")

        print("\n\n\n")

#%%
# Discriminability as a metric
n_subsamples = 10
n_per_subsample = 2 ** 12
# n_per_subsample = None
metric = "cosine"


def get_int_labels(index):
    y = index.get_level_values("Neuron_type").values
    uni_y = np.unique(y)
    name_map = dict(zip(uni_y, range(len(uni_y))))
    y = np.vectorize(name_map.get)(y)
    return y


discrim_result_rows = []
for curr_params in params:
    print(curr_params)
    print()
    model = models_by_params[curr_params]
    for mode in ["test"]:
        if mode == "train":
            S = S_train_by_params[curr_params]
            y = get_int_labels(index_train)
        elif mode == "test":
            S = S_test_by_params[curr_params]
            y = get_int_labels(index_test)

        for i in range(n_subsamples):
            if n_per_subsample is None:
                n_per_subsample = len(S)
            subsample_inds = np.random.choice(
                len(S), size=n_per_subsample, replace=False
            )

            # compute discriminability
            dist_S = pairwise_distances(S[subsample_inds], metric=metric)
            currtime = time.time()
            discrim = DiscrimOneSample(is_dist=True)
            tstat, _ = discrim.test(dist_S, y[subsample_inds], reps=0)
            print(f"{time.time() - currtime:.3f} elapsed for discriminability.")

            # save results
            metrics = {}
            metrics["method"] = curr_params[0]
            metrics["tstat"] = tstat
            metrics["n_components"] = curr_params[1]
            metrics["discrim_resample"] = i
            metrics["mode"] = mode
            metrics["sparsity_param"] = curr_params[2]
            discrim_result_rows.append(metrics)

        print()

discrim_results = pd.DataFrame(discrim_result_rows)
discrim_results

#%%
discrim_results["params"] = list(
    zip(
        discrim_results["method"],
        discrim_results["n_components"],
        discrim_results["sparsity_param"],
    )
)
discrim_results["pretty_params"] = ""
for i, row in discrim_results.iterrows():
    method = row["method"]
    if method == "PCA":
        discrim_results.loc[i, "pretty_params"] = "PCA"
    else:
        if method == "SCA":
            symbol = r"$\gamma$"
        elif method == "SparsePCA":
            symbol = r"$\alpha$"
        discrim_results.loc[i, "pretty_params"] = (
            row["method"] + ", " + symbol + "=" + f"{row['sparsity_param']:.0f}"
        )

#%%
discrim_results = discrim_results.sort_values(
    ["method", "n_components", "sparsity_param"]
)
discrim_results
#%%
# red_shades = sns.color_palette("Reds", n_colors=len(gammas)+1)[-2:-1]
# gammas = np.unique(discrim_results[discrim_results['method'] == 'SCA']['sparsity_param'])
# alphas = np.unique(discrim_results[discrim_results['method'] == 'SparsePCA']['sparsity_param'])
# push = 2
# blue_shades = sns.color_palette("Blues", n_colors=len(gammas)+push)[push:]
# green_shades = sns.color_palette("Greens", n_colors=len(alphas)+push)[push:][::-1]

# shades = red_shades + blue_shades + green_shades
# palette = dict(zip(discrim_results['params'], shades))
# palette

#%%
metrics = pd.DataFrame(metric_rows)
metrics["params"] = list(
    zip(metrics["method"], metrics["n_components"], metrics["sparsity_param"])
)
metrics = metrics.set_index("params")
metrics

#%%
plot_results = discrim_results[discrim_results["mode"] == "test"].copy()
plot_results = plot_results.groupby("params").mean()
plot_results = pd.concat(
    (plot_results, metrics.drop(["n_components", "sparsity_param", "method"], axis=1)),
    axis=1,
)
plot_results = (
    plot_results.reset_index()
    .rename({"level_0": "method"}, axis=1)
    .drop(["level_1", "level_2"], axis=1)
)
plot_results = plot_results.sort_values(["method", "n_components", "sparsity_level"])
#%%

plot_results["pretty_params"] = ""
for i, row in plot_results.iterrows():
    method = row["method"]
    if method == "PCA":
        plot_results.loc[i, "pretty_params"] = "PCA"
    else:
        if method == "SCA":
            symbol = r"$\gamma$"
        elif method == "SparsePCA":
            symbol = r"$\alpha$"
        plot_results.loc[i, "pretty_params"] = (
            row["method"] + ", " + symbol + "=" + f"{row['sparsity_param']:.0f}"
        )
plot_results


#%%

red_shades = sns.color_palette("Reds", n_colors=len(gammas) + 1)[-2:-1]
gammas = np.unique(plot_results[plot_results["method"] == "SCA"]["sparsity_param"])
alphas = np.unique(
    plot_results[plot_results["method"] == "SparsePCA"]["sparsity_param"]
)
push = 2
blue_shades = sns.color_palette("Blues", n_colors=len(gammas) + push)[push:]
green_shades = sns.color_palette("Greens", n_colors=len(alphas) + push)[push:]

shades = red_shades + blue_shades + green_shades
palette = dict(zip(plot_results["pretty_params"], shades))
palette

line_palette = dict(
    zip(
        ["PCA", "SCA", "SparsePCA"], [red_shades[-1], blue_shades[-1], green_shades[-1]]
    )
)
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=plot_results,
    x="p_nonzero_cols",
    y="tstat",
    hue="pretty_params",
    palette=palette,
    ax=ax,
    s=100,
)
handles, labels = ax.get_legend_handles_labels()
handles = handles[:11]
labels = labels[:11]
sns.lineplot(
    data=plot_results,
    x="p_nonzero_cols",
    y="tstat",
    hue="method",
    zorder=-1,
    palette=line_palette,
)
ax.get_legend().remove()
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc="upper left")
ax.set(ylabel="Discriminability", xlabel="# of genes used")
stashfig("discriminability-vs-n_genes-new")

#%%

# plot_results["p_nonzero_cols_jitter"] = plot_results[
#     "p_nonzero_cols"
# ] + np.random.uniform(-0.02, 0.02, size=len(plot_results))
# mean_results = plot_results.groupby("params").mean()

# gammas = np.unique(plot_results["gamma"])

# palette = dict(zip(gammas, sns.color_palette("deep", 10)))
# blue_shades = sns.color_palette("Blues", n_colors=len(gammas))[1:]
# palette = dict(zip(gammas[:-1], blue_shades))
# red_shades = sns.color_palette("Reds", n_colors=len(gammas))[1:]
# palette[np.inf] = red_shades[-1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=plot_results,
    x="p_nonzero_cols",
    y="tstat",
    hue="params",
    palette=palette,
    ax=ax,
    s=10,
)
# sns.scatterplot(
#     data=mean_results,
#     x="p_nonzero_cols",
#     y="tstat",
#     hue="params",
#     palette=palette,
#     ax=ax,
#     marker="_",
#     s=200,
#     linewidth=4,
#     legend=False,
# )
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set(ylabel="Discriminability", xlabel="# of genes used")
stashfig("discriminability-vs-n_genes-new")

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=plot_results,
    x="p_nonzero",
    y="p_nonzero_cols",
    ax=ax,
    hue="params",
    palette=palette,
)
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=plot_results,
    x="p_nonzero",
    y="train_time",
    ax=ax,
    hue="params",
    palette=palette,
)

#%%
from sklearn.decomposition import PCA
from scipy.stats import ortho_group

n_components = 10
pca = PCA(n_components=n_components)
pca.fit_transform(X_train[: 2 ** 11])
Y = pca.components_.T

#%%
rows = []
for i in range(1000):
    R = ortho_group.rvs(n_components)
    loading_l1 = l1_norm(Y @ R)
    rows.append({"rotation": "random", "l1_norm": loading_l1})

from sparse_decomposition.decomposition.decomposition import _varimax

Y_varimax = _varimax(Y)
loading_l1 = l1_norm(Y_varimax)
rows.append({"rotation": "varimax", "l1_norm": loading_l1})
results = pd.DataFrame(rows)
#%%
import matplotlib.transforms as transforms

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(
    data=results[results["rotation"] == "random"],
    x="l1_norm",
    ax=ax,
    stat="density",
    kde=True,
    element="step",
)
ax.axvline(loading_l1, color="darkred", linestyle="--", linewidth=2)
# ax.axvline(l1_norm(Y), color='blue', linestyle='--', linewidth=2)
ax.annotate(
    "Varimax\nrotation",
    (loading_l1 + 2, 0.8),
    (20, 20),
    xycoords=transforms.blended_transform_factory(ax.transData, ax.transAxes),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)
ax.annotate(
    "Random\nrotation",
    (605, 0.6),
    (-100, 20),
    xycoords=transforms.blended_transform_factory(ax.transData, ax.transAxes),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)
ax.set(xlabel=r"$\|\|YR\|\|_{1}$ (element-wise norm)", ylabel="", yticks=[])
ax.spines["left"].set_visible(False)
stashfig("random-rotations")
