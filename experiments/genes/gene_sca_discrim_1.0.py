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
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from graspologic.plot import pairplot
from sparse_decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import calculate_explained_variance_ratio
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


max_iter = 15
tol = 1e-4
n_components_range = [30]  # 60, 120]
params = []
S_train_by_params = {}
S_test_by_params = {}
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
        S_test_by_params[curr_params] = S_test

        # save metrics
        metrics = compute_metrics(sca)
        metrics["gamma"] = gamma
        metrics["n_components"] = n_components
        metric_rows.append(metrics)

        print("\n\n\n")

#%%
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
            result = {
                "tstat": tstat,
                "n_components": curr_params[0],
                "gamma": curr_params[1],
                "discrim_resample": i,
                "mode": mode,
            }
            discrim_result_rows.append(result)

        print()
discrim_results = pd.DataFrame(discrim_result_rows)
discrim_results

#%%
discrim_results["params"] = list(
    zip(discrim_results["n_components"], discrim_results["gamma"])
)
discrim_results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.stripplot(data=discrim_results, x="params", y="tstat", hue="mode", ax=ax)
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

plot_results = discrim_results[discrim_results["mode"] == "test"]
plot_results = plot_results.groupby("params").mean()

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
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set(ylabel="Discriminability", xlabel="# of genes used")

#%%


def calc_n_tests(positivity, n_per_batch):
    n_samples = 1000
    n_batches = n_samples / n_per_batch
    p_all_negative = (1 - positivity) ** n_per_batch
    p_retest = 1 - p_all_negative
    n_tests_original = n_batches
    n_tests_repeat = p_retest * n_batches * n_per_batch
    n_tests = n_tests_original + n_tests_repeat
    return n_tests / n_samples


rows = []
for positivity in np.linspace(0.01, 0.15, 10):
    for n_per_batch in np.arange(2, 20, 1):
        n_tests = calc_n_tests(positivity, n_per_batch)
        rows.append(
            {"positivity": positivity, "n_per_batch": n_per_batch, "n_tests": n_tests}
        )

results = pd.DataFrame(rows)
argmins = results.groupby("positivity")["n_tests"].idxmin()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="n_per_batch", y="n_tests", hue="positivity", ax=ax)
sns.scatterplot(
    data=results.loc[argmins],
    x="n_per_batch",
    y="n_tests",
    hue="positivity",
    ax=ax,
    linewidth=0,
    marker="o",
    s=50,
)
handles, labels = ax.get_legend_handles_labels()
n_show = len(handles) // 2
handles = handles[:n_show]
labels = labels[:n_show]

dummy_marker = Line2D([0], [0], color="black", lw=0, marker="o")
handles.append(dummy_marker)
labels.append("Minimum")

ax.get_legend().remove()
ax.legend(
    handles=handles,
    labels=labels,
    bbox_to_anchor=(1, 1),
    loc="upper left",
    title=r"$P_{infected}$",
)
ax.axhline(1, linestyle="--", color="dimgrey")
ax.text(1.5, 1.01, "No batch testing", ha="left", va="bottom", color="dimgrey")
ax.set(xlabel="Batch size", ylabel="# tests required per sample")
ax.xaxis.set_major_locator(plt.IndexLocator(base=4, offset=2))
stashfig("batch-testing")
