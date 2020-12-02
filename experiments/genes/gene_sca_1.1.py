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


fig_dir = Path("sparse_new_basis/results/gene_sca_1.1")


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
# TODO plot the distribution of frequencies by cell type

#%%
neuron_index = sequencing_df.index
y = sequencing_df.index.get_level_values(level="Neuron_type").values

# stratify=y will try to set the distribution of class labels the same for train/test
X_train, X_test, index_train, index_test = train_test_split(
    X, neuron_index, stratify=y, train_size=2 ** 14
)

#%%
n_components = 20
max_iter = 10
# gamma = 20
gamma = 100
sca_params = f"-n_components={n_components}-max_iter={max_iter}-gamma={gamma}"
pca_params = f"-n_components={n_components}"

#%% center and scale training data
currtime = time.time()
scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
X_train = scaler.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to scale and center data.")

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
neuron_types = index_train.get_level_values("Neuron_type").values
neuron_type_palette = dict(zip(np.unique(neuron_types), cc.glasbey_light))

n_show = 5


def make_plot_df(X, labels=None):
    columns = [f"Dimension {i+1}" for i in range(X.shape[1])]
    plot_df = pd.DataFrame(data=X, columns=columns)
    if labels is not None:
        plot_df["labels"] = labels
    return plot_df


pg = sns.PairGrid(
    data=make_plot_df(X_pca[:, :n_show], neuron_types),
    hue="labels",
    palette=neuron_type_palette,
    corner=True,
)
pg.map_lower(sns.scatterplot, alpha=0.7, linewidth=0, s=10)
pg.set(xticks=[], yticks=[])
pg.fig.suptitle("PCA")

axes = pg.axes
fig = pg.fig
gs = fig._gridspecs[0]
for i in range(len(axes)):
    axes[i, i].remove()
    axes[i, i] = None
    ax = fig.add_subplot(gs[i, i])
    axes[i, i] = ax
    ax.axis("off")
    p_nonzero = np.count_nonzero(X_pca[:, i]) / len(X_pca)
    text = f"{p_nonzero:.2f}"
    if i == 0:
        text = "Proportion\nnonzero:\n" + text
    ax.text(0.5, 0.5, text, ha="center", va="center")

stashfig("pairplot-pca-celegans-genes" + pca_params)

#%%



pg = sns.PairGrid(
    data=make_plot_df(X_sca[:, :n_show], neuron_types),
    hue="labels",
    palette=neuron_type_palette,
    corner=True,
)
# hide_indices = np.tril_indices_from(axes, 1)
# for i, j in zip(*hide_indices):
#     axes[i, j].remove()
#     axes[i, j] = None

pg.map_lower(sns.scatterplot, alpha=0.7, linewidth=0, s=10)
pg.set(xticks=[], yticks=[])
pg.fig.suptitle("SCA")

axes = pg.axes
fig = pg.fig
gs = fig._gridspecs[0]
for i in range(len(axes)):
    axes[i, i].remove()
    axes[i, i] = None
    ax = fig.add_subplot(gs[i, i])
    axes[i, i] = ax
    ax.axis("off")
    p_nonzero = np.count_nonzero(X_sca[:, i]) / len(X_sca)
    text = f"{p_nonzero:.2f}"
    if i == 0:
        text = "Proportion\nnonzero:\n" + text
    ax.text(0.5, 0.5, text, ha="center", va="center")

stashfig("pairplot-sca-celegans-genes" + sca_params)

#%% train vs test PVE
# TODO this one not really done, not sure if worth showing
X_test = scaler.transform(X_test)

X_test_pca = pca.transform(X_test)
explained_variance_pca = calculate_explained_variance_ratio(X_test, pca.components_.T)

X_test_sca = sca.transform(X_test)
explained_variance_sca = calculate_explained_variance_ratio(X_test, sca.components_.T)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(explained_variance_pca)
plt.plot(explained_variance_sca)

#%%

X_transformed = X_sca
columns = [f"component_score_{i}" for i in range(n_components)]
neuron_df = pd.DataFrame(index=index_train, data=X_transformed, columns=columns,)
neuron_df["neuron_type"] = scrna_meta.loc[index_train, "Neuron_type"]
for c in columns:
    neuron_df["abs_" + c] = np.abs(neuron_df[c])


#%%

i = 19
neuron_types = np.unique(neuron_df["neuron_type"])
n_neuron_types = len(neuron_types)
n_per_row = 20
n_rows = int(np.ceil(n_neuron_types / n_per_row))
fig, axs = plt.subplots(
    n_rows, 2, figsize=(10, 2.5 * n_rows), gridspec_kw=dict(width_ratios=[0.6, 0.3])
)
y_max = neuron_df[f"component_score_{i}"].max()
y_min = neuron_df[f"component_score_{i}"].min()
y_range = y_max - y_min
y_max += 0.05 * y_range
y_min -= 0.05 * y_range


component = sca.components_[i]
nonzero_inds = np.nonzero(component)[0]
magnitude_sort_inds = np.argsort(np.abs(component[nonzero_inds]))[::-1]
nonzero_inds = nonzero_inds[magnitude_sort_inds]
select_gene_index = gene_index[nonzero_inds]
select_genes = gene_df.loc[select_gene_index].copy()
select_genes["component_ind"] = np.arange(len(gene_index))[nonzero_inds]

select_gene_names = select_genes["gene_symbol"]
select_annotation_genes = annotation_genes[
    annotation_genes["gene"].isin(select_gene_names)
]

select_genes = select_genes.reset_index().set_index("gene_symbol")
select_genes["cell_annotations"] = ""

for _, row in select_annotation_genes.iterrows():
    select_genes.loc[row["gene"], "cell_annotations"] += str(row["neuron_class"]) + ","

median_mags = neuron_df.groupby("neuron_type")[f"abs_component_score_{i}"].agg(
    np.median
)
median_mags = median_mags.sort_values(ascending=False)
neuron_types = median_mags.index.values

for row in range(n_rows):
    row_neuron_types = neuron_types[n_per_row * row : n_per_row * (row + 1)]
    ax = axs[row, 0]
    # sns.violinplot(
    #     data=neuron_df[neuron_df["neuron_type"].isin(row_neuron_types)],
    #     x="neuron_type",
    #     y=f"component_score_{i}",
    #     hue="neuron_type",
    #     palette=neuron_type_palette,
    #     ax=ax,
    #     inner=None,
    # )
    sns.stripplot(
        data=neuron_df[neuron_df["neuron_type"].isin(row_neuron_types)],
        x="neuron_type",
        y=f"component_score_{i}",
        hue="neuron_type",
        hue_order=row_neuron_types,  # ensures sorting stays the same
        order=row_neuron_types,  # ensures sorting stays the same
        palette=neuron_type_palette,
        ax=ax,
        s=2,
    )
    ax.get_legend().remove()
    ax.set(xlim=(-1, n_per_row), ylim=(y_min, y_max), xlabel="", ylabel="", yticks=[])
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=45)
    for tick in ax.get_xticklabels():
        text = tick.get_text()
        tick.set_color(neuron_type_palette[text])

axs[2, 0].set_ylabel(f"Component {i + 1} score")
plt.tight_layout()

ax = axs[0, 1]

y = 0

last_text = None
for gene, row in select_genes.iterrows():
    sign = np.sign(component[row["component_ind"]])
    gene_text = "+" if sign > 0 else "-"
    gene_text += gene
    text = ax.text(0, 1 - y, gene_text, transform=ax.transAxes, verticalalignment="top")

    if row["cell_annotations"] != "":

        cell_types = row["cell_annotations"].split(",")
        x = 0
        for cell_type in cell_types[:-1]:
            if cell_type in neuron_type_palette:
                ax.text(
                    0.6 + x,
                    1 - y,
                    cell_type,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    color=neuron_type_palette[cell_type],
                )
                x += 0.2
    y += 0.2


[ax.axis("off") for ax in axs[:, 1]]

stashfig(
    f"scores-by-type-component={i+1}-n_components={n_components}-gamma={gamma}-max_iter={max_iter}"
)


#%%
gene_crosstab = pd.crosstab(
    index=annotation_genes["neuron_class"], columns=annotation_genes["gene"]
)
sns.clustermap(gene_crosstab)
#%%

for i, component in enumerate(sca.components_):
    sort_inds = np.argsort(np.abs(component))[::-1]
    select_genes = gene_index[sort_inds][:20]
    print(gene_df.loc[select_genes])
    plt.figure()
    plt.plot(np.sort(component))

#%%
