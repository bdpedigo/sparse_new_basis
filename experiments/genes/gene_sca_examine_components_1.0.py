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
from sparse_matrix_analysis import SparseComponentAnalysis
from sparse_matrix_analysis.utils import calculate_explained_variance_ratio
from sparse_new_basis.data import load_scRNAseq
from sparse_new_basis.plot import savefig, set_theme

set_theme()


fig_dir = Path("sparse_new_basis/results/gene_sca_examine_components_1.0")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
output_dir = Path("sparse_new_basis/experiments/genes/outputs")

var_thresh = 0.01
train_size = 2 ** 14
n_components = 20
max_iter = 20
with_mean = True
with_std = True
seed = 8888

global_params = (
    f"var_thresh={var_thresh}-train_size={train_size}-n_components={n_components}"
    f"-max_iter={max_iter}-with_std={with_std}-seed={seed}"
)
output_dir = output_dir / global_params


if not os.path.isdir(output_dir):
    print(f"{output_dir} is not a directory... creating.")
    os.mkdir(output_dir)
    os.mkdir(output_dir / "data")
    os.mkdir(output_dir / "models")

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

with open(output_dir / Path("data") / "sequencing_df.pkl", "wb") as f:
    pickle.dump(sequencing_df, f)

with open(output_dir / Path("data") / "index_train.pkl", "wb") as f:
    pickle.dump(index_train, f)

with open(output_dir / Path("data") / "index_test.pkl", "wb") as f:
    pickle.dump(index_test, f)


#%% center and scale training data
currtime = time.time()
scaler = StandardScaler(with_mean=with_mean, with_std=with_std, copy=False)
X_train = scaler.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to scale and center data.")

with open(output_dir / Path("models") / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

#%%
np.random.seed(seed)
currtime = time.time()
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train)
print(f"{time.time() - currtime:.3f} elapsed to fit PCA model.")

#%%
np.random.seed(seed)
gammas = [
    n_components,
    100,
    250,
    500,
    int(np.sqrt(X_train.shape[1]) * n_components),
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
        n_components=n_components, max_iter=_max_iter, gamma=gamma
    )
    X_sca = sca.fit_transform(X_train)
    print(f"{time.time() - currtime:.3f} elapsed.")
    models_by_gamma[gamma] = sca
    Xs_by_gamma[gamma] = X_sca
    plt.figure()
    plt.plot(sca._Z_diff_norms_)
    plt.plot(sca._Y_diff_norms_)

    model_name = f"sca_gamma={gamma}"
    with open(output_dir / Path("models") / f"{model_name}.pkl", "wb") as f:
        pickle.dump(sca, f)

    print()

#%%

gamma = 20.0
with open(output_dir / Path("models") / f"sca_gamma={gamma}.pkl", "rb") as f:
    sca = pickle.load(f)

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

stashfig("pairplot-pca-celegans-genes")

#%%

X_sca = Xs_by_gamma[20]
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

stashfig("pairplot-sca-celegans-genes")

#%% train vs test PVE
# TODO this one not really done, not sure if worth showing
# X_test = scaler.transform(X_test)

# X_test_pca = pca.transform(X_test)
# explained_variance_pca = calculate_explained_variance_ratio(X_test, pca.components_.T)

# X_test_sca = sca.transform(X_test)
# explained_variance_sca = calculate_explained_variance_ratio(X_test, sca.components_.T)

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# plt.plot(explained_variance_pca)
# plt.plot(explained_variance_sca)

#%%
gamma = 100
sca = models_by_gamma[gamma]
X_transformed = Xs_by_gamma[gamma]


def make_neuron_df(X_transformed):
    columns = [f"component_score_{i}" for i in range(n_components)]
    neuron_df = pd.DataFrame(index=index_train, data=X_transformed, columns=columns,)
    # neuron_df["neuron_type"] = scrna_meta.loc[index_train, "Neuron_type"]
    neuron_df = neuron_df.reset_index(level="Neuron_type")
    neuron_df.rename(columns={"Neuron_type": "neuron_type"}, inplace=True)
    for c in columns:
        neuron_df["abs_" + c] = np.abs(neuron_df[c])
    return neuron_df


def make_genes_annotations(component):
    nonzero_inds = np.nonzero(component)[0]
    magnitude_sort_inds = np.argsort(np.abs(component[nonzero_inds]))[::-1]
    nonzero_inds = nonzero_inds[magnitude_sort_inds]
    select_gene_names = gene_index[nonzero_inds].copy()
    select_genes = pd.DataFrame(select_gene_names)
    select_gene_names = select_gene_names.values

    # select_genes = gene_df.loc[select_gene_index].copy()
    select_genes["component_val"] = component[nonzero_inds]
    select_genes["component_ind"] = nonzero_inds
    select_genes = select_genes.set_index("genes")
    # select_gene_names = select_genes["gene_symbol"]
    select_annotation_genes = annotation_df[
        annotation_df["gene"].isin(select_gene_names)
    ]

    # select_genes = select_genes.reset_index().set_index("gene_symbol")
    select_genes["cell_annotations"] = ""

    for _, row in select_annotation_genes.iterrows():
        select_genes.loc[row["gene"], "cell_annotations"] += (
            str(row["neuron_class"]) + ","
        )

    return select_genes


neuron_df = make_neuron_df(X_transformed)


#%%

neuron_df = make_neuron_df(X_transformed)

for i in range(n_components):
    component = sca.components_[i].copy()
    sign = np.sign(np.max(component[np.nonzero(component)]))
    component *= sign  # flip to positive at least for plotting
    select_genes = make_genes_annotations(component)
    # also flip the scores for plotting
    # select_genes["component_val"] = select_genes["component_val"] * sign
    neuron_df[f"component_score_{i}"] *= sign
    median_mags = neuron_df.groupby("neuron_type")[f"abs_component_score_{i}"].agg(
        np.median
    )
    median_mags = median_mags.sort_values(ascending=False)
    neuron_types = median_mags.index.values

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(6, 8),
        gridspec_kw=dict(height_ratios=[0.4, 0.2, 0.4], hspace=0.06),
    )

    y_max = neuron_df[f"component_score_{i}"].max()
    y_min = neuron_df[f"component_score_{i}"].min()
    y_range = y_max - y_min
    y_max += 0.05 * y_range
    y_min -= 0.05 * y_range

    n_per_row = 20

    row_neuron_types = neuron_types[:n_per_row]
    ax = axs[0]
    sns.stripplot(
        data=neuron_df[neuron_df["neuron_type"].isin(row_neuron_types)],
        x="neuron_type",
        y=f"component_score_{i}",
        hue="neuron_type",
        hue_order=row_neuron_types,  # ensures sorting stays the same
        order=row_neuron_types,  # ensures sorting stays the same
        palette=neuron_type_palette,
        jitter=0.35,
        ax=ax,
        s=3,
        alpha=0.7,
    )
    ax.get_legend().remove()
    ax.set(
        xlim=(-1, n_per_row),
        ylim=(y_min, y_max),
        xlabel=f"Top {n_per_row} cell types",
        ylabel="Score",
        yticks=[0],
        yticklabels=[0],
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.tick_params(length=0, labelsize="xx-small")
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="center",
    )
    for tick in ax.get_xticklabels():
        text = tick.get_text()
        tick.set_color(neuron_type_palette[text])

    ax = axs[2]
    plot_select_genes = select_genes.reset_index()
    plot_select_genes = plot_select_genes.iloc[:n_per_row]
    plot_select_genes["x"] = range(len(plot_select_genes))
    sns.scatterplot(
        data=plot_select_genes, x="x", y="component_val", color="black", s=30
    )
    ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(n_per_row)))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(plot_select_genes["genes"].values))
    ax.tick_params(length=0, labelsize="xx-small")
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="center",
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.yaxis.set_major_locator(plt.FixedLocator([0]))
    ax.yaxis.set_major_formatter(plt.FixedFormatter([0]))
    ax.set(
        xlim=(-1, n_per_row), xlabel=f"Top {n_per_row} genes", ylabel="Component",
    )

    annot_ax = axs[1]
    annot_ax.set_zorder(-100)
    sns.utils.despine(ax=annot_ax, left=True, bottom=True)

    annot_ax.set(xlim=(-1, n_per_row), ylim=(0, 1.5), xticks=[], yticks=[], ylabel="")

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for x, row in plot_select_genes.iterrows():
        if row["cell_annotations"] != "":
            cell_types = np.unique(row["cell_annotations"].split(",")[:-1])
            cell_types = [
                cell_type
                for cell_type in cell_types
                if cell_type in neuron_type_palette
            ]
            y_last = y_min
            for c, cell_type in enumerate(cell_types):
                if cell_type in neuron_type_palette:
                    y_top = y_last + y_range / len(cell_types)
                    ax.fill_between(
                        (x - 0.5, x + 0.5),
                        y_last,
                        y_top,
                        color=neuron_type_palette[cell_type],
                        alpha=1,
                        zorder=-1,
                        facecolor="white",
                    )
                    y_last = y_top

                    cell_loc = np.where(row_neuron_types == cell_type)[0]
                    if len(cell_loc) > 0:
                        annot_ax.plot(
                            [x, cell_loc[0]],
                            [0.02, 1],
                            color=neuron_type_palette[cell_type],
                        )

    fig.suptitle(f"Component {i + 1}", y=0.93)

    stashfig(f"component_{i+1}_relationplot-gamma={gamma}", format="png")

