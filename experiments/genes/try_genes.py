#%%
import time
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from graspologic.plot import pairplot
from sparse_decomposition import SparseComponentAnalysis

data_dir = Path("sparse_new_basis/data/BP_Barabasi_Share/ScRNAData")

# gene expression data
sequencing_loc = data_dir / "Celegans_ScRNA_OnlyLabeledNeurons.csv"
sequencing_df = pd.read_csv(sequencing_loc, skiprows=[1])
currtime = time.time()
sequencing_df = sequencing_df.pivot(index="genes", columns="neurons", values="Count")
sequencing_df = sequencing_df.T.fillna(0)
print(f"{time.time() - currtime} elapsed")

# TODO drop the background RNA from table S2 in the paper

# info about the genes themselves
gene_loc = data_dir / "GSE136049_gene_annotations.csv"
gene_df = pd.read_csv(gene_loc)
gene_df["genes"] = range(1, len(gene_df) + 1)
gene_df = gene_df.set_index("genes")
gene_df = gene_df.loc[sequencing_df.columns]

# metadata for each neuron in the gene expression data
class_map_loc = data_dir / "Labels2_CElegansScRNA_onlyLabeledNeurons.csv"
scrna_meta = pd.read_csv(class_map_loc)
scrna_meta = scrna_meta.set_index("OldIndices")

# single neuron connectome data
connectome_loc = data_dir / "NeuralWeightedConn.csv"
adj_df = pd.read_csv(connectome_loc, index_col=None, header=None)
adj = adj_df.values

# metadata for neurons in the connectome
label_loc = data_dir / "NeuralWeightedConn_Labels.csv"
connectome_meta = pd.read_csv(label_loc)
connectome_meta["cell_name"] = connectome_meta["Var1"].map(lambda x: x.strip("'"))
connectome_meta["broad_type"] = connectome_meta["Var2"].map(lambda x: x.strip("'"))
connectome_meta["cell_type"] = connectome_meta["Var3"].map(lambda x: x.strip("'"))
connectome_meta["neurotransmitter"] = connectome_meta["Var4"].map(
    lambda x: x.strip("'")
)
connectome_meta["cell_type_index"] = connectome_meta["Var5"]
broad_labels = connectome_meta["broad_type"].values

#%%
X = sequencing_df.values.copy()

var_thresh = VarianceThreshold(threshold=0.01)
X = var_thresh.fit_transform(X)
gene_index = gene_df.index
gene_index = gene_index[var_thresh.get_support()]
X = StandardScaler(with_mean=True, with_std=True, copy=False).fit_transform(X)

#%%

# n_per_class = 10
# neuron_sample = scrna_meta.groupby("CellTypeIndex").sample(n=n_per_class).index
neuron_index = sequencing_df.index
y = scrna_meta["Neuron_type"].values

X_train, X_test, index_train, index_test = train_test_split(
    X, neuron_index, stratify=y, train_size=2 ** 14
)

#%%
n_components = 20
max_iter = 10
gamma = 20
sca = SparseComponentAnalysis(n_components=n_components, max_iter=max_iter, gamma=gamma)

# subsample = 2 ** 12
# if subsample:
#     subsample_inds = np.random.choice(len(X), replace=False, size=subsample)
#     X_sub = X[subsample_inds, :]
# else:
#     X_sub = X

currtime = time.time()
X_transformed = sca.fit_transform(X_train)
print(f"{time.time() - currtime} elapsed")


#%%

for i in range(10):
    print(np.count_nonzero(sca.components_[i]) / len(sca.components_[i]))
    sort_inds = np.argsort(np.abs(sca.components_[i]))[::-1]
    plt.figure()
    plt.plot(np.sort(sca.components_[i]), marker="o", linewidth=0, markersize=1)
    # print(gene_df.loc[select_genes])

#%%


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


fig_dir = Path("sparse_new_basis/results/try_genes")


def stashfig(
    name,
    *args,
    transparent=False,
    facecolor="w",
    dpi=300,
    pad_inches=0.25,
    bbox_inches="tight",
    **kwargs,
):
    plt.savefig(
        fig_dir / name,
        transparent=transparent,
        facecolor=facecolor,
        dpi=dpi,
        pad_inches=pad_inches,
        bbox_inches=bbox_inches,
    )


neuron_type_palette = dict(
    zip(np.unique(scrna_meta.loc[index_train, "Neuron_type"]), cc.glasbey_light)
)
pairplot(
    X_transformed[:, :],
    labels=scrna_meta.loc[index_train, "Neuron_type"].values,
    palette=neuron_type_palette,
    diag_kind=None,
)
stashfig("pairplot-sca-celegans-genes")

#%%
for i, component in enumerate(sca.components_):
    print(i)
    component_scores = X_transformed[:, i]
    sort_inds = np.argsort(np.abs(component_scores))[::-1]
    print(scrna_meta.loc[index_train[sort_inds[:100]]]["Neuron_type"])

i = 20  # used to be 10
sca.components_[i]
sort_inds = np.argsort(np.abs(sca.components_[i]))[::-1]
select_genes = gene_index[sort_inds][:20]
gene_df.loc[select_genes]

#%%

columns = [f"component_score_{i}" for i in range(n_components)]
neuron_df = pd.DataFrame(index=index_train, data=X_transformed, columns=columns,)
neuron_df["neuron_type"] = scrna_meta.loc[index_train, "Neuron_type"]
for c in columns:
    neuron_df["abs_" + c] = np.abs(neuron_df[c])

#%%
annotation_genes = pd.read_csv(data_dir / "annotation_genes.csv")
nt_annotation_genes = annotation_genes.melt(
    id_vars=["neuron_class", "neuron_type"],
    value_vars=[f"nt_gene_{i}" for i in range(3)],
    value_name="gene",
).dropna(axis=0)
nt_annotation_genes = nt_annotation_genes.drop("variable", axis=1)
nt_annotation_genes["gene_type"] = "neurotransmitter"

other_annotation_genes = annotation_genes.melt(
    id_vars=["neuron_class", "neuron_type"],
    value_vars=[f"gene_{i}" for i in range(12)],
    value_name="gene",
).dropna(axis=0)
other_annotation_genes = other_annotation_genes.drop("variable", axis=1)
other_annotation_genes["gene_type"] = "other"

annotation_genes = pd.concat((nt_annotation_genes, other_annotation_genes), axis=0)
annotation_genes

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
