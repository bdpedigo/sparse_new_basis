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
from sparse_new_basis.plot import savefig, set_theme

#%%
fig_dir = Path("sparse_new_basis/results/try_genes")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


set_theme()

#%%
data_dir = Path("sparse_new_basis/data/BP_Barabasi_Share/ScRNAData")

#%%
# gene expression data
sequencing_loc = data_dir / "Celegans_ScRNA_OnlyLabeledNeurons.csv"
sequencing_df = pd.read_csv(sequencing_loc, skiprows=[1])
currtime = time.time()
sequencing_df = sequencing_df.pivot(index="genes", columns="neurons", values="Count")
sequencing_df = sequencing_df.T.fillna(0)
print(f"{time.time() - currtime} elapsed to load sequencing data")
sequencing_df

# TODO drop the background RNA from table S2 in the paper

#%%
# info about the genes themselves
gene_loc = data_dir / "GSE136049_gene_annotations.csv"
gene_df = pd.read_csv(gene_loc)
gene_df["genes"] = range(1, len(gene_df) + 1)
gene_df = gene_df.set_index("genes")
gene_df = gene_df.loc[sequencing_df.columns]  # some gene #s werent used
gene_df

#%%
sequencing_df.rename(columns=gene_df["gene_symbol"], inplace=True)
sequencing_df

#%%
# annotations for the individual genes
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
# metadata for each neuron in the gene expression data
class_map_loc = data_dir / "Labels2_CElegansScRNA_onlyLabeledNeurons.csv"
scrna_meta = pd.read_csv(class_map_loc)
scrna_meta = scrna_meta.set_index("OldIndices")
scrna_meta

#%%
# class-wise connectome
connectome_loc = data_dir / "scRNAClassConnectome.csv"
adj_df = pd.read_csv(connectome_loc, index_col=None, header=None)
adj = adj_df.values
adj_df

#%%
# labels for the classes
label_loc = data_dir / "Connectome_scRNAClassDescriptors.csv"
label_df = pd.read_csv(label_loc)
label_df["neuron_class"] = label_df["neuron_class"].map(lambda x: x.strip("'"))
label_df["broad_class"] = label_df["broad_class"].map(lambda x: x.strip("'"))
label_df["neurotransmitter"] = label_df["neurotransmitter"].map(lambda x: x.strip("'"))
label_df

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

# #%%
# n_components = 20
# max_iter = 10
# gamma = 20
# sca = SparseComponentAnalysis(n_components=n_components, max_iter=max_iter, gamma=gamma)

# currtime = time.time()
# X_transformed = sca.fit_transform(X_train)
# print(f"{time.time() - currtime} elapsed")


#%%

neuron_type_palette = dict(
    zip(np.unique(scrna_meta.loc[index_train, "Neuron_type"]), cc.glasbey_light)
)

#%%
from graspologic.embed import LaplacianSpectralEmbed

lse = LaplacianSpectralEmbed(form="R-DAD", n_components=16)
U, V = lse.fit_transform(adj)
Y = np.concatenate((U, V), axis=1)

#%%
embed_map = dict(zip(label_df["neuron_class"], Y))

#%%
neuron_types = scrna_meta.loc[index_train, "Neuron_type"]

Y_expanded = neuron_types.map(embed_map)
Y_expanded = np.stack(Y_expanded.values)
Y_expanded


# %%
Y_expanded = StandardScaler(with_mean=True, with_std=True, copy=False).fit_transform(
    Y_expanded
)
#%%
from sklearn.cross_decomposition import PLSSVD, CCA

currtime = time.time()
n_components = 5
model = PLSSVD(n_components=n_components)
X_scores, Y_scores = model.fit_transform(X_train, Y_expanded)
print(f"{time.time() - currtime} elapsed")

import matplotlib.pyplot as plt

for i in range(n_components):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(
        x=X_scores[:, i],
        y=Y_scores[:, i],
        hue=neuron_types,
        palette=neuron_type_palette,
        ax=ax,
        legend=False,
    )
    ax.set(ylabel="Connectivity score", xlabel="Gene score")
#%%
pairplot(X_scores, labels=neuron_types.values, palette=neuron_type_palette)

#%%
# could try taking the median over the classes from the gene data
# before running CCA/PLS
