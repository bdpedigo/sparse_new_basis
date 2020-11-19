#%%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sparse_matrix_analysis import SparseComponentAnalysis

data_dir = Path("sparse_new_basis/data/BP_Barabasi_Share/ScRNAData")

# gene expression data
sequencing_loc = data_dir / "Celegans_ScRNA_OnlyLabeledNeurons.csv"
sequencing_df = pd.read_csv(sequencing_loc, skiprows=[1])
currtime = time.time()
sequencing_df = sequencing_df.pivot(index="genes", columns="neurons", values="Count")
sequencing_df = sequencing_df.T.fillna(0)
print(f"{time.time() - currtime} elapsed")

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
sca = SparseComponentAnalysis(n_components=10, max_iter=10, gamma=10)

# subsample = 2 ** 12
# if subsample:
#     subsample_inds = np.random.choice(len(X), replace=False, size=subsample)
#     X_sub = X[subsample_inds, :]
# else:
#     X_sub = X

currtime = time.time()
X_transformed = sca.fit_transform(X_train)
print(f"{time.time() - currtime} elapsed")

for i in range(10):
    print(np.count_nonzero(sca.components_[i]) / len(sca.components_[i]))
    sort_inds = np.argsort(np.abs(sca.components_[i]))[::-1]
    select_genes = gene_index[sort_inds][:10]  # top 10 in magnitude loading
    print(gene_df.loc[select_genes])
    print()

#%%

import matplotlib.pyplot as plt

#

# #%%
for i in range(10):
    print(np.count_nonzero(sca.components_[i]) / len(sca.components_[i]))
    sort_inds = np.argsort(np.abs(sca.components_[i]))[::-1]
    plt.figure()
    plt.plot(np.sort(sca.components_[i]), marker="o", linewidth=0, markersize=1)
    # print(gene_df.loc[select_genes])

#%%

from graspologic.plot import pairplot
import colorcet as cc

pairplot(
    X_transformed[:, :],
    labels=scrna_meta.loc[index_train, "Neuron_type"].values,
    palette=cc.glasbey_light,
    diag_kind=None,
)

# sca = SparseComponentAnalysis()
#%%
# len(sequencing_df.index[np.nonzero(sca.components_[0])])
