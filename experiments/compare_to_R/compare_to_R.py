#%%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ortho_group

from sparse_decomposition import SparseComponentAnalysis
from sparse_decomposition.utils import (
    l1_norm,
    soft_threshold,
    proportion_variance_explained,
)
from sparse_new_basis.plot import savefig, set_theme
from sparse_new_basis.R import sca_R_epca, setup_R, sma_R_epca


set_theme()

epca = setup_R()


def sca_R(*args, **kwargs):
    return sca_R_epca(epca, *args, **kwargs)


fig_dir = Path("sparse_new_basis/results/compare_to_R")


def stashfig(name, *args, **kwargs):
    savefig(fig_dir, name, *args, **kwargs)


#%%
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

    X = S @ Y.T + np.random.normal(0, 0.01, size=(n, n))  # TODO variance right?
    return X


center = True
scale = False
max_iter = 1
k_range = np.arange(2, d + 1, 4)
n_replicates = 400
rows = []
for i in range(n_replicates):
    X = sample_data()
    if center:
        X -= np.mean(X, axis=0)
    for k in k_range:
        gamma = k * 2.5
        sca = SparseComponentAnalysis(
            n_components=k, gamma=gamma, max_iter=max_iter, tol=0
        )
        Z_hat_sca = sca.fit_transform(X)
        Y_hat_sca = sca.components_.T

        pve = proportion_variance_explained(X, Y_hat_sca)
        rows.append(
            {
                "replicate": i,
                "k": k,
                "pve": pve,
                "method": "SCA",
                "n_nonzero": np.count_nonzero(Y_hat_sca),
            }
        )

        Z_hat_r, Y_hat_r, outs = sca_R(
            X,
            k=k,
            gamma=gamma,
            center=False,
            scale=scale,
            max_iter=max_iter,
            return_all=True,
            epsilon=0,
        )
        pve_r = np.asarray(outs[2])[-1]
        pve = proportion_variance_explained(X, Y_hat_r)
        rows.append({"replicate": i, "k": k, "pve": pve, "method": "r-SCA", 'n_nonzero':np.count_nonzero(Y_hat_r)})

results = pd.DataFrame(rows)

# # %%
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# colors = sns.color_palette("deep", 10)

# palette = {"SCA": colors[0], "r-SCA": colors[2]}
# sns.lineplot(
#     x="k",
#     y="pve",
#     data=results,
#     ci=None,
#     ax=ax,
#     markers=["o", ","],
#     hue="method",
#     palette=palette,
#     style="method",
# )
# dummy_palette = dict(
#     zip(results["replicate"].unique(), n_replicates * [palette["SCA"]])
# )
# sns.lineplot(
#     x="k",
#     y="pve",
#     data=results[results["method"] == "SCA"],
#     ci=None,
#     ax=ax,
#     alpha=0.3,
#     hue="replicate",
#     palette=dummy_palette,
#     legend=False,
#     lw=1,
# )
# dummy_palette = dict(
#     zip(results["replicate"].unique(), n_replicates * [palette["r-SCA"]])
# )
# sns.lineplot(
#     x="k",
#     y="pve",
#     data=results[results["method"] == "r-SCA"],
#     ci=None,
#     ax=ax,
#     alpha=0.3,
#     hue="replicate",
#     palette=dummy_palette,
#     legend=False,
#     lw=1,
# )
# ax.set(
#     yticks=[0.25, 0.5, 0.75],
#     xticks=[4, 8, 12, 16],
#     ylabel="PVE",
#     xlabel="# of PCs",
#     ylim=(0.05, 0.95),
# )
# stashfig("PVE-by-rank-r-vs-mine")

# #%%

sca_results = results[results["method"] == "SCA"]
r_sca_results = results[results["method"] == "r-SCA"]
diff_results = sca_results.copy()
diff_results["diff_pve"] = sca_results["pve"].values - r_sca_results["pve"].values
diff_results["diff_nnz"] = sca_results["n_nonzero"].values - r_sca_results["n_nonzero"].values
diff_results["k_jitter"] = diff_results["k"] + np.random.uniform(
    -0.5, 0.5, size=len(diff_results)
)
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# sns.scatterplot(x="k_jitter", y="diff", data=diff_results, ax=ax, s=20, alpha=0.7, lw=0)
# ax.axhline(0, linestyle=":", color="black", zorder=-1)
# ax.set(ylabel="(Python - R) PVE", xlabel="# of PCs")
# stashfig("PVE-diff-by-rank")

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.violinplot(x="k", y="diff", data=diff_results, ax=ax, width=0.9, cut=0, linewidth=1)
sns.stripplot(
    x="k", y="diff", data=diff_results, ax=ax, alpha=0.7, lw=0, size=3, jitter=0.3
)
ax.axhline(0, linestyle=":", color="black", zorder=-1)
ax.set(ylabel="(Python - R) PVE", xlabel="# of PCs")

stashfig("PVE-diff-violin")

#%%
