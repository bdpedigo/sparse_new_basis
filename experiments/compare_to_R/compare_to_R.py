#%%

# setting temporary PATH variables
import os

# path to your R installation
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
# path depends on where you installed Python. Mine is the Anaconda distribution
os.environ[
    "R_USER"
] = "/Users/bpedigo/miniconda3/envs/sparse/lib/python3.7/site-packages/rpy2"


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import seaborn as sns
from rpy2.robjects.packages import importr
from scipy.stats import ortho_group

from sparse_matrix_analysis import sparse_component_analysis
from sparse_matrix_analysis.utils import l1_norm, soft_threshold


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

rpy2.robjects.numpy2ri.activate()

# load the module we need
epca = importr("epca")


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

    X = S @ Y.T + np.random.normal(0, 0.03, size=(n, n))  # TODO variance right?
    return X


def proportion_variance_explained(X, Y):
    X = X.copy()
    # X -= X.mean(axis=0)[None, :]
    return (np.linalg.norm(X @ Y @ np.linalg.inv(Y.T @ Y) @ Y.T, ord="fro") ** 2) / (
        np.linalg.norm(X, ord="fro") ** 2
    )


def r_sca(X, n_components=2, gamma=None, center=True, scale=True):
    out = epca.sca(X, k=n_components, gamma=gamma, center=center, scale=scale)
    loadings = np.asarray(out[0])
    scores = np.asarray(out[1])
    return scores, loadings


center = True
scale = False
k_range = np.arange(2, d + 1, 2)
n_replicates = 5
rows = []
for i in range(n_replicates):
    X = sample_data()
    # X -= np.mean(X, axis=0) # TODO ?
    for k in k_range:
        gamma = k * 2.5
        Z_hat_sca, Y_hat_sca = sparse_component_analysis(
            X, n_components=k, gamma=gamma, max_iter=100, scale=scale, center=center
        )
        pve = proportion_variance_explained(X, Y_hat_sca)
        rows.append({"replicate": i, "k": k, "pve": pve, "method": "SCA"})

        Z_hat_r, Y_hat_r = r_sca(
            X, n_components=k, gamma=gamma, center=center, scale=scale
        )
        pve = proportion_variance_explained(X, Y_hat_r)
        rows.append({"replicate": i, "k": k, "pve": pve, "method": "r-SCA"})

        # this just verifies that PVE calc seems reasonable
        # pca_obj = PCA(n_components=k, svd_solver="full")
        # Z_hat = pca_obj.fit_transform(X)
        # assert pca_obj.explained_variance_ratio_.sum() - pve < 1e-10

results = pd.DataFrame(rows)

# %%


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = sns.color_palette("deep", 10)

palette = {"SCA": colors[0], "r-SCA": colors[2]}
sns.lineplot(
    x="k",
    y="pve",
    data=results,
    ci=None,
    ax=ax,
    markers=["o", ","],
    hue="method",
    palette=palette,
    style="method",
)
dummy_palette = dict(
    zip(results["replicate"].unique(), n_replicates * [palette["SCA"]])
)
sns.lineplot(
    x="k",
    y="pve",
    data=results[results["method"] == "SCA"],
    ci=None,
    ax=ax,
    alpha=0.3,
    hue="replicate",
    palette=dummy_palette,
    legend=False,
    lw=1,
)
dummy_palette = dict(
    zip(results["replicate"].unique(), n_replicates * [palette["r-SCA"]])
)
sns.lineplot(
    x="k",
    y="pve",
    data=results[results["method"] == "r-SCA"],
    ci=None,
    ax=ax,
    alpha=0.3,
    hue="replicate",
    palette=dummy_palette,
    legend=False,
    lw=1,
)
ax.set(
    yticks=[0.25, 0.5, 0.75],
    xticks=[4, 8, 12, 16],
    ylabel="PVE",
    xlabel="# of PCs",
    ylim=(0.05, 0.95),
)
plt.savefig("PVE-by-rank-r-vs-mine", transparent=False, facecolor="w")
