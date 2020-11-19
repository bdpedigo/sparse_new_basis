#%%

# setting temporary PATH variables
import os

os.chdir("./")
print(os.getcwd())
os.environ[
    "R_HOME"
] = "/Library/Frameworks/R.framework/Resources/"  # path to your R installation
os.environ[
    "R_USER"
] = "/Users/bpedigo/miniconda3/envs/sparse/lib/python3.7/site-packages/rpy2"  # path depends on where you installed Python. Mine is the Anaconda distribution
print("here")
# importing rpy2
import rpy2.robjects as robjects

print("now here")
print(robjects.r)


from rpy2.robjects.packages import importr


# import rpy2.robjects.packages as rpackages

# # import R's utility package
# utils = rpackages.importr("utils")

# select a mirror for R packages
# utils.chooseCRANmirror(ind=1)

# utils.install_packages("epca")

epca = importr("epca")
import numpy as np
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

# A = np.arange(100).reshape((10, 10))
A = np.random.randn(10, 10)
out = epca.sca(A)
print(out)
loadings = out[0]
scores = out[1]
print(loadings)
print(np.asarray(loadings))
#%%
def setup_R():
    import os

    # os.chdir("./")
    # print(os.getcwd())
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
    os.environ[
        "R_USER"
    ] = "/Users/bpedigo/miniconda3/envs/sparse/lib/python3.7/site-packages/rpy2"
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    epca = importr("epca")
    return epca


def run_sca_R(epca, **kws):
    out = epca.sca(**kws)
    loadings = out[0]
    scores = out[1]
    return np.asarray(scores), np.asarray(loadings)
