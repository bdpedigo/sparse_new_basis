import numpy as np


def setup_R():
    import os

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


def sca_R_epca(epca, *args, return_all=False, **kwargs):
    out = epca.sca(*args, **kwargs)
    Y = np.asarray(out[0])
    Z = np.asarray(out[1])
    if return_all:
        return Z, Y, out
    else:
        return Z, Y


def sma_R_epca(epca, *args, return_all=False, **kwargs):
    out = epca.sma(*args, **kwargs)
    Z = np.asarray(out[0])
    B = np.asarray(out[1])
    Y = np.asarray(out[2])
    if return_all:
        return Z, B, Y, out[3:]
    else:
        return Z, B, Y
