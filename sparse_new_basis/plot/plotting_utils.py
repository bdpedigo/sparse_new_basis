import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_RC_PARAMS = {
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


def set_theme(rc_dict=DEFAULT_RC_PARAMS):
    for key, val in rc_dict.items():
        mpl.rcParams[key] = val
    context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
    sns.set_context(context)


def savefig(
    fig_dir,
    name,
    transparent=False,
    facecolor="w",
    dpi=300,
    pad_inches=0.25,
    bbox_inches="tight",
    format="png",
    **kwargs,
):
    plt.savefig(
        fig_dir / name,
        transparent=transparent,
        facecolor=facecolor,
        dpi=dpi,
        pad_inches=pad_inches,
        bbox_inches=bbox_inches,
        **kwargs,
    )
