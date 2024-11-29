#!/usr/bin/env python

import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from setup import (colors, condition_cmap, configure_matplotlib_style,
                   save_pgf_trim)
from statsmodels.stats.multitest import multipletests

configure_matplotlib_style()

# Read BNCI2014-009 dataset
df_bnci2014_009 = pd.read_csv("data/covert_align/results_BNCI2014-009_cross.csv")
df_bnci2014_009_riemann = pd.read_csv(
    "data/covert_align/results_BNCI2014-009_erpcov-ts-logreg_cross.csv"
)
df_bnci2014_009_riemann = df_bnci2014_009_riemann[
    df_bnci2014_009_riemann["model"] == "Riemann"
]
df_bnci2014_009 = pd.concat(
    [df_bnci2014_009, df_bnci2014_009_riemann],
    ignore_index=True,
)

df_bnci2014_009["dataset"] = "BNCI2014-009"

df_cvsa_erp = pd.read_csv("data/covert_align/results_CVSA-ERP_cross.csv")
df_cvsa_erp_riemann = pd.read_csv(
    "data/covert_align/results_CVSA-ERP_erpcov-ts-logreg_cross.csv"
)
df_cvsa_erp_riemann = df_cvsa_erp_riemann[df_cvsa_erp_riemann["model"] == "Riemann"]
df_cvsa_erp = pd.concat(
    [df_cvsa_erp, df_cvsa_erp_riemann],
    ignore_index=True,
)
df_cvsa_erp["session"] = 1
# Read CVSA-ERP dataset
df_cvsa_erp["dataset"] = "CVSA-ERP"
df = pd.concat([df_bnci2014_009, df_cvsa_erp], ignore_index=True)
# Clean up dataset
df = df[df["train_cond"] == df["test_cond"]]
df["condition"] = df["train_cond"]
df = df[df["condition"] != "mix"]
replace = {
    "S": "overt",
    "G": "covert",
    "split/1": "split ($d=1$)",
    "split/2": "split ($d=2$)",
    "split/3": "split ($d=3$)",
}
df["condition"].replace(replace, inplace=True)
replace = {
    "wCBLE": "WCBLE",
}
df["model"].replace(replace, inplace=True)

# Plot

df["roc_auc"] = df["roc_auc"] * 100
index = [
    "dataset",
    "subject",
    "session",
    "condition",
    "fold",
]
df_plot = df.pivot(
    index=index,
    columns="model",
    values="roc_auc",
)
contrasts = [("WCBLE", "tLDA"), ("WCBLE", "CBLE"), ("CBLE", "tLDA")]
for contrast in contrasts:
    df_plot[contrast] = df_plot[contrast[1]] - df_plot[contrast[0]]
df_plot = df_plot.reset_index()
df_plot = df_plot.melt(
    id_vars=index[:-1],
    value_vars=contrasts,
    var_name="contrast",
    value_name="roc_auc_diff",
)
mpl.rcParams["lines.linewidth"] = 0.7

df_plot = df_plot.dropna()
plot_index = [
    "dataset",
    "subject",
    "session",
    "condition",
    "contrast",
]
df_plot = df_plot.groupby(plot_index)
df_plot = df_plot["roc_auc_diff"].aggregate(np.nanmean)
df_plot = df_plot.reset_index()
df_plot["contrast"] = pd.Categorical(
    df_plot["contrast"], categories=contrasts, ordered=True
)
df_plot["order"] = df_plot["condition"].copy()
df_plot["order"].replace(
    {
        "overt": 0,
        "covert": 1,
        "split ($d=1$)": 2,
        "split ($d=2$)": 3,
        "split ($d=3$)": 4,
        "mix": 5,
    },
    inplace=True,
)
df_plot = df_plot.sort_values("order")

g = sns.FacetGrid(
    data=df_plot,
    row="dataset",
    col="contrast",
    sharey="row",
    # margin_titles=True,
    gridspec_kws=dict(height_ratios=[5, 2]),
    row_order=["CVSA-ERP", "BNCI2014-009"],
)


def pointstripplot(*args, **kwargs):
    stripplot_kwargs = kwargs.copy()
    stripplot_kwargs["zorder"] = 1
    stripplot_kwargs["color"] = "lightgray"
    stripplot_kwargs["size"] = 3
    stripplot_kwargs["hue"] = None
    stripplot_kwargs["legend"] = False
    sns.stripplot(*args, **stripplot_kwargs)

    pointplot_kwargs = kwargs.copy()
    pointplot_kwargs["join"] = False
    pointplot_kwargs["palette"] = condition_cmap
    pointplot_kwargs["n_boot"] = 1000
    sns.pointplot(*args, **pointplot_kwargs)


g.map_dataframe(
    pointstripplot,
    x="roc_auc_diff",
    y="condition",
    hue="condition",
)

g.despine(left=False, right=False)
g.set(
    xlim=[-0.1 * 100, 0.1 * 100],
    xticks=[-0.05 * 100, 0, 0.05 * 100],
    xlabel="$\Delta$ ROC-AUC",
    # ylabel="VSA condition",
    ylabel=None,
)
# g.set_titles(
#    row_template="{row_name}",
#    col_template="$\\leftarrow$ {col_name[0]} vs. {col_name[1]} $\\rightarrow$",
# )


def annot(ax, l1, l2):
    ax.text(
        0.02,
        0.99,
        f"$\\leftarrow$ {l1}",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.98,
        f"{l2} $\\rightarrow$",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )


annot(g.axes[0, 0], "WCBLE", "tLDA")
annot(g.axes[0, 1], "WCBLE", "CBLE")
annot(g.axes[0, 2], "CBLE", "tLDA")


g.set_titles(template="")
for ax in g.axes.flatten():
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="y", which="both", length=0)
    ax.axvline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()])

g.axes[0, 0].set_title("CVSA-ERP")
g.axes[1, 0].set_title("BNCI2014-009")
g.fig.set_layout_engine("constrained")


save_pgf_trim(
    g.fig, g.axes[0, 0], "figures/covert_align/roc_auc_diff.pgf", rows=2, columns=3
)
