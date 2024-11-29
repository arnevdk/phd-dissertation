#!/usr/bin/env python
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from setup import configure_matplotlib_style, model_cmap, save_pgf_trim
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

# from statannotations.Annotator import Annotator

configure_matplotlib_style()


# Read data
results_CVSA_ERP = pd.read_csv("data/covert_align/results_CVSA-ERP_blocks.csv.bak")
results_CVSA_ERP_riemann = pd.read_csv(
    "data/covert_align/results_CVSA-ERP_riemann_blocks.csv"
)
# results_CVSA_ERP_riemann["task"] = results_CVSA_ERP_riemann["condition"]
results_CVSA_ERP = pd.concat(
    [
        results_CVSA_ERP_riemann,
        results_CVSA_ERP,
    ]
)
results_CVSA_ERP["dataset"] = "CVSA-ERP"
results_CVSA_ERP["condition"] = results_CVSA_ERP["condition"].replace(
    {
        "split/1": "split ($d=1$)",
        "split/2": "split ($d=2$)",
        "split/3": "split ($d=3$)",
    }
)

results_geospell = pd.read_csv("data/covert_align/results_BNCI2014-009_blocks.csv")

results_geospell_riemann = pd.read_csv(
    "data/covert_align/results_BNCI2014-009_riemann_blocks.csv"
)
# results_geospell_riemann["task"] = results_geospell_riemann["condition"]
results_geospell = pd.concat([results_geospell_riemann, results_geospell])
geospell_dataset_name = "BNCI2014-009"
results_geospell["dataset"] = geospell_dataset_name
results_geospell["condition"] = results_geospell["condition"].replace(
    {"S": "overt", "G": "covert"}
)
#
#
df = pd.concat([results_geospell, results_CVSA_ERP])
df = df.replace(
    {
        "XDAWNCov-TS-LogReg": "XDAWNCov-TS-LR",
        "ERPCov-TS-LogReg": "ERPCov-TS-LR",
    }
)
# df = results_covert
df["n_trials"] = pd.Categorical(
    df["n_trials"], categories=list(range(1, 10 + 1)), ordered=True
)
df = df.rename(columns={"n_trials": "n_trials"})
replace = {
    "wCBLE": "WCBLE",
}
df["model"].replace(replace, inplace=True)

df["model"] = pd.Categorical(
    df["model"],
    categories=[
        "WCBLE",
        "CBLE",
        "tLDA",
        "XDAWNCov-TS-LR",
        "ERPCov-TS-LR",
    ],
    ordered=True,
)
# df["condition"] = pd.Categorical(
#    df["condition"],
#    categories=["overt", "covert", "split ($d=1$)", "split ($d=2$)", "split ($d=3$)"],
#    ordered=True,
# )

df["accuracy"] = df["accuracy"] * 100

# Plot
df = df.dropna(subset=["accuracy"])
df = df.reset_index()

fig, axs = plt.subplots(
    2,
    4,
    sharey=True,
    gridspec_kw=dict(width_ratios=[8, 10, 10, 10]),
)

plots = [
    ((0, 0), "BNCI2014-009", "overt"),
    ((1, 0), "BNCI2014-009", "covert"),
    #
    ((0, 1), "CVSA-ERP", "overt"),
    ((0, 2), "CVSA-ERP", "covert"),
    ((1, 1), "CVSA-ERP", "split ($d=1$)"),
    ((1, 2), "CVSA-ERP", "split ($d=2$)"),
    ((1, 3), "CVSA-ERP", "split ($d=3$)"),
]
for pos, dataset, condition in plots:
    ax = axs[pos]
    df_plot = df.loc[((df["dataset"] == dataset) & (df["condition"] == condition))]
    sns.lineplot(
        data=df_plot,
        x="n_trials",
        y="accuracy",
        hue="model",
        palette=model_cmap,
        legend=pos == (0, 2),
        ci=None,
        ax=ax,
    )
    ax.set_xlabel("\# repetitions")
    ax.text(
        0.95,
        0.05,
        condition,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([0, 100])
    if dataset == "BNCI2014-009":
        ax.set_xticks([2, 4, 6, 8])
    else:
        ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([str(int(t)) for t in ax.get_yticks()])

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)

axs[0, 0].set_ylabel("Accuracy (\\%)")
axs[1, 0].set_ylabel("Accuracy (\\%)")
axs[0, 0].set_title("CVSA-ERP")
axs[0, 1].set_title("BNCI2014-009")
for ax in axs[0, :]:
    ax.set_xlabel("")
    ax.set_xticklabels([])
axs[0, -1].axis("off")

legend = axs[0, 2].legend(loc="center left", bbox_to_anchor=(1.1, 0.5), frameon=False)
legend.set_in_layout(False)
save_pgf_trim(
    fig,
    axs[0, 0],
    "figures/covert_align/block_accuracy.pgf",
)
