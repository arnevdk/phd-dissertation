#!/usr/bin/env python
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.gridspec import GridSpec
from setup import (configure_matplotlib_style, diverging_cmap, save_pgf_trim,
                   textwidth_in)
from statsmodels.stats.multitest import multipletests

configure_matplotlib_style()

# Read data
df_our = pd.read_csv("data/covert_align//results_CVSA-ERP_cross.csv")
# df_our_riemann = pd.read_csv("data/results_CVSA-ERP_erpcov-ts-logreg_cross.csv")
# df_our_riemann = df_our_riemann[df_our_riemann["model"] == "Riemann"]
# df_our = pd.concat(
#    [
#        df_our,
#        df_our_riemann,
#    ],
#    ignore_index=True,
# )
df_our["dataset"] = "CVSA-ERP"
df_our["session"] = 1


# for sub_exclude in EXCLUDE["CVSA-ERP"]:
#    df_our = df_our[df_our["subject"] != sub_exclude]

df_aloise2012 = pd.read_csv("data/covert_align/results_BNCI2014-009_cross.csv")
# df_aloise2012_riemann = pd.read_csv(
#    "data/results_BNCI2014-009_erpcov-ts-logreg_cross.csv"
# )
# df_aloise2012_riemann = df_aloise2012_riemann[
#    df_aloise2012_riemann["model"] == "Riemann"
# ]
# df_aloise2012 = pd.concat(
#    [df_aloise2012, df_aloise2012_riemann],
#    ignore_index=True,
# )
df_aloise2012["dataset"] = "BNCI2014-009"
df_aloise2012["session"] = df_aloise2012["session"].astype(int)
df = pd.concat([df_our, df_aloise2012])

df["roc_auc"] = df["roc_auc"] * 100

replace = {
    "S": "overt",
    "G": "covert",
    "split/1": "split ($d=1$)",
    "split/2": "split ($d=2$)",
    "split/3": "split ($d=3$)",
    "mix": "combined",
}
df["train_cond"].replace(replace, inplace=True)
df["test_cond"].replace(replace, inplace=True)
categories = [
    "overt",
    "covert",
    "split ($d=1$)",
    "split ($d=2$)",
    "split ($d=3$)",
    "combined",
]
df["train_cond"] = pd.Categorical(df["train_cond"], categories=categories, ordered=True)
df["test_cond"] = pd.Categorical(df["test_cond"], categories=categories, ordered=True)
replace = {
    "wCBLE": "WCBLE",
}
df["model"].replace(replace, inplace=True)


df = df.pivot_table(
    index=[
        "dataset",
        "train_cond",
        "test_cond",
        "subject",
        "session",
        "fold",
    ],
    columns="model",
    values="roc_auc",
    aggfunc="mean",
    dropna=True,
)

# Calculate differences
df_diff = pd.DataFrame()
models = ["tLDA", "CBLE", "tLDA"]
pairs = [("WCBLE", "tLDA"), ("WCBLE", "CBLE"), ("CBLE", "tLDA")]
for model1, model2 in pairs:
    df_diff[f"{model1} - {model2}"] = df[model1] - df[model2]
df_diff = df_diff.groupby(["dataset", "train_cond", "test_cond"])
df_diff = df_diff.aggregate("mean")
df_diff = df_diff.dropna()

df_diff = df_diff.melt(
    var_name="comparison", value_name="roc_auc_diff", ignore_index=False
)
df_diff = df_diff.set_index(["comparison"], append=True)


# Calculate p-values
def wilcoxon(df, col1, col2):
    print(len(df))
    stat = scipy.stats.wilcoxon(df[col1], df[col2])
    return stat


df_stat = pd.DataFrame()
df_g = df.groupby(["dataset", "train_cond", "test_cond"])
for model1, model2 in pairs:
    # TODO: replace with setting directly in df_diff
    df_stat[f"{model1} - {model2}"] = df_g.apply(wilcoxon, model1, model2)
df_stat = df_stat.melt(var_name="comparison", value_name="stat", ignore_index=False)
df_stat = df_stat.set_index(["comparison"], append=True)["stat"]
df_diff["stat"] = df_stat
df_diff["stat"], df_diff["p"] = list(zip(*df_diff["stat"]))


def p_multipletests(p_vals, *args, **kwargs):
    reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
        p_vals, *args, **kwargs
    )
    return pvals_corrected


# Correct for multiple tests
df_diff["p_corr"] = df_diff.groupby(["dataset", "comparison"])["p"].transform(
    p_multipletests,
    alpha=0.05,
    method="fdr_bh",
)

# Reverse last pair
rows = df_diff.index.get_level_values("comparison") == "tLDA - CBLE"
df_diff.loc[rows, "roc_auc_diff"] *= -1
df_diff.rename(index={"tLDA - CBLE": "CBLE - tLDA"}, inplace=True)


# Generate annotations
# df_diff["star"] = "\n"
# df_diff.loc[df_diff["p_corr"] <= 0.05, "star"] = "\n*"
# df_diff.loc[df_diff["p_corr"] <= 0.01, "star"] = "\n**"
# df_diff.loc[df_diff["p_corr"] <= 0.001, "star"] = "\n***"
# df_diff.loc[df_diff["p_corr"] <= 0.0001, "star"] = "\n****"
# df_diff["annot"] = df_diff["roc_auc_diff"].combine(
#    df_diff["star"], lambda d, s: f"{d:.{2}f}{s}"
# )
#
df_diff["annot"] = [f"{d:.{2}f}" for d in df_diff["roc_auc_diff"]]


def draw_annot_heatmap(row_var, col_var, val_var, annot_var, **kwargs):
    df = kwargs.pop("data")
    df_val = df.pivot(index=row_var, columns=col_var, values=val_var)
    df_annot = df.pivot(index=row_var, columns=col_var, values=annot_var)
    # df_mask = df.pivot(index=row_var, columns=col_var, values=mask_var)
    kwargs.setdefault("annot", df_annot)
    kwargs.setdefault("fmt", "")
    ## kwargs.setdefault("mask", df_mask)
    ax = sns.heatmap(df_val, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)


df_diff = df_diff.reset_index()

aspect = 2.7
fg_kws = dict(
    # row="dataset",
    col="comparison",
    gridspec_kws=dict(wspace=0.1),
)
map_df_kws = dict(
    func=draw_annot_heatmap,
    row_var="train_cond",
    col_var="test_cond",
    val_var="roc_auc_diff",
    annot_var="annot",
    cbar=False,
    square=True,
    center=0,
    vmin=-0.05 * 100,
    vmax=0.05 * 100,
    cmap=diverging_cmap,
    annot_kws={"fontsize": 8},
)
g = sns.FacetGrid(
    df_diff[df_diff["dataset"] == "CVSA-ERP"],
    height=(textwidth_in + 0.0) / aspect,
    **fg_kws,
)
g.map_dataframe(**map_df_kws)
g.set_titles(col_template="{col_name}")
g.set(xlabel="Evaluation", ylabel="Training")

plt.savefig(
    "figures/covert_align/cross_cvsa-erp.pgf",
    bbox_inches="tight",
    transparent=True,
    pad_inches=0,
)

g = sns.FacetGrid(
    df_diff[df_diff["dataset"] == "BNCI2014-009"],
    height=((textwidth_in / 2) + 0.0) / aspect,
    **fg_kws,
)
g.map_dataframe(**map_df_kws)
g.set_titles(col_template="{col_name}")
g.set(xlabel="Evaluation", ylabel="Training")

plt.savefig(
    "figures/covert_align/cross_bnci2014-009.pgf",
    bbox_inches="tight",
    transparent=True,
    pad_inches=0,
)
