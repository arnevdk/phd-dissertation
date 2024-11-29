#!/usr/bin/env python
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from setup import (condition_cmap, configure_matplotlib_style, save_pgf_trim,
                   textwidth_in)
from statannotations.Annotator import Annotator

configure_matplotlib_style()

# ROC AUC score
# df_covert = pd.read_csv("data/results_covert_cross.csv")
# df_covert["dataset"] = "CVSA-ERP"
# df_covert["session"] = 1
# df_aloise2012 = pd.read_csv("data/results_arico2014_cross.csv")
# df_aloise2012["dataset"] = "BNCI2014-009"
# df = pd.concat([df_covert, df_aloise2012], ignore_index=True)
# df = df[df["train_task"] == df["test_task"]]
#
# df["condition"] = df["train_task"]
replace = {
    "S": "overt",
    "G": "covert",
    "split/1": "split\n($d=1$)",
    "split/2": "split\n($d=2$)",
    "split/3": "split\n($d=3$)",
}
# df["condition"].replace(replace, inplace=True)

#
# df = df.pivot(
#    index=["dataset", "subject", "session", "condition", "fold"],
#    columns="model",
#    values="roc_auc",
# )
# df = df.groupby(["dataset", "subject", "session", "condition"])
# df = df.aggregate("mean")

## counting accuracy
# df_count = pd.read_csv("data/counting_accuracy.csv")
# df_count["condition"].replace(replace, inplace=True)
# df_count["dataset"] = "covert"
# df_count["accuracy"] = (
#    1 - np.abs(df_count["actual"] - df_count["count"]) / df_count["actual"]
# )
# df_count = df_count.groupby(["dataset", "session", "subject", "condition"])
# df["count_accuracy"] = df_count["accuracy"].aggregate("mean")

# Target jitter
df_jitter = pd.read_csv("data/covert_align/latency.csv")
df_jitter["dataset"].replace(
    {"covert": "CVSA-ERP", "aloise2012": "BNCI2014-009"},
    inplace=True,
)
df_jitter = df_jitter[df_jitter["event_name"] == "target"]
df_jitter["condition"] = df_jitter["task"]
df_jitter["condition"].replace(replace, inplace=True)
df_jitter = df_jitter.groupby(["dataset", "subject", "session", "condition"])
df = df_jitter["target_latency"].aggregate(scipy.stats.iqr)
df = df.reset_index()
df = df.dropna()
df["condition"] = pd.Categorical(
    df["condition"],
    categories=[
        "overt",
        "covert",
        "split\n($d=1$)",
        "split\n($d=2$)",
        "split\n($d=3$)",
    ],
    ordered=True,
)

# Plot


def barplot_statannot(**kwargs):
    ax = sns.barplot(**kwargs)
    df = kwargs["data"]
    conditions = np.unique(df["condition"])
    pairs = []
    for c1 in range(len(conditions)):
        cond1 = conditions[c1]
        for c2 in range(c1 + 1, len(conditions)):
            cond2 = conditions[c2]
            pairs.append((cond1, cond2))
    annot = Annotator(ax, pairs, **kwargs)
    annot.configure(
        test="Mann-Whitney",
        hide_non_significant=True,
        comparisons_correction="bonf",
        line_width=1,
        line_height=0.1,
    )
    test = annot.apply_test()
    print(test)
    ax, test_results = annot.annotate()


g = sns.FacetGrid(
    df,
    col="dataset",
    sharex=False,
    gridspec_kws=dict(width_ratios=[5, 2]),
    col_order=["CVSA-ERP", "BNCI2014-009"],
)
g.map_dataframe(
    barplot_statannot,
    x="condition",
    y="target_latency",
    palette=condition_cmap,
    # width=0.5,
)
# g.set_titles(col_template="{col_name}")
g.axes[0, 0].set_title("CVSA-ERP", loc="left", weight="bold")
g.axes[0, 1].set_title("BNCI2014-009", loc="left", weight="bold")
g.axes[0, 1].set_yticklabels([str(round(t, 2)) for t in g.axes[0, 1].get_yticklabels()])

g.set_titles(col_template="")
g.set_axis_labels(x_var="VSA condition", y_var="Target latency IQR (s)")
save_pgf_trim(
    g.figure, g.axes[0, 0], "figures/covert_align/jitter.pgf", width=0.6 * textwidth_in
)
