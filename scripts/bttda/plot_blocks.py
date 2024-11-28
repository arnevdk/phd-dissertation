import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import colors, configure_matplotlib_style, save_pgf_trim

configure_matplotlib_style()

df_select = pd.read_csv("data/bttda/block_erp_select.csv")
df_select = pd.concat(
    [df_select, pd.read_csv("data/bttda/block_mi_select.csv")], ignore_index=True
)
df_select = pd.concat(
    [df_select, pd.read_csv("data/bttda/block_lr_select.csv")], ignore_index=True
)
df_select = df_select.reset_index()
df_train = pd.read_csv("data/bttda/block_erp_train.csv")
df_train = pd.concat(
    [df_train, pd.read_csv("data/bttda/block_mi_train.csv")], ignore_index=True
)
df_train = pd.concat(
    [df_train, pd.read_csv("data/bttda/block_lr_train.csv")], ignore_index=True
)
df_train = df_train.reset_index()

df_select = df_select.groupby(["dataset", "subject", "session", "block"])
df_select = df_select[["train_score", "val_score", "test_score"]].aggregate("mean")
df_select = df_select.reset_index()
idx = ["dataset", "subject", "session"]
df_score_train = df_select.groupby(idx)["train_score"].aggregate("first")
df_score_val = df_select.groupby(idx)["val_score"].aggregate("first")
df_score_test = df_select.groupby(idx)["test_score"].aggregate("first")
df_diff = df_select.set_index(idx + ["block"])
df_diff["validation"] = df_diff["val_score"] - df_score_val
df_diff["training"] = df_diff["train_score"] - df_score_train
df_diff["test"] = df_diff["test_score"] - df_score_test
df_diff = df_diff.reset_index()
df_diff = df_diff.melt(
    id_vars=idx + ["block"],
    value_vars=["training", "validation", "test"],
    var_name="split",
    value_name="roc_auc_diff",
)

df_diff["n_blocks"] = df_diff["block"] + 1
df_diff["roc_auc_diff_%"] = df_diff["roc_auc_diff"] * 100
fig, axs = plt.subplots(2, 2, layout="tight", sharex=True)
lineplot_kwargs = dict(
    x="n_blocks",
    hue="dataset",
    errorbar=None,
)

df_train["n_blocks"] = df_train["block"]

datasets = [["BNCI2014-008", "BNCI2015-003"], ["BNCI2014-001", "BNCI2014-004"]]
for row, selection in enumerate(datasets):

    sns.lineplot(
        data=df_train[df_train["dataset"].isin(selection)],
        y="nmse",
        ax=axs[row, 0],
        legend=True,
        **lineplot_kwargs
    )
    axs[row, 0].set_ylabel("NMSE")
    axs[row, 0].set_xticklabels([str(round(t, 1)) for t in axs[row, 0].get_xticks()])
    axs[row, 0].set_yticklabels([str(round(t, 1)) for t in axs[row, 0].get_yticks()])
    axs[row, 0].legend(loc="upper right", bbox_to_anchor=(1.075, 1))

    sns.lineplot(
        data=df_diff[df_diff["dataset"].isin(selection)],
        y="roc_auc_diff_%",
        style="split",
        ax=axs[row, 1],
        legend=False,
        **lineplot_kwargs
    )
    axs[row, 1].axhline(0, color=colors["lightgray"], linestyle="--", linewidth=1)
    if not row:
        axs[row, 1].set_ylim([0, None])
    if not row:
        axs[row, 1].set_ylabel("$\\Delta$ ROC-AUC (\\%.)")
    else:
        axs[row, 1].set_ylabel("$\\Delta$ accuracy (\\%.)")

    axs[row, 1].set_xticklabels([str(int(t)) for t in axs[row, 1].get_xticks()])
    axs[row, 1].set_yticklabels([str(int(t)) for t in axs[row, 1].get_yticks()])

    if not row:
        line_styles = ["solid", "dashed", "dotted"]
        labels = ["training", "validation", "test"]
        handles = [
            plt.Line2D([0], [0], linestyle=style, color=colors["darkgray"])
            for style in line_styles
        ]
        legend = axs[row, 1].legend(handles, labels, frameon=True, loc="upper left")
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_boxstyle("Square")  # Remove rounded corners
for ax in axs.flatten():
    ax.set_xlabel("$b$")
    ax.set_xlim([1, 10])

fig.tight_layout()
save_pgf_trim(fig, axs[0, 0], "figures/bttda/blocks.pgf", rows=2.2, columns=2)
