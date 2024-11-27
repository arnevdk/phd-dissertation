import os

import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import (colors, configure_matplotlib_style, save_pgf_trim,
                   textwidth_in)

configure_matplotlib_style()

iteration_out_path = "data/stbf_struct/iterations.csv"


classifier_models = ["stbf_kron_toep_shrunk", "stbf_loocv", "stbf_emp", "xdawn_rg"]
train_blocks_range = range(1, 9 + 1)

accuracy_out_path = "data/stbf_struct/accuracy.csv"

accuracy = pd.read_csv(accuracy_out_path)
accuracy["accuracy"] = accuracy["accuracy"] * 100
accuracy.rename(
    columns={
        "n_train_blocks": "Amount of training data",
        "accuracy": "Accuracy (\\%)",
        "classifier_model": "Classifier",
    },
    inplace=True,
)
model_rename = {
    "stbf_emp": "STBF-emp",
    "stbf_loocv": "STBF-shrunk",
    "stbf_kron_toep_shrunk": "STBF-struct",
    "xdawn_rg": "XDAWN+RG",
}
accuracy.replace(model_rename, inplace=True)


def plot_accuracy(
    ax, trial_accuracy, show_chance_label=False, show_legend=False, title=None
):
    ax = sns.lineplot(
        data=trial_accuracy,
        x="Amount of training data",
        y="Accuracy (\\%)",
        hue="Classifier",
        style="Classifier",
        ax=ax,
        legend=show_legend,
        errorbar=None,
        dashes=False,
        markers="sDoP",
        markersize=3,
        mew=0,
    )
    ax.axhline(100 / 9, color=colors["lightgray"], linestyle="--", linewidth=1)
    ax.set_ylim([0, 100])
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()])

    if show_chance_label:
        ax.annotate(
            "chance",
            fontsize=9,
            xy=(0.05, 1.2 / 9),
            ha="left",
            va="bottom",
            xycoords="axes fraction",
            color=colors["darkgray"],
            bbox=dict(facecolor="white", linewidth=0, pad=0),
        )
    if title is not None:
        ax.set_title(title)
    if show_legend:
        plt.legend(
            title=None,
            frameon=True,  # Enable the frame (to make it white)
            framealpha=1,  # Opaque background
            edgecolor="none",
        )
        plt.setp(ax.get_legend().get_texts(), fontsize="7")  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize="7")
    ax.set_facecolor("white")
    return ax


fig, ax = plt.subplots(1, 1, constrained_layout=True)

plot_accuracy(
    ax,
    trial_accuracy=accuracy[accuracy["n_runs"] == 1],
    show_chance_label=True,
    show_legend=True,
)
legend = ax.get_legend()
legend.set_frame_on(True)  # Ensure the frame is displayed
legend.get_frame().set_alpha(0.0)
ax.set_yticklabels([str(int(t)) for t in ax.get_yticks()])
save_pgf_trim(
    fig,
    ax,
    "figures/stbf_struct/accuracy.pgf",
    width=2.5,
    height=2,
)

fig, axs = plt.subplots(5, 3, constrained_layout=True, sharex=True, sharey=True)
for n_runs in range(1, 15 + 1):
    ax = plot_accuracy(
        axs.flatten()[n_runs - 1],
        trial_accuracy=accuracy[accuracy["n_runs"] == n_runs],
        show_chance_label=n_runs == 13,
        show_legend=n_runs == 15,
        title="1 trial" if n_runs == 1 else f"{n_runs} trials",
    )
    if n_runs in [1, 4, 10, 13]:
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
save_pgf_trim(
    fig,
    axs[0, 0],
    "figures/stbf_struct/accuracy_all.pgf",
    width=textwidth_in * 1.05,
    rows=8,
    columns=3,
)
