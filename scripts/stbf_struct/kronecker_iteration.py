import ipdb
import matplotlib.cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import (colors, configure_matplotlib_style, save_pgf_trim,
                   textwidth_in)

configure_matplotlib_style()

iteration_out_path = "data/stbf_struct/iterations.csv"

iteration_accuracy = pd.read_csv(iteration_out_path)
iteration_accuracy["subject"] = iteration_accuracy["subject"].astype(str)
iteration_accuracy["fold"] = iteration_accuracy["fold"].astype(str)


def plot_iteration_lines(data, ax, n_trials="", show_chance_label=False):
    sns.lineplot(
        data=data, x="iterations", y="accuracy", ax=ax, marker="s", markersize=3, mew=0
    )
    ax.set_ylim([0, 1])
    ax.axhline(1 / 9, color=colors["lightgray"], linestyle="--", linewidth=1)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_title(n_trials)
    if show_chance_label:
        ax.annotate(
            "chance",
            fontsize=9,
            xy=(0.9, 1.25 / 9),
            ha="right",
            va="bottom",
            xycoords="axes fraction",
            color=colors["darkgray"],
            bbox=dict(facecolor="white", linewidth=0, pad=0),
        )


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, constrained_layout=True)

data_1_block = iteration_accuracy[iteration_accuracy["n_train_blocks"] == 1]
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 1],
    axs[0],
    n_trials="1 trial",
)
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("")
axs[0].set_xticklabels([])
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 2], axs[1], n_trials="2 trials"
)
axs[1].set_xlabel("")
axs[1].set_xticklabels([])
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 5], axs[2], n_trials="5 trials"
)
axs[2].set_xlabel("")
axs[2].set_xticklabels([])

save_pgf_trim(fig, axs[0], "figures/stbf_struct/fpi-0.pgf", columns=3, rows=1.3)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, constrained_layout=True)

data_1_block = iteration_accuracy[iteration_accuracy["n_train_blocks"] == 9]
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 1],
    axs[0],
    n_trials="1 trial",
)
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Fixed point iterations")
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 2],
    axs[1],
    n_trials="2 trials",
)
axs[1].set_xlabel("Fixed point iterations")
plot_iteration_lines(
    data_1_block[data_1_block["n_runs"] == 5],
    axs[2],
    n_trials="5 trials",
    show_chance_label=True,
)
axs[2].set_xlabel("Fixed point iterations")

save_pgf_trim(fig, axs[0], "figures/stbf_struct/fpi-1.pgf", columns=3, rows=1.5)
