import os

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from setup import (colors, configure_matplotlib_style, save_pgf_trim,
                   textwidth_in)

configure_matplotlib_style()


classifier_models = ["stbf_kron_toep_shrunk", "stbf_loocv", "stbf_emp", "xdawn_rg"]

ns_channels = [32]

training_time_path = "data/stbf_struct/training_time.csv"

training_time = pd.read_csv(training_time_path)
training_time.loc[training_time["classifier_model"] == "xdawn_rg", "elapsed"] -= (
    1.035 - 0.2
)

training_time.rename(
    columns={
        "elapsed": "Training time (s)",
    },
    inplace=True,
)
model_rename = {
    "stbf_kron_toep_shrunk": "STBF-struct",
    "stbf_loocv": "STBF-shrunk",
    "stbf_emp": "STBF-emp",
    "xdawn_rg": "XDAWN+RG",
}
training_time.replace(model_rename, inplace=True)


def plot_training_time(data, ax, label=None, show_legend=False, show_y_label=False):
    sns.stripplot(
        data=data,
        y="classifier_model",
        x="Training time (s)",
        hue="classifier_model",
        ax=ax,
        size=2,
        zorder=0,
    )
    sns.boxplot(
        data=data,
        y="classifier_model",
        x="Training time (s)",
        color=colors["darkgray"],
        ax=ax,
        fill=False,
        showfliers=False,
        linewidth=1,
        zorder=1,
        width=0.5,
    )

    ax.set_ylabel("")


fig, ax = plt.subplots(1, 1, constrained_layout=True)
plot_training_time(training_time[training_time["n_channels"] == 32], ax)

fig.savefig("figures/stbf_struct/training_time.eps")
save_pgf_trim(fig, ax, "figures/stbf_struct/training_time.pgf", columns=2)
