import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mne import combine_evoked, pick_info, read_evokeds
from mne.transforms import apply_trans
from mne.viz import plot_compare_evokeds
from mne.viz.utils import plot_sensors
from setup import (colors, condition_cmap, configure_matplotlib_style,
                   save_pgf_trim)

configure_matplotlib_style()

df = pd.read_csv("data/covert_align/evokeds.csv")

for ind in df.index:
    path = df.loc[ind, "evoked_path"]
    path = os.path.join("data", "covert_align", path)
    evoked = read_evokeds(path)[0]
    assert not np.any(np.isnan(evoked.data))
    evoked = evoked.apply_baseline((-0.2, 0))
    df.loc[ind, "evoked"] = evoked
df = df.groupby(["dataset", "condition", "label"])
df = df["evoked"].aggregate(lambda x: combine_evoked(list(x), weights="equal"))
df = df.reset_index()

# Plot
plt.rcParams.update(
    {
        "lines.linewidth": 1.5,
    }
)


def plot_sensor_picks(ax, pick):
    plot_sensors(
        evoked.info,
        kind="topomap",
        ch_type="eeg",
        block=False,
        show=False,
        axes=ax,
        show_names=False,
        pointsize=0.2,
    )
    ch_idc = [i for i, ch in enumerate(evoked.ch_names) if ch in pick]
    plot_sensors(
        pick_info(evoked.info, ch_idc),
        kind="topomap",
        ch_type="eeg",
        block=False,
        show=False,
        axes=ax,
        show_names=False,
        pointsize=2,
    )
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])


def plot_evoked(ax, target, non_target, color, ylim, distractor=None):
    target_contrast = combine_evoked([target, non_target], weights=[1, -1])
    ax.plot(
        target_contrast.times,
        target_contrast.get_data().mean(axis=0) * 1e6,
        color=color,
    )
    # ax.plot(
    #    non_target.times,
    #    non_target.get_data().mean(axis=0) * 1e6,
    #    color="grey",
    # )
    if distractor is not None:
        distractor_contrast = combine_evoked([distractor, non_target], weights=[1, -1])
        ax.plot(
            distractor_contrast.times,
            distractor_contrast.get_data().mean(axis=0) * 1e6,
            color=colors["darkgray"],
        )

    ax.set_yticks([ylim[0], 0, ylim[1]])
    ax.set_ylim(ylim)


picks = [["Pz", "Cz", "Fz"], ["PO7", "PO8"]]
conds = [
    ("CVSA-ERP", "overt", (0, 0), (-3, 8)),
    ("CVSA-ERP", "covert", (0, 1), (-3, 8)),
    ("CVSA-ERP", "split ($d=1$)", (1, 0), (-3, 8)),
    ("CVSA-ERP", "split ($d=2$)", (1, 1), (-3, 8)),
    ("CVSA-ERP", "split ($d=3$)", (1, 2), (-3, 8)),
    ("BNCI2014-009", "overt", (2, 0), (-30, 30)),
    ("BNCI2014-009", "covert", (2, 1), (-30, 30)),
]
fig = plt.figure()
gs = GridSpec(3, 6, figure=fig, width_ratios=[1] * 6)

axs = np.empty((3, 6), dtype=object)
for pi, pick in enumerate(picks):
    ax = fig.add_subplot(gs[0, pi * 3 + 2])
    plot_sensor_picks(ax, pick)
    for dataset, condition, pos, ylim in conds:
        ax = fig.add_subplot(gs[pos[0], pi * 3 + pos[1]])
        axs[pos[0], pi * 3 + pos[1]] = ax
        color = condition_cmap[condition]
        df_cond = df.query(f"condition == '{condition}' and dataset == '{dataset}'")
        target = df_cond.query(f"label == 'target'")["evoked"].values[0]
        non_target = df_cond.query(f"label == 'non-target'")["evoked"].values[0]
        distractor = None
        if condition.startswith("split"):
            distractor = df_cond.query(f"label == 'distractor'")["evoked"].values[0]
        plot_evoked(
            ax,
            target.copy().pick(pick),
            non_target.copy().pick(pick),
            color,
            ylim,
            distractor=distractor.copy().pick(pick) if distractor is not None else None,
        )
        ax.axhline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
        ax.axvline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
        # ax.text(
        #    0.99,
        #    0.99,
        #    condition,
        #    ha="right",
        #    va="top",
        #    transform=ax.transAxes,
        #    color=color,
        # )
        ax.set_xlim([-0.2, 1])
        ax.set_title(f"{condition}")

# ax.set_xticks([])
for ax in axs.flatten():
    if ax is not None:
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
for r, c in [(-1, 0), (-1, 1), (1, 2), (-1, 3), (-1, 4), (1, 5)]:
    axs[r, c].set_xticklabels([str(round(t, 1)) for t in axs[r, c].get_xticks()])
    axs[r, c].set_xlabel("Time (s)")
for r in [0, 1, 2]:
    axs[r, 0].set_yticklabels([str(round(t, 1)) for t in axs[r, 0].get_yticks()])
    axs[r, 0].set_ylabel("Amplitude (µV)")

plt.draw()

annot_kwargs = dict(
    ha="left",  # Horizontal alignment ('left', 'center', 'right')
    va="bottom",
    color="black",  #
)
text = axs[0, 0].text(
    0, 1.3, "\\textbf{\\large CVSA-ERP}", transform=axs[0, 0].transAxes, **annot_kwargs
)
text.set_in_layout(False)
text = axs[-1, 0].text(
    0,
    1.3,
    "\\textbf{\\large BNCI2014-009}",
    transform=axs[-1, 0].transAxes,
    **annot_kwargs,
)
text.set_in_layout(False)
save_pgf_trim(fig, axs[-1, 0], "figures/covert_align/grand_avg.pfg", rows=1.3, pad=0.2)
