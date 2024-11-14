import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from mne import read_epochs
from setup import colors, configure_matplotlib_style, pagesize, savefig_kws

configure_matplotlib_style()


subjects = {
    "PA1": "PA01",
    "PB1": "PB01",
    "PB2": "PB02",
    "PB4": "PB04",
    "PC4": "PC04",
}
conditions = ["overt", "covert", "free"]


N_TARGETS = 6
# INTERFACE_RADIUS = 12.28 / 2  # degrees of FOV
INTERFACE_RADIUS = 20 / 2  # degrees of FOV
TARGET_RADIUS = 4.15 / 2  # degrees of FOV
TARGET_INTENSE_RADIUS = 5.60 / 2  # degrees of FOV
FONT_SIZE = 0.5  # degrees of FOG


def get_target_pos(pos):
    if pos == "center":
        return [0, 0]
    elif isinstance(pos, int) and pos >= 0 and pos < 6:
        angle = pos * 2 * np.pi / N_TARGETS
        x = INTERFACE_RADIUS * np.cos(angle)
        y = INTERFACE_RADIUS * np.sin(angle)
        return [x, y]
    else:
        raise ValueError


xedges = np.load("data/patients/gaze/gaze_xedges.npy")
yedges = np.load("data/patients/gaze/gaze_yedges.npy")

fig, axs = plt.subplots(
    len(subjects),
    len(conditions),
    sharex=True,
    sharey=True,
    figsize=pagesize,
    layout="constrained",
)
axs = axs.reshape((len(subjects), len(conditions)))
for i, (subject, sub) in enumerate(subjects.items()):
    for j, condition in enumerate(conditions):
        axs[i, j].set_title(f"{subject}, {condition} VSA")
        hist = np.load(f"data/patients/gaze/gaze_sub-{sub}_cond-{condition}.npy")
        axs[i, j].pcolormesh(xedges, yedges, hist.T)
        axs[i, j].set_aspect("equal")
        axs[i, j].set_axis_off()
        for target in ["center", 0, 1, 2, 3, 4, 5]:
            pos = get_target_pos(target)
            if condition=='overt':
                c=colors["lightgray"] if target == "center" else colors["accent1"]
            elif condition=='covert':
                c=colors["accent1"] if target == "center" else colors["lightgray"]
            elif condition == 'free':
                c=colors["lightgray"]
            axs[i, j].scatter(
                pos[0],
                pos[1],
                s=50,
                c=c,
                marker="+",
                linewidth=0.5,
            )


# Step: Add a white scale bar to the top-left plot in the bottom-right corner
ax_top_left = axs[0, 0]  # Get the top-left subplot
scale_bar_length = 5
x_start = -17  # Starting position of the scale bar on the x-axis
y_start = -17  # Position on the y-axis near the bottom

# Add the white scale bar (a Rectangle) to the top-left plot
scale_bar = Rectangle(
    (x_start, y_start),
    scale_bar_length,
    0.5,
    linewidth=5,
    edgecolor="none",
    facecolor=colors["lightgray"],
)
ax_top_left.add_patch(scale_bar)

# Add a label indicating the length of the scale bar with 'degrees' unit
ax_top_left.text(
    x_start + scale_bar_length / 2,
    y_start + 0.5,
    f"{scale_bar_length}°",
    color=colors["lightgray"],
    ha="center",
    va="bottom",
)

fig.savefig("figures/patients/fig_gaze.pgf", **savefig_kws)
fig.savefig("figures/patients/fig_gaze.png", dpi=300, **savefig_kws)
