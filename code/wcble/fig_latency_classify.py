import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import configure_matplotlib_style, savefig_kws, set_size

configure_matplotlib_style()
figsize = set_size(subplots=(1, 3))


# Latency
df = pd.read_csv("data/wcble/results_align.csv")
df = df[df["stc"] == "sine"]

facet_kws = dict(ylim=[0, 1], xlim=[0, -31])
fg = sns.relplot(
    data=df,
    x="snr",
    y="rho",
    col="jitter",
    # row="stc",
    hue="method",
    kind="line",
    facet_kws=facet_kws,
)
fg.set_titles(col_template="$\\sigma={col_name}$s")
fg.set_xlabels("SNR (dB)")
fg.set_ylabels("$\\rho$")
fg.figure.set_size_inches(*figsize)
# fg.tight_layout()

plt.savefig("figures/wcble/fig_latency.pgf", bbox_inches="tight", pad_inches=0)

# Accuracy
df = pd.read_csv("data/wcble/results_classify.csv")
df = pd.concat(
    [df, pd.read_csv("data/wcble/results_classify_tlda.csv")], ignore_index=True
)
df = df[df["stc"] == "sine"]
df["accuracy"] *= 100

facet_kws = dict(ylim=[0, 100], xlim=[0, -31])
fg = sns.relplot(
    data=df,
    x="snr",
    y="accuracy",
    col="jitter",
    # row="stc",
    hue="method",
    kind="line",
    facet_kws=facet_kws,
)
fg.set_titles(col_template="$\\sigma={col_name}$s")
fg.set_xlabels("SNR (dB)")
fg.set_ylabels("accuracy (%)")
fg.figure.set_size_inches(*figsize)
# fg.tight_layout()


# Force drawing of the figure to get the correct bounding boxes
plt.draw()
fig = fg.figure
# Get the bounding box of the axis, in figure coordinatesσ = 0.1s
bbox = fig.axes[0].get_tightbbox(fig.canvas.get_renderer())
# Convert the bounding box units to figure-relative coordinates
bbox_in_fig_units = bbox.transformed(fig.transFigure.inverted())
# Calculate the amount of space required to align the left spine with the figure border
left_margin_required = bbox_in_fig_units.x0
# Adjust the figure layout automatically by adding enough left margin
fig.subplots_adjust(left=left_margin_required)

fig.savefig("figures/wcble/fig_accuracy.pgf", format="pgf", **savefig_kws)
