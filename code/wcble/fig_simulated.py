import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne import read_epochs
from setup import configure_matplotlib_style, darkgray, pagesize, savefig_kws

configure_matplotlib_style()

epochs = read_epochs("data/wcble/simulated_epochs.fif")
meta = epochs.metadata

epochs.pick(["C3", "STIM"])
epochs.crop(-2, 2)

snrs = meta["snr"].unique()
# snrs = [0, -16, -31]
jitters = meta["jitter"].unique()


img_kwargs = dict(
    vmin=-60,
    vmax=60,
    picks="C3",
)


def latency(epochs):
    ground_truth = np.zeros(len(epochs))
    for i, epoch in enumerate(epochs):
        stim_chan = epoch[-1, :]
        ground_truth[i] = np.nonzero(stim_chan)[0]
    ground_truth = ground_truth / epochs.info["sfreq"] + epochs.tmin
    return ground_truth


def order(epochs, n_epochs=100):
    latency_ = latency(epochs)
    order = np.zeros(len(epochs), dtype=int)
    order[:n_epochs] = np.argsort(latency_[:n_epochs])
    order[n_epochs:] = np.arange(n_epochs) + n_epochs
    return order


def plot_img(ax, epochs):
    order_ = order(epochs)
    epochs.plot_image(
        order=order_,
        colorbar=False,
        evoked=False,
        axes=ax,
        show=False,
        **img_kwargs,
    )
    ax_img.axhline(100, linestyle="--", linewidth=0.5, color=darkgray)

    ax_img.set_title("")
    ax_img.set_xticks([])
    ax_img.set_xlabel("")
    ax_img.set_yticks([])


def plot_ts(ax, epochs):
    x = epochs.times
    y = epochs["non-target"].pick("C3").get_data().squeeze() * 1e6
    df = pd.DataFrame(y.T, index=x)
    df.index.name = "time"
    df = df.melt(ignore_index=False, value_name="amplitude")
    sns.lineplot(data=df.reset_index(), x="time", y="amplitude", ax=ax_ts)
    y = epochs["target"].pick("C3").get_data().squeeze() * 1e6
    df = pd.DataFrame(y.T, index=x)
    df.index.name = "time"
    df = df.melt(ignore_index=False, value_name="amplitude")
    sns.lineplot(data=df.reset_index(), x="time", y="amplitude", ax=ax_ts)


stc = "sine"

height_ratios = []
for _ in range(len(snrs)):
    height_ratios.extend([3, 1])  # Closer together for image+timeseries
    height_ratios.append(0.75)
gridspec_kw = dict(height_ratios=height_ratios, hspace=0)
shape = (len(height_ratios), len(jitters))
fig, axs = plt.subplots(
    *shape,
    sharex="col",
    sharey="row",
    gridspec_kw=gridspec_kw,
    figsize=pagesize,
)
axs = axs.reshape(shape)

# Hide every third row that is used as a spacer
for i in range(2, len(axs), 3):
    for j in range(len(jitters)):  # Loop over columns (since axs is 2D)
        axs[i, j].axis("off")

for j, snr in enumerate(snrs):
    for k, jitter in enumerate(jitters):
        prefix = f"{stc}/{snr}/{jitter}"
        print(prefix)
        curr_epochs = epochs[prefix]

        ax_img = axs[j * 3, k]
        ax_ts = axs[(j * 3) + 1, k]

        plot_img(ax_img, curr_epochs)
        label = f"SNR={snr}dB, $\sigma$={jitter}s"
        ax_img.set_title(label)

        if k:
            ax_img.set_ylabel("")
        else:
            ax_img.set_ylabel("epochs")

        plot_ts(ax_ts, curr_epochs)
        # if k:
        ax_ts.set_ylabel("")
        ax_ts.set_ylim([-10, 10])
        ax_ts.set_yticks([])

        ax_ts.set_xticks([-2, -1, 0, 1, 2])
        if j == len(snrs) - 1:
            ax_ts.set_yticklabels([])
        else:
            ax_ts.set_xlabel("Time (s)")

fig.savefig(f"figures/wcble/simulated-{stc}.pdf", **savefig_kws)
