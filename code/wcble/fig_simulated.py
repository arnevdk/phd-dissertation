import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne import read_epochs
from setup import pagesize

epochs = read_epochs("data/wcble/simulated_epochs.fif")
meta = epochs.metadata

epochs.pick(["C3", "STIM"])
epochs.crop(-2, 2)

stcs = meta["waveform"].unique()
snrs = meta["snr"].unique()
# snrs = [0, -16, -31]
jitters = meta["jitter"].unique()


img_kwargs = dict(
    vmin=-50,
    vmax=50,
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


for i, stc in enumerate(stcs):
    shape = (2 * len(snrs), len(jitters))
    fig, axs = plt.subplots(
        *shape,
        sharex="col",
        sharey="row",
        # layout="tight",
        height_ratios=[3, 1] * len(snrs),
        figsize=pagesize,
    )
    axs = axs.reshape(shape)
    fig.suptitle(stc)

    for j, snr in enumerate(snrs):
        for k, jitter in enumerate(jitters):
            prefix = f"{stc}/{snr}/{jitter}"
            print(prefix)
            curr_epochs = epochs[prefix]

            ax_img = axs[j * 2, k]
            ax_ts = axs[(j * 2) + 1, k]

            order_ = order(curr_epochs)
            curr_epochs.plot_image(
                order=order_,
                colorbar=False,
                evoked=False,
                axes=ax_img,
                show=False,
                **img_kwargs,
            )
            label = f"SNR={snr}dB, $\sigma$={jitter}s"
            ax_img.text(
                0.05,
                0.95,
                label,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_img.transAxes,
            )
            ax_img.axhline(100, color="black", linestyle="--", linewidth=0.5)

            ax_img.set_title("")
            ax_img.set_xticks([])
            ax_img.set_xlabel("")
            ax_img.set_yticks([])

            # if k:
            #    ax_img.set_ylabel("")
            ax_img.set_ylabel("")

            x = curr_epochs.times
            y = curr_epochs["non-target"].pick("C3").get_data().squeeze() * 1e6
            df = pd.DataFrame(y.T, index=x)
            df.index.name = "time"
            df = df.melt(ignore_index=False, value_name="amplitude")
            sns.lineplot(data=df.reset_index(), x="time", y="amplitude", ax=ax_ts)
            y = curr_epochs["target"].pick("C3").get_data().squeeze() * 1e6
            df = pd.DataFrame(y.T, index=x)
            df.index.name = "time"
            df = df.melt(ignore_index=False, value_name="amplitude")
            sns.lineplot(data=df.reset_index(), x="time", y="amplitude", ax=ax_ts)

            ax_ts.axvline(0, color="black", linestyle="--", linewidth=0.5)

            # if k:
            ax_ts.set_ylabel("")
            ax_ts.set_yticks([])

            ax_ts.set_xticks([-2, -1, 0, 1, 2])
            if j == len(snrs) - 1:
                ax_ts.set_yticklabels([])
            else:
                ax_ts.set_xlabel("Time (s)")

    fig.savefig(f"figures/wcble/simulated-{stc}.pdf")
