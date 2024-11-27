import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mcbf.alg.covariance import (KroneckerToeplitzCovariance,
                                 SpatiotemporalEmpiricalCovariance,
                                 SpatiotemporalLOOCVShrunkCovariance)
from mne import read_epochs
from mne.decoding import Scaler
from setup import (colors, configure_matplotlib_style, diverging_cmap,
                   textwidth_in)

configure_matplotlib_style()

# Read and process data
epochs = read_epochs("data/stbf_struct/sub-01_task-oddball_proc-32hz_eeg.fif")
epochs = epochs[:100]
epochs.pick_channels(["Fz", "Cz", "Pz", "Oz"])
epochs.crop(tmin=0.1, tmax=0.6)
scaler = Scaler(scalings="median", with_mean=False)
epochs_data = scaler.fit_transform(epochs.get_data())

# Calculate covariances
emp_cov = SpatiotemporalEmpiricalCovariance()
emp_cov.fit(epochs_data)
shrunk_cov = SpatiotemporalLOOCVShrunkCovariance()
shrunk_cov.fit(epochs_data)
kron_toep_cov = KroneckerToeplitzCovariance(decimate=1)
kron_toep_cov.fit(epochs_data)


def plot_cov(cov, i, figsize):
    heatmap_kws = dict(
        square=True,
        center=0,
        cmap=diverging_cmap,
        cbar=False,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(cov, ax=ax, **heatmap_kws)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines[["top", "bottom", "left", "right"]].set_visible(True)
    fig.savefig(
        f"figures/stbf_struct/covs-{i}.eps", bbox_inches="tight", pad_inches=0.05
    )


large_covs = [
    emp_cov.covariance_,
    shrunk_cov.covariance_,
    kron_toep_cov.covariance_,
    emp_cov.precision_,
    shrunk_cov.precision_,
    kron_toep_cov.precision_,
]
small_covs = [
    kron_toep_cov.spatial_.covariance_,
    kron_toep_cov.spatial_.precision_,
    kron_toep_cov.temporal_.covariance_,
    kron_toep_cov.temporal_.precision_,
]

for i, cov in enumerate(large_covs):
    plot_cov(cov, i, (textwidth_in / 4, textwidth_in / 4))

for i, cov in enumerate(small_covs):
    plot_cov(cov, i + len(large_covs), (textwidth_in / 10, textwidth_in / (4 * 3)))
