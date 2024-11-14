from setup import colors, configure_matplotlib_style, pagesize, primary_cmap, secondary_cmap

from mcbf.alg.covariance import SpatiotemporalEmpiricalCovariance, \
    SpatiotemporalLOOCVShrunkCovariance, KroneckerToeplitzCovariance
from mne import read_epochs
from mne.decoding import Scaler
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

configure_matplotlib_style()

epochs = read_epochs('data/stbf_struct/sub-01_task-oddball_proc-32hz_eeg.fif')
epochs = epochs[:100]
epochs.pick_channels(['Fz', 'Cz', 'Pz', 'Oz'])
epochs.crop(tmin=0.1, tmax=0.6)
scaler = Scaler(scalings='median', with_mean=False)
epochs_data = scaler.fit_transform(epochs.get_data())


cov_plot_kwargs = dict(cmap=secondary_cmap, vmin=-1, vmax=1)
inv_cov_plot_kwargs = dict(cmap=secondary_cmap)
savefig_kwargs = dict(format='eps')

emp_cov = SpatiotemporalEmpiricalCovariance()
emp_cov.fit(epochs_data)
shrunk_cov = SpatiotemporalLOOCVShrunkCovariance()
shrunk_cov.fit(epochs_data)
kron_toep_cov = KroneckerToeplitzCovariance(decimate=1)
kron_toep_cov.fit(epochs_data)
print(f'Shrinkage: {shrunk_cov.shrinkage_}')

fig = plt.figure(constrained_layout=True, figsize=(11.1,5))
gs = GridSpec(4, 4, figure=fig,
              width_ratios=[2 / 7, 2 / 7, 2 / 7, 1 / 7])

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 12

def plot_cov_pub(data, ax, label, **kwargs):
    data = np.flip(data, axis=1)
    opt = np.max(np.abs(data))
    kwargs.setdefault('vmin', -opt)
    kwargs.setdefault('vmax', opt)
    im = ax.pcolormesh(data, cmap=secondary_cmap, **kwargs)
    ax.set_aspect('equal')
    #fig.colorbar(im, ax=ax)
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    #ax.text(0.07, 1.15, label, transform=ax.transAxes,
    #        fontsize=16, fontweight='bold', va='top', ha='right')
    ax.set_title(label, loc='left', fontsize=16, fontweight='bold')
    return im

cov_kwargs = dict(vmin=-1.1, vmax=1.1)
pad=6

ax = fig.add_subplot(gs[:2, 0])
plot_cov_pub(emp_cov.covariance_, ax=ax, label='A', **cov_kwargs)
#ax.annotate('Empirical', xy=(0.5, 1), xytext=(0, pad),
#            xycoords='axes fraction', textcoords='offset points',
#            size='large', ha='center', va='baseline')
ax = fig.add_subplot(gs[:2, 1])
plot_cov_pub(shrunk_cov.covariance_, ax=ax, label='C', **cov_kwargs)
#ax.annotate('Shrunk', xy=(0.5, 1), xytext=(0, pad),
#            xycoords='axes fraction', textcoords='offset points',
#            size='large', ha='center', va='baseline')
ax = fig.add_subplot(gs[:2, 2])
plot_cov_pub(kron_toep_cov.covariance_, ax=ax, label='E', **cov_kwargs)
#ax.annotate('Structured', xy=(0.5, 1), xytext=(0, pad),
#            xycoords='axes fraction', textcoords='offset points',
#            size='large', ha='center', va='baseline')


ax = fig.add_subplot(gs[2:, 0])
plot_cov_pub(emp_cov.precision_, ax=ax, label='B')
ax = fig.add_subplot(gs[2:, 1])
plot_cov_pub(shrunk_cov.precision_, ax=ax, label='D')
ax = fig.add_subplot(gs[2:, 2])
plot_cov_pub(kron_toep_cov.precision_, ax=ax, label='F')

ax = fig.add_subplot(gs[0, 3])
plot_cov_pub(kron_toep_cov.spatial_.covariance_, ax=ax, label='G', **cov_kwargs)
ax = fig.add_subplot(gs[1, 3])
plot_cov_pub(kron_toep_cov.spatial_.precision_, ax=ax, label='H')
ax = fig.add_subplot(gs[2, 3])
plot_cov_pub(kron_toep_cov.temporal_.covariance_, ax=ax, label='I', **cov_kwargs)
ax = fig.add_subplot(gs[3, 3])
plot_cov_pub(kron_toep_cov.temporal_.precision_, ax=ax, label='J')

fig.savefig('figures/stbf_struct/covs.eps', format='eps', bbox_inches='tight')
