import pdb

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import plot_topomap
from setup import (colors, configure_matplotlib_style, diverging_cmap,
                   save_pgf_trim, textwidth_in)

configure_matplotlib_style()
# plt.rcParams["axes.prop_cycle"] = "cycler('color', 'krb')"


for b in range(2):
    # Plot reconstructed contrast
    contrast_rec = mne.read_evokeds(f"data/bttda/forward/contrast_rec_block-{b}.fif")[0]
    contrast_rec.apply_baseline((None, 0))
    contrast_rec.nave = None

    fig, ax = plt.subplots(1, 1)
    contrast_rec.plot(axes=ax, show=False)
    ax.set_title("")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_yticks([-2, 0, 2])
    ax.set_ylim([-2, 2])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim([-0.2, 1])
    ax.axvline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
    ax.axhline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)

    fig.set_size_inches(0.3 * textwidth_in, 0.25 * textwidth_in)
    fig.savefig(
        f"figures/bttda/forward_block-{b}_contrast.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )

    # Plot temporal aps
    fig, ax = plt.subplots(1, 1)
    aps_tmp = np.load(f"data/bttda/forward/ap_block-{b}_mode-{1}.npy")
    ax.axvline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
    ax.axhline(0, color=colors["lightgray"], linestyle="--", linewidth=1, zorder=-1)
    ax.plot(contrast_rec.times, aps_tmp)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([str(int(t)) for t in ax.get_yticks()])
    ax.set_ylim([-1.5, 1.5])
    ax.set_ylabel("Scale")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([str(round(t, 1)) for t in ax.get_xticks()])
    ax.set_xlabel("Time (s)")
    ax.set_xlim([-0.2, 1])
    save_pgf_trim(
        fig,
        ax,
        f"figures/bttda/forward_block-{b}_ap-tmp.pgf",
        width=0.25 * textwidth_in,
        height=0.25 * textwidth_in,
    )

    # Plot spatial aps
    aps_sp = np.load(f"data/bttda/forward/ap_block-{b}_mode-{0}.npy")
    vmax = np.max(np.abs(aps_sp))
    for r in range(aps_sp.shape[-1]):
        fig, ax = plt.subplots(1, 1)
        plot_topomap(
            aps_sp[:, r],
            contrast_rec.info,
            axes=ax,
            show=False,
            cmap=diverging_cmap,
            vlim=(-vmax, vmax),
        )
        save_pgf_trim(
            fig,
            ax,
            f"figures/bttda/forward_block-{b}_ap-sp-{r}.pgf",
            width=0.15 * textwidth_in,
            height=0.15 * textwidth_in,
        )
