import matplotlib.pyplot as plt
import pdb
import pandas as pd
import seaborn as sns
from setup import configure_matplotlib_style,textwidth_in, save_pgf_trim

configure_matplotlib_style()


# Latency
df = pd.read_csv("data/wcble/results_align.csv")
df = df.replace({'tlda': 'tLDA', 'CBLE-median':'CBLE', 'WCBLE-median':'WCBLE'})
df = df[df["stc"] == "sine"]

facet_kws = dict(ylim=[0, 1], xlim=[0, -31], despine=False, legend_out=False,
                 )
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
save_pgf_trim(fg.fig, fg.axes[0,0], 'figures/wcble/fig_latency.pgf', height=2)


# Accuracy
df = pd.read_csv("data/wcble/results_classify.csv")
df = pd.concat(
    [df, pd.read_csv("data/wcble/results_classify_tlda.csv")], ignore_index=True
)
df = df[df["stc"] == "sine"]
df["accuracy"] *= 100

df = df.replace({'tlda': 'tLDA', 'CBLE-median':'CBLE', 'WCBLE-median':'WCBLE'})

facet_kws = dict(ylim=[0, 100], xlim=[0, -31], despine=False, legend_out=False)
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

save_pgf_trim(fg.fig, fg.axes[0,0], 'figures/wcble/fig_accuracy.pgf', height=2)
