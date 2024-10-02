import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import colors, configure_matplotlib_style, savefig_kws

configure_matplotlib_style()

df = []
for pipeline in ["tLDA", "WCBLE", "XDAWNCov+TS+LDA"]:
    df_pipe = pd.read_csv(f"data/patients/decode/decode_results_{pipeline}.csv")
    df_pipe["pipeline"] = pipeline
    df.append(df_pipe)
df = pd.concat(df, ignore_index=True)
df = df[df["subject"] != "PD01"]
df = df[df["subject"] != "PB03"]

df_avg = df.copy()
df_avg["subject"] = "avg."
df = pd.concat([df, df_avg], ignore_index=True)

pdb.set_trace()

fg = sns.catplot(
    data=df,
    x="subject",
    y="test_score",
    hue="pipeline",
    kind="bar",
    col="condition",
    legend="auto",
    legend_out=False,
)
fg.set(ylim=(0, 1))
plt.legend(
    bbox_to_anchor=(-0.4, -0.3), loc="upper center", ncol=3
)  # Adjust ncol to fit all labels in one line
fg.axes[0, 0].legend_.remove()
for ax in fg.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0.5, color=colors["darkgray"], linestyle="--", linewidth=1)
fg.set_axis_labels("patient", "ROC-AUC")
fg.set_titles("{col_name} VSA")
fig = fg.figure
fig.set_size_inches(6.5, 2)
fig.savefig("figures/patients/fig_decode.pgf", **savefig_kws)
