import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import colors, configure_matplotlib_style, save_pgf_trim

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

df = df.replace({
    'PA01': 'PA1',
    'PB01': 'PB1',
    'PB02': 'PB2',
    'PB04': 'PB4',
    'PC02': 'PC2',
    'PC03': 'PC3',
    'PC04': 'PC4',

})

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

fg._legend.set_frame_on(True)
fg._legend.get_frame().set_facecolor("white")
fg._legend.get_frame().set_alpha(1)  # Ensure opaque background (non-transparent)



for ax in fg.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0.5, color=colors["darkgray"], linestyle="--", linewidth=1)
fg.set_axis_labels("patient", "ROC-AUC")
fg.set_titles("{col_name} VSA")
fig = fg.figure
save_pgf_trim(fig, fg.axes[0,0], "figures/patients/fig_decode.pgf", height=2)
