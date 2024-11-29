import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import colors, configure_matplotlib_style, save_pgf_trim

configure_matplotlib_style()

df = pd.read_csv("data/patients/results_cross.csv")
df = df[df["subject"] != "PD01"]
df = df[df["subject"] != "PB03"]
print(df)

df_avg = df.copy()
df_avg["subject"] = "avg."
df = pd.concat([df, df_avg], ignore_index=True)

df = df.replace(
    {
        "PA01": "PA1",
        "PB01": "PB1",
        "PB02": "PB2",
        "PB04": "PB4",
        "PC02": "PC2",
        "PC03": "PC3",
        "PC04": "PC4",
        "overt": "overt VSA",
        "covert": "covert VSA",
        "free": "free VSA",
    }
)

df["test setting"] = df["cond_test"]
df["train setting"] = df["cond_train"]

df["test_score"] = df["test_score"] * 100

fg = sns.catplot(
    data=df,
    x="subject",
    y="test_score",
    hue="train setting",
    kind="bar",
    col="test setting",
    legend="auto",
    legend_out=False,
    col_order=["overt VSA", "covert VSA", "free VSA"],
)
fg.axes[0, 0].set_yticks([0, 20, 40, 60, 80, 100])
fg.axes[0, 0].set_yticklabels([str(int(t)) for t in fg.axes[0, 0].get_yticks()])
fg.set(ylim=(0, 100))
for ax in fg.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(50, color=colors["darkgray"], linestyle="--", linewidth=1)
fg.set_axis_labels("participant", "ROC-AUC (\\%)")
fg.set_titles("{col_name}")
legend = fg.axes[0, 0].get_legend()
legend.set_title("")
legend.set_frame_on(True)
legend.get_frame().set_linewidth(0)
legend.get_frame().set_boxstyle("Square")  # Remove rounded corners


fig = fg.figure
save_pgf_trim(fig, fg.axes[0, 0], "figures/patients/fig_cross.pgf", height=2)
