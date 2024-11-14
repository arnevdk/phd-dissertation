import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import colors, configure_matplotlib_style, save_pgf_trim

configure_matplotlib_style()

df = pd.read_csv('data/patients/results_cross.csv')
df = df[df["subject"] != "PD01"]
df = df[df["subject"] != "PB03"]
print(df)

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
    'overt': 'overt VSA',
    'covert': 'covert VSA',
    'free': 'free VSA',
})

df['test setting'] = df['cond_test']
df['train setting'] = df['cond_train']

fg = sns.catplot(
    data=df,
    x="subject",
    y="test_score",
    hue="train setting",
    kind="bar",
    col="test setting",
    legend="auto",
    legend_out=False,
    col_order=['overt VSA', 'covert VSA', 'free VSA']
)
fg.set(ylim=(0, 1))
for ax in fg.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0.5, color=colors["darkgray"], linestyle="--", linewidth=1)
fg.set_axis_labels("patient", "ROC-AUC")
fg.set_titles("{col_name}")


fg._legend.set_frame_on(True)
fg._legend.get_frame().set_facecolor("white")
fg._legend.get_frame().set_alpha(1)  # Ensure opaque background (non-transparent)

fig = fg.figure
save_pgf_trim(fig, fg.axes[0,0], "figures/patients/fig_cross.pgf", height=2)
