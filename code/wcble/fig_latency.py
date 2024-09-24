import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/wcble/results_align.csv")
print(df)
sns.relplot(
    data=df, x="snr", y="rho", col="jitter", row="stc", hue="method", kind="line"
)
plt.savefig('figures/wcble/fig_latency.pdf')
