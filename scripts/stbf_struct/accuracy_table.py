import os

import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from setup import (colors, configure_matplotlib_style, save_pgf_trim,
                   textwidth_in)

configure_matplotlib_style()

iteration_out_path = "data/stbf_struct/iterations.csv"


classifier_models = ["stbf_kron_toep_shrunk", "stbf_loocv", "stbf_emp", "xdawn_rg"]
train_blocks_range = range(1, 9 + 1)

data_path = "data/stbf_struct/accuracy.csv"

df = pd.read_csv(data_path)
df["accuracy"] = df["accuracy"] * 100
df = df.groupby(["classifier_model", "n_runs", "n_train_blocks"])
df = df["accuracy"].aggregate("mean")
df = df.reset_index()

for model in df["classifier_model"].unique():
    model_df = df[df["classifier_model"] == model]
    model_df = model_df.pivot(
        columns="n_train_blocks", index="n_runs", values="accuracy"
    )
    model_df.index.names = ["\\# trials"]
    model_df.columns.names = ["\\# training blocks"]
    model_df.to_latex(
        f"chapters/stbf_struct-acc-{model}.tex",
        header=True,
        index=True,
        index_names=True,
        float_format="%.2f",
        column_format="@{}l" + "c" * 9 + "@{}",
    )
    print(model_df)
