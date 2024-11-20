import pandas as pd

for dataset in ["CVSA-ERP", "BNCI2014-009"]:
    df = pd.read_csv(f"data/covert_align/results_{dataset}_cross.csv")

    replace = {
        "S": "overt",
        "G": "covert",
        "split/1": "split ($d=1$)",
        "split/2": "split ($d=2$)",
        "split/3": "split ($d=3$)",
        "mix": "combined",
    }
    df["train_cond"].replace(replace, inplace=True)
    df["test_cond"].replace(replace, inplace=True)
    if dataset == "CVSA-ERP":
        categories = [
            "overt",
            "covert",
            "split ($d=1$)",
            "split ($d=2$)",
            "split ($d=3$)",
            "combined",
        ]
    else:
        categories = [
            "overt",
            "covert",
            "combined",
        ]

    df["train_cond"] = pd.Categorical(
        df["train_cond"], categories=categories, ordered=True
    )
    df["test_cond"] = pd.Categorical(
        df["test_cond"], categories=categories, ordered=True
    )

    df = df.groupby(["train_cond", "model", "test_cond"])["roc_auc"].aggregate("mean")
    df = df.reset_index()
    df["roc_auc"] *= 100
    df = df.pivot(index=["model", "train_cond"], columns="test_cond", values="roc_auc")
    df.index.names = ["classifier", "train"]
    df.columns.names = ["test"]
    df = df.dropna()
    df.to_latex(
        f"chapters/covert_align_cross-{dataset}.tex",
        header=True,
        index=True,
        index_names=True,
        float_format="%.2f",
        column_format="@{}ll" + "c" * len(df.columns) + "@{}",
    )
    print(df)
