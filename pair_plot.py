import sys

import pandas as pd
import seaborn as sns


def main(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop(df.columns[[1, 2, 3, 4]], axis=1)
    pg = sns.pairplot(
        df,
        diag_kind="hist",
        hue="Hogwarts House",
        palette={
            "Gryffindor": "red",
            "Hufflepuff": "yellow",
            "Ravenclaw": "blue",
            "Slytherin": "green",
        },
    )
    pg.savefig("images/pair_plot.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python3 pair_plot.py dataset/dataset_train.csv`."
        )
        exit(1)

    main(sys.argv[1])
