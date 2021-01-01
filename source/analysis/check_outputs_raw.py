import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from evaluations.calc_conll import get_score


def get_stats(folder):
    rows = []
    for file_name in glob(f"{folder}/*.out"):
        if not os.path.isfile(file_name):
            continue

        score = get_score(file_name)
        if score is not None:
            rows.append(score)
    return pd.DataFrame(rows, columns=["name", "epoch", "step", "conll"])


def check_folder(folder):
    stats = get_stats(folder)
    # with open(r"stats.dmp", "wb") as f:
    #    pickle.dump(stats, f)

    df = stats
    df = df[((df["step"] + 1) % 6) == 0]
    df = df[df["epoch"] >= 0]
    df["composite_index"] = df["epoch"] * 1000 + df["step"]
    df = df.sort_values("composite_index")
    df_pivot = df.pivot("composite_index", "name", "conll")
    pivot = df_pivot.reset_index(drop=True)
    window = 10
    alpha = 0.1
    values = pivot.ewm(alpha=alpha, adjust=False).mean()
    _ = values.plot(figsize=(10, 5), style=['-', '--', '.', ':', '-.', '+'], color="k")

    # Vertical lines for epochs
    new_epoch = pivot.index[[False] + list(np.diff(df_pivot.index) > 800)]
    for x in new_epoch:
        _ = plt.axvline(x, linestyle=':', color="grey")

    _ = plt.title(f"Performance no dataset de mini-validação")
    _ = plt.xlabel("Etapa")
    _ = plt.ylabel(r"Score CoNLL - Média móvel exponencial $\alpha$={}".format(alpha))
    _ = plt.legend(["baseline", "lexico_efi", "glove_efi", "bert_surface", "span_surface_2E6", "span_surface"])
    # _ = plt.legend(["glove_efi", "glove_surface"])
    plt.show()

    _ = pivot["baseline"].plot(figsize=(10, 5), color="k")
    for x in new_epoch:
        _ = plt.axvline(x, linestyle=':', color="grey")
    _ = plt.title("Performance no dataset de mini-validação - Baseline")
    _ = plt.xlabel("Etapa")
    _ = plt.ylabel(f"Score CoNLL")

    plt.show()


if __name__ == "__main__":
    check_folder(r"D:\ProjetoFinal\avaliacao")
