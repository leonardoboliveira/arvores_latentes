import os
import pandas as pd
import matplotlib.pyplot as plt


def get_info(root):
    df = pd.DataFrame(list(check_folder(root)), columns=["name", "epoch", "step", "time"])
    df["grand_step"] = (df["epoch"] * 1000) + df["step"]
    pivot = df.pivot(index="grand_step", columns="name", values="time")
    # pivot = pivot.ffill()
    intervals = pivot.diff().dropna()
    return intervals


def check_folder(root):
    for file in os.listdir(root):
        if "predicted" not in file:
            continue

        spliited = file.split(".")
        if len(spliited) < 3:
            continue
        name = spliited[0]
        epoch = int(spliited[1])
        step = int(spliited[2])
        time = os.path.getmtime(f"{root}/{file}")
        yield name, epoch, step, time


def clean(df):
    df = df[df.index > 1000]
    mean = df.mean()
    std = df.std()
    df = df[df < mean + 3 * std]
    df = df[df > mean - 3 * std]
    return df


if __name__ == "__main__":
    df = get_info(r"d:\ProjetoFinal\output\times")
    clean(df).hist(bins=15)
    plt.show()
    print("Done")
