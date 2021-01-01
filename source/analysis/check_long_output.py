import os
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_score(root, file_name):
    perf_re = re.compile(r"^.*:(\d+\.\d+).*$")

    with open(os.path.join(root, file_name), "r") as f:
        name = file_name.split(".")[0]
        epoch = file_name.split(".")[1]
        for line in f:
            if "Performance" in line:
                match = perf_re.match(line)
                perf = float(match.group(1))
                return name, int(epoch), perf

    return None, 0, 0


def get_stats(folder):
    stats = {}
    for file_name in tqdm(os.listdir(folder)):
        if not os.path.isfile(os.path.join(folder, file_name)):
            continue

        name, epoch, perf = get_score(folder, file_name)
        if name is not None:
            if epoch not in stats:
                stats[epoch] = {}
            stats[epoch][name] = perf
    return stats


def check_folder(folder):
    stats = get_stats(folder)
    df = pd.DataFrame(stats).T
    df = df.sort_index()
    df.rolling(5).mean().plot()
    plt.show()


if __name__ == "__main__":
    check_folder(r"D:\ProjetoFinal\output\long")
