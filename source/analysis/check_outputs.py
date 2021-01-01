import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def get_score(file_name):
    conll = None
    recall = []
    precision = []

    with open(file_name, "r") as f:
        name = os.path.basename(file_name).split(".")[0]
        for line in f:
            if "conll" in line:
                conll = float(line.split("\t")[-1])
            if "muc" in line or \
                    "bcub" in line or \
                    "ceafm" in line or \
                    "blanc" in line:
                stats = line.split("\t")
                recall.append(float(stats[1]))
                precision.append(float(stats[2]))

    if conll is None:
        return None, None, None, None
    return name, np.mean(recall), np.mean(precision), conll


def get_stats(folder):
    stats = {}
    for file_name in glob(f"{folder}/*.out"):
        if not os.path.isfile(file_name):
            continue

        name, r, p, c = get_score(file_name)
        if name is not None:
            if name not in stats:
                stats[name] = []
            stats[name].append((r, p, c))
    return stats


def check_folder(folder):
    stats = get_stats(folder)
    names = list(stats.keys())

    names.sort(key=lambda x: np.mean([y[-1] for y in stats[x]]))

    r1 = np.arange(len(names))
    barWidth = 0.3

    conll = None
    conll_errs = None
    conll_max = None
    for idx, name in enumerate(["recall", "precision", "conll"]):
        r2 = [x + barWidth * idx for x in r1]
        values = [np.mean([y[idx] for y in stats[x]]) for x in names]
        errs = [np.std([y[idx] for y in stats[x]]) for x in names]
        plt.barh(r2, values, xerr=errs, label=name, height=barWidth)
        if name == "conll":
            conll = values
            conll_errs = errs
            conll_max = [np.max([y[idx] for y in stats[x]]) for x in names]

    plt.yticks([x + barWidth for x in r1], labels=names)
    for i, v in enumerate(conll):
        plt.text(v + 10, i + .5, "{:1.2f} +- {:1.2f} [{:1.2f}]".format(v, conll_errs[i], conll_max[i]), color='blue',
                 fontweight='bold')

    plt.gcf().subplots_adjust(left=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    check_folder(r"Z:\temp\a")
