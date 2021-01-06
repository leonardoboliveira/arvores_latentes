import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import interp1d
import os
import pickle
import sys


def get_frequencies(file_name, bins):
    iter_csv = pd.read_csv(file_name, header=None, chunksize=1000)
    base_df = None
    for df in iter_csv:
        df = df[df[0] != "-"]
        df = df.iloc[:, 2:]
        if len(df) == 0:
            continue
        # print(f"{df.shape} {len(bins)}")
        assert max(bins) >= df.max().max(), f"Max in bins:{max(bins)}, max in df: {df.max().max()}"
        assert min(bins) <= df.min().min(), f"Min in bins:{min(bins)}, min in df: {df.min().min()}"

        ret = df.apply(lambda x: np.histogram(x, bins=bins)[0], axis=0).reset_index(drop=True)

        if base_df is None:
            base_df = ret
        else:
            base_df += ret

        if base_df.isnull().sum().sum() > 0:
            print("Something is not ok")

    return base_df


def get_frequencies_for_all(files, bins):
    base = None
    for file_name in tqdm(files):
        if base is None:
            base = get_frequencies(file_name, bins)
        else:
            base += get_frequencies(file_name, bins)

    return base


def plot(freq, idx):
    idx = 2
    ax = freq[idx][-1:1].plot(color="grey")
    ax2 = (freq[idx][-1:1].cumsum() / freq[idx][-1:1].sum()).plot(ax=ax, secondary_y=True, color="k")
    ax2.axhline(0.2, color='k', linestyle="--")
    ax2.axhline(0.3, color='k', linestyle="--")
    _ = ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))


def plot_hist(counts, detail_bins):
    detail_bins = detail_bins[:-1]
    bin_count = []

    prev_bound = -10000
    x_values = np.arange(-2, 2, 0.2)
    for next_bound in x_values:
        mask = (detail_bins >= prev_bound) & (detail_bins < next_bound)
        bin_count.append(counts[mask].sum())
        prev_bound = next_bound

    mask = (detail_bins >= prev_bound)
    bin_count.append(counts[mask].sum())

    bin_count = np.array(bin_count)

    x_values = np.concatenate((x_values, [2]))
    names = ["{:.2f}".format(x) for x in x_values]
    r1 = np.arange(len(names))
    barWidth = 0.8 / counts.shape[1]

    xnew = np.linspace(-2, 2, num=100, endpoint=True)

    for idx in range(counts.shape[1]):
        r2 = [x + barWidth * idx for x in r1]
        # plt.bar(r2, bin_count[:, idx], width=barWidth)
        f2 = interp1d(x_values, bin_count[:, idx], kind='cubic')
        plt.plot(xnew, f2(xnew), label=idx, markevery=1)

    # plt.xticks([x + barWidth for x in r1], labels=names, rotation='vertical')
    # plt.gcf().subplots_adjust(left=0.3)
    plt.title("Distribution for some features")
    plt.xlabel("Feature Value")
    plt.ylabel("Number of Occurrences")
    plt.legend()
    plt.show()


def get_distribution(files, total_bins):
    detail_bins = np.arange(-20, 20, 0.001)
    frequencies = get_frequencies_for_all(files, detail_bins)

    acc_frequencies = frequencies / frequencies.sum()
    acc_frequencies = acc_frequencies.cumsum()
    values = []
    step = 1 / total_bins

    for bin in range(total_bins):
        new_cum_count = (acc_frequencies <= (bin + 1) * step).sum()
        this_values = [detail_bins[x] for x in new_cum_count]
        values.append(this_values)

    return np.array(values)


def get_files(path):
    for r, d, f in os.walk(path):
        for file_name in f:
            yield os.path.join(r, file_name)


if __name__ == "__main__":
    # files = [r"C:\Users\loliveira\Desktop\train.span"]
    f_in = r"D:\ProjetoFinal\data\train\spanbert"
    f_out = r"D:\GDrive\Puc\Projeto Final\Code\extra_files\deciles_span.csv"

    if len(sys.argv) == 3:
        _, f_in, f_out = sys.argv

    dist = get_distribution(get_files(f_in), 10)
    np.savetxt(f_out, dist, delimiter=",")

    if "DEBUG" in os.environ:
        plt.plot(dist.transpose())
        plt.show()
