from cort.core import corpora, mention_extractor
import pandas as pd
import os
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    x = dfs.dropna()
    values = x["next"] - x["first"]
    count, bins = np.histogram(values, range=(0, 50))
    x = range(len(count))

    fig, ax = plt.subplots(figsize=(10, 5))
    _ = ax.bar(x, count, color="grey")
    _ = plt.xticks(x, [f"{x:.0f}" for x in bins[1:-1]] + ['50+'])
    ax2 = ax.twinx()
    _ = ax2.plot(np.cumsum(count / sum(count)), color="k")
    _ = ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    _ = plt.title("Histograma de distâncias entre menções de um mesmo cluster")
    _ = plt.xlabel("Distância entre menções")
    _ = ax.set_ylabel("Número de menções")
    _ = ax2.set_ylabel("Acumulado do total de menções")
    plt.show()


def build_dataframe_in_path(path_in):
    dfs = []
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if "conll" not in file_name:
                continue
            dfs.append(get_flat_sizes(os.path.join(r, file_name)))

    return pd.concat(dfs)


def get_flat_sizes(file_path):
    doc_names = []
    cluster_id = []
    cluster_start = []
    cluster_end = []

    for doc_name, clusters in get_cluster_sizes(file_path):
        for c_id, c_first, c_next in clusters:
            doc_names.append(doc_name)
            cluster_id.append(c_id)
            cluster_start.append(c_first)
            cluster_end.append(c_next)
    return pd.DataFrame(
        {"doc": doc_names,
         "ids": cluster_id,
         "first": cluster_start,
         "next": cluster_end})


def get_clusters_in_doc(doc):
    last_in_clusters = {}
    steps = []
    for idx, mention in enumerate(mention_extractor.extract_system_mentions(doc)):
        cluster_id = mention.attributes["annotated_set_id"]
        if cluster_id in last_in_clusters:
            steps.append([cluster_id, last_in_clusters[cluster_id], idx])

        last_in_clusters[cluster_id] = idx
    return doc.identifier, steps


def get_cluster_sizes(file_path):
    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            yield get_clusters_in_doc(doc)


if __name__ == "__main__":
    path = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train"
    analize_clusters_in_path(r"D:\ProjetoFinal\data\debug")
