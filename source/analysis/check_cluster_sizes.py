from cort.core import corpora, mention_extractor
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    values = dfs["end"] - dfs["start"]
    count, bins = np.histogram(values, range=(0, 300))
    x = range(len(count))

    fig, ax = plt.subplots(figsize=(10, 5))
    _ = ax.bar(x, count, color="grey")
    _ = plt.xticks(x, [f"{x:.0f}" for x in bins[1:-1]] + ['300+'])
    ax2 = ax.twinx()
    _ = ax2.plot(np.cumsum(count / sum(count)), color="k")
    _ = ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    _ = plt.title("Histograma de distâncias entre menções no cluster")
    _ = plt.xlabel("Número máximo de menções dentro do cluster")
    _ = ax.set_ylabel("Número de clusters")
    _ = ax2.set_ylabel("Acumulado do total de clusters")
    plt.show()


def build_dataframe_in_path(path_in):
    dfs = []
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue
            dfs.append(get_flat_sizes(os.path.join(r, file_name)))

    return pd.concat(dfs)


def get_flat_sizes(file_path):
    doc_names = []
    cluster_id = []
    cluster_start = []
    cluster_end = []
    cluster_count = []
    for doc_name, clusters in get_cluster_sizes(file_path):
        for c_id, c_range in clusters.items():
            c_start, c_end, c_count = c_range
            doc_names.append(doc_name)
            cluster_id.append(c_id)
            cluster_start.append(c_start)
            cluster_end.append(c_end)
            cluster_count.append(c_count)
    return pd.DataFrame(
        {"doc": doc_names,
         "ids": cluster_id,
         "start": cluster_start,
         "end": cluster_end,
         "count": cluster_count})


def get_clusters_in_doc(doc):
    clusters = {}
    for idx, mention in enumerate(mention_extractor.extract_system_mentions(doc)):
        cluster_id = mention.attributes["annotated_set_id"]
        if cluster_id not in clusters:
            clusters[cluster_id] = [idx, idx, 1]
        else:
            clusters[cluster_id][1] = idx
            clusters[cluster_id][2] += 1
    return doc.identifier, clusters


def get_cluster_sizes(file_path):
    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            yield get_clusters_in_doc(doc)


if __name__ == "__main__":
    analize_clusters_in_path(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train")
