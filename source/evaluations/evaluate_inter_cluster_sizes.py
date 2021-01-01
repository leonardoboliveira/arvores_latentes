from cort.core import corpora, mention_extractor
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    with open(r"z:\temp\inter_cluster_sizes.dmp", "wb") as f:
        pickle.dump(dfs, f)

    info = (dfs["next"] - dfs["first"])
    ax = np.clip(info, 0, 80).hist(bins=20)
    # ax = info.hist(bins=20)
    # ax.get_figure().savefig("z:/temp/teste.png")
    plt.xlabel("Number of mentions in cluster")
    plt.title("Cluster size frequency")
    print(f"Percentiles:{np.percentile(info, [0, 25, 50, 75, 100])}")
    plt.show()
    return dfs


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
    analize_clusters_in_path(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train")
