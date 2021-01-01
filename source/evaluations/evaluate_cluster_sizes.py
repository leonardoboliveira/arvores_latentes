from cort.core import corpora, mention_extractor
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    with open(r"z:\temp\cluster_sizes.dmp", "wb") as f:
        pickle.dump(dfs, f)

    info = ((dfs["end"] - dfs["start"] + 1) / dfs["count"])
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
