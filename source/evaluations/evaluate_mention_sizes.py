import os
import pandas as pd
from cort.core import corpora
import matplotlib.pyplot as plt
import numpy as np
import pickle


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    with open(r"z:\temp\mention_size.dmp", "wb") as f:
        pickle.dump(dfs, f)

    ax = np.clip(dfs["len"], 0, 20).hist(bins=10)
    # ax.get_figure().savefig("z:/temp/teste.png")
    plt.xlabel("Mention length")
    plt.title("Mention length frequency")
    plt.show()
    return dfs


def build_dataframe_in_path(path_in):
    dfs = []
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue
            dfs.append(get_flat_lens(os.path.join(r, file_name)))

    return pd.concat(dfs)


def get_flat_lens(file_path):
    doc_names = []
    sentence_id = []
    sentence_len = []
    for doc_name, lens in get_sentece_lens(file_path):
        for c_id, c_len in lens.items():
            doc_names.append(doc_name)
            sentence_id.append(c_id)
            sentence_len.append(c_len)
    return pd.DataFrame(
        {"doc": doc_names,
         "sentence_id": sentence_id,
         "len": sentence_len})


def get_sentece_lens(file_path):
    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            yield get_lens_in_doc(doc)


def get_lens_in_doc(doc):
    def calc_len(pos):
        span = doc.annotated_mentions[pos].span
        return span.end - span.begin + 1

    print(doc.identifier)
    clusters = {i: calc_len(i) for i in range(len(doc.annotated_mentions))}
    return doc.identifier, clusters


if __name__ == "__main__":
    analize_clusters_in_path(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train")
