import os
import pandas as pd
from cort.core import corpora, mention_extractor
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pickle


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    with open(r"z:\temp\mention_size.dmp", "wb") as f:
        pickle.dump(dfs, f)

    values = dfs["len"]
    count, bins = np.histogram(values, range=(0, 6), bins=list(range(0, 8)))
    x = range(len(count) - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    _ = ax.bar(x, count[1:], color="grey")
    _ = plt.xticks(x, [f"{x:.0f}" for x in bins[1:-2]] + ['6+'])
    ax2 = ax.twinx()
    _ = ax2.plot(np.cumsum(count / sum(count))[1:], color="k")
    _ = ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    _ = plt.title("Tamanho das menções em palavras")
    _ = plt.xlabel("Número de palavras na menção")
    _ = ax.set_ylabel("Número de menções")
    _ = ax2.set_ylabel("Acumulado do total de menções")
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
    def calc_len(mention):
        span = mention.span
        return span.end - span.begin + 1

    print(doc.identifier)
    clusters = {i: calc_len(mention) for i, mention in enumerate(mention_extractor.extract_system_mentions(doc)) if
                not mention.is_dummy()}
    return doc.identifier, clusters


if __name__ == "__main__":
    analize_clusters_in_path(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train")
