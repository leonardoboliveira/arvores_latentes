import os
import pandas as pd
from cort.core import corpora, mention_extractor
import matplotlib.pyplot as plt
import pickle


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    with open(r"z:\temp\mention_size.dmp", "wb") as f:
        pickle.dump(dfs, f)

    if "len" not in dfs:
        print("Error")

    ax = dfs["len"].hist(bins=10)
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
    mention_id = []
    distance = []
    for doc_name, distances in get_mention_distances(file_path):
        for c_id, c_dist in distances.items():
            doc_names.append(doc_name)
            mention_id.append(c_id)
            distance.append(c_dist)
    return pd.DataFrame(
        {"doc": doc_names,
         "mention_id": mention_id,
         "distance": distance})


def get_mention_distances(file_path):
    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            yield get_lens_in_doc(doc)


def get_lens_in_doc(doc):
    def calc_len(mention):
        span = mention.span
        return span.end - span.begin + 1

    # mentions = doc.annotated_mentions
    mentions = mention_extractor.extract_system_mentions(doc)
    print(doc.identifier)
    sizes = {i: calc_len(mention) for i, mention in enumerate(mentions) if not mention.is_dummy()}
    return doc.identifier, sizes


if __name__ == "__main__":
    analize_clusters_in_path(r"D:\gdrive\puc\projeto final\Datasets\conll\train")
