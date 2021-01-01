import os
import pandas as pd
from cort.core import corpora, mention_extractor
import pickle


def analize_clusters_in_path(path_in):
    dfs = build_dataframe_in_path(path_in)
    true_positives = dfs["intersection"].sum()
    false_positives = dfs["system"].sum()
    false_negative = dfs["annotated"].sum()

    precision = (true_positives) / (true_positives + false_positives)
    recall = (true_positives) / (true_positives + false_negative)

    no_mention = ((dfs["intersection"] == 0) & (dfs["annotated"] == 0)).sum()
    no_mention = no_mention / len(dfs)

    print(f"Total:{true_positives + false_negative}")
    print(f"Extracted:{true_positives + false_positives}")
    print(f"Correct:{true_positives}")
    print(f"Precision:{precision}, Recall:{recall}, No Mentions:{no_mention}")

    with open(r"z:/temp/dfs.dmp", "wb") as f:
        pickle.dump(dfs, f)


def build_dataframe_in_path(path_in):
    dfs = []
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue
            dfs.append(check_file(os.path.join(r, file_name)))

    return pd.concat(dfs)


def check_doc(doc):
    system = set(mention_extractor.extract_system_mentions(doc))
    annotated = set(doc.annotated_mentions)

    only_system = len(system - annotated)
    only_annotated = len(annotated - system)
    intersection = len(system.intersection(annotated))

    return only_system, only_annotated, intersection


def check_file(file_path):
    ids = []
    only_system = []
    only_annotated = []
    intersec = []

    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            ids.append(doc.identifier)
            system, annotated, inter = check_doc(doc)
            only_system.append(system)
            only_annotated.append(annotated)
            intersec.append(inter)

    return pd.DataFrame({"id": ids, "system": only_system, "annotated": only_annotated, "intersection": intersec})


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
