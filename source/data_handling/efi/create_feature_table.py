import os
from cort.core import corpora, mention_extractor
from cort.util import import_helper
from cort.coreference.cost_functions import cost_based_on_consistency
from extension.instance_extractors import InstanceExtractor, split_features
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_mention_features(mention, mention_features):
    features = {}
    if mention.is_dummy():
        return features
    o_features, bert_features = split_features([feature(mention) for feature in mention_features])
    for name, value in o_features:
        features[name] = value

    for name, values in bert_features:
        for idx, value in enumerate(values):
            features[f"{name}.{idx}"] = values

    return features


def arc_to_features(arc, mention_features, pairwise_features):
    features = {  # "arc": str(repr(arc)),
        "consistent": arc[0].decision_is_consistent(arc[1])}

    for prefix, mention in zip(["ana", "ante"], arc):
        for name, value in get_mention_features(mention, mention_features).items():
            features[f"{prefix}_{name}"] = value

    if not arc[1].is_dummy():
        o_features, bert_features = split_features([feature(arc[0], arc[1]) for feature in pairwise_features])
        for name, value in o_features:
            features[name] = value

        for name, values in bert_features:
            for idx, value in enumerate(values):
                features[f"{name}_{idx}"] = values

    return features


def get_data_from_file(file_name, features_file):
    mention_features, pairwise_features = import_helper.get_features(features_file)
    extract_substructures_limited = import_helper.import_from_path(
        "extension.antecedent_trees.extract_substructures_limited")
    extractor = InstanceExtractor(extract_substructures_limited, mention_features, pairwise_features,
                                  cost_based_on_consistency)
    os.environ["MAX_DISTANCE"] = '40'
    corpus = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))
    data = []
    for doc in tqdm(corpus, desc="Documents"):
        doc.system_mentions = mention_extractor.extract_system_mentions(doc)
        substructures = extractor.extract_substructures(doc)
        for struct in substructures:
            for arc in tqdm(struct, desc="Arc"):
                yield arc_to_features(arc, mention_features, pairwise_features)


def write_to_file(file_name, features_file, out_file):
    def write(f, names, row):
        if "arc" in row:
            f.write(f'"{row["arc"]}",')
        f.write(f"{','.join([str(row[x]) for x in names if x != 'arc'])}\n")

    gen = get_data_from_file(file_name, features_file)
    first = next(gen)
    names = first.keys()
    with open(out_file, "w", encoding='utf-8') as f:
        f.write(f"{','.join(names)}\n")
        write(f, names, first)
        for row in gen:
            write(f, names, row)


if __name__ == "__main__":
    features_file = r"D:\GDrive\Puc\Projeto Final\Code\extra_files\features.txt"
    root = r"D:\ProjetoFinal\data\train\conll\train.conll.train.0"
    out_file = r"D:\ProjetoFinal\data\decision_tree\span_0.csv"
    write_to_file(root, features_file, out_file)
