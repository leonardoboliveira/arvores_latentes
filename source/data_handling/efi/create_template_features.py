import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time
import json
import pickle
from tqdm import tqdm

NUM_CLASSES = 5


def get_template_and_features(file_name, dot_file=None):
    def index_marks(nrows, chunk_size):
        return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

    def split(dfm, chunk_size):
        indices = index_marks(dfm.shape[0], chunk_size)
        return np.split(dfm, indices)

    templates = []
    df = pd.read_csv(file_name, dtype='category').dropna()
    chunks = split(df, int(len(df) / 5))
    for c in chunks:
        templates += get_templates(c, dot_file=dot_file)

    features = list(get_all_features(df, templates))

    return templates, features


def to_derived_features(corref, template):
    unique_values = corref[template].drop_duplicates()
    unique_values = unique_values.reset_index(drop=True)
    # r = unique_values.apply(lambda row: [int(row.name * NUM_CLASSES + x) for x in row])
    # assert r.max().max() <= 600 * NUM_CLASSES
    return unique_values


def get_all_features(df, templates):
    if "consistent" in df:
        corref = df[df["consistent"] == 'True']
    else:
        corref = df[df.iloc[:, -1] == 1]

    for template in templates:
        yield to_derived_features(corref, template)


def get_sub_templates(t, start):
    if t[start] < 0:
        return 1, []
    left_c, left_template = get_sub_templates(t, start + 1)
    right_c, right_template = get_sub_templates(t, start + 1 + left_c)

    ref_id = int(t[start] / 1)
    this_template = [ref_id]
    all_templates = [this_template]
    for child_templates in left_template + right_template:
        all_templates.append(this_template + child_templates)

    return left_c + right_c + 1, all_templates


def grid_search(X, Y):
    for ln in [8, 16, 32, 64, 128, 256, None]:
        for md in [5, 10, 20, 30, None]:
            clf = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=ln, random_state=1, max_depth=md)
            start = time.time()
            scores = cross_val_score(clf, X, Y, cv=5, n_jobs=11, scoring="balanced_accuracy")
            end = time.time()
            print(f"Precision {ln} - {md}: %0.2f (+/- %0.2f) - {end - start}" % (scores.mean(), scores.std() * 2))
    exit(1)


def get_templates(full, dot_file=None, hist_file=None):
    if "consistent" in full:
        Y = full["consistent"]
        X = full[[x for x in full.columns if x not in ["arc", "consistent"]]]
        cat_columns = X.select_dtypes(['category']).columns
        X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    else:
        Y = full.iloc[:, -1]
        X = full.iloc[:, :-1]

    print(f"X.shape:{X.shape}")
    if X.shape[0] == 0:
        return []

    # grid_search(X,Y)
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=None, random_state=1, max_depth=20)
    clf.fit(X, Y)

    if dot_file:
        dot_data = tree.export_graphviz(clf, out_file=dot_file,
                                        feature_names=X.columns,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph
    if hist_file:
        S = X.sum(axis=0)
        print(f"S.shape:{S.shape}")
        plt.hist(S)
        plt.savefig(hist_file)

    templates = get_sub_templates(clf.tree_.feature, 0)
    templates = set([frozenset(x) for x in templates[1]])
    templates = set([x for x in templates if (1 < len(x) < 6)])

    def gen(x):
        for l in x:
            yield [X.columns[k] for k in l]

    return [list(reversed(x)) for x in gen(templates)]


if __name__ == "__main__":
    tqdm.pandas()
    templates, features = get_template_and_features(r"Z:\temp\head.csv", )
    with open(r"D:\ProjetoFinal\data\decision_tree\head.txt", "w") as f:
        json.dump(templates, f)

    with open(r"D:\ProjetoFinal\data\decision_tree\head.dat", "wb") as f:
        pickle.dump(features, f)
