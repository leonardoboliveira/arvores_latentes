from extension.c_bins_features import bins_embedding
import numpy as np
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

_embedded = None
_derived_features = None


def __load_features():
    global _embedded
    global _derived_features
    file_name = os.environ["INDUCED_PICKLED"]
    if not os.path.isfile(file_name):
        print(f"Not found induced file {file_name}")
        exit(1)
    with open(file_name, "rb") as f:
        _derived_features = pickle.load(f)
        Embedder = InduceEmbedder  # import_helper.import_from_path("extension.embedder.InduceEmbedder")
        _embedded = Embedder(_derived_features)


def get_features(full):
    global _embedded
    if _embedded is None:
        __load_features()

    for f in _embedded.get_features(full):
        yield [str(x) for x in f]


def induced(mention1, mention2):
    e1 = bins_embedding(mention1)[1]
    e2 = bins_embedding(mention2)[1]
    full = np.concatenate([e1, e2])
    assert sum(full) == 600.0, f"Missing data:{sum(full)}"
    ind = get_features(full)
    return "induced", ind


class TemplateEmbedder:
    def __init__(self, template):
        self.names = template.columns
        self.values = template.values

        generator = list
        for x in range(len(self.names)):
            self.structure = defaultdict(generator)
            generator = lambda: defaultdict(generator)

        for row in template.itertuples():
            x = self.structure
            for idx, v in enumerate(row[1:-1]):
                x = x[v]
            x[row[-1]] = "^".join([f"{x}={y}" for x, y in zip(self.names, row[1:])])

    def derived_match(self, basic_features, as_dict):
        for name, value in zip(self.names, basic_features):
            if as_dict[name] != value:
                return False

        return True

    def template_match(self, as_dict):
        level = self.structure
        for idx, name in enumerate(self.names):
            if name not in as_dict:
                return None
            search_for = as_dict[name]
            level = level.get(search_for, None)
            if level is None:
                # print(f"Give up on {idx}/{len(self.names)}")
                return None
        return level

    def template_match_old(self, as_dict):
        for name in self.names:
            if name not in as_dict:
                return None

        for basic_features in self.values:
            if self.derived_match(basic_features, as_dict):
                return "^".join([f"{x}={y}" for x, y in zip(self.names, basic_features)])
        return None


class InduceEmbedder:

    def __init__(self, derived_features):
        self.derived_features = [TemplateEmbedder(t) for t in tqdm(derived_features, desc="Templates")]

    def get_features(self, full):
        features = []
        if len(full) == 0:
            return features

        splited = [x.split("=") for x in full]
        as_dict = {x[0]: "=".join(x[1:]) for x in splited}
        for template in self.derived_features:
            matched = template.template_match(as_dict)
            if matched is not None:
                features.append(matched)

        return features


def induce_features(basic_features):
    global _embedded
    if _embedded is None:
        __load_features()

    return _embedded.get_features(basic_features)
