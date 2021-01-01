import os
from cort.core import corpora


def get_words_in_doc(doc):
    def is_valid(x):
        if not x.isnumeric():
            return False
        if ":" in x:
            return False
        if "." in x:
            return False
        if "," in x:
            return False
        if "*" in x:
            return False
        return True

    tokens = set([x for x in doc.tokens if is_valid(x)])
    for mention in doc.annotated_mentions:
        tokens.add("_".join(mention.attributes["head"]))
    return tokens


def process_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
        for doc in corpus.documents:
            yield get_words_in_doc(doc)


def build_vocab(path_in):
    tokens = set()
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue
            for tk in process_file(os.path.join(r, file_name)):
                tokens = tokens.union(tk)

    return tokens


if __name__ == "__main__":
    path_in = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\development"
    out_file = r"D:\GDrive\Puc\Projeto Final\models\bert\custom\vocab.txt"

    tokens = build_vocab(path_in)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
