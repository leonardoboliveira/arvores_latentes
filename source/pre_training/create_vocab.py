from cort.core import corpora
from tqdm import tqdm
from cort.core import mention_extractor


def vocab_from_file(file_name):
    heads = set()
    reference = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))
    for doc in tqdm(reference.documents):
        for m in doc.annotated_mentions:
            heads = heads.union(set(m.attributes["head"]))

        for m in mention_extractor.extract_system_mentions(doc):
            if "head" in m.attributes:
                heads = heads.union(set(m.attributes["head"]))

    return heads


def build_from_base(base_vocab):
    vocab = set()
    with open(base_vocab, "r", encoding="utf-8") as f:
        for line in f:
            vocab.add(line.strip("\n").strip("\r"))
    return vocab


def save_vocab_to_file(files, base_vocab, vocab_file):
    if base_vocab is not None:
        vocab = build_from_base(base_vocab)
    else:
        vocab = set()

    print(f"Starting with {len(vocab)} words")
    for file_name in files:
        vocab = vocab.union(vocab_from_file(file_name))
        print(f"New count:{len(vocab)}")

    with open(vocab_file, "w", encoding="utf-8") as out:
        out.write("\n".join(vocab))


if __name__ == "__main__":
    files = [r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train.conll"]
    base_vocab = r"D:\GDrive\Puc\Projeto Final\models\wwm_cased_L-24_H-1024_A-16\vocab.txt"
    save_vocab_to_file(files, None, "../../extra_files/head_token_vocab.txt")
