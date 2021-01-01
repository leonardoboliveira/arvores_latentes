import numpy as np
import os
import random

_bert_db = {}
vocab_file = None

EMB_SIZE = 10000
INC_STEP = 10000 / 50000


def __load_bert_db():
    global _bert_db
    global vocab_file
    vocab_file = os.environ["VOCAB_FILE"]  # + str(random.randint(1, 10000))

    # print(f"Loading vocab from {vocab_file}")
    if not os.path.isfile(vocab_file):
        return

    with open(vocab_file, "r", encoding="utf-8") as f:
        pos = 0
        inc = 1
        for line in f:
            _bert_db[line.strip("\n").strip("\r")] = (pos, inc * INC_STEP)
            pos = pos + 1
            if pos >= EMB_SIZE:
                pos = 0
                inc += 1

    vocab_file = os.environ["VOCAB_FILE"] + "_" + str(random.randint(1, 10000))


def save_bert():
    global vocab_file
    with open(vocab_file, "w", encoding="utf-8") as f:
        for word in sorted(_bert_db.keys(), key=lambda x: _bert_db[x]):
            f.write(word + "\n")


def get_embedding(words):
    global _bert_db

    if len(_bert_db) == 0:
        __load_bert_db()

    update_file = False
    for word in words:
        if word not in _bert_db:
            print(f"NiV:{word}")
            _bert_db[word] = len(_bert_db)
            update_file = True
    if update_file:
        save_bert()

    base = np.zeros(EMB_SIZE)
    for word in words:
        pos, inc = _bert_db[word]
        base[pos] += inc
    return base


def bert_pair_head_embedding(mention1, mention2):
    head1 = bert_head_embedding(mention1)[1]
    head2 = bert_head_embedding(mention2)[1]

    if len(head1) != len(head2):
        del mention1.attributes["bert_head_embedding"]
        del mention2.attributes["bert_head_embedding"]
        return bert_pair_head_embedding(mention1, mention2)

    return "bert_pair_head_embedding", head1 + head2


def bert_head_embedding(mention):
    def calc(mention):
        return get_embedding(mention.attributes["head"])

    return bert_aux("bert_head_embedding", mention, calc)


def bert_aux(name, mention, fn):
    if name in mention.attributes:
        attr = mention.attributes[name]
    else:
        if mention.is_dummy():
            attr = np.ones(len(_bert_db))
        else:
            attr = fn(mention)
        mention.attributes[name] = attr

    return name, attr
