from data_handling.create_bert_ts import create_tokenizer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def check_file(tokenizer, in_file):
    sizes = []

    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if (len(line) == 0) or ("#begin" in line) or ("#end" in line):
                continue
            splitted = line.split()
            if len(splitted) <= 3:
                continue
            word = splitted[3]
            tokens = tokenizer.tokenize(word)
            sizes.append(len(tokens))
    return sizes


def check_files(tokenizer, files):
    sizes = []
    for file in files:
        sizes += check_file(tokenizer, file)
    return sizes


def get_files(path):
    for r, d, f in tqdm(os.walk(path)):
        for file_name in f:
            yield os.path.join(r, file_name)


def check_path(path):
    tokenizer = create_tokenizer()
    sizes = check_files(tokenizer, get_files(path))
    with open(r"z:\temp\sizes.dmp", "wb") as f:
        pickle.dump(sizes, f)
    plt.hist(sizes)
    plt.show()


if __name__ == "__main__":
    check_path(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train\data\english\annotations")
