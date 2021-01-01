import os


def get_word_set_from_file(file_name):
    def get_words_from_file(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                if ("#begin" in line) or ("#end" in line) or (len(line) == 0):
                    continue
                splitted = line.split()
                if len(splitted) < 4:
                    continue
                yield splitted[3]

    return set(get_words_from_file(file_name))


def get_word_set_from_path(path):
    all_sets = set()
    for r, d, f in os.walk(path):
        for file_name in f:
            if "conll" not in file_name:
                continue
            all_sets = all_sets.union(get_word_set_from_file(os.path.join(r, file_name)))

    return all_sets


if __name__ == "__main__":
    all_words = get_word_set_from_path(r"D:\ProjetoFinal\data\train\conll")
    print(len(all_words))
    with open(r"D:\ProjetoFinal\output\full_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_words)))
