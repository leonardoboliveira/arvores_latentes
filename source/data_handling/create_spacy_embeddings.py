import sys
import os
from tqdm import tqdm
import numpy as np
import spacy


def train_file_to_list(file):
    """
    :param file: file name to be read
    :return: list of all lines
    """
    with open(file, "r", encoding="utf8") as f:
        return f.readlines()


class Document:
    def __init__(self, first_line, window, stride):
        self.lines = []
        self.sentences = []
        self.current_sentence = []
        self.doc_name = first_line.replace("#begin document ", "").strip("\r").strip("\n")
        self.window = window
        self.stride = stride

    def add_line(self, line):
        line = Line(line)
        self.lines.append(line)
        if line.end_of_sentence():
            self.sentences.append(self.save_sentence())
        else:
            self.current_sentence.append(line)

    def save_sentence(self):
        sentence = []
        sentence_id = len(self.sentences)
        for line in self.current_sentence:
            line.sentence_id = sentence_id
            sentence.append(line.word)

        self.current_sentence.clear()
        return " ".join(sentence)

    def finish_doc(self):
        start_index = 0
        while start_index < len(self.lines):
            window = min(self.window, len(self.lines) - start_index)

            current_length = 0
            end_index = start_index
            text = ""
            while (current_length < window) and (end_index < len(self.lines)):
                if self.lines[end_index].word is not None:
                    text += " "
                    current_length += 1
                    text += self.lines[end_index].word
                end_index += 1
            text = text.strip()

            if len(text) > 0:
                print(f"Processing \n'{text}'")
                tokens = nlp(text)
                token_start = 0
                print(f"Tokenized\n {[x.text for x in tokens]}")
                print(f"Embeddings Sum\n {[float(sum(x.vector)) for x in tokens]}")
                print(f"Saving embeddings from {start_index} to {end_index}")
                for idx, line in enumerate(self.lines[start_index: end_index]):
                    token_start = line.save_token_slice(tokens, token_start, (start_index, end_index, idx))

            start_index += self.stride

        for line in self.lines:
            line.choose_embedding()

    def write(self, out_file):
        self.finish_doc()
        for line in self.lines:
            out_file.write(f'{self.doc_name},{line}\n')


class Line:
    def __init__(self, line):
        self.line = line
        self.embeddings = {}
        self.tokens = {}
        self.sentence_id = -1
        self.chosen_embedding = None

        splitted = line.split()
        if len(splitted) < 3:
            self.eos = True
            self.word = None
        else:
            self.eos = False
            self.word = splitted[3]

    def end_of_sentence(self):
        return self.eos

    def get_tokens(self, all_tokens, token_start):
        if all_tokens[token_start].text == self.word:
            return [all_tokens[token_start].text]
        while all_tokens[token_start].text.strip() == "":
            token_start += 1

        new_token = all_tokens[token_start].text
        token_end = token_start
        while new_token != self.word:
            token_end += 1
            new_token += all_tokens[token_end].text.strip()

        token_end += 1
        return [x.text for x in all_tokens[token_start:token_end]]

    def save_token_slice(self, all_tokens, token_start, span):
        if self.word is None:
            return token_start
        tokens = self.get_tokens(all_tokens, token_start)
        token_end = token_start + len(tokens)

        token_slice = np.array([x.vector for x in all_tokens[token_start:token_end]])
        embeddings = list(np.mean(token_slice, axis=0))

        if sum(embeddings) == 0:
            print(f"All empty:{token_start}:{token_end}:{min(embeddings)} ,{max(embeddings)} >> {self.line}")

        # assert sum(embeddings) != 0, "All empty"
        assert EMBEDDING_SIZE == len(embeddings), f"Wrong shape:{len(embeddings)}"

        self.embeddings[span] = embeddings
        self.tokens[span] = tokens
        return token_end

    def choose_embedding(self):
        if self.word is None:
            return

        self.chosen_embedding = None
        better_score = -1
        for span in self.embeddings:
            score = min(span[2], span[1] - span[0] - span[2])
            if score > better_score:
                better_score = score
                self.chosen_embedding = self.embeddings[span]
            else:  # When it starts decreasing, will only get smaller
                break
        if self.chosen_embedding is None:
            print("No embedding")

        assert self.chosen_embedding is not None, f"No embedding {self.line}"

    def __str__(self):
        if self.word is None:
            return self.empty_line(False)
        embed = ",".join([str(x) for x in self.chosen_embedding])
        if self.word == '"':
            return f"'{self.word}',{embed}"
        return f'"{self.word}",{embed}'

    @staticmethod
    def empty_line(include_doc_name=True, word=""):
        empty = f'"{word}",{",".join(["0"] * EMBEDDING_SIZE)}'
        if include_doc_name:
            empty = "-," + empty

        return empty


def create_embedding(file_in, file_out, window, stride):
    train_list = train_file_to_list(file_in)
    document = None

    with open(file_out, "w", encoding="utf8") as f:
        for line in tqdm(train_list, "Reading"):
            if "#begin" in line:
                document = Document(line, window, stride)
                f.write(Line.empty_line(word="#begin") + "\n")
                continue
            if "#end" in line:
                document.write(f)
                document = None
                f.write(Line.empty_line(word="#end") + "\n")
                continue
            if document is None:
                f.write(Line.empty_line())
            else:
                document.add_line(line)


def create_path_embedding(path_in, path_out, window, stride):
    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue
            real_out = r.replace(path_in, path_out)
            os.makedirs(real_out, exist_ok=True)
            create_embedding(os.path.join(r, file_name), os.path.join(real_out, file_name), window, stride)


if __name__ == "__main__":
    # en_core_web_md
    # en_trf_xlnetbasecased_lg
    # en_trf_bertbaseuncased_lg
    # spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg")
    EMBEDDING_SIZE = 300

    if len(sys.argv) != 3:
        in_file = r"D:\ProjetoFinal\data\train\train_gold\train\data\english\annotations"
        out_file = r"D:\ProjetoFinal\data\train\glove2"
    else:
        in_file, out_file = sys.argv[1:]

    window = 128  # 384
    stride = 128
    create_path_embedding(in_file, out_file, window, stride)
