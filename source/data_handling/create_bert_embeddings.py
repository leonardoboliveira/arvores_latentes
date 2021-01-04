from bert_serving.client import BertClient
import sys
import os
import glob
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import tokenization
import numpy as np
from pre_training.constants import BERT_MODEL_HUB
import logging
import multiprocessing

logging.basicConfig(level=logging.DEBUG)

EMBEDDING_SIZE = 1024


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def train_file_to_list(file):
    """
    :param file: file name to be read
    :return: list of all lines
    """
    print(f"Opening {file}")
    with open(file, "r", encoding="utf8") as f:
        return f.readlines()


class Document:
    def __init__(self, tokenizer, first_line):
        self.tokenizer = tokenizer
        self.lines = []
        self.sentences = []
        self.current_sentence = []
        self.doc_name = first_line.replace("#begin document ", "").strip("\r").strip("\n")

    @property
    def file_name(self):
        # (bc/cctv/00/cctv_0000); part 000
        splitted = self.doc_name.split(";")
        begin = splitted[0].strip("(").strip(")")
        end = splitted[1].replace(" part ", "")
        return f"{begin}_{end}"

    def file_exists(self, folder_to_check):
        path = os.path.join(folder_to_check, self.file_name)
        return len(glob.glob(f"{path}.*")) > 0

    def open_file(self, out_path, suffix):
        splitted = self.file_name.split("/")
        path = os.path.join(out_path, *splitted[:-1])
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        file_name = os.path.join(path, f"{splitted[-1]}.{suffix}")
        return open(file_name, "w", encoding="utf-8")

    def add_line(self, line):
        line = Line(self.tokenizer, line)
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

    def finish_doc(self, bc):
        sentences = []
        token_count = 0
        current_sentence = ""
        for line in self.lines:
            if token_count + len(line.tokens) > 256:
                sentences.append(current_sentence.strip(" "))
                current_sentence = ""
                token_count = 0

            if line.word is not None:
                current_sentence += line.word + " "
                token_count += len(line.tokens)
        sentences.append(current_sentence.strip(" "))

        all_tokens, real_tokens = bc.encode(sentences, show_tokens=True)

        token_idx = 1  # First token is [CLS]
        sentence_idx = 0
        for line in self.lines:
            token_idx, sentence_idx = line.save_token_slice(all_tokens, real_tokens, token_idx, sentence_idx)

    def write(self, out_file):
        for line in self.lines:
            out_file.write(f'{self.doc_name},{line}\n')


class Line:
    def __init__(self, tokenizer, line):
        self.tokenizer = tokenizer
        self.line = line
        self.embeddings = []
        self.sentence_id = -1

        splitted = line.split()
        if len(splitted) < 3:
            self.eos = True
            self.word = None
            self.tokens = []
        else:
            self.eos = False
            self.word = splitted[3]
            self.tokens = tokenizer.tokenize(self.word)

    def end_of_sentence(self):
        return self.eos

    def save_token_slice(self, all_tokens, real_tokens, token_start, idx_sentence):
        if self.word is None:
            return token_start, idx_sentence

        token_end = self.find_last_token_2(idx_sentence, real_tokens, token_start)
        if token_end <= token_start:
            print("NOK")

        assert token_end > token_start, f"Not found token for '{self.line}' starting from {token_start}"

        token_slice = np.array(all_tokens[idx_sentence, token_start:token_end])
        self.tokens = real_tokens[idx_sentence][token_start:token_end]
        self.embeddings = list(np.mean(token_slice, axis=0))

        assert EMBEDDING_SIZE == len(self.embeddings), f"Wrong shape:{len(self.embeddings)}"
        if sum(self.embeddings) == 0:
            print(f"All empty")
        assert sum(self.embeddings) != 0, "No valid embeddings"

        if token_end >= len(real_tokens[idx_sentence]) - 1:  # Last one is [SEP]
            token_end = 1  # First is [CLS]
            idx_sentence += 1

        return token_end, idx_sentence

    def find_last_token_2(self, idx_sentence, real_tokens, token_start):
        idx_token = token_start
        word_token = 0
        for idx_token in range(token_start, len(real_tokens[idx_sentence])):
            if (word_token >= len(self.tokens)) or \
                    (self.tokens[word_token] != real_tokens[idx_sentence][idx_token]):
                break
            word_token += 1
        return idx_token

    def find_last_token(self, idx_sentence, real_tokens, token_start):
        idx_token = token_start
        word_copy = self.word
        for idx_token in range(token_start, len(real_tokens[idx_sentence])):
            current_token = real_tokens[idx_sentence][idx_token]
            if current_token == "[UNK]" and "[UNK]" in self.tokens:  # unk case
                word_copy = ''
                continue

            if current_token[:2] == "##":
                current_token = current_token[2:]

            sz = len(current_token)
            if word_copy[:sz] != current_token:
                break
            word_copy = word_copy[sz:]
        return idx_token

    def __str__(self):
        if self.word is None:
            return self.empty_line(False)
        embed = ",".join([str(x) for x in self.embeddings])
        if self.word == '"':
            return f"'{self.word}',{embed}"
        return f'"{self.word}",{embed}'

    @staticmethod
    def empty_line(include_doc_name=True, word=""):
        empty = f'"{word}",{",".join(["0"] * EMBEDDING_SIZE)}'
        if include_doc_name:
            empty = "-," + empty

        return empty


def get_documents(folder_in, tokenizer, folder_to_check):
    for root, dirs, files in os.walk(folder_in):
        for file_in in files:
            if file_in[:-5] != "conll":
                continue
            train_list = train_file_to_list(file_in)
            document = None
            for line in tqdm(train_list, "Reading"):
                if "#begin" in line:
                    document = Document(tokenizer, line)
                    continue
                if "#end" in line:
                    if not document.file_exists(folder_to_check):
                        yield document
                    document = None
                    continue
                if document is not None:
                    document.add_line(line)


class MyRunner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, doc):
        with BertClient() as bc:
            doc.finish_doc(bc)
        return doc


def create_embedding(file_in, folder_out, suffix):
    print("Building tokenizer")
    tokenizer = create_tokenizer_from_hub_module()
    print("Waiting for BertServer")
    documents = get_documents(file_in, tokenizer, folder_out)

    with multiprocessing.Pool() as pool:
        for doc in pool.imap_unordered(MyRunner(tokenizer), documents):
            with doc.open_file(folder_out, suffix) as f:
                f.write(Line.empty_line(word="#begin") + "\n")
                doc.write(f)
                f.write(Line.empty_line(word="#end") + "\n")


def start_server(flags):
    print("Starting Server")
    from bert_serving.server.helper import get_args_parser
    from bert_serving.server import BertServer

    args = get_args_parser().parse_args(flags)
    server = BertServer(args)
    server.start()
    print("Started")
    return server


def build_server_flags(model_dir, ckpt_name):
    params = ["-model_dir", model_dir,
              "-ckpt_name", ckpt_name,
              "-pooling_strategy", "NONE",
              "-cased_tokenization",
              "-show_tokens_to_client",
              "-max_seq_len", "NONE"]

    if "DEBUG" not in os.environ:
        params += ["-cpu"]
        params += ["-num_worker", str(int(os.cpu_count() * 0.3))]

    if "TMPDIR" in os.environ:
        params += ["-graph_tmp_dir", os.environ["TMPDIR"]]

    return params


if __name__ == "__main__":
    model_dir = r"D:\GDrive\Puc\Projeto Final\models\spanbert_large"
    f_in = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\test"
    f_out = r"D:\ProjetoFinal\data\test\spanbert"
    tag = "span"
    model_name = "mode.max.ckpt"

    if len(sys.argv) == 6:
        _, model_dir, model_name, f_in, f_out, tag = sys.argv

    server = start_server(build_server_flags(model_dir, model_name))
    create_embedding(f_in, f_out, tag)

    server.close()
