import extension.bins_features as bins_features
import multiprocessing
import glob
import sys


def get_docs_in_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if "#begin" in line:
                yield line.strip("#begin document").strip("\r").strip("\n").strip(" ")


def get_embedding(doc_id):
    bins_features.get_embedding(doc_id, 0, 1)


def get_embeddings_in_file(file_name):
    pool = multiprocessing.Pool()
    pool.map(get_embedding, get_docs_in_file(file_name))


def get_embedding_in_path(path):
    for file_name in glob.glob(path):
        get_embeddings_in_file(file_name)


if __name__ == "__main__":
    # get_embeddings_in_file(r"D:\GDrive\Puc\Projeto Final\Datasets\conll\devel.conll")
    get_embedding_in_path(sys.argv[1])
