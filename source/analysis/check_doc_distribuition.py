import pandas as pd
from cort.core import corpora, mention_extractor
import os


def get_doc_info(file_name):
    with open(file_name, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)
    for doc in corpus.documents:
        yield file_name, doc.identifier, len(doc.annotated_mentions)


def get_df_for_path(root):
    rows = []
    for file_name in os.listdir(root):
        if "conll" not in file_name:
            continue
        print(f"Reading {file_name}")
        for row in get_doc_info(os.path.join(root, file_name)):
            rows.append(row)

    return pd.DataFrame(data=rows, columns=["FileName", "ID", "Mentions"])


if __name__ == "__main__":
    df = get_df_for_path(r"Z:\temp\conll")
    df.to_csv(r"z:\temp\distribuition_train.csv")
