import re
import os
import sys
from tqdm import tqdm


def create(out_path, doc_id, suffix):
    splitted = doc_id.split("/")
    path = os.path.join(out_path, *splitted[:-1])
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    file_name = os.path.join(path, f"{splitted[-1]}.{suffix}")
    return open(file_name, "a", encoding="utf-8")


def get_document_name(line):
    """
    The the document name from the first line.
    :param train_list: list of all lines in the document
    :return: the document name
    """
    name_re = re.compile(r".*\((.*)\).*")
    match = name_re.match(line)
    return match.group(1)


def split_encoded(in_file, out_path, suffix):
    current_writer = None
    current_doc = None
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if "#begin" in line:
                next_line = next(f)
                doc_id = get_document_name(next_line)
                if doc_id != current_doc:
                    if current_writer is not None:
                        current_writer.close()
                    current_doc = doc_id
                    current_writer = create(out_path, doc_id, suffix)
                current_writer.write(line)
                current_writer.write(next_line)
            else:
                current_writer.write(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        in_file = r"Z:\temp\glove\devel.glove2"
        out_path = r"D:\ProjetoFinal\data\devel\glove"
        suffix = "glove"
    else:
        in_file, out_path, suffix = sys.argv[1:]

    split_encoded(in_file, out_path, suffix)
