import os
import compress_pickle
from collections import defaultdict
from tqdm import tqdm


def get_file_info(file_name):
    struct, info = compress_pickle.load(file_name)
    if len(struct) == 0:
        return None
    for arc in struct[0]:
        nn, n, v = info[arc][0]
        size = len(nn) + len(n)
        if size > 0:
            return size


def count_on_path(path):
    data = defaultdict(int)
    for r, d, f in tqdm(os.walk(path), desc="Full"):
        for file_name in tqdm(f, leave=False, desc=r):
            if ".lzma" not in file_name:
                continue
            size = get_file_info(os.path.join(r, file_name))
            if size is None:
                print(f"Empty {file_name}")
                continue
            data[size] += 1
        assert len(data) <= 1, f"{r} {data}"

    return data


def check_folders_in(root):
    done = ["glove_40"]
    return {folder: count_on_path(os.path.join(root, folder)) for folder in os.listdir(root) if folder not in done}


if __name__ == "__main__":
    data = check_folders_in(r"D:\ProjetoFinal\pickle\pickle\glove_40_efi\bc")
    print(data)
