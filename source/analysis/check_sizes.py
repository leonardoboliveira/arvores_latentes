import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_info(root, ref):
    data = {name: check_folder(f"{root}/{name}") for name in os.listdir(root)}

    for name, df in tqdm(data.items(), desc="Merging"):
        df["name"] = name
    dfs = pd.concat(data.values())
    dfs = dfs.reset_index().pivot(index="doc", columns="name", values="size")
    print("Normalizing")
    norm = dfs.div(dfs[ref], axis=0)
    return norm[[x for x in norm if x != ref]]


def check_folder(root):
    df = pd.DataFrame(check_folder_df(root), columns=["doc", "size"])
    return df.set_index("doc")


def check_folder_df(root):
    for r, d, f in tqdm(os.walk(root), desc=root):
        for file_name in f:
            name = f"{r.replace(root, '')}/{file_name}"
            statinfo = os.stat(f"{r}/{file_name}")
            yield name.strip(".lzma"), statinfo.st_size


if __name__ == "__main__":
    df = get_info(r"D:\ProjetoFinal\pickle\analysis", "lex_induction")
    df.hist()
    plt.show()
    print("Done")
