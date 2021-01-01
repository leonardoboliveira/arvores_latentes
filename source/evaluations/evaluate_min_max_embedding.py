import pandas as pd
from tqdm import tqdm


def get_extremes(file_name):
    min_value = float("inf")
    max_value = float("-inf")
    iter_csv = pd.read_csv(file_name, header=None, chunksize=100000)
    for chunk in tqdm(iter_csv):
        slice = chunk.iloc[:, 2:]
        this_min = slice.min().min()
        this_max = slice.max().max()
        min_value = min(min_value, this_min)
        max_value = max(max_value, this_max)

    return min_value, max_value


if __name__ == "__main__":
    min_value, max_value = get_extremes(r"C:\Users\loliveira\Desktop\train.B2")
    print("min:{}, max:{}".format(min_value, max_value))
