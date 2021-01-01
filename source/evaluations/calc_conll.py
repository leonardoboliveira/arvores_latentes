import numpy as np
import sys
import os


def get_score(file_name):
    f1 = []

    with open(file_name, "r") as f:
        use_metric = False
        for line in f:
            splitted = line.split()
            if "METRIC" in line:
                use_metric = splitted[1].strip(":") in ["muc", "bcub", "ceafe"]
            elif "Coreference" in line and use_metric:
                f1.append(splitted[-1])

    f1 = [float(x.strip("%")) for x in f1]
    base = os.path.basename(file_name).split(".")
    name = base[0]
    if len(base) < 4:
        epoch = -1
        step = -1
    else:
        epoch = int(base[1])
        step = int(base[2])

    if len(f1) == 0:
        conll = 0
    else:
        conll = np.mean(f1)

    return name, epoch, step, conll


if __name__ == "__main__":
    file_name = sys.argv[1]
    name, echo, step, conll = get_score(file_name)
    print(conll)
