import sys
import numpy as np
import matplotlib.pyplot as plt

in_file_name = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train.conll"  # sys.argv[1]

counter = 0
counts = []
with open(in_file_name, "r", encoding="utf-8") as in_file:
    for line in in_file:
        if "#begin" in line:
            counter = 0
        elif "#end" in line:
            counts.append(counter)
        else:
            counter += 1

    ax = plt.hist(np.clip(counts, 0, 512), bins=30)
    # ax.get_figure().savefig("z:/temp/teste.png")
    plt.xlabel("Document length")
    plt.title("Document length frequency")
    plt.show()
