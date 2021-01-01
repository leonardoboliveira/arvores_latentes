import random
from tqdm import tqdm
import sys
from cort.core import corpora


def get_list_of_files(out_path, file_name, num_bins, open_mode="w"):
    def get_file(f_type, i=0):
        return open(f"{out_path}/{file_name}.{f_type}.{i}", open_mode, encoding="utf-8")

    all_bins = [get_file("test")]
    all_bins += [get_file("train", i) for i in range(num_bins)]

    return all_bins


def split_file(in_path, out_path, file_name, num_bins, skip_no_annotations):
    all_bins = get_list_of_files(out_path, file_name, num_bins)
    idx = list(range(len(all_bins)))

    current_doc = []
    with open(f"{in_path}/{file_name}", "r", encoding="utf-8") as f:
        has_annotations = False
        for line in tqdm(f):
            if "#begin" in line:
                current_doc = []
                has_annotations = not skip_no_annotations
            elif (len(line) > 10) and (line.split()[-1] != "-") and ("#end" not in line):
                has_annotations = True

            current_doc.append(line)

            if "#end" in line:
                chosen_idx = random.choice(idx)
                if has_annotations or (chosen_idx == 0):
                    file = all_bins[chosen_idx]
                    for l in current_doc:
                        file.write(l.strip("\r\n") + "\n")
                else:
                    print("No annotations")

    for file in all_bins:
        file.close()


def check_files(out_path, file_name, num_bins):
    print("Checking")
    all_bins = get_list_of_files(out_path, file_name, num_bins, "r")
    for file in all_bins:
        corpora.Corpus.from_file("reference", file)
        file.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        in_path = r"D:\GDrive\Puc\Projeto Final\Datasets\conll"
        out_path = r"D:\ProjetoFinal\data\debug_in"
        file_name = "train.conll"
        partitions = 9
        skip_no_annotations = True
    else:
        in_path, out_path, file_name, partitions, skip_no_annotations = sys.argv[1:]
        partitions = int(partitions)
        skip_no_annotations = bool(skip_no_annotations)

    split_file(in_path, out_path, file_name, partitions, skip_no_annotations)

    # check_files(out_path, file_name, partitions)
