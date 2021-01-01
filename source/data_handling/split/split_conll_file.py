import sys
import os

prefix = sys.argv[1]
in_file_name = sys.argv[2]
out_folder = sys.argv[3]
max_count = int(sys.argv[4])

current_file = None
counter = 0
count_in_file = 0
with open(in_file_name, "r", encoding="utf-8") as in_file:
    for line in in_file:
        if "#begin" in line:
            if current_file is None:
                current_file = open(f"{out_folder}/{prefix}.{counter}", "w", encoding="utf-8")
                count_in_file = 0

        count_in_file += 1
        current_file.write(line)

        if "#end" in line:
            if count_in_file >= max_count:
                current_file.close()
                counter += 1
                current_file = None

if current_file:
    current_file.close()
