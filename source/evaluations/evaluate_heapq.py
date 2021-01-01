import heapq
from cort.core import corpora
import functools


class MyStruct:
    def __init__(self):
        self.docs = []
        self.total = 0

    def add_doc(self, doc):
        self.docs.append(doc)
        self.total += len(doc.annotated_mentions)

    def __lt__(self, other):
        return self.total < other.total


li = [[list(range(5))], [list(range(7))], [list(range(9))], [list(range(1))], [list(range(3))]]
heapq.heapify(li)
print(list(li))

print(heapq.heappop(li))

root = r"D:\GDrive\Puc\Projeto Final\Datasets\conll"
file_name = f"{root}/bc.conll"
file_name = r"D:\GDrive\Puc\Projeto Final\Code\test_files\cctv_0000.gold_conll"
reference = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))

li = [MyStruct() for i in range(3)]
heapq.heapify(li)

print(f"# Docs:{len(reference.documents)}")
for doc in reference.documents:
    min_list = heapq.heappop(li)
    min_list.add_doc(doc)
    heapq.heappush(li, min_list)

print([x.total for x in li])
