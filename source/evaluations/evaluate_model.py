from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms
from cort.core import corpora, mention_extractor
from boilerplate import mentions

root = r"D:\GDrive\Puc\Projeto Final\Datasets\conll"
file_name = f"{root}/bc.conll"
file_name = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\test.conll"
# file_name = r"Z:\temp\teste.conll"
reference = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))

with open(file_name, "r", encoding="utf-8") as f:
    lines = f.readlines()

mentions = mentions.build_mention_list(lines)

total = 0
idx = 0
print(f"# Docs:{len(reference.documents)}")
for doc in reference.documents:
    total += len(doc.annotated_mentions)
    extracted = mention_extractor.extract_system_mentions(doc)
    print(f"{doc.identifier}:{len(doc.annotated_mentions)}")
    for m in doc.annotated_mentions:
        print(mentions[idx].start_pos - m.span.begin)

    print(f"BoilerPlate:{len(mentions)}, Cort:{total}")
