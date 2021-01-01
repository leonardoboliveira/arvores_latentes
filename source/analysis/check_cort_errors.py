from cort.core import corpora
from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms
from analysis import plotting
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib

file_name = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\devel.conll"
reference = corpora.Corpus.from_file("reference", open(file_name, encoding="utf-8"))

extractor = error_extractors.ErrorExtractor(
    reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)


def add_files(extractor, name, prefix):
    tree = corpora.Corpus.from_file(name, open(f"{prefix}.predicted", encoding="utf-8"))
    tree.read_antecedents(open(f"{prefix}.ante", encoding="utf-8"))
    extractor.add_system(tree)


path = r"D:\ProjetoFinal\error_check"
add_files(extractor, "baseline", f"{path}/baseline_lex_efi")
add_files(extractor, "span", f"{path}/span_induction_E6")
add_files(extractor, "glove", f"{path}/glove_efi_2000")
add_files(extractor, "bert", f"{path}/bert_bins")

errors = extractor.get_errors()

errors_by_type = errors.categorize(
    lambda error: error[0].attributes['type']
)
# errors_by_type.visualize("glove")

span = errors_by_type["span"]["precision_errors"]["all"]
glove = errors_by_type["glove"]["precision_errors"]["all"]
baseline = errors_by_type["baseline"]["precision_errors"]["all"]
bert = errors_by_type["bert"]["precision_errors"]["all"]

patterns = None  # ["-", ".", "", '\\']
font = {'size': 16}
matplotlib.rc('font', **font)

plotting.plot(
    [("glove", [(cat, len(errs)) for cat, errs in glove.items()]),
     ("span", [(cat, len(errs)) for cat, errs in span.items()]),
     ("baseline", [(cat, len(errs)) for cat, errs in baseline.items()]),
     ("bert", [(cat, len(errs)) for cat, errs in bert.items()]),
     ],
    "Precision Errors",
    "Type of anaphor",
    "Number of Errors",
    patterns=patterns)

span = errors_by_type["span"]["recall_errors"]["all"]
glove = errors_by_type["glove"]["recall_errors"]["all"]
baseline = errors_by_type["baseline"]["recall_errors"]["all"]
bert = errors_by_type["bert"]["recall_errors"]["all"]

plotting.plot(
    [("glove", [(cat, len(errs)) for cat, errs in glove.items()]),
     ("span", [(cat, len(errs)) for cat, errs in span.items()]),
     ("baseline", [(cat, len(errs)) for cat, errs in baseline.items()]),
     ("bert", [(cat, len(errs)) for cat, errs in bert.items()]),
     ],
    "Recall Errors",
    "Type of anaphor",
    "Number of Errors",
    patterns=patterns)
plt.show()

for system in ["span", "glove", "baseline", "bert"]:
    nom_rec_errs = errors_by_type[system]["recall_errors"]["all"]["NOM"]
    all_heads = [" ".join(err[0].attributes["head"]).lower() for err in nom_rec_errs]
    most_common = Counter(all_heads).most_common(10)
    print(system, most_common)

plt.show()
