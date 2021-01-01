import re
import tempfile


def run_cort(bert_file, input_file):
    pass


def text_to_lines(text):
    pass


def build_bert_file(text):
    pass


def build_input_file(text):
    _, output = tempfile.mkstemp()
    with open(output, "w") as f:
        f.write("#begin document fake_doc\n")
        f.writelines(text_to_lines(text))
        f.write("#end document\n")

    return output


def build_in_files(text):
    return build_bert_file(text), build_input_file(text)


def do_coref(text):
    bert_file, input_file = build_in_files(text)
    return run_cort(bert_file, input_file)


def discover_coref(text):
    ante_file = do_coref(text)
    arcs = read_arcs(ante_file)
    clusters = create_clusters(arcs)
    return mark_text(text, clusters)


def read_arcs(file_name):
    pattern = re.compile(r"\((\d+), (\d+)\)+")

    with open(file_name, encoding="utf-8") as f:
        for line in f:
            ana, ante = pattern.findall(line)
            ana_begin, ana_end = ana
            ante_begin, ante_end = ante
            yield ((int(ana_begin), int(ana_end)), (int(ante_begin), int(ante_end)))


def mark_text(text, clusters):
    tokenized = text.split()

    def add_mark(pos, mark, before):
        if before:
            tokenized[pos] = mark + tokenized[pos]
        else:
            tokenized[pos] = tokenized[pos] + mark

    mention_map = {}
    for id_cluster, cluster in enumerate(clusters):
        for mention in cluster:
            mention_map[mention] = id_cluster

    for mention in sorted(mention_map, key=lambda x: -x[0]):
        begin, end = mention
        id_cluster = mention_map[mention]

        add_mark(begin, f"<cluster_{id_cluster}>", True)
        add_mark(end, f"</cluster_{id_cluster}>", False)

    return " ".join(tokenized)


def create_clusters(arcs):
    map_to_cluster = {}

    for ana, ante in arcs:
        cluster_ante = map_to_cluster[ante] if ante in map_to_cluster else set()
        cluster_ana = map_to_cluster[ana] if ana in map_to_cluster else set()

        cluster = cluster_ante.union(cluster_ana)

        cluster.add(ana)
        cluster.add(ante)

        for mention in cluster:
            map_to_cluster[mention] = cluster

    clusters = []
    for c in map_to_cluster.values():
        if c not in clusters:
            clusters.append(c)

    return clusters
