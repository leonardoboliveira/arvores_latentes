import collections
import os
from boilerplate import loader
from boilerplate import mentions
from boilerplate.mentions import CONLL_WORD_COLUMN
import random
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import tokenization
from pre_training.constants import BERT_MODEL_HUB, MAX_SEQ_LEN, MAX_CLASSES, MAX_NUM_PREDICTIONS
import numpy as np

SEP = "[SEP]"


class Span:
    def __init__(self, tokenizer, start_pos, words):
        self.start_pos = start_pos
        self.words = words
        self.end_pos = start_pos + len(words) - 1
        self.tokens = [tokenizer.tokenize(x) for x in words]
        self.tokens = [item for sublist in self.tokens for item in sublist]

    def token_len(self):
        return len(self.tokens)

    def __str__(self):
        return f"({self.start_pos}, {self.end_pos})"


class Cluster:
    def __init__(self):
        self.first_mention = None  # The first occurring mention
        self.mentions = []  # All other mentions in cluster

    def add_mention(self, mention):
        if mention in self.mentions:
            return

        if self.first_mention is None:  # First Mention
            self.first_mention = mention
        elif self.first_mention.compare(mention) > 0:
            self.mentions.append(self.first_mention)  # Save the mention that was the first
            self.first_mention = mention  # Set the new one
        else:
            self.mentions.append(mention)  # Just save the mention


class Line:
    def __init__(self, line):
        self.line = line
        self.word = None
        if isinstance(line, str):
            if ("#begin" in line) or ("#end" in line):
                return
            split = line.split()
            if len(split) >= CONLL_WORD_COLUMN:
                self.word = split[CONLL_WORD_COLUMN]

    def __str__(self):
        return self.line


class TreeNode:
    def __init__(self, mention, parent=None):
        self.mention = mention
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_node(self)

    def add_node(self, node):
        self.children.append(node)
        node.parent = self

    def add_mention(self, mention):
        node = TreeNode(mention)
        self.add_node(node)
        return node

    def delete(self):
        for c in self.children:
            self.parent.add_node(c)

        try:
            self.parent.children.remove(self)
        except ValueError:
            pass

    def get_mention_subtree(self):
        my_map = {}
        if self.mention is not None:
            my_map[self.mention.start_pos] = [self.mention.end_pos]

        for child in self.children:
            child_map = child.get_mention_subtree()
            for start, list_end in child_map.items():
                if start not in my_map:
                    my_map[start] = list_end
                else:
                    my_map[start] += list_end
        return my_map

    def __str__(self):
        return f"TreeNode ({self.parent}, {self.mention})"


class Tree:
    def __init__(self):
        self.root = TreeNode(None)
        self.mention_map = {}

    def add(self, mention, parent=None):
        if parent is None:
            return self.add(mention, self.root)

        return TreeNode(mention, parent)

    def get_parent_id(self, substructure_id, mention, max_depth=1000, skip_root=False):
        if max_depth <= 0:
            return []

        if mention not in self.mention_map:
            print("Not in mapping")
            return None

        node = self.mention_map[mention]
        parent = node.parent
        if parent is None or parent.mention is None:
            if skip_root:
                return []
            return [0]

        parent_id = parent.mention.get_seq_id(substructure_id)
        assert parent_id < MAX_CLASSES, f"Class too big {parent_id}"
        return [parent_id] + self.get_parent_id(substructure_id, parent.mention, max_depth - 1, True)

    @staticmethod
    def to_bits(value):
        return 1 << value  # Doing shift bitwise

    def update_mapping(self, new_map):
        self.mention_map.update(new_map)

    def remove_mention(self, mention):
        node = self.mention_map[mention]
        node.delete()

    def get_clusters(self):
        for child in self.root.children:
            yield child.get_mention_subtree()


class SuperStructure:
    def __init__(self, tokenizer, lines, use_cls_sep):
        self.structure_map = self.build_map(tokenizer, lines, use_cls_sep)
        self.size = len(lines)
        self.mentions = []

    def build_map(self, tokenizer, lines, use_cls_sep):
        all_groups = []
        for start, end in self.find_doc_spans(lines):
            spans = self.detect_spans(tokenizer, lines[start:end])
            # -1 because of [CLS] in the beginning and [SEP] in the end
            span_grouped = self.group_spans(spans, MAX_SEQ_LEN - 1)
            all_groups += [(x[0] + start, x[1] + start) for x in span_grouped]

        return {x: self.build_substructure(tokenizer, lines, x, use_cls_sep) for x in all_groups}

    @staticmethod
    def find_doc_spans(lines):
        start = -1
        end = -1
        for idx, line in enumerate(lines):
            if "#begin" in line:
                yield (start, end)
                start = idx
            elif "#end" in line:
                end = idx + 1

        yield (start, end)

    def add_mention(self, mention):
        self.mentions.append(mention)

        mention.master_id = len(self.mentions)

        for span, structure in self.structure_map.items():
            if span[0] <= mention.start_pos and span[1] >= mention.end_pos:
                structure.add_mention(mention)

    def get_random_examples(self, max_depth, use_full_seq):
        for structure in self.structure_map.values():
            yield structure.get_random_example(max_depth, use_full_seq)

    @staticmethod
    def build_substructure(tokenizer, lines, span, use_cls_sep):
        return Structure(tokenizer, lines[span[0]:(span[1] + 1)], span[0], use_cls_sep)

    @staticmethod
    def group_spans(spans, max_length):
        total_length = 0
        groups = []
        current_group = []

        def current_rep():
            start = current_group[0].start_pos
            end = current_group[-1].end_pos
            return start, end

        for span in spans:
            inc = span.token_len() + 1
            if total_length + inc > max_length:
                groups.append(current_rep())

                while total_length + inc > (max_length / 2) and len(current_group) > 0:
                    total_length -= current_group.pop(0).token_len() + 1

            if inc < max_length:  # If a single sentence is too big, skip it
                current_group.append(span)
                total_length += inc
            else:
                print(f"Skipping long sentence {total_length}, {inc}")

            assert total_length <= max_length, f"New length:{total_length}"

        if len(current_group) > 0:  # Last sentences
            assert sum([x.token_len() for x in current_group]) <= max_length
            groups.append(current_rep())

        return groups

    @staticmethod
    def detect_spans(tokenizer, lines):
        spans = []
        running = False
        start = 0
        words = []
        for idx, line in enumerate(lines):
            line = line.strip("\n").strip("\r")
            if "#begin" in line:
                continue
            if len(line) == 0 or "#end" in line:
                if running:
                    running = False
                    spans.append(Span(tokenizer, start, words))
                    words = []
                continue
            if not running:
                start = idx
                running = True
            words.append(line.split()[CONLL_WORD_COLUMN])

        return spans

    def __len__(self):
        return sum([len(s) for s in self.structure_map.values()])


class Structure:
    def __init__(self, tokenizer, lines, first_position, use_cls_sep):
        self.cls_token = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.sep_token = tokenizer.convert_tokens_to_ids([SEP])[0]
        self.use_cls_sep = use_cls_sep
        self.lines = [Line(x) for x in lines]
        self.clusters = {}
        self.flat = [[] for x in range(len(lines))]
        self.first_position = first_position
        self.tokens = [[SEP] if l.word is None else tokenizer.tokenize(l.word) for l in self.lines]
        if not use_cls_sep:
            for tk in self.tokens:
                if SEP in tk:
                    tk.remove(SEP)
        self.tokens = [tokenizer.convert_tokens_to_ids(t) for t in self.tokens]

        if sum([len(x) for x in self.tokens]) > MAX_SEQ_LEN:
            print("Too much")

    def __len__(self):
        return sum([len(x) for x in self.flat])

    def add_mention(self, mention):
        mention_id = mention.mention_id

        if mention_id not in self.clusters:
            self.clusters[mention_id] = Cluster()

        self.clusters[mention_id].add_mention(mention)
        self.flat[mention.start_pos - self.first_position].append(mention)

    def get_random_example(self, max_depth, use_full_seq):
        tree = self.get_random_tree()
        ids, parents = self.build_vectors(tree, max_depth, use_full_seq)

        tokens = []
        seq_ids = []
        labels = []
        if self.use_cls_sep:
            tokens.append(self.cls_token)
            seq_ids.append(0)
            labels.append([])

        ex_id = 0
        for index, line in enumerate(self.lines):
            # if line.word is None:
            #    continue
            index_id = ids[index]
            parent_id = parents[index]
            for tk in self.tokens[index]:
                tokens.append(tk)
                seq_ids.append(index_id)
                labels.append(parent_id)
                ex_id += sum(parent_id) * tk

        if len(labels) > MAX_SEQ_LEN:
            print("Too much")

        assert len(labels) <= MAX_SEQ_LEN, f"Too big {len(labels)}, {self.first_position}"

        return to_example(tokens, seq_ids, labels), ex_id

    def build_vectors(self, tree, max_depth, use_full_seq):
        ids = [0] * len(self.flat)
        parents = [[] for i in range(len(self.flat))]

        counter = 1
        for i in range(len(ids)):
            mentions = self.flat[i]
            if len(mentions) == 0:
                continue
            mentions.sort(key=lambda x: x.end_pos)
            for mention in mentions:
                # mention = self.clean_inconsistency(mentions, tree)
                end_pos = mention.end_pos + 1 - self.first_position
                start_pos = mention.start_pos - self.first_position

                parent_id = tree.get_parent_id(self.first_position, mention, max_depth)

                for j in range(start_pos, end_pos):
                    ids[j] = counter
                parents[j] += parent_id
                mention.set_seq_id(self.first_position, counter)
                counter += 1

        if use_full_seq:  # Will override all values in ids
            counter = 0
            for i in range(len(ids)):
                ids[i] = counter
                if self.sep_token in self.tokens[i]:
                    counter += 1

        return ids, parents

    def get_random_tree(self):
        tree = Tree()
        for cluster in self.clusters.values():
            subtree, mapping = self.get_random_subtree(cluster)
            tree.root.add_node(subtree)
            tree.update_mapping(mapping)
        return tree

    @staticmethod
    def get_random_subtree(cluster):
        mapping = {}
        root = TreeNode(cluster.first_mention)
        mapping[cluster.first_mention] = root

        max_height = 0
        for mention in cluster.mentions:
            next_root = root
            height = 1
            rnd = random.random()
            last = round(rnd * max_height)
            for i in range(last):
                # only descent in tree in children that occur before the current mention
                possible_children = [x for x in next_root.children if x.mention.compare(mention) < 0]

                if len(possible_children) == 0:
                    break
                next_root = random.choice(possible_children)
                height += 1
            node = next_root.add_mention(mention)
            mapping[mention] = node

            if height >= max_height:
                max_height = height

        return root, mapping


def to_k_hot(values):
    output = np.zeros((MAX_NUM_PREDICTIONS, MAX_CLASSES), dtype=np.int64)

    for tk, value in enumerate(values):
        for idx in value:
            output[tk, idx] = 1

    return output


def create_khot_feature(values):
    values = to_k_hot(values)
    o_shape = values.shape
    values = values.reshape((o_shape[0] * o_shape[1]))
    return create_int_feature(values)


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def to_example(tokens, seq_ids, labels):
    features = collections.OrderedDict()
    masks = [1] * len(tokens)

    missing = [0] * (MAX_SEQ_LEN - len(tokens))
    for vec in [tokens, seq_ids, masks]:
        vec += missing

    features["input_ids"] = create_int_feature(tokens)
    features["segment_ids"] = create_int_feature(seq_ids)
    features["input_mask"] = create_int_feature(masks)

    lm_positions = np.where([len(k) > 0 for k in labels])[0]
    masked_lm_ids = [labels[x] for x in lm_positions]
    masked_lm_weights = np.ones(len(masked_lm_ids), dtype=np.int64)

    assert len(masked_lm_weights) <= MAX_NUM_PREDICTIONS, f"Too much predictions:{len(masked_lm_weights)}"

    missing = np.zeros(MAX_NUM_PREDICTIONS - len(masked_lm_weights), dtype=np.int64)
    lm_positions = np.concatenate([lm_positions, missing])
    masked_lm_weights = np.concatenate([masked_lm_weights, missing])
    masked_lm_ids += [[]] * len(missing)

    features["masked_lm_positions"] = create_int_feature(lm_positions)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["masked_lm_ids"] = create_khot_feature(masked_lm_ids)

    return tf.train.Example(features=tf.train.Features(feature=features))


def build_structure(tokenizer, lines, use_cls_sep):
    structure = SuperStructure(tokenizer, lines, use_cls_sep)

    for mention in build_mention_list(lines):
        structure.add_mention(mention)

    return structure


def build_mention_list(lines):
    return build_mention_list_boilerplate(lines)


def build_mention_list_boilerplate(lines):
    return mentions.build_mention_list(lines)


def build_mention_list_cort(lines):
    from cort.core import corpora
    from cort.core import mention_extractor

    document_as_strings = []

    current_document = ""

    for line in lines:
        if line.startswith("#begin") and current_document != "":
            document_as_strings.append(current_document)
            current_document = ""
        current_document += line

    document_as_strings.append(current_document)

    training_corpus = corpora.Corpus("test", sorted([corpora.from_string(doc) for
                                                     doc in document_as_strings]))
    for idx, doc in enumerate(training_corpus):
        for mention in mention_extractor.extract_system_mentions(doc):
            # print(f"New Mention:{mention}")

            if mention.span is None:
                continue

            words = mention.attributes["tokens"]
            # The cort convension is not aligned to ours. It is 0 based, and does not count begin/end/blank lines
            # So, to adjust, need:
            # +1 for #begin
            # +1 for 1-based
            # +sentence_id to account for spaces between sentenced
            start_pos = mention.span.begin + 2 + mention.attributes["sentence_id"]
            end_pos = mention.span.end + 2 + mention.attributes["sentence_id"]
            my_mention = mentions.Mention(f"{idx}_{start_pos}", words, start_pos, end_pos)
            yield my_mention


def write_examples(file_path, num_examples, tokenizer, out_file, max_depth, use_full_seq, use_cls_sep):
    lines = loader.train_file_to_list(file_path)
    structure = build_structure(tokenizer, lines, use_cls_sep)

    if len(structure) == 0:
        print(f"No examples for {file_path}")
        return

    hit = 0
    with tf.io.TFRecordWriter(out_file) as writer:
        found = set()
        for _ in range(num_examples):
            for example, ex_id in structure.get_random_examples(max_depth, use_full_seq):
                print('.', end='')
                if ex_id in found:
                    hit += 1
                    if hit > 3:
                        return
                    continue
                else:
                    hit = 0
                found.add(ex_id)
                writer.write(example.SerializeToString())

            if len(found) >= num_examples:
                break


def create_tokenizer():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def build_files(path_in, path_out, num_examples=1000, max_depth=1000, use_full_seq=True, use_cls_sep=True):
    print("Creating tokenizer")
    tokenizer = create_tokenizer()

    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue

            write_examples(os.path.join(r, file_name), num_examples, tokenizer, f"{path_out}/{file_name}.tsv",
                           max_depth, use_full_seq, use_cls_sep)


if __name__ == "__main__":
    root = "D:/GDrive/Puc/Projeto Final/Datasets/"

    path_out = r"D:\ProjetoFinal\data\finetune"
    # path_out = f"{root}/finetuning/positional/devel_tf_full_seq/"
    path_in = f"{root}/conll/development"

    build_files(path_in, path_out, max_depth=1, use_full_seq=False, use_cls_sep=False)
