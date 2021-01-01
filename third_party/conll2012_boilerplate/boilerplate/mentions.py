"""
Mentions Module

This module will handle the parse of the files and creation of a MentionsPair class that will be used to generate
the output files.
"""

import difflib
import re

from tqdm import tqdm

CONLL_DOC_ID_COLUMN = 0
CONLL_PART_NUM_COLUMN = 1
CONLL_WORD_SEQ = 2
CONLL_WORD_COLUMN = 3
CONLL_POS_COLUMN = 4
CONLL_PARSE_BIT_COLUMN = 5
CONLL_LEMMA_COLUMN = 6
CONLL_FRAMESET_ID_COLUMN = 7
CONLL_SENSE_COLUMN = 8
CONLL_SPEAKER_COLUMN = 9
CONLL_NAMED_COLUMN = 10


class Mention:
    """
    This class represents a mention with all its features. It can be extended if more information is needed
    """

    def __init__(self, m_id, words, start_pos, end_pos):
        self.words = words
        self.mention_id = m_id
        self.doc_id = m_id.split('_')[0]
        self.mention = " ".join(words)
        self.first_word = words[0]
        self.last_word = words[-1]
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.seq_ids = {}

    def compare(self, other):
        start_diff = self.start_pos - other.start_pos
        if start_diff != 0:
            return start_diff
        return self.end_pos - other.end_pos

    def __str__(self):
        return f"Mention ({self.doc_id},{self.start_pos},{self.end_pos})"

    def set_seq_id(self, substructre_id, seq_id):
        self.seq_ids[substructre_id] = seq_id

    def get_seq_id(self, substructre_id):
        return self.seq_ids[substructre_id]


class MentionPair:
    """
    This class represents a pair of mentions. It can be extended if more information is needed
    """

    def __init__(self, mention1, mention2):
        self.mention1 = mention1
        self.mention2 = mention2
        self.coref = mention1.mention_id == mention2.mention_id

    def get_info_vector(self):
        """
        Return the information to enable the recovery of the original mention pair
        :return: [start/end position for both mentions]
        """
        return [self.mention1.start_pos, self.mention1.end_pos, self.mention2.start_pos, self.mention2.end_pos]


def check_usable_pairs(mention_list, i, j):
    """
    Default implementation for checking if two mentions should be considered as a pair. These pairs are not necessarily
    correferences
    :param mention_list: list of mentions
    :param i: first index
    :param j: second index
    :return: True if mention i and mention j should be paired
    """
    if mention_list[i].doc_id != mention_list[j].doc_id:
        return False
    if mention_list[i].mention_id == mention_list[j].mention_id:
        return True
    # Skipping some indexes so not all pairs will be generated
    if j % 2 == 0 or j % 3 == 0 or j % 5 == 0 or j % 7 == 0 or j % 11 == 0:
        return False
    return True


def build_mention_list(train_list, fill_information=None):
    """
    Build a list of dictionaries with the information about each mention. The mentions are not yet grouped

    :param train_list: list of lines in the document
    :param fill_information: function to add more information into the cluster
    :return:
    """
    mentions = []
    cluster_start, start_pos, cluster_end, end_pos = _get_mention(train_list)
    mention_cluster = _create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos)
    for m in tqdm(mention_cluster, desc="mentions"):
        m_id, start_pos, end_pos = m

        mention_words = get_mention_words(train_list, start_pos, end_pos)
        mention = Mention(m_id, mention_words, start_pos, end_pos)

        # Building features
        mention.mention_start = start_pos
        mention.mention_end = end_pos
        mention.pre_words = _get_preceding_words(train_list, start_pos)
        mention.next_words = _get_next_words(train_list, end_pos)
        mention.mention_sentence = _mention_sentence(train_list, start_pos)
        mention.speaker = train_list[start_pos - 1].split()[CONLL_SPEAKER_COLUMN]

        # This will allow external info to be added
        if fill_information:
            fill_information(mention)

        mentions.append(mention)

    mentions = sorted(mentions, key=lambda k: k.mention_start)
    mentions = _check_mention_contain(mentions)
    mentions = _get_index(mentions)
    return mentions


def _add_extra_pair_info(mention_pair_list, train_list, increment_mention_pair=None):
    """
    Adds distance information about the mention pairs.
    'overlap' : True (1) if the second element overlaps the first
    'speaker' : True (1) if both mentions have the same speaker
    'mention_exact_match' : True(1) if both mentions are from the same sentence
    'mention_partial_match' : True(1) if the sentences are similar

    :param mention_pair_list: list of pair of mentions
    :param train_list: list of lines in the document
    :param increment_mention_pair: function to add more information to each pair
    :return:
    """
    for p in mention_pair_list:
        p.overlap = p.mention1.overlap == p.mention2.mention_id
        p.speaker = p.mention1.speaker == p.mention2.speaker
        p.mention_exact_match = p.mention1.mention == p.mention2.mention

        seq = difflib.SequenceMatcher(None, p.mention1.mention, p.mention2.mention)
        score = seq.ratio()
        p.mention_partial_match = score > 0.6

        if increment_mention_pair:
            increment_mention_pair(p, train_list)

    return mention_pair_list


def _get_mention(train_list):
    """
    Creates four auxiliary lists with IDs and positions. These lists are not completely related to each other. It will
    simply gather the open/close parenthesis in the order it appears in the document.
    Check the formatting of a CoNLL document in the site

        * Cluster start: ID of the opening parenthesis cluster
        * Start pos: line of starting position
        * Cluster end: ID of the closing parenthesis cluster (
        * End pos: line

    :param train_list: List of lines of the file
    :return: cluster_start, start_pos, cluster_end, end_pos
    """
    cluster_start = []
    start_pos = []
    cluster_end = []
    end_pos = []
    i = 1
    for line in train_list:
        if line == '\n' or line == '-':  # Ignore empty lines
            i += 1
            continue
        part_number = line.split()[CONLL_PART_NUM_COLUMN]  # Part number column
        coref_col = line.split()[-1]  # Coref column
        for j in range(len(coref_col)):  # There can be more than one parenthesis. Tracking all of them
            if coref_col[j] == '(':
                cluster_start.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[j + 1:])[0]))
                start_pos.append(i)
            if coref_col[j] == ')':
                cluster_end.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[:j])[-1]))
                end_pos.append(i)
        i += 1
    return cluster_start, start_pos, cluster_end, end_pos


def _create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos):
    """
    Builds a list of mentions. One mention per item. The clusters are not yet grouped
     * First position is the id = [document_cluster]
     * Second position is the starting line
     * Third position in the ending line (counting from header).
     Use function get_mention to build the lists properly.

    :param cluster_start: List of IDs of starting cluster IDs (opening parenthesis)
    :param start_pos: List of positions for the starting cluster ID
    :param cluster_end: List of IDs for ending cluster (closing parenthesis)
    :param end_pos: List of positions for the ending cluster ID
    :return: List with three items and length of the same size of the input lists
    """
    cluster_start_end_list = []
    for start, pos in zip(cluster_start, start_pos):  # Join ID and position
        cluster = [start, pos]
        i = 0
        for i in range(len(cluster_end)):  # Search where is the closing parenthesis
            if cluster_end[i] == start:
                cluster.append(end_pos[i])  # Found it. Add to the pairing
                break
        del cluster_end[i]  # Remove from original list. This will prevent from being used again
        del end_pos[i]
        cluster_start_end_list.append(cluster)
    return cluster_start_end_list


def get_mention_words(train_list, pos1, pos2):
    """
    Gets the list of words between these lines
    :param train_list: lines of the document
    :param pos1: initial line
    :param pos2: final line
    :return: List of all words
    """
    mention = []
    for line_no in range(pos1 - 1, pos2):
        word = train_list[line_no].split()[CONLL_WORD_COLUMN]
        mention.append(word)
    return mention


def _get_preceding_words(train_list, pos, max_words=5):
    """
    Get the previous max_words in the document (if they exists)
    :param train_list: list of lines
    :param pos: word position (numer of the line)
    :param max_words: max words to look ahead
    :return:
    """
    word_part = train_list[pos - 1].split()[CONLL_PART_NUM_COLUMN]
    num_words = 0
    word = []
    for i in range(2, pos):
        if train_list[pos - i] != '\n':
            doc_id = train_list[pos - i].split()[CONLL_DOC_ID_COLUMN]  # Checking if begin/end of document was reached
            if doc_id == '#begin' or doc_id == '#end':
                break
            part_no = train_list[pos - i].split()[CONLL_PART_NUM_COLUMN]
            if part_no == word_part:  # Will add only words for the same part
                word.append(train_list[pos - i].split()[CONLL_WORD_COLUMN])
                num_words += 1
            if num_words == max_words:
                break
    return word


def _get_next_words(train_list, pos, max_words=5):
    """
    Get the next max_words in the document (if they exists)
    :param train_list: list of lines
    :param pos: word position (numer of the line)
    :param max_words: max words to look ahead
    :return:
    """
    pos = pos - 1
    word_part = train_list[pos].split()[CONLL_PART_NUM_COLUMN]
    num_words = 0
    word = []
    for i in range(1, len(train_list) - pos):
        if train_list[pos + i] != '\n':
            doc_id = train_list[pos + i].split()[CONLL_DOC_ID_COLUMN]  # Checking if begin/end of document was reached
            if doc_id == '#begin' or doc_id == '#end':
                break
            part_no = train_list[pos + i].split()[CONLL_PART_NUM_COLUMN]
            if part_no == word_part:
                word.append(train_list[pos + i].split()[CONLL_WORD_COLUMN])
                num_words += 1
            if num_words == max_words:
                break
        i += 1
    return word


def _mention_sentence(train_list, pos):
    """
    Gets the sentence that contains the mention in this line
    :param train_list: list of lines
    :param pos: line of reference
    :return: string with sentence
    """
    pos = pos - 1
    i = 1
    end = 0
    while True:
        if train_list[pos - i] == '\n':
            start = pos - i
            break
        if train_list[pos - i].split()[CONLL_DOC_ID_COLUMN] == '#begin':
            start = pos - i
            break
        i += 1
    start += 2
    i = 1

    while True:
        if train_list[pos + i] == '\n':
            end = pos + i
            break
        if train_list[pos + i].split()[CONLL_DOC_ID_COLUMN] == '#end':
            start = pos + i
            break
        i += 1
    sentence = get_mention_words(train_list, start, end)
    return " ".join(sentence)


def _check_mention_contain(mention_list):
    """
    Adds information about mentions containing/overlapping other mentions. The difference between contain and overlap
    will be defined by the end position. In this context, overlapping is not reflexive.
    The keys 'contained'/'overlap' will have the id of the container/overlapped mention or False if it is not contained

    :param mention_list: list of dictionaries, one item per mention. Each item must have the keys:
        mention_start. mention_end and id

    :return: the modified list
    """
    for i in range(0, len(mention_list)):
        start = mention_list[i].mention_start
        end = mention_list[i].mention_end

        for j in range(0, len(mention_list)):
            c_start = mention_list[j].mention_start
            c_end = mention_list[j].mention_end

            if c_start == start and c_end == end:
                continue

            if c_start >= start:  # If starts after the ref
                if c_end <= end:  # and end before, it is contained by the ref
                    mention_list[j].contained = mention_list[i].mention_id
                if c_start <= end:  # if starts before the end of the ref, only overlaps
                    mention_list[j].overlap = mention_list[i].mention_id

    for k in range(0, len(mention_list)):  # Filling missing info
        if not hasattr(mention_list[k], 'contained'):
            mention_list[k].contained = False
        if not hasattr(mention_list[k], 'overlap'):
            mention_list[k].overlap = False

    return mention_list


def _get_index(mentions):
    """
    Adds the 'index' and 'mention_position'.
    'index' is a counter for each mention in each document
    'mention_position' is the index in a [0,1] interval (0 is the first, 1 is the last).
    :param mentions: list of all mentions
    :return: the same list of input, with objects changed
    """
    doc_count = '0'
    count = 0
    i = 0
    mentions_in_each_doc = []
    for m in mentions:
        if m.doc_id == doc_count:  # Keep counting while it's the same doc
            count += 1
        else:
            mentions_in_each_doc.append(count)  # save the counter
            doc_count = m.doc_id
            count = 1
        m.index = count
    mentions_in_each_doc.append(count)  # save last counter

    # Now will transform into [0,1,...]
    doc_count = '0'
    for m in mentions:
        if m.doc_id == doc_count:
            m.mention_position = m.index / mentions_in_each_doc[i]
        else:
            doc_count = m.doc_id
            i += 1
            m.mention_position = m.index / mentions_in_each_doc[i]

    return mentions


def get_mention_pairs(train_list, increment_mention_info=None, increment_mention_pair=None,
                      use_pair=check_usable_pairs):
    """
    Builds a list of pair of mentions for the file. Each pair may or may not have a coreference. Each position
    simulates an object with the first two positions being mentions and the following are dictionaries with extra
    features

    :param train_list: list of lines in the file
    :param increment_mention_info: function to add more information to the mention
    :param increment_mention_pair: function to add more information to the mention pair
    :param use_pair: function to define if two mentions should be paired or not
    :return: list of objects
    """
    mention_list = build_mention_list(train_list, increment_mention_info)
    mention_pair_list = []
    for i in tqdm(range(1, len(mention_list)), desc="mention pair"):
        for j in range(0, i):
            if use_pair(mention_list, i, j):
                pair = MentionPair(mention_list[i], mention_list[j])
                mention_pair_list.append(pair)

    # Adding extra info
    mention_pair_list = _add_extra_pair_info(mention_pair_list, train_list, increment_mention_pair)

    return mention_pair_list
