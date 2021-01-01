"""
This module is an example of a custom implementation for extending the mentions classes. It is not part of
the main code and can be ignored.

This module uses spaCy en model. If you get an error,
run python -m download en

More info in  https://spacy.io/usage

"""
import numpy as np
import spacy
from num2words import num2words

# Loading spaCy
nlp = spacy.load('en_core_web_lg')


def increment_mention_pair(p, train_list):
    """
    Adds extra information about the mention pair
    :param p: mention pair to be used
    :param train_list: list of all lists in fhe file
    """
    count = 0
    m1 = p.mention1.mention_start
    m2 = p.mention2.mention_start
    if m1 < m2:
        for t in range(m1, m2 + 1):
            if train_list[t] == '\n':
                count += 1
    p.sentence_dist_count = _distance(count)

    p.mention_dist_count = _distance(p.mention1.index - p.mention2.index)
    p.head_match = p.mention1.head_word == p.mention2.head_word


def increment_mention(mention):
    """
    Adding information about head word and mention type
    :param mention:
    :return: None
    """

    mention_words = mention.mention

    doc = nlp(mention_words)
    if mention_words.isdigit() or mention_words == 'its' or mention_words.lower() == 'that' or mention_words.lower() == 'this':
        mention.head_word = ''
    else:
        if len(list(doc.noun_chunks)) > 0:
            mention.head_word = list(doc.noun_chunks)[0].root.head.text
        else:
            mention.head_word = ''
    mention.mention_type = _mention_type(doc, mention_words).tolist()
    mention.mention_length = _get_mention_length(mention_words)


def _get_mention_length(mention_words):
    """
    Length of the mention in words (i.e. if the len(mention_words) is 2, returns "two"
    :param mention_words:
    :return:
    """
    mention_len = len(mention_words)
    len_in_words = num2words(mention_len)
    return len_in_words


# pronoun: [1, 0, 0, 0]
# proper:  [0, 1, 0, 0]
# nominal(common noun): [0, 0, 1, 0]
# list:    [0, 0, 0, 1]
def _mention_type(doc, mention):
    """
    Identifies the mention type
    :param doc: the mention converted to tokens
    :param mention: the mention as string
    :return: one-hot identifier as this
        * pronoun: [1, 0, 0, 0]
        * proper:  [0, 1, 0, 0]
        * nominal(common noun): [0, 0, 1, 0]
        * list:    [0, 0, 0, 1]
    """
    # pos 0: pronoun, pos 1: proper noun, pos 2: common noun
    token_type = [0, 0, 0]
    for token in doc:
        if token.pos_ == 'PRON':
            token_type[0] += 1
        elif token.pos_ == 'PROPN':
            token_type[1] += 1
        elif token.pos_ == 'NOUN':
            token_type[2] += 1
    m = max(token_type)
    a = [i for i, j in enumerate(token_type) if j == m]
    is_dominant = m >= len(mention.split()) / 2
    if is_dominant:
        if a[0] == 0:
            return np.array([1, 0, 0, 0])
        if a[0] == 1:
            return np.array([0, 1, 0, 0])
        if a[0] == 2:
            return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


def _distance(a):
    """
    Represents a distance as a vector
    :param a:
    :return: a vector with 10 positions
    """
    d = np.zeros(10)
    d[a == 0, 0] = 1
    d[a == 1, 1] = 1
    d[a == 2, 2] = 1
    d[a == 3, 3] = 1
    d[a == 4, 4] = 1
    d[(5 <= a) & (a < 8), 5] = 1
    d[(8 <= a) & (a < 16), 6] = 1
    d[(16 <= a) & (a < 32), 7] = 1
    d[(a >= 32) & (a < 64), 8] = 1
    d[a >= 64, 9] = 1
    return d.tolist()
