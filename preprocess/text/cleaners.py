"""
Cleaners are transformations that run over the input text at both training and eval time.

Text cleaners for Croatian language.
"""


import re
from .numbers import normalize_numbers

_whitespace_re = re.compile(r'\s+')

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('prof', 'profesor'),
    ('dr', 'doktor'),
    ('sc', 'znanosti'),
    ('mg', 'miligrama'),
    ('gđa', 'gospođa'),
    ('gđica', 'gospođica'),
    ('gdin', 'gospodin'),
]]

abbreviation_letters = {
    "A": "a", "B": "be", "C": "ce", "Č": "če", "Ć": "će", "D": "de", "Đ": "đe",
    "E": "e", "F": "ef", "G": "ge", "H": "ha", "I": "i", "J": "je", "K": "ka",
    "L": "el", "M": "em", "N": "en", "O": "o", "P": "pe", "R": "er", "S": "es",
    "Š": "še", "T": "te", "U": "u", "V": "ve", "Z": "ze", "Ž": "že"
}

abbreviation_exceptions = {
    "NATO": "nato",
    "UNESCO": "unesko",
    "SFRJ": "sfrj",
}


def is_abbreviation(word):
    """Check if a word is an abbreviation (all uppercase and more than one letter)."""
    return word.isupper() and len(word) > 1


def expand_abbreviations(text):
    """Expand common abbreviations like 'dr.', 'mr.' to full words."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_abbreviation_exceptions(word):
    """Expand exceptions for abbreviations that are pronounced as full words."""
    if word in abbreviation_exceptions:
        return abbreviation_exceptions[word]
    elif is_abbreviation(word):
        return ''.join([abbreviation_letters.get(letter, letter) for letter in word])
    else:
        return word


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def croatian_cleaners(text):
    """Pipeline for Croatian text, including number and abbreviation expansion."""
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    words = text.split()
    expanded_words = []
    for word in words:
        word_body = word.rstrip('.,!?')
        punctuation = word[len(word_body):] if len(word_body) < len(word) else ''

        expanded_word = expand_abbreviation_exceptions(word_body) + punctuation
        expanded_words.append(expanded_word)

    text = ' '.join(expanded_words)
    text = collapse_whitespace(text)
    text = lowercase(text)
    return text
