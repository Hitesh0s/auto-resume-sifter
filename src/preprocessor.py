import re
import string
import functools

import nltk
import spacy

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _nltk_stopwords

_STOP_WORDS: set[str] = set(_nltk_stopwords.words("english"))


@functools.lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text: str) -> str:
    """Clean, tokenize, remove stopwords, and lemmatize text.

    Returns a single whitespace-joined string of lemmas, ready for TF-IDF.
    """
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = _nlp()(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in _STOP_WORDS and len(token.lemma_) > 1
    ]
    return " ".join(tokens)
