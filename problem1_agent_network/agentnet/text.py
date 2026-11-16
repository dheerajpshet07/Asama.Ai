"""Tiny text normalisation shared by topic detection and the hashing embedder."""
from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from into over under about as is are was were be been
    being this that these those it its we our us you your they their them he she his her i me my mine
    will would can could should may might must shall do does did done have has had not no nor so than too
    very just also any all each every some such only own same other more most both few here there when
    where which who whom what why how up down out off again further then once now new get got via per
    across within upon onto
    """.split()
)
# "off" is meaningful here ("5% off"), so it is not a stopword for us.
STOPWORDS = STOPWORDS - {"off"}


@lru_cache(maxsize=50_000)
def stem(word: str) -> str:
    """Very light plural stripping. Good enough to match 'deliveries' with
    'delivery' and 'offers' with 'offer' without pulling in a stemmer."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    words = _WORD.findall(text.lower())
    if not keep_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [stem(w) for w in words]


def phrase_text(text: str) -> str:
    """Normalised text padded with spaces, for whole-phrase substring matching.
    Stopwords stay in, so 'tie up' or 'new year' only match as whole phrases."""
    return " " + " ".join(tokens(text, keep_stopwords=True)) + " "


def phrase(term: str) -> str:
    return " ".join(tokens(term, keep_stopwords=True))
