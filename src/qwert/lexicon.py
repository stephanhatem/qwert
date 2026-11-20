"""Real-word check, used to push noise like "Frrff" and "Zzzt" down the ranking."""

from __future__ import annotations

from typing import Callable

WordCheck = Callable[[str], bool]


def load_word_check() -> WordCheck | None:
    """Return ``is_word(word)``, or None if no word list is installed.

    Uses the optional ``wordfreq`` package (``pip install qwert[stats]``):
    a word counts as real if wordfreq has seen it at all. That is
    deliberately lenient, so names (Rex) and rare words (zax) pass and
    only letter-noise fails. The macOS system dictionary is not used as
    a fallback because it lacks plurals and would flag "crabs".
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    return lambda word: zipf_frequency(word, "en") > 0
