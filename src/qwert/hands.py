"""The QWERTY hand mapping: which hand types which key.

This module is the single statement of the constraint. It has no
dependencies, so both the mask builder and the verifier can read the
tables while keeping their checking logic separate.

Conventions (standard touch typing, Shift ignored):

* Letters: the split given in the project brief.
* Digits: 1-5 left, 6-0 right.
* Punctuation belongs to the hand of its key, shifted or not, so ``!``
  (Shift+1) is left and ``?`` (Shift+/) is right.
* Space is always allowed. Newline is treated the same way, as a
  judgment call: it is whitespace, and banning it makes chat models
  behave oddly. Tab and every other control character are banned.
"""

from __future__ import annotations

import string
from enum import Enum


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Punctuation(str, Enum):
    FREE = "free"      # any punctuation
    STRICT = "strict"  # only punctuation typed by the chosen hand
    NONE = "none"      # no punctuation at all


LETTERS: dict[Hand, frozenset[str]] = {
    Hand.LEFT: frozenset("qwertasdfgzxcvb"),
    Hand.RIGHT: frozenset("yuiophjklnm"),
}

DIGITS: dict[Hand, frozenset[str]] = {
    Hand.LEFT: frozenset("12345"),
    Hand.RIGHT: frozenset("67890"),
}

PUNCTUATION: dict[Hand, frozenset[str]] = {
    Hand.LEFT: frozenset("`~!@#$%"),
    Hand.RIGHT: frozenset("^&*()-_=+[]{}\\|;:'\",<.>/?"),
}

WHITESPACE: frozenset[str] = frozenset(" \n")

# Typographic punctuation that models emit in place of ASCII, mapped to
# the key a person would press for it. All of these are right-hand keys.
TYPOGRAPHIC: dict[str, str] = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en and em dash
    "…": ".",                  # ellipsis
}

ALL_PUNCTUATION: frozenset[str] = frozenset(string.punctuation) | frozenset(TYPOGRAPHIC)


def char_allowed(ch: str, hand: Hand, punctuation: Punctuation) -> bool:
    """Return True if the single character ``ch`` satisfies the constraint."""
    if ch in WHITESPACE:
        return True
    if ch.isalpha():
        # Non-ASCII letters are not on the keyboard, and some of them
        # lower-case to ASCII (the Kelvin sign becomes "k"), so check first.
        return ch.isascii() and ch.lower() in LETTERS[hand]
    if ch.isdigit():
        return ch.isascii() and ch in DIGITS[hand]
    if ch in ALL_PUNCTUATION:
        if punctuation is Punctuation.FREE:
            return True
        if punctuation is Punctuation.STRICT:
            return TYPOGRAPHIC.get(ch, ch) in PUNCTUATION[hand]
        return False
    return False
