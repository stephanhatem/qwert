"""Verifier: check any string against the one-hand constraint.

Deliberately independent of the token mask. It reads the key tables in
``hands`` (they are the specification) but does not call
``hands.char_allowed``, which is what the mask builder uses. Instead it
builds an explicit whitelist of characters and tests membership, so a
logic slip on one side is caught by the other. ``tests/test_verify.py``
checks that the two agree on every ASCII character.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .hands import (
    ALL_PUNCTUATION,
    DIGITS,
    LETTERS,
    PUNCTUATION,
    TYPOGRAPHIC,
    WHITESPACE,
    Hand,
    Punctuation,
)


@dataclass(frozen=True)
class Violation:
    index: int
    char: str

    def __str__(self) -> str:
        return f"{self.char!r} at position {self.index}"


@lru_cache(maxsize=None)
def allowed_characters(hand: Hand, punctuation: Punctuation) -> frozenset[str]:
    """Every character that may appear in valid text, both cases included."""
    letters = LETTERS[hand]
    chars = set(WHITESPACE) | letters | {c.upper() for c in letters} | DIGITS[hand]
    if punctuation is Punctuation.FREE:
        chars |= ALL_PUNCTUATION
    elif punctuation is Punctuation.STRICT:
        keys = PUNCTUATION[hand]
        chars |= keys | {fancy for fancy, key in TYPOGRAPHIC.items() if key in keys}
    return frozenset(chars)


def verify(
    text: str,
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
) -> list[Violation]:
    """Return every character of ``text`` that breaks the constraint."""
    allowed = allowed_characters(Hand(hand), Punctuation(punctuation))
    return [Violation(i, ch) for i, ch in enumerate(text) if ch not in allowed]


def is_valid(
    text: str,
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
) -> bool:
    return not verify(text, hand, punctuation)
