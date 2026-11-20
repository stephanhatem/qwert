"""Rerank candidates by how natural the *unmasked* model finds them.

    score = mean per-token log-probability
            - repeat_weight  * share of repeated words
            - nonword_weight * share of non-words

The log-probability comes from the same model with no mask, so it
measures fluency rather than rewarding text the mask forced. Higher
(closer to zero) is better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .hands import Hand, Punctuation
from .verify import verify

WORD = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Scored:
    text: str
    score: float
    mean_logprob: float
    repeat_fraction: float
    nonword_fraction: float | None = None  # None: not measured (no word list)

    @property
    def chars(self) -> int:
        return len(self.text)


def repeat_fraction(text: str) -> float:
    """Share of words that repeat an earlier word: 0.0 none, near 1.0 a loop."""
    words = [w.lower() for w in WORD.findall(text)]
    if not words:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def nonword_fraction(text: str, is_word: Callable[[str], bool]) -> float:
    """Share of words that ``is_word`` rejects, e.g. "Frrff" or "Zzzt"."""
    words = [w.lower() for w in WORD.findall(text)]
    if not words:
        return 0.0
    return sum(not is_word(w) for w in words) / len(words)


def mean_logprob(model: Any, prompt_ids: Sequence[int], token_ids: Sequence[int]) -> float:
    """Mean log-probability of ``token_ids`` given the prompt, no mask.

    One forward pass over prompt + candidate; position i predicts token i+1.
    """
    import mlx.core as mx

    if not token_ids:
        return float("-inf")
    start = len(prompt_ids)
    inputs = mx.array([*prompt_ids, *token_ids])[None]
    logits = model(inputs)[0, start - 1 : start - 1 + len(token_ids)].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = mx.array(list(token_ids))[:, None]
    return mx.take_along_axis(logprobs, targets, axis=-1).mean().item()


def rerank(
    candidates: Sequence[Any],
    score_fn: Callable[[Sequence[int]], float],
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
    top: int = 3,
    repeat_weight: float = 2.0,
    min_chars: int = 1,
    is_word: Callable[[str], bool] | None = None,
    nonword_weight: float = 3.0,
) -> tuple[list[Scored], int]:
    """Verify, score and sort candidates. Returns (best ``top``, rejected count).

    ``candidates`` need ``.text`` and ``.tokens``; ``score_fn`` maps token
    ids to a mean log-probability. Every candidate goes through the
    verifier, and anything that fails is dropped and counted, never shown.
    Duplicates and texts shorter than ``min_chars`` are skipped. Without
    ``is_word`` there is no non-word penalty.
    """
    rejected = 0
    seen: set[str] = set()
    scored: list[Scored] = []
    for candidate in candidates:
        text = candidate.text
        if len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        if verify(text, hand, punctuation):
            rejected += 1
            continue
        logprob = score_fn(candidate.tokens)
        repeats = repeat_fraction(text)
        nonwords = nonword_fraction(text, is_word) if is_word else None
        score = logprob - repeat_weight * repeats - nonword_weight * (nonwords or 0.0)
        scored.append(Scored(text, score, logprob, repeats, nonwords))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top], rejected
