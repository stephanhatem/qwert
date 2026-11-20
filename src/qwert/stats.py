"""Vocabulary statistics for a mask: what survives the constraint."""

from __future__ import annotations

from dataclasses import dataclass

from .mask import TokenMask, TokenizerLike, token_text


@dataclass(frozen=True)
class VocabStats:
    vocab_size: int
    allowed_tokens: int   # ordinary text tokens that survive
    stop_tokens: int      # allowed special tokens
    whole_words: int      # distinct surviving whole words
    top_words: list[str]
    ranked_by: str        # how top_words was ordered

    @property
    def allowed_fraction(self) -> float:
        return self.allowed_tokens / self.vocab_size if self.vocab_size else 0.0


def surviving_words(tokenizer: TokenizerLike, mask: TokenMask) -> list[str]:
    """Distinct surviving whole words, in token-id order.

    A "whole word" is a token that starts a word (leading space) and is
    purely alphabetic. That still admits word-initial fragments such as
    " str"; a tokenizer alone cannot tell those from real words.
    """
    seen: dict[str, None] = {}
    for token_id in mask.text_ids:
        text = token_text(tokenizer, token_id)
        word = text[1:]
        if text.startswith(" ") and word.isalpha():
            seen.setdefault(word.lower(), None)
    return list(seen)


def rank_words(words: list[str]) -> tuple[list[str], str]:
    """Order words most-common first.

    Uses the optional ``wordfreq`` package when installed, which also
    drops fragments that are not English words. Otherwise falls back to
    token-id order. Lower ids are *often* more frequent in BPE-style
    vocabularies, but that is a heuristic I could not verify for every
    tokenizer, so the result is labelled accordingly.
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return words, "token id (approximate; install qwert[stats] for real frequencies)"
    scored = [(zipf_frequency(w, "en"), w) for w in words]
    ranked = [w for freq, w in sorted(scored, key=lambda p: -p[0]) if freq > 0]
    return ranked, "English word frequency (wordfreq)"


def vocab_stats(tokenizer: TokenizerLike, mask: TokenMask, top: int = 40) -> VocabStats:
    words, ranked_by = rank_words(surviving_words(tokenizer, mask))
    return VocabStats(
        vocab_size=mask.vocab_size,
        allowed_tokens=len(mask.text_ids),
        stop_tokens=len(mask.stop_ids),
        whole_words=len(words),
        top_words=words[:top],
        ranked_by=ranked_by,
    )


def format_stats(stats: VocabStats) -> str:
    return "\n".join(
        [
            f"vocabulary size:   {stats.vocab_size}",
            f"surviving tokens:  {stats.allowed_tokens} ({stats.allowed_fraction:.1%})"
            f" + {stats.stop_tokens} stop token(s)",
            f"whole words:       {stats.whole_words}",
            f"top words by {stats.ranked_by}:",
            "  " + " ".join(stats.top_words),
        ]
    )
