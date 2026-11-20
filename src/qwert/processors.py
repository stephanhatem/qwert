"""Logits processors for mlx-lm.

mlx-lm (checked against 0.32.2, ``mlx_lm/sample_utils.py``) defines

    LogitsProcessor = Callable[[mx.array, mx.array], mx.array]

called as ``processor(tokens, logits)`` where ``tokens`` is the 1-D array
of every token so far *including the prompt*, and ``logits`` has shape
``(1, vocab)``. Processors run before the sampler.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from .mask import TokenMask, allowed_lookup

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


def make_mask_processor(mask: TokenMask) -> LogitsProcessor:
    """Set the logit of every banned token to -inf."""
    allowed: mx.array | None = None

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        nonlocal allowed
        width = logits.shape[-1]
        if allowed is None or allowed.shape[0] != width:
            # Built on first use because the model's output width can be
            # larger than the tokenizer vocabulary (padded embeddings).
            allowed = mx.array(allowed_lookup(mask, width))
        return mx.where(allowed, logits, -mx.inf)

    return processor


def make_no_repeat_ngram_processor(n: int) -> LogitsProcessor:
    """Ban any token that would complete an n-gram already generated.

    Only generated tokens count; the prompt is skipped by remembering how
    long ``tokens`` was on the first call. Make a fresh processor for
    each generation.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    prompt_length: int | None = None

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        nonlocal prompt_length
        if prompt_length is None:
            prompt_length = len(tokens)
        banned = banned_next_tokens(tokens[prompt_length:].tolist(), n)
        if banned:
            logits[:, mx.array(banned)] = -mx.inf
        return logits

    return processor


def banned_next_tokens(generated: list[int], n: int) -> list[int]:
    """Tokens that would repeat an n-gram if appended to ``generated``."""
    if len(generated) < n:
        return []
    prefix = generated[len(generated) - (n - 1):] if n > 1 else []
    banned = {
        generated[i + n - 1]
        for i in range(len(generated) - n + 1)
        if generated[i:i + n - 1] == prefix
    }
    return sorted(banned)
