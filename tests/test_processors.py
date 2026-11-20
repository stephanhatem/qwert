"""Logits-processor tests. Skipped where MLX is not installed."""

import pytest

mx = pytest.importorskip("mlx.core")

from fake_tokenizer import FakeTokenizer  # noqa: E402

from qwert.hands import Hand  # noqa: E402
from qwert.mask import build_mask  # noqa: E402
from qwert.processors import (  # noqa: E402
    banned_next_tokens,
    make_mask_processor,
    make_no_repeat_ngram_processor,
)


def test_mask_processor_bans_everything_outside_the_mask():
    mask = build_mask(FakeTokenizer(), Hand.LEFT)
    width = mask.vocab_size + 4  # model output wider than the tokenizer
    out = make_mask_processor(mask)(mx.array([1, 2]), mx.zeros((1, width)))
    kept = [i for i, v in enumerate(out[0].tolist()) if v == 0.0]
    assert kept == list(mask.allowed)
    assert all(v == float("-inf") for i, v in enumerate(out[0].tolist()) if i not in kept)


def test_banned_next_tokens():
    assert banned_next_tokens([1, 2, 3, 1, 2], 3) == [3]
    assert banned_next_tokens([1, 2, 3, 4], 3) == []
    assert banned_next_tokens([5, 6, 5, 7, 5], 2) == [6, 7]
    assert banned_next_tokens([4, 4], 1) == [4]
    assert banned_next_tokens([1, 2], 3) == []


def test_no_repeat_ngram_ignores_the_prompt():
    processor = make_no_repeat_ngram_processor(2)
    prompt = mx.array([7, 8, 7])
    first = processor(prompt, mx.zeros((1, 10)))
    assert first[0].tolist() == [0.0] * 10  # 7 -> 8 is in the prompt: not banned
    later = processor(mx.array([7, 8, 7, 3, 4, 3]), mx.zeros((1, 10)))
    assert [i for i, v in enumerate(later[0].tolist()) if v != 0.0] == [4]
