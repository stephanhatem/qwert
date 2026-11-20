"""Masked sampling with mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .hands import LETTERS, Hand
from .mask import TokenMask, _as_id_list
from .processors import make_mask_processor, make_no_repeat_ngram_processor


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 120
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.15   # 1.0 disables
    repetition_context: int = 64
    no_repeat_ngram: int = 3           # 0 disables
    seed: int | None = None


@dataclass(frozen=True)
class Candidate:
    text: str
    tokens: tuple[int, ...]  # generated ids, without the stop token
    finished: bool           # True if the model stopped by itself


def constraint_hint(hand: Hand) -> str:
    letters = " ".join(sorted(LETTERS[hand]))
    return (
        f"Write using only words that can be typed with the {hand.value} hand "
        f"on a QWERTY keyboard. The only letters available are: {letters}."
    )


def build_prompt(tokenizer: Any, prompt: str, hand: Hand, hint: bool = True) -> list[int]:
    """Tokenise the user prompt through the model's chat template.

    The hint tells the model about the constraint. It goes in the user
    turn because not every chat template (older Gemma) has a system role.
    """
    content = f"{constraint_hint(hand)}\n\n{prompt}" if hint else prompt
    if not getattr(tokenizer, "chat_template", None):
        return _as_id_list(tokenizer.encode(content))
    # enable_thinking=False: reasoning tags cannot be spelled with one hand,
    # so a "thinking" template would only waste the masked model's tokens.
    # Templates without that variable ignore it.
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_id_list(ids)


def _processors(mask: TokenMask, config: SamplingConfig) -> list:
    penalty = config.repetition_penalty if config.repetition_penalty != 1.0 else None
    processors = make_logits_processors(
        repetition_penalty=penalty,
        repetition_context_size=config.repetition_context,
    )
    if config.no_repeat_ngram > 0:
        processors.append(make_no_repeat_ngram_processor(config.no_repeat_ngram))
    # The mask goes last so nothing after it can revive a banned token.
    processors.append(make_mask_processor(mask))
    return processors


def sample_one(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    mask: TokenMask,
    config: SamplingConfig,
) -> Candidate:
    stop_ids = set(mask.stop_ids)
    sampler = make_sampler(temp=config.temperature, top_p=config.top_p)
    tokens: list[int] = []
    finished = False
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=config.max_tokens,
        sampler=sampler,
        logits_processors=_processors(mask, config),
    ):
        if token in stop_ids:
            finished = True
            break
        tokens.append(int(token))
    return Candidate(tokenizer.decode(tokens).strip(), tuple(tokens), finished)


def sample_candidates(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    mask: TokenMask,
    config: SamplingConfig,
    n: int,
    on_candidate: Callable[[int, Candidate], None] | None = None,
) -> list[Candidate]:
    """Sample ``n`` candidates one after another.

    Sequential on purpose: it is simple and memory use stays flat.
    mlx-lm also has a ``BatchGenerator`` that would be faster; I have
    not measured it with per-sequence logits processors.
    """
    if config.seed is not None:
        mx.random.seed(config.seed)
    candidates: list[Candidate] = []
    for index in range(n):
        candidates.append(sample_one(model, tokenizer, prompt_ids, mask, config))
        if on_candidate is not None:
            on_candidate(index + 1, candidates[-1])
    return candidates
