"""Static vocabulary mask: which token ids may be generated.

The tokenizer is duck-typed (see ``TokenizerLike``) so the builder works
with a Hugging Face tokenizer, mlx-lm's ``TokenizerWrapper`` around one,
or the tiny fake used in the tests. Nothing here imports MLX.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .hands import Hand, Punctuation, char_allowed

# Bump when the rules below change, so stale cache files are ignored.
MASK_VERSION = 1

# Word-start markers used in raw vocabulary pieces:
# SentencePiece (Gemma, Llama) and byte-level BPE (Qwen, GPT-2).
SPACE_MARKERS = ("▁", "Ġ")
REPLACEMENT_CHAR = "�"


class TokenizerLike(Protocol):
    all_special_ids: list[int]

    def __len__(self) -> int: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def convert_ids_to_tokens(self, ids: int) -> str: ...


@dataclass(frozen=True)
class TokenMask:
    allowed: tuple[int, ...]   # every id that may be sampled, sorted
    stop_ids: tuple[int, ...]  # the allowed special tokens (EOS, chat template)
    vocab_size: int

    @property
    def text_ids(self) -> tuple[int, ...]:
        """Allowed ids that are ordinary text rather than special tokens."""
        stops = set(self.stop_ids)
        return tuple(i for i in self.allowed if i not in stops)


def token_text(tokenizer: TokenizerLike, token_id: int) -> str:
    """Decode one token to the text it contributes mid-sentence.

    Decoding a lone SentencePiece token drops its leading space, so the
    space is restored from the raw piece. A partial UTF-8 sequence
    (byte-fallback token) decodes to U+FFFD, which callers treat as banned.
    """
    text = tokenizer.decode([token_id])
    piece = tokenizer.convert_ids_to_tokens(token_id)
    if isinstance(piece, str) and piece.startswith(SPACE_MARKERS) and not text.startswith(" "):
        text = " " + text
    return text


def text_allowed(text: str, hand: Hand, punctuation: Punctuation) -> bool:
    """True if a token's text may be emitted. Empty text is banned."""
    if not text or REPLACEMENT_CHAR in text:
        return False
    return all(char_allowed(ch, hand, punctuation) for ch in text)


def special_ids(tokenizer: TokenizerLike) -> set[int]:
    """Ids of control tokens, which are never judged by their spelling.

    Without this, a token such as ``<s>`` would pass a left-hand mask.
    """
    ids = set(tokenizer.all_special_ids)
    added: dict[int, Any] = getattr(tokenizer, "added_tokens_decoder", None) or {}
    ids.update(i for i, tok in added.items() if getattr(tok, "special", False))
    return ids


def eos_ids(tokenizer: TokenizerLike) -> set[int]:
    # mlx-lm's TokenizerWrapper exposes ``eos_token_ids`` (a set, which can
    # hold several end tokens); plain HF tokenizers only ``eos_token_id``.
    ids = set(getattr(tokenizer, "eos_token_ids", None) or ())
    single = getattr(tokenizer, "eos_token_id", None)
    if single is not None:
        ids.add(single)
    return ids


def _as_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "keys"):  # BatchEncoding or dict
        encoded = encoded["input_ids"]
    return [int(i) for i in encoded]


def chat_template_ids(tokenizer: TokenizerLike) -> set[int]:
    """Special tokens the chat template itself uses (turn markers, BOS).

    Found by rendering a dummy conversation and keeping the special ids
    that appear. Returns an empty set for models without a chat template.
    """
    if not getattr(tokenizer, "chat_template", None):
        return set()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "a"}]
    found: set[int] = set()
    for msgs, add_prompt in ((messages, False), (messages[:1], True)):
        ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            msgs, tokenize=True, add_generation_prompt=add_prompt
        )
        found.update(_as_id_list(ids))
    # <unk> can show up when the dummy text is out of vocabulary; it is
    # never a template token.
    found.discard(getattr(tokenizer, "unk_token_id", None))
    return found & special_ids(tokenizer)


def build_mask(
    tokenizer: TokenizerLike,
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
) -> TokenMask:
    """Walk the vocabulary once and decide every token."""
    specials = special_ids(tokenizer)
    stops = eos_ids(tokenizer) | chat_template_ids(tokenizer)
    vocab_size = len(tokenizer)
    allowed: list[int] = []
    for token_id in range(vocab_size):
        if token_id in stops:
            allowed.append(token_id)
        elif token_id in specials:
            continue
        elif text_allowed(token_text(tokenizer, token_id), hand, punctuation):
            allowed.append(token_id)
    return TokenMask(tuple(allowed), tuple(sorted(stops)), vocab_size)


# --- cache -----------------------------------------------------------------


def cache_dir() -> Path:
    root = os.environ.get("QWERT_CACHE_DIR")
    return Path(root) if root else Path.home() / ".cache" / "qwert"


def cache_path(model: str, hand: Hand, punctuation: Punctuation, root: Path | None = None) -> Path:
    if Path(model).exists():  # a local model: key on its full path
        model = str(Path(model).resolve())
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "--", model).strip("-.")
    name = f"{safe_model}--{Hand(hand).value}--{Punctuation(punctuation).value}.json"
    return (root or cache_dir()) / "masks" / name


def save_mask(mask: TokenMask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MASK_VERSION,
        "vocab_size": mask.vocab_size,
        "stop_ids": list(mask.stop_ids),
        "allowed": list(mask.allowed),
    }
    path.write_text(json.dumps(payload))


def load_mask(path: Path, vocab_size: int) -> TokenMask | None:
    """Read a cached mask; None if missing, unreadable, or for another vocab."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != MASK_VERSION or payload.get("vocab_size") != vocab_size:
        return None
    return TokenMask(tuple(payload["allowed"]), tuple(payload["stop_ids"]), vocab_size)


def load_or_build_mask(
    tokenizer: TokenizerLike,
    model: str,
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
    root: Path | None = None,
) -> TokenMask:
    """Cached per model + hand + punctuation mode."""
    path = cache_path(model, hand, punctuation, root)
    mask = load_mask(path, len(tokenizer))
    if mask is None:
        mask = build_mask(tokenizer, hand, punctuation)
        save_mask(mask, path)
    return mask


def allowed_lookup(mask: TokenMask, size: int | None = None) -> list[bool]:
    """Dense boolean table, optionally padded/truncated to ``size`` entries."""
    table = [False] * (size if size is not None else mask.vocab_size)
    for token_id in _in_range(mask.allowed, len(table)):
        table[token_id] = True
    return table


def _in_range(ids: Iterable[int], size: int) -> Iterable[int]:
    return (i for i in ids if 0 <= i < size)
