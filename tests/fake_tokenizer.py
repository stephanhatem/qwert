"""A tiny stand-in tokenizer so tests need no model download.

It mimics the parts of a Hugging Face tokenizer that qwert touches:
raw pieces with a SentencePiece word-start marker, single-token decode
that drops the leading space, byte-fallback pieces that decode to
U+FFFD, special tokens, and a chat template.
"""

from __future__ import annotations

from dataclasses import dataclass

MARKER = "▁"

PIECES: list[str] = [
    "<pad>",            # 0  special, banned
    "<s>",              # 1  special BOS; spells a left-hand letter
    "</s>",             # 2  special EOS
    "<start_of_turn>",  # 3  special, chat template
    "<end_of_turn>",    # 4  special, chat template
    "<unused0>",        # 5  special, not in the template: banned
    MARKER + "tree",    # 6  left
    MARKER + "sweater", # 7  left
    "ed",               # 8  left, word continuation
    MARKER + "Great",   # 9  left, capitalised
    MARKER + "you",     # 10 right
    MARKER + "moon",    # 11 right
    "ly",               # 12 right, word continuation
    MARKER + "the",     # 13 mixed
    MARKER,             # 14 a bare space
    "\n",               # 15 newline
    ".",                # 16 right-hand punctuation
    "!",                # 17 left-hand punctuation
    "<0xE2>",           # 18 byte fallback: partial UTF-8
    "café",        # 19 non-ASCII letter
    MARKER + "12",      # 20 left digits
    MARKER + "67",      # 21 right digits
    "\t",               # 22 tab
    MARKER + "\U0001f600",  # 23 emoji
    "user",             # 24 mixed; used by the chat template
    MARKER + "a",       # 25 left
]


@dataclass
class AddedToken:
    content: str
    special: bool


class FakeTokenizer:
    eos_token_id = 2
    chat_template = "fake"

    def __init__(self) -> None:
        self.pieces = list(PIECES)
        self.all_special_ids = [0, 1, 2]
        # Like Gemma: some control tokens are only flagged in added_tokens_decoder.
        self.added_tokens_decoder = {
            i: AddedToken(self.pieces[i], special=True) for i in (0, 1, 2, 3, 4, 5)
        }
        self.decode_calls = 0

    def __len__(self) -> int:
        return len(self.pieces)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[token_id]

    def decode(self, token_ids: list[int]) -> str:
        self.decode_calls += 1
        text = "".join(
            "�" if self.pieces[i].startswith("<0x") else self.pieces[i]
            for i in token_ids
        ).replace(MARKER, " ")
        # SentencePiece decoders drop the space at the start of a sequence.
        return text[1:] if text.startswith(" ") else text

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_):
        ids = [1]
        for message in messages:
            ids += [3, 24, 25, 4]
        if add_generation_prompt:
            ids += [3]
        return ids
