# qwert

Constrained text generation where every letter is typed by **one hand**
on a QWERTY keyboard.

```
left hand:   q w e r t  a s d f g  z x c v b      digits 1-5
right hand:  y u i o p  h j k l    n m            digits 6-0
```

Runs locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx)
via `mlx-lm`. Phase 1 (this code) forces the constraint with a static
token mask, then picks the most natural results by reranking. Phase 2
(a LoRA fine-tune) is designed but not built: see
[docs/phase2-lora.md](docs/phase2-lora.md).

## Setup

Needs Python 3.10+ and, for generation, an Apple Silicon Mac.

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev,stats]"
```

```bash
pytest
```

The tests use a tiny fake tokenizer, so they download nothing. The
first `qwert generate` downloads the model from Hugging Face.

## Usage

```bash
qwert generate --hand left --prompt "Write a short poem about the sea" -n 20 --top 3 --max-tokens 120
```

Each result is printed with its score and its length in characters:

```
[1] score -2.413  (logprob -2.313, repeats 5%)  87 chars
<text>
```

Other commands:

```bash
qwert stats --hand right
```

```bash
qwert verify --hand left "great sweater"
```

Useful `generate` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `default` | MLX repo id, local path, or an alias (below). Also `$QWERT_MODEL`. |
| `--punctuation` | `free` | `free` any punctuation, `strict` only the chosen hand's keys, `none` |
| `-n` / `--top` | 20 / 3 | candidates to sample / results to print |
| `--temperature`, `--top-p` | 0.9, 0.95 | sampling |
| `--repetition-penalty` | 1.15 | mlx-lm's penalty over the last `--repetition-context` (64) tokens; 1.0 disables |
| `--no-repeat-ngram` | 3 | never repeat a token n-gram of this size; 0 disables |
| `--repeat-weight` | 2.0 | rerank penalty for repeated words |
| `--seed` | none | reproducible sampling |
| `--no-hint` | off | do not tell the model about the constraint in the prompt |
| `--adapter-path` | none | LoRA adapter, for Phase 2 |

## Models

The model is never hard-coded: pass `--model`, or set `QWERT_MODEL`.

| Alias | Repo | Use |
| --- | --- | --- |
| `default` | `mlx-community/gemma-4-E4B-it-qat-4bit` | ~4B, fast, small |
| `quality` | `mlx-community/gemma-4-12B-it-qat-4bit` | ~12B, better prose |
| `fallback` | `mlx-community/gemma-3-4b-it-qat-4bit` | older, if Gemma 4 will not load |
| `fallback-quality` | `mlx-community/gemma-3-12b-it-qat-4bit` | older 12B |

Why Gemma: instruction-tuned, no "thinking" mode to switch off, and a
large SentencePiece vocabulary (about 262k entries according to
mlx-lm's Gemma 4 model code), which gives the mask more whole words to
keep. Quantisation-aware-trained (QAT) 4-bit builds lose less quality
than ordinary 4-bit ones. A 12B model at 4 bits is roughly 7-8 GB of
weights, which fits in 16 GB with the browser closed; anything much
larger does not.

**What I could not verify.** The Gemma 4 repo names come from a web
search of the mlx-community "Gemma 4 QAT" collection, and mlx-lm 0.32.2
contains Gemma 4 model code. The machine this was written on has no
access to Hugging Face, so I have not downloaded or loaded either
model, and I do not know their exact sizes or speed on an M4. If a
model fails to load, try `--model fallback`, or any Qwen build such as
`mlx-community/Qwen3-4B-4bit` (Qwen templates have a thinking mode;
qwert asks the template to turn it off, but I have not tested that on a
real Qwen model).

## How it works

1. **Mask** (`mask.py`). Walk the tokenizer vocabulary once. Decode each
   token and allow it only if every character passes the hand rule.
   Byte-fallback / partial-UTF-8 tokens, non-ASCII letters, tabs and
   emoji are banned. Special tokens are never judged by spelling (`<s>`
   would otherwise pass as left-handed): EOS and the tokens the chat
   template uses are allowed, the rest banned. The mask is cached in
   `~/.cache/qwert/masks/` per model + hand + punctuation mode.
2. **Generate** (`processors.py`, `generate.py`). The mask is an mlx-lm
   logits processor that sets banned logits to `-inf`, alongside a
   repetition penalty and a no-repeat-n-gram processor. Generation stops
   at any allowed special token.
3. **Rerank** (`rerank.py`). Each candidate is scored by the same model
   with **no mask**: mean per-token log-probability, minus a penalty for
   repeated words. Higher is better.
4. **Verify** (`verify.py`). Every candidate is checked by a verifier
   that is written separately from the mask. Anything that fails is
   dropped and reported, never printed.

The mlx-lm API was checked against the source of version 0.32.2:
a logits processor is `f(tokens, logits) -> logits`, where `tokens`
includes the prompt and `logits` has shape `(1, vocab)`; `generate_step`
takes `sampler=` and `logits_processors=`.

### Rules that were my judgment calls

- Newline is allowed like space. Tab is not.
- Punctuation belongs to the hand of its key, Shift ignored: left has
  only `` ` ~ ! @ # $ % ``; everything else, including `. , ? ' "`, is
  right-hand. So `--punctuation strict` with the left hand cannot end a
  sentence with a full stop.
- Curly quotes, dashes and the ellipsis character count as the ASCII
  key they stand for.
- Digits: 1-5 left, 6-0 right, in every punctuation mode.

## Left vs right: what to expect

**I have not been able to run a real model**, so these are expectations
from the keyboard itself, not measurements. Run `qwert stats` for both
hands and replace this section with real numbers.

- **Left is much easier.** It has 15 letters including `e a t r s d`,
  the backbone of English: *a, we, as, at, are, was, were, see, get,
  best, great, after, water, street*. Full, if odd, sentences are
  possible ("We saw a great sweater").
- **Right is hard.** 11 letters, the vowels `i o u` (and `y`) but no
  `e`, `a`, `t` or `s`: *I, you, in, on, my, no, oh, up, him, only,
  look, moon, pink, jolly*. No plurals in `-s`, no past tense in `-ed`.
  Expect shorter, stranger output, lower scores and more looping.
- **Neither hand has** *the, and, is, it, to, of, that, for, with* —
  each mixes hands. So even good output reads like a telegram.

Things worth checking once it runs: the surviving-token share for each
hand, whether the right hand needs a stronger repetition penalty, and
whether `--no-hint` helps or hurts.

## Layout

```
src/qwert/
  hands.py       key tables and the per-character rule
  verify.py      independent verifier
  mask.py        vocabulary walk, special tokens, cache
  stats.py       surviving-vocabulary report
  processors.py  mlx-lm logits processors (mask, no-repeat n-gram)
  generate.py    prompt building and masked sampling
  rerank.py      unmasked scoring and ranking
  models.py      model aliases and loading
  cli.py         command line
  finetune.py    Phase 2 stub
tests/           pytest, with a fake tokenizer
docs/phase2-lora.md
```

Only `processors.py`, `generate.py` and the scoring function in
`rerank.py` import MLX, so everything else is testable anywhere.
