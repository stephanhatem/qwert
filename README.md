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
than ordinary 4-bit ones. The `quality` model is an 11 GB
download, which is tight on a 16 GB machine: close other apps first,
and expect it to be much slower than `default` if macOS starts
swapping.

**What has been verified.** The `default` model downloads, loads and
generates on a 16 GB M4 (see results below). The mlx-lm API was read
from the 0.32.2 source. **Not yet verified:** the `quality` model (the
repo name comes from a web search of the mlx-community "Gemma 4 QAT"
collection; it has not been loaded), and the fallbacks. Any other MLX
repo works with `--model`, for example `mlx-community/Qwen3-4B-4bit`.
Qwen templates have a thinking mode; qwert asks the template to turn it
off, but that is untested on a real Qwen model.

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

## Left vs right: results

Measured on a MacBook Air M4 (16 GB) with the `default` model,
`mlx-community/gemma-4-E4B-it-qat-4bit`, punctuation `free`. This is
one run per hand with default settings (`-n 20 --top 3`), so treat the
scores as an indication, not a benchmark.

Vocabulary (`qwert stats`, 262,144 tokens in total):

| | Left | Right |
| --- | --- | --- |
| Surviving tokens | 13,946 (5.3%) | 7,731 (2.9%) |
| Whole English words | 2,865 | 813 |
| Most common | a was be as are at we were get after see great best state better | in i you on my up no only him look oh john million july upon |

The whole-word counts include a few single letters (`s`, `d`, `t`),
which the tokenizer stores as word-start tokens.

Best three of 20, prompt "Write a short poem about the sea" (left) and
"... about the moon" (right):

| | Left | Right |
| --- | --- | --- |
| Scores | -1.37, -1.59, -1.77 | -1.96, -2.33, -2.35 |
| Length | 72-79 chars | 45-57 chars |

Left, best result:

```
Sea vast, free.

Waves crest fast,
Wet breeze gas.

Far west,
Safe rest.
```

Right, second result (the first was shorter and stranger):

```
Moon, you loom,
No hum, no jump.
Only milk,
Loom in lull.
```

What this shows:

- **Left is easier, as expected.** Nearly twice the tokens and about
  3.5 times the whole words, with `e a t r s d` available. The poems
  rhyme (*crest / west / rest / best*) and mostly make sense.
- **Right is harder but not hopeless.** No `e`, `a`, `t` or `s`, so no
  plurals and no past tense. Output is shorter, scores are about 0.6
  lower, and non-words slip in ("in lu"). Still, "Moon, you loom" is a
  real line.
- **Candidates cluster.** All three left-hand results opened with
  "Sea vast". A higher `--temperature` (1.1) and a larger `-n` should
  give more variety; not yet tried.
- **The mask shows in places**: "Wet breeze gas", "Only milk". The
  model takes the nearest allowed word.
- **Neither hand has** *the, and, is, it, to, of, that, for, with* —
  each mixes hands. So even good output reads like a telegram.

Not yet tried: the `quality` model, `--punctuation strict` and `none`,
and whether `--no-hint` helps or hurts.

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
