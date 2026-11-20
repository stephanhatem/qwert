# Phase 2 design note: LoRA fine-tune for one-handed vocabulary

Status: **not built**. `src/qwert/finetune.py` is a stub and
`qwert finetune` only points here.

## Why

Phase 1 forces the model: at every step the mask removes most of what
the model wanted to say, and it has to improvise with what is left. It
never planned for the constraint, so it drifts into word salad and
loops. A small LoRA adapter should teach the model to *plan* in the
restricted vocabulary, so that the tokens it wants are mostly tokens it
is allowed. The mask stays on at inference as a guarantee; the adapter
only makes the mask bite less often.

One adapter per hand (left and right vocabularies barely overlap).

## Training data

Target: a few thousand examples per hand. Three sources, in order of
how much I trust them.

1. **Public-domain text that already satisfies the constraint.**
   Take Project Gutenberg / Wikisource text, split into sentences and
   clauses, and keep those where `verify()` returns no violations.
   Expect very few full sentences, especially for the right hand, so
   also slide a window over the text and keep maximal valid runs of at
   least ~4 words ("we were at a great sea") as phrase-level examples.

2. **Near misses, repaired.** Keep sentences where only one or two
   words violate the constraint, then repair them: drop the offending
   word if the sentence still reads, or replace it with an allowed word.
   A simple way to choose the replacement is Phase 1 itself: mask the
   slot and let the masked model fill it, then keep the result only if
   the unmasked rerank score is above a threshold. Every repaired
   sentence goes through `verify()` again.

3. **The best Phase 1 outputs.** Run `qwert generate` over a list of
   varied prompts with a large `-n`, and keep only top-ranked results
   above a score threshold and with a low repeat fraction. This is
   self-distillation, with a known risk: the model learns its own
   habits, including its favourite few words. Cap this source at about
   a third of the set and deduplicate aggressively (drop examples that
   share a long word n-gram with one already kept).

Format: mlx-lm's `chat` JSONL (`{"messages": [...]}`), one file each
for `train.jsonl`, `valid.jsonl`, `test.jsonl` in one directory. The
user turn carries the same constraint hint that `build_prompt` uses in
Phase 1 plus a task ("Write a sentence about ..."), so training and
inference prompts match. For source 1 and 2 the task has to be invented
from the text (a topic word, or a generic "write a sentence"); a mix of
specific and generic tasks is fine.

Hold out prompts, not just examples: the evaluation prompts below must
not appear in training in any form.

## Training

Use mlx-lm's own tooling rather than a custom loop, on the ~4B model
(the 16 GB machine has room to train it; the 12B model is for
inference only):

```bash
pip install "mlx-lm[train]"
mlx_lm.lora --model <4B model> --train --data data/left \
    --mask-prompt --iters 600 --adapter-path adapters/left
```

Checked against the `LORA.md` shipped in mlx-lm 0.32.2: `--train`,
`--data`, `--iters`, `--adapter-path`, `--mask-prompt` (loss on the
completion only, which is what we want), `--fine-tune-type`, and a YAML
`--config` all exist, and a quantized `--model` means QLoRA is used
automatically. **Not verified:** that LoRA works on the Gemma 4 model
family specifically (the supported-families list in that file names
"Gemma" without versions), sensible values for rank, learning rate,
batch size and number of layers, and memory use on 16 GB. Start from
the defaults, watch validation loss, and lower batch size / number of
tuned layers if memory is tight.

The training loss is ordinary next-token loss with no mask. An open
question worth one experiment: applying the mask inside the loss
(renormalising over allowed tokens) would match inference exactly, but
needs a custom training loop, so try the plain version first.

Inference then needs no new code: `qwert generate --adapter-path
adapters/left` already passes the adapter to `mlx_lm.load`.

## Comparing Phase 1 and Phase 2

Same prompts, same sampling settings, same seed, same `-n`, mask on in
both. Use a fixed list of ~50 held-out prompts.

The judge must not move. Phase 1's rerank score is the mean per-token
log-probability under the *unmasked base model*. Keep that exact judge
for both arms: score Phase 2 outputs with the base model **without the
adapter**. If the adapted model scored its own outputs, it would be
grading its own homework, and the numbers would not be comparable.
(This needs a small change: load the base model a second time, or
score after generation with the adapter removed.)

Report per prompt and averaged:

- best-of-N score and mean score of all N candidates (the mean shows
  whether the typical sample improved, not just the lucky one);
- repeat fraction, and distinct words across all outputs (to catch
  collapse onto a few favourite words);
- share of candidates that finish by themselves before `--max-tokens`;
- **mask pressure**: probability mass the model puts on allowed tokens
  before masking, averaged over steps. This is the most direct measure
  of "learned instead of forced". Phase 1 does not record it yet; it is
  a few lines in the mask processor.

And read the outputs. A higher log-probability does not prove the text
means anything; twenty side-by-side pairs judged blind is a better
final check than any of the numbers above.
