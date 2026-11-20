"""Command-line interface: ``qwert generate | stats | verify | finetune``."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

from .hands import Hand, Punctuation
from .models import ENV_VAR, MODEL_ALIASES, resolve_model
from .verify import verify


def _add_constraint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand", choices=[h.value for h in Hand], required=True)
    parser.add_argument(
        "--punctuation",
        choices=[p.value for p in Punctuation],
        default=Punctuation.FREE.value,
        help="free: any; strict: only the chosen hand's keys; none (default: free)",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    aliases = ", ".join(MODEL_ALIASES)
    parser.add_argument(
        "--model",
        help=f"MLX repo id, local path, or alias ({aliases}). "
        f"Falls back to ${ENV_VAR}, then 'default'.",
    )
    parser.add_argument("--adapter-path", help="LoRA adapter directory (Phase 2)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the token mask")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="sample candidates, rerank, print the best")
    _add_constraint_args(gen)
    _add_model_args(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("-n", "--num-candidates", type=int, default=20)
    gen.add_argument("--top", type=int, default=3, help="how many results to print")
    gen.add_argument("--max-tokens", type=int, default=120)
    gen.add_argument("--temperature", type=float, default=0.9)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--repetition-penalty", type=float, default=1.15, help="1.0 disables")
    gen.add_argument("--repetition-context", type=int, default=64)
    gen.add_argument("--no-repeat-ngram", type=int, default=3, help="0 disables")
    gen.add_argument("--repeat-weight", type=float, default=2.0,
                     help="rerank penalty per unit of repeated-word fraction")
    gen.add_argument("--nonword-weight", type=float, default=3.0,
                     help="rerank penalty per unit of non-word fraction; 0 disables "
                     "(needs the wordfreq package)")
    gen.add_argument("--no-examples", action="store_true",
                     help="do not show the model example sentences in the prompt")
    gen.add_argument("--min-chars", type=int, default=1, help="skip shorter candidates")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--no-hint", action="store_true",
                     help="do not tell the model about the constraint in the prompt")
    gen.set_defaults(func=cmd_generate)

    stats = sub.add_parser("stats", help="report which vocabulary survives the mask")
    _add_constraint_args(stats)
    _add_model_args(stats)
    stats.add_argument("--words", type=int, default=40, help="how many top words to list")
    stats.set_defaults(func=cmd_stats)

    ver = sub.add_parser("verify", help="check a string against the constraint")
    _add_constraint_args(ver)
    ver.add_argument("text")
    ver.set_defaults(func=cmd_verify)

    fine = sub.add_parser("finetune", help="Phase 2 (not implemented yet)")
    fine.set_defaults(func=cmd_finetune)
    return parser


def _load(args: argparse.Namespace):
    """Load model, tokenizer and mask for the chosen constraint."""
    from .mask import build_mask, load_or_build_mask
    from .models import load_model

    hand, punctuation = Hand(args.hand), Punctuation(args.punctuation)
    name = resolve_model(args.model)
    print(f"model: {name}", file=sys.stderr)
    model, tokenizer = load_model(name, args.adapter_path)
    print("building token mask (first run for this model is slower)...", file=sys.stderr)
    if args.no_cache:
        mask = build_mask(tokenizer, hand, punctuation)
    else:
        mask = load_or_build_mask(tokenizer, name, hand, punctuation)
    return model, tokenizer, mask, hand, punctuation


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import SamplingConfig, build_prompt, sample_candidates
    from .lexicon import load_word_check
    from .rerank import mean_logprob, rerank

    model, tokenizer, mask, hand, punctuation = _load(args)
    config = SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_context=args.repetition_context,
        no_repeat_ngram=args.no_repeat_ngram,
        seed=args.seed,
    )
    prompt_ids = build_prompt(
        tokenizer, args.prompt, hand, hint=not args.no_hint, examples=not args.no_examples
    )
    is_word = load_word_check() if args.nonword_weight else None
    if args.nonword_weight and is_word is None:
        print("note: wordfreq is not installed, so non-words are not penalised "
              "(pip install 'qwert[stats]')", file=sys.stderr)
    started = time.perf_counter()

    def progress(done: int, candidate) -> None:
        elapsed = time.perf_counter() - started
        print(f"\rsampling {done}/{args.num_candidates}  ({elapsed:.0f}s)",
              end="", file=sys.stderr, flush=True)

    print("sampling...", end="", file=sys.stderr, flush=True)
    candidates = sample_candidates(
        model, tokenizer, prompt_ids, mask, config, args.num_candidates, progress
    )
    print("\nscoring...", file=sys.stderr, flush=True)
    results, rejected = rerank(
        candidates,
        lambda tokens: mean_logprob(model, prompt_ids, tokens),
        hand,
        punctuation,
        top=args.top,
        repeat_weight=args.repeat_weight,
        min_chars=args.min_chars,
        is_word=is_word,
        nonword_weight=args.nonword_weight,
    )
    if rejected:
        print(f"warning: {rejected} candidate(s) failed the verifier and were dropped",
              file=sys.stderr)
    if not results:
        print("no valid candidates; try a higher -n or --max-tokens", file=sys.stderr)
        return 1
    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank}] score {result.score:.3f}  "
            f"(logprob {result.mean_logprob:.3f}, repeats {result.repeat_fraction:.0%}, "
            f"non-words {result.nonword_fraction:.0%})  "
            f"{result.chars} chars"
        )
        print(result.text)
        print()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .stats import format_stats, vocab_stats

    _, tokenizer, mask, hand, punctuation = _load(args)
    print(f"hand: {hand.value}   punctuation: {punctuation.value}")
    print(format_stats(vocab_stats(tokenizer, mask, top=args.words)))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    violations = verify(args.text, Hand(args.hand), Punctuation(args.punctuation))
    if not violations:
        print("ok")
        return 0
    print(f"{len(violations)} violation(s): " + ", ".join(str(v) for v in violations))
    return 1


def cmd_finetune(args: argparse.Namespace) -> int:
    from .finetune import DESIGN_NOTE

    print(f"Phase 2 (LoRA fine-tuning) is not implemented yet. See {DESIGN_NOTE}.",
          file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
