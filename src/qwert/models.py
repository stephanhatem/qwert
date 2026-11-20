"""Model selection. The model is always a CLI/config choice, never fixed.

Resolution order: ``--model`` flag, then the ``QWERT_MODEL`` environment
variable, then the ``default`` alias below.

NOTE ON VERIFICATION. These repo names were found by web search (they
are listed in the mlx-community "Gemma 4 QAT" collection), and mlx-lm
0.32.2 ships ``gemma4`` / ``gemma4_text`` model code. I could not
download or load them where this was written (no Hugging Face access),
so exact download sizes and load behaviour on a 16 GB machine are
unconfirmed. If a repo fails to load, pass any other MLX repo or local
path with ``--model``; ``fallback`` and ``fallback-quality`` are older
Gemma 3 builds kept as known-good alternatives.
"""

from __future__ import annotations

import os
from typing import Any

MODEL_ALIASES: dict[str, str] = {
    # ~4B, 4-bit: fast, leaves plenty of headroom in 16 GB.
    "default": "mlx-community/gemma-4-E4B-it-qat-4bit",
    # ~12B, 4-bit: noticeably better prose, several GB of weights.
    "quality": "mlx-community/gemma-4-12B-it-qat-4bit",
    "fallback": "mlx-community/gemma-3-4b-it-qat-4bit",
    "fallback-quality": "mlx-community/gemma-3-12b-it-qat-4bit",
}

ENV_VAR = "QWERT_MODEL"


def resolve_model(name: str | None = None) -> str:
    """Turn a flag value, alias, or nothing into a repo id or local path."""
    chosen = name or os.environ.get(ENV_VAR) or "default"
    return MODEL_ALIASES.get(chosen, chosen)


def load_model(name: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load ``(model, tokenizer)`` with mlx-lm. Imported lazily: MLX is heavy."""
    from mlx_lm import load

    return load(name, adapter_path=adapter_path)
