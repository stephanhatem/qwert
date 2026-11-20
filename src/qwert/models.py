"""Model selection. The model is always a CLI/config choice, never fixed.

Resolution order: ``--model`` flag, then the ``QWERT_MODEL`` environment
variable, then the ``default`` alias below.

NOTE ON VERIFICATION. ``default`` has been run on a 16 GB M4. The
other repo names come from a web search (the mlx-community "Gemma 4
QAT" collection) or from memory, and have not been loaded. If one
fails, pass any other MLX repo or local path with ``--model``.
"""

from __future__ import annotations

import os
from typing import Any

MODEL_ALIASES: dict[str, str] = {
    # ~4B, 4-bit: fast, leaves plenty of headroom in 16 GB.
    "default": "mlx-community/gemma-4-E4B-it-qat-4bit",
    # ~12B, 4-bit: an 11 GB download. Did not run usably on a 16 GB M4 Air.
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
