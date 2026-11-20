"""Phase 2 stub: LoRA fine-tuning for one-handed vocabulary.

Not built yet on purpose. The plan is in ``docs/phase2-lora.md``.
The functions below mark where the pieces will go.
"""

from __future__ import annotations

from pathlib import Path

from .hands import Hand, Punctuation

DESIGN_NOTE = "docs/phase2-lora.md"


def build_dataset(
    sources: list[Path],
    out_dir: Path,
    hand: Hand,
    punctuation: Punctuation = Punctuation.FREE,
) -> None:
    """Filter text into train/valid/test JSONL for ``mlx_lm.lora``."""
    raise NotImplementedError(f"Phase 2 is not implemented yet; see {DESIGN_NOTE}")


def train(model: str, data_dir: Path, adapter_dir: Path) -> None:
    """Run LoRA training with mlx-lm's tooling."""
    raise NotImplementedError(f"Phase 2 is not implemented yet; see {DESIGN_NOTE}")
