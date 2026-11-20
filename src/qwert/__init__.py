"""qwert: text generation where every letter is typed by one hand on QWERTY."""

from .hands import Hand, Punctuation
from .verify import Violation, is_valid, verify

__all__ = ["Hand", "Punctuation", "Violation", "is_valid", "verify"]
__version__ = "0.1.0"
