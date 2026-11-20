"""Hand-written example sentences shown to the model in the prompt.

The point is to demonstrate how to keep sentences flowing without the
usual joining words ("the", "and", "it" all mix hands): the left hand
leans on "a", "as", "at", "after", "we were", "was"; the right hand on
"I", "you", "in", "on", "my", "no", "only".

``tests/test_examples.py`` runs every line through the verifier.
"""

from __future__ import annotations

from .hands import Hand

EXAMPLES: dict[Hand, tuple[str, ...]] = {
    Hand.LEFT: (
        "Fred sweats as fast cars race west at a vast desert.",
        "We were sad after Dad's sweater was wasted.",
        "Greta tasted a sweet red beet, saw a few crabs, was scared.",
        "Stars are far. Trees are bare. We rest.",
        "A brave bee, Bert, was sad: a bear ate a tree's best sweets. "
        "Bert sewed a red vest, faced a bear, saved a tree. Bees cared after.",
    ),
    Hand.RIGHT: (
        "Look up, you. No moon in July, only pink.",
        "Polly, join in! Hop on my lumpy pony.",
        "Hi, puppy! Jump up, hop in, nip my pinky.",
        "Milky moon, you loom on my hill.",
        "In my opinion, Molly, you look jolly in pink.",
    ),
}


def examples_block(hand: Hand) -> str:
    lines = "\n".join(f"- {line}" for line in EXAMPLES[hand])
    return (
        "Examples of sentences in this style (do not copy them; "
        f"write something new):\n{lines}"
    )
