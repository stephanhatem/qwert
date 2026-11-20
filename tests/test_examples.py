import pytest

from qwert.examples import EXAMPLES, examples_block
from qwert.hands import Hand
from qwert.verify import verify


@pytest.mark.parametrize("hand", list(Hand))
def test_every_example_satisfies_its_own_constraint(hand):
    assert len(EXAMPLES[hand]) >= 3
    for line in EXAMPLES[hand]:
        assert verify(line, hand) == [], line


def test_examples_block_lists_each_example():
    block = examples_block(Hand.RIGHT)
    assert all(f"- {line}" in block for line in EXAMPLES[Hand.RIGHT])
