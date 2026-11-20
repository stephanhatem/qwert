import pytest

from qwert.hands import Hand, Punctuation, char_allowed
from qwert.verify import Violation, is_valid, verify

FREE, STRICT, NONE = Punctuation.FREE, Punctuation.STRICT, Punctuation.NONE


def test_valid_left_hand_text():
    assert is_valid("Sweater dressed, a brave cat ate a few great red beets!", Hand.LEFT)


def test_valid_right_hand_text():
    assert is_valid("Oh you jolly pink onion, hum in my pool.", Hand.RIGHT)


def test_reports_each_violation_with_its_position():
    assert verify("tree you", Hand.LEFT) == [
        Violation(5, "y"), Violation(6, "o"), Violation(7, "u"),
    ]


def test_case_is_ignored():
    assert is_valid("SWEATER Vest", Hand.LEFT)
    assert not is_valid("SWEATER Vest", Hand.RIGHT)


def test_punctuation_modes():
    text = "fast cars."
    assert is_valid(text, Hand.LEFT, FREE)
    assert verify(text, Hand.LEFT, STRICT) == [Violation(9, ".")]
    assert verify(text, Hand.LEFT, NONE) == [Violation(9, ".")]
    assert is_valid("fast cars!", Hand.LEFT, STRICT)
    assert is_valid("fast cars", Hand.LEFT, NONE)


def test_digits_follow_the_hand():
    assert is_valid("we set 123", Hand.LEFT)
    assert not is_valid("we set 678", Hand.LEFT)
    assert is_valid("only 90", Hand.RIGHT)


def test_accepts_plain_strings_for_hand_and_mode():
    assert is_valid("tree", "left", "none")


def test_off_keyboard_characters_fail():
    assert not is_valid("café", Hand.LEFT)
    assert not is_valid("K", Hand.RIGHT)  # Kelvin sign
    assert not is_valid("a\tb", Hand.LEFT)
    assert is_valid("", Hand.LEFT)


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("mode", list(Punctuation))
def test_verifier_agrees_with_the_mask_rule(hand, mode):
    """Two separate implementations of the rule must give the same answer."""
    samples = [chr(i) for i in range(0x250)] + list("‘’“”–—…K١\U0001f600")
    for ch in samples:
        assert is_valid(ch, hand, mode) == char_allowed(ch, hand, mode), repr(ch)
