import string

import pytest

from qwert.hands import DIGITS, LETTERS, PUNCTUATION, Hand, Punctuation, char_allowed

FREE, STRICT, NONE = Punctuation.FREE, Punctuation.STRICT, Punctuation.NONE


def test_letters_match_the_brief():
    assert "".join(sorted(LETTERS[Hand.LEFT])) == "abcdefgqrstvwxz"
    assert "".join(sorted(LETTERS[Hand.RIGHT])) == "hijklmnopuy"


def test_hands_partition_the_keyboard():
    for table, everything in (
        (LETTERS, string.ascii_lowercase),
        (DIGITS, string.digits),
        (PUNCTUATION, string.punctuation),
    ):
        assert not table[Hand.LEFT] & table[Hand.RIGHT]
        assert table[Hand.LEFT] | table[Hand.RIGHT] == set(everything)


@pytest.mark.parametrize("ch", "qazQAZ")
def test_left_letters_any_case(ch):
    assert char_allowed(ch, Hand.LEFT, FREE)
    assert not char_allowed(ch, Hand.RIGHT, FREE)


@pytest.mark.parametrize("ch", "pmyPMY")
def test_right_letters_any_case(ch):
    assert char_allowed(ch, Hand.RIGHT, FREE)
    assert not char_allowed(ch, Hand.LEFT, FREE)


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("mode", list(Punctuation))
def test_space_and_newline_always_allowed(hand, mode):
    assert char_allowed(" ", hand, mode)
    assert char_allowed("\n", hand, mode)
    assert not char_allowed("\t", hand, mode)


def test_digits_follow_the_hand_in_every_mode():
    for mode in Punctuation:
        assert char_allowed("3", Hand.LEFT, mode)
        assert not char_allowed("3", Hand.RIGHT, mode)
        assert char_allowed("8", Hand.RIGHT, mode)
        assert not char_allowed("8", Hand.LEFT, mode)


def test_punctuation_modes():
    # "." is a right-hand key, "!" (Shift+1) a left-hand key.
    assert char_allowed(".", Hand.LEFT, FREE)
    assert not char_allowed(".", Hand.LEFT, STRICT)
    assert char_allowed(".", Hand.RIGHT, STRICT)
    assert char_allowed("!", Hand.LEFT, STRICT)
    assert not char_allowed("!", Hand.RIGHT, STRICT)
    for hand in Hand:
        assert not char_allowed(".", hand, NONE)
        assert not char_allowed("!", hand, NONE)


def test_typographic_punctuation_maps_to_its_key():
    assert char_allowed("’", Hand.LEFT, FREE)       # curly apostrophe
    assert char_allowed("’", Hand.RIGHT, STRICT)
    assert not char_allowed("’", Hand.LEFT, STRICT)
    assert not char_allowed("—", Hand.RIGHT, NONE)  # em dash


@pytest.mark.parametrize("ch", ["é", "K", "١", "\U0001f600", "中", "\x00"])
def test_everything_off_the_keyboard_is_banned(ch):
    # e-acute, Kelvin sign (lower-cases to "k"), Arabic digit, emoji, CJK, NUL
    for hand in Hand:
        assert not char_allowed(ch, hand, FREE)
