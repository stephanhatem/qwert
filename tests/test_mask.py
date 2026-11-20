from fake_tokenizer import FakeTokenizer

from qwert.hands import Hand, Punctuation
from qwert.mask import (
    allowed_lookup,
    build_mask,
    cache_path,
    chat_template_ids,
    load_or_build_mask,
    special_ids,
    token_text,
)
from qwert.stats import surviving_words, vocab_stats
from qwert.verify import is_valid

FREE, STRICT, NONE = Punctuation.FREE, Punctuation.STRICT, Punctuation.NONE
STOPS = (1, 2, 3, 4)  # BOS, EOS and the two turn markers


def test_token_text_restores_the_leading_space():
    tok = FakeTokenizer()
    assert token_text(tok, 6) == " tree"
    assert token_text(tok, 8) == "ed"
    assert token_text(tok, 14) == " "


def test_special_tokens_found_from_both_sources():
    tok = FakeTokenizer()
    assert special_ids(tok) == {0, 1, 2, 3, 4, 5}
    assert chat_template_ids(tok) == {1, 3, 4}


def test_left_mask_free_punctuation():
    mask = build_mask(FakeTokenizer(), Hand.LEFT, FREE)
    assert mask.stop_ids == STOPS
    assert mask.text_ids == (6, 7, 8, 9, 14, 15, 16, 17, 20, 25)
    assert mask.vocab_size == 26


def test_right_mask_free_punctuation():
    mask = build_mask(FakeTokenizer(), Hand.RIGHT, FREE)
    assert mask.text_ids == (10, 11, 12, 14, 15, 16, 17, 21)


def test_strict_and_none_punctuation():
    tok = FakeTokenizer()
    assert 17 in build_mask(tok, Hand.LEFT, STRICT).allowed      # "!" is left
    assert 16 not in build_mask(tok, Hand.LEFT, STRICT).allowed  # "." is right
    assert 16 in build_mask(tok, Hand.RIGHT, STRICT).allowed
    none = build_mask(tok, Hand.LEFT, NONE)
    assert 16 not in none.allowed and 17 not in none.allowed
    assert 14 in none.allowed and 15 in none.allowed  # space, newline


def test_special_tokens_are_not_judged_by_spelling():
    mask = build_mask(FakeTokenizer(), Hand.LEFT, FREE)
    assert 0 not in mask.allowed  # <pad>
    assert 5 not in mask.allowed  # <unused0>
    # <s> spells only a left-hand letter, yet is allowed solely as a stop token.
    assert 1 in mask.stop_ids and 1 not in mask.text_ids


def test_bans_bytes_non_ascii_tabs_emoji_and_mixed_words():
    for hand in Hand:
        allowed = set(build_mask(FakeTokenizer(), hand, FREE).allowed)
        assert not allowed & {13, 18, 19, 22, 23, 24}


def test_any_sequence_of_allowed_tokens_passes_the_verifier():
    tok = FakeTokenizer()
    for hand in Hand:
        for mode in Punctuation:
            mask = build_mask(tok, hand, mode)
            assert is_valid(tok.decode(list(mask.text_ids)), hand, mode)


def test_allowed_lookup_pads_and_truncates():
    mask = build_mask(FakeTokenizer(), Hand.LEFT, FREE)
    table = allowed_lookup(mask, 30)
    assert len(table) == 30 and table[6] and not table[13] and not table[29]
    assert len(allowed_lookup(mask, 7)) == 7


def test_cache_is_per_model_hand_and_mode(tmp_path):
    paths = {
        cache_path(model, hand, mode, tmp_path)
        for model in ("org/model-a", "org/model-b")
        for hand in Hand
        for mode in Punctuation
    }
    assert len(paths) == 12


def test_cache_round_trip_skips_the_vocabulary_walk(tmp_path):
    tok = FakeTokenizer()
    built = load_or_build_mask(tok, "org/fake", Hand.LEFT, FREE, tmp_path)
    assert cache_path("org/fake", Hand.LEFT, FREE, tmp_path).exists()
    fresh = FakeTokenizer()
    cached = load_or_build_mask(fresh, "org/fake", Hand.LEFT, FREE, tmp_path)
    assert cached == built
    assert fresh.decode_calls == 0


def test_cache_ignored_when_vocabulary_size_changes(tmp_path):
    load_or_build_mask(FakeTokenizer(), "org/fake", Hand.LEFT, FREE, tmp_path)
    bigger = FakeTokenizer()
    bigger.pieces.append("▁we")
    mask = load_or_build_mask(bigger, "org/fake", Hand.LEFT, FREE, tmp_path)
    assert mask.vocab_size == 27 and 26 in mask.allowed


def test_stats_count_tokens_and_whole_words():
    tok = FakeTokenizer()
    mask = build_mask(tok, Hand.LEFT, FREE)
    assert surviving_words(tok, mask) == ["tree", "sweater", "great", "a"]
    stats = vocab_stats(tok, mask)
    assert stats.allowed_tokens == 10
    assert stats.stop_tokens == 4
    assert set(stats.top_words) <= {"tree", "sweater", "great", "a"}
