from dataclasses import dataclass

import pytest

from qwert.hands import Hand
from qwert.models import MODEL_ALIASES, resolve_model
from qwert.rerank import nonword_fraction, repeat_fraction, rerank


@dataclass
class Cand:
    text: str
    tokens: tuple = ()


def test_repeat_fraction():
    assert repeat_fraction("a great sweater") == 0.0
    assert repeat_fraction("were were were were") == pytest.approx(0.75)
    assert repeat_fraction("Tree, tree.") == pytest.approx(0.5)
    assert repeat_fraction("...") == 0.0


def test_rerank_orders_by_score_and_penalises_repeats():
    scores = {"a great sweater": -3.0, "were were were were": -1.0, "a few cats": -2.0}
    cands = [Cand(t) for t in scores]
    best, rejected = rerank(cands, lambda _: 0.0, Hand.LEFT, top=3)
    assert rejected == 0 and best[-1].text == "were were were were"

    by_text = iter(scores.values())
    best, _ = rerank(cands, lambda _: next(by_text), Hand.LEFT, top=2)
    assert [b.text for b in best] == ["a few cats", "were were were were"]
    assert best[1].score == pytest.approx(-1.0 - 2.0 * 0.75)
    assert best[0].chars == len("a few cats")


def test_rerank_drops_invalid_duplicate_and_short_candidates():
    cands = [Cand("a great sweater"), Cand("a great sweater"), Cand("hello"), Cand("")]
    best, rejected = rerank(cands, lambda _: -1.0, Hand.LEFT)
    assert [b.text for b in best] == ["a great sweater"]
    assert rejected == 1  # "hello" failed the verifier


def test_model_resolution(monkeypatch):
    monkeypatch.delenv("QWERT_MODEL", raising=False)
    assert resolve_model(None) == MODEL_ALIASES["default"]
    assert resolve_model("quality") == MODEL_ALIASES["quality"]
    assert resolve_model("some/other-model") == "some/other-model"
    monkeypatch.setenv("QWERT_MODEL", "env/model")
    assert resolve_model(None) == "env/model"
    assert resolve_model("quality") == MODEL_ALIASES["quality"]


REAL = {"a", "fat", "cat", "sat", "rex", "ate"}


def test_nonword_fraction():
    assert nonword_fraction("A fat cat sat.", REAL.__contains__) == 0.0
    assert nonword_fraction("Frrff. Zzzt! Cat sat.", REAL.__contains__) == pytest.approx(0.5)
    assert nonword_fraction("...", REAL.__contains__) == 0.0


def test_rerank_penalises_non_words():
    cands = [Cand("Frrff zzzt cat sat"), Cand("Rex ate")]
    best, _ = rerank(cands, lambda _: -2.0, Hand.LEFT, is_word=REAL.__contains__, nonword_weight=3.0)
    assert [b.text for b in best] == ["Rex ate", "Frrff zzzt cat sat"]
    assert best[1].nonword_fraction == pytest.approx(0.5)
    assert best[1].score == pytest.approx(-2.0 - 3.0 * 0.5)
    # Without a word list the penalty is off and the order is the input order.
    best, _ = rerank(cands, lambda _: -2.0, Hand.LEFT)
    assert best[0].text == "Frrff zzzt cat sat" and best[0].nonword_fraction is None


def test_wordfreq_word_check_if_installed():
    pytest.importorskip("wordfreq")
    from qwert.lexicon import load_word_check

    is_word = load_word_check()
    assert all(is_word(w) for w in ("rex", "starved", "zax", "a"))
    assert not any(is_word(w) for w in ("frrff", "frrr", "zzzt"))
