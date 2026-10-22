import math

import pytest

from llm_eval.pairwise import (
    Comparison,
    bradley_terry,
    bt_to_elo,
    head_to_head,
    parse_comparisons,
    win_table,
)
from llm_eval.records import DataError


def games(a, b, a_wins, b_wins, ties=0):
    return (
        [Comparison(a, b, 1.0)] * a_wins
        + [Comparison(a, b, 0.0)] * b_wins
        + [Comparison(a, b, 0.5)] * ties
    )


def test_parse_winner_forms():
    rows = [
        {"model_a": "x", "model_b": "y", "winner": "x"},
        {"model_a": "x", "model_b": "y", "winner": "b"},
        {"model_a": "x", "model_b": "y", "winner": "TIE"},
        {"model_a": "x", "model_b": "y", "winner": "model_a"},
    ]
    assert [c.outcome for c in parse_comparisons(rows)] == [1.0, 0.0, 0.5, 1.0]


def test_parse_errors():
    with pytest.raises(DataError):
        parse_comparisons([{"model_a": "x", "model_b": "y", "winner": "z"}])
    with pytest.raises(DataError):
        parse_comparisons([{"model_a": "x", "model_b": "x", "winner": "x"}])
    with pytest.raises(DataError):
        parse_comparisons([{"model_a": "x", "winner": "x"}])


def test_win_table_counts_and_rates():
    t = win_table(games("a", "b", 3, 1, 2))
    assert (t["a"].wins, t["a"].losses, t["a"].ties) == (3, 1, 2)
    assert (t["b"].wins, t["b"].losses, t["b"].ties) == (1, 3, 2)
    assert t["a"].win_rate == pytest.approx(4 / 6)
    assert t["a"].win_rate + t["b"].win_rate == pytest.approx(1.0)


def test_bt_two_players_closed_form():
    # MLE for two players: p_a / p_b = wins_a / wins_b = 3.
    bt = bradley_terry(games("a", "b", 3, 1))
    assert bt.converged
    assert bt.strengths["a"] == pytest.approx(0.75, abs=1e-8)
    assert bt.prob("a", "b") == pytest.approx(0.75, abs=1e-8)


def test_bt_ties_count_half():
    # 2 wins + 2 ties vs 0 wins + 2 ties -> W_a = 3, W_b = 1
    bt = bradley_terry(games("a", "b", 2, 0, 2))
    assert bt.strengths["a"] == pytest.approx(0.75, abs=1e-8)


def test_bt_symmetric_cycle_gives_equal_strengths():
    comps = games("a", "b", 2, 1) + games("b", "c", 2, 1) + games("c", "a", 2, 1)
    bt = bradley_terry(comps)
    for v in bt.strengths.values():
        assert v == pytest.approx(1 / 3, abs=1e-8)


def _three_player():
    return games("a", "b", 7, 3) + games("b", "c", 6, 4) + games("a", "c", 8, 2, 1)


def test_bt_satisfies_mle_stationarity():
    # At the MLE, each model's observed wins equal its expected wins:
    #   W_i = sum_j n_ij * p_i / (p_i + p_j)
    comps = _three_player()
    bt = bradley_terry(comps, tol=1e-13)
    w = head_to_head(comps)
    p = bt.strengths
    for i in p:
        observed = sum(v for (x, _), v in w.items() if x == i)
        expected = sum((w[(i, j)] + w[(j, i)]) * p[i] / (p[i] + p[j]) for (x, j) in w if x == i)
        assert observed == pytest.approx(expected, abs=1e-8)
    assert sum(p.values()) == pytest.approx(1.0)


def test_bt_is_a_likelihood_maximum():
    comps = _three_player()
    bt = bradley_terry(comps)
    w = head_to_head(comps)

    def ll(p):
        return sum(v * math.log(p[i] / (p[i] + p[j])) for (i, j), v in w.items() if v > 0)

    assert ll(bt.strengths) == pytest.approx(bt.log_likelihood)
    for m in bt.strengths:
        for eps in (0.01, -0.01):
            q = dict(bt.strengths)
            q[m] *= math.exp(eps)
            assert ll(q) < bt.log_likelihood


def test_bt_ranking_order():
    bt = bradley_terry(_three_player())
    assert [m for m, _ in bt.ranking()] == ["a", "b", "c"]


def test_bt_undefeated_requires_prior():
    comps = games("a", "b", 3, 0) + games("b", "c", 2, 1)
    with pytest.raises(DataError, match="prior"):
        bradley_terry(comps)
    bt = bradley_terry(comps, prior=0.5)
    assert bt.converged
    assert bt.ranking()[0][0] == "a"
    assert all(v > 0 for v in bt.strengths.values())


def test_bt_disconnected_graph_rejected():
    comps = games("a", "b", 1, 1) + games("c", "d", 1, 1)
    with pytest.raises(DataError, match="disconnected"):
        bradley_terry(comps, prior=1.0)


def test_bt_to_elo_scale():
    elo = bt_to_elo({"a": 0.75, "b": 0.25})
    # 400 * log10(3) ~= 190.85 rating points apart, centred on 1000
    assert elo["a"] - elo["b"] == pytest.approx(400 * math.log10(3))
    assert (elo["a"] + elo["b"]) / 2 == pytest.approx(1000)
