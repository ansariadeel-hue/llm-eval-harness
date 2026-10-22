import json

import pytest

from llm_eval.records import DataError, load_jsonl, parse_response_records
from llm_eval.rubric import DEFAULT_RUBRIC, Rubric

RUBRIC = Rubric.from_dict(
    {
        "name": "t",
        "criteria": [
            {"name": "accuracy", "weight": 4},
            {"name": "relevance", "weight": 3},
            {"name": "coherence", "weight": 2},
            {"name": "safety", "weight": 1, "min_acceptable": 3},
        ],
    }
)


def test_weights_are_normalised():
    assert RUBRIC.weights == pytest.approx(
        {"accuracy": 0.4, "relevance": 0.3, "coherence": 0.2, "safety": 0.1}
    )


def test_weighted_score_reference_value():
    # weighted mean on the 1-5 scale = .4*5 + .3*4 + .2*3 + .1*5 = 4.3 -> (4.3-1)/4 = 0.825
    s = RUBRIC.score([{"accuracy": 5, "relevance": 4, "coherence": 3, "safety": 5}])
    assert s.overall == pytest.approx(0.825)
    assert s.percent == pytest.approx(82.5)
    assert s.missing == [] and s.flags == []


def test_extremes_map_to_zero_and_one():
    lo = {c.name: 1 for c in RUBRIC.criteria}
    hi = {c.name: 5 for c in RUBRIC.criteria}
    assert RUBRIC.score([lo]).overall == pytest.approx(0.0)
    assert RUBRIC.score([hi]).overall == pytest.approx(1.0)


def test_multiple_raters_are_averaged():
    s = RUBRIC.score(
        [
            {"accuracy": 5, "relevance": 5, "coherence": 5, "safety": 5},
            {"accuracy": 3, "relevance": 5, "coherence": 5, "safety": 5},
        ]
    )
    assert s.per_criterion["accuracy"] == pytest.approx(4.0)
    assert s.n_raters == 2
    assert s.overall == pytest.approx(1 - 0.4 * 0.25)


def test_missing_criteria_renormalise_weights():
    s = RUBRIC.score([{"accuracy": 5, "relevance": 1}])
    # (4*1 + 3*0) / 7
    assert s.overall == pytest.approx(4 / 7)
    assert s.missing == ["coherence", "safety"]


def test_no_ratings_gives_none():
    s = RUBRIC.score([])
    assert s.overall is None and s.percent is None


def test_safety_gate_flag():
    s = RUBRIC.score([{"accuracy": 5, "relevance": 5, "coherence": 5, "safety": 2}])
    assert s.flags == ["safety<3"]


def test_rating_out_of_range_and_unknown_criterion():
    with pytest.raises(DataError):
        RUBRIC.score([{"accuracy": 6}])
    with pytest.raises(DataError):
        RUBRIC.score([{"humour": 3}])


def test_custom_scale_normalisation():
    r = Rubric.from_dict({"criteria": [{"name": "x", "weight": 1, "scale": [0, 10]}]})
    assert r.score([{"x": 7}]).overall == pytest.approx(0.7)


@pytest.mark.parametrize(
    "bad",
    [
        {"criteria": []},
        {"criteria": [{"name": "a", "weight": 0}]},
        {"criteria": [{"name": "a", "weight": 1}, {"name": "a", "weight": 1}]},
        {"criteria": [{"name": "a", "weight": 1, "scale": [5, 1]}]},
        {"no": "criteria"},
    ],
)
def test_invalid_rubrics_rejected(bad):
    with pytest.raises(DataError):
        Rubric.from_dict(bad)


def test_default_rubric_has_four_criteria():
    assert [c.name for c in DEFAULT_RUBRIC.criteria] == ["accuracy", "relevance", "coherence", "safety"]
    assert sum(DEFAULT_RUBRIC.weights.values()) == pytest.approx(1.0)


def test_load_jsonl_and_parse(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"id": 1, "prompt": "p", "response": "r", "model": "m", "ratings": {"accuracy": 4}},
        {"id": 2, "prompt": "p", "response": "r", "model": "m", "ratings": [{"accuracy": 4}, {"accuracy": 2}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    recs = parse_response_records(load_jsonl(p))
    assert [r.id for r in recs] == ["1", "2"]
    assert recs[0].ratings == [{"accuracy": 4.0}]
    assert len(recs[1].ratings) == 2


def test_load_jsonl_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": 1}\n{not json}\n', encoding="utf-8")
    with pytest.raises(DataError, match=":2:"):
        load_jsonl(p)


def test_parse_rejects_missing_fields_and_duplicates():
    with pytest.raises(DataError, match="response"):
        parse_response_records([{"id": 1, "prompt": "p", "model": "m"}])
    row = {"id": 1, "prompt": "p", "response": "r", "model": "m"}
    with pytest.raises(DataError, match="duplicate"):
        parse_response_records([row, dict(row)])
    with pytest.raises(DataError, match="number"):
        parse_response_records([{**row, "ratings": {"accuracy": "high"}}])
