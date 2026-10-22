import json
from pathlib import Path

import pytest

from llm_eval.cli import main
from llm_eval.report import format_table

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_score_cli_table(capsys):
    rc = main(["score", str(EXAMPLES / "responses.jsonl"), "--rubric", str(EXAMPLES / "rubric.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "general-qa-v1" in out
    rows = [l.split() for l in out.splitlines() if l.split()[:1] in (["1"], ["2"], ["3"])]
    assert [r[1] for r in rows] == ["model-a", "model-b", "model-c"]


def test_score_cli_json_leaderboard_values(capsys):
    rc = main(
        ["score", str(EXAMPLES / "responses.jsonl"), "--rubric", str(EXAMPLES / "rubric.json"), "--json", "--details"]
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    board = {m["model"]: m for m in data["leaderboard"]}
    assert [m["model"] for m in data["leaderboard"]] == ["model-a", "model-b", "model-c"]
    assert board["model-a"]["check_pass_rate"] == 1.0
    assert board["model-c"]["refusals"] == 1 and board["model-c"]["empty"] == 1
    assert board["model-b"]["flagged"] == 1
    assert board["model-b"]["scored"] == 4  # one record has no ratings
    assert len(data["records"]) == 15


def test_compare_cli_json(capsys):
    rc = main(["compare", str(EXAMPLES / "pairwise.jsonl"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["converged"]
    assert data["comparisons"] == 18
    assert [m["model"] for m in data["models"]] == ["model-a", "model-b", "model-c"]
    assert sum(m["bt_strength"] for m in data["models"]) == pytest.approx(1.0, abs=1e-5)
    a = data["models"][0]
    assert (a["wins"], a["losses"], a["ties"]) == (8, 2, 2)


def test_compare_cli_table(capsys):
    assert main(["compare", str(EXAMPLES / "pairwise.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "Bradley-Terry" in out and "P(model-a beats model-b)" in out


def test_cli_error_exit_code(tmp_path, capsys):
    bad = tmp_path / "x.jsonl"
    bad.write_text('{"id": 1, "prompt": "p", "model": "m"}\n', encoding="utf-8")
    assert main(["score", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_format_table_alignment():
    out = format_table(["name", "n"], [["alpha", 1], ["b", 10]]).splitlines()
    assert out[2] == "alpha   1"
    assert out[3] == "b      10"
