"""Per-record evaluation, per-model leaderboard and plain-text tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .checks import CheckConfig, CheckResult, run_checks
from .pairwise import BTResult, ModelRecord, bt_to_elo
from .records import ResponseRecord
from .rubric import Rubric, RubricScore


@dataclass
class RecordResult:
    record: ResponseRecord
    score: RubricScore
    checks: list[CheckResult]

    @property
    def checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "model": self.record.model,
            "score": None if self.score.overall is None else round(self.score.overall, 6),
            "per_criterion": {k: round(v, 4) for k, v in self.score.per_criterion.items()},
            "missing_criteria": self.score.missing,
            "flags": self.score.flags,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


@dataclass
class ModelSummary:
    model: str
    n: int = 0
    n_scored: int = 0
    score_sum: float = 0.0
    criterion_sums: dict[str, float] = field(default_factory=dict)
    criterion_counts: dict[str, int] = field(default_factory=dict)
    checks_passed: int = 0
    refusals: int = 0
    empty: int = 0
    flagged: int = 0

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.n_scored if self.n_scored else None

    @property
    def check_pass_rate(self) -> float:
        return self.checks_passed / self.n if self.n else 0.0

    def criterion_means(self) -> dict[str, float]:
        return {k: self.criterion_sums[k] / self.criterion_counts[k] for k in self.criterion_sums}

    def to_dict(self) -> dict[str, Any]:
        ms = self.mean_score
        return {
            "model": self.model,
            "responses": self.n,
            "scored": self.n_scored,
            "mean_score": None if ms is None else round(ms, 6),
            "criterion_means": {k: round(v, 4) for k, v in self.criterion_means().items()},
            "check_pass_rate": round(self.check_pass_rate, 6),
            "refusals": self.refusals,
            "empty": self.empty,
            "flagged": self.flagged,
        }


def evaluate(records: list[ResponseRecord], rubric: Rubric) -> list[RecordResult]:
    cfg = CheckConfig.from_dict(rubric.checks)
    return [RecordResult(r, rubric.score(r.ratings), run_checks(r, cfg)) for r in records]


def leaderboard(results: list[RecordResult]) -> list[ModelSummary]:
    by_model: dict[str, ModelSummary] = {}
    for res in results:
        s = by_model.setdefault(res.record.model, ModelSummary(res.record.model))
        s.n += 1
        if res.score.overall is not None:
            s.n_scored += 1
            s.score_sum += res.score.overall
        for k, v in res.score.per_criterion.items():
            s.criterion_sums[k] = s.criterion_sums.get(k, 0.0) + v
            s.criterion_counts[k] = s.criterion_counts.get(k, 0) + 1
        s.checks_passed += res.checks_passed
        failed = set(res.failed_checks())
        s.refusals += "no_refusal" in failed
        s.empty += "nonempty" in failed
        s.flagged += bool(res.score.flags)
    return sorted(
        by_model.values(),
        key=lambda m: (-(m.mean_score if m.mean_score is not None else -1.0), -m.check_pass_rate, m.model),
    )


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def numeric(s: str) -> bool:
        try:
            float(s.rstrip("%"))
            return True
        except ValueError:
            return s == "-"

    # Text columns are left-aligned, numeric columns right-aligned.
    right = [all(numeric(r[i]) for r in cells[1:]) for i in range(len(headers))]

    def line(r: list[str]) -> str:
        out = [c.rjust(widths[i]) if right[i] else c.ljust(widths[i]) for i, c in enumerate(r)]
        return "  ".join(out).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), sep] + [line(r) for r in cells[1:]])


def render_leaderboard(summaries: list[ModelSummary], rubric: Rubric) -> str:
    crit = [c.name for c in rubric.criteria]
    headers = ["rank", "model", "n", "score"] + crit + ["checks", "refusal", "empty", "flagged"]
    rows = []
    for rank, s in enumerate(summaries, 1):
        means = s.criterion_means()
        ms = s.mean_score
        rows.append(
            [rank, s.model, s.n, "-" if ms is None else f"{100 * ms:.1f}"]
            + [f"{means[c]:.2f}" if c in means else "-" for c in crit]
            + [f"{100 * s.check_pass_rate:.0f}%", s.refusals, s.empty, s.flagged]
        )
    return format_table(headers, rows)


def render_failures(results: list[RecordResult]) -> str:
    lines = []
    for r in results:
        bad = [c for c in r.checks if not c.passed]
        if bad or r.score.flags:
            parts = [f"{c.name} ({c.detail})" if c.detail else c.name for c in bad]
            parts += [f"rubric flag {f}" for f in r.score.flags]
            lines.append(f"  {r.record.id} [{r.record.model}]: " + "; ".join(parts))
    return "\n".join(lines) if lines else "  none"


def render_pairwise(table: dict[str, ModelRecord], bt: BTResult) -> str:
    elo = bt_to_elo(bt.strengths)
    rows = []
    for rank, (model, strength) in enumerate(bt.ranking(), 1):
        rec = table[model]
        rows.append(
            [rank, model, rec.games, rec.wins, rec.losses, rec.ties,
             f"{100 * rec.win_rate:.1f}%", f"{strength:.4f}", f"{elo[model]:.0f}"]
        )
    return format_table(
        ["rank", "model", "games", "W", "L", "T", "win_rate", "bt_strength", "elo"], rows
    )
