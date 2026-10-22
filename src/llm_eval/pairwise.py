"""Pairwise preference aggregation: win rates and a Bradley-Terry model.

The Bradley-Terry model says P(i beats j) = p_i / (p_i + p_j). Strengths are
fitted by maximum likelihood with Hunter's (2004) MM algorithm:

    p_i <- W_i / sum_{j != i} n_ij / (p_i + p_j)

where W_i is i's (possibly fractional) win total and n_ij the number of games
between i and j. A tie counts as half a win for each side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .records import DataError


@dataclass(frozen=True)
class Comparison:
    model_a: str
    model_b: str
    outcome: float  # 1.0 = a wins, 0.0 = b wins, 0.5 = tie
    prompt_id: str | None = None


_A = {"a", "model_a", "left", "1"}
_B = {"b", "model_b", "right", "2"}
_TIE = {"tie", "draw", "both", "equal", "none"}


def parse_comparisons(rows: Iterable[dict[str, Any]]) -> list[Comparison]:
    comps: list[Comparison] = []
    for row in rows:
        where = f"line {row.get('_line', '?')}"
        for key in ("model_a", "model_b", "winner"):
            if key not in row:
                raise DataError(f"{where}: missing required field '{key}'")
        a, b = str(row["model_a"]), str(row["model_b"])
        if a == b:
            raise DataError(f"{where}: a model cannot be compared with itself ({a})")
        w = str(row["winner"]).strip()
        wl = w.lower()
        if w == a or wl in _A:
            outcome = 1.0
        elif w == b or wl in _B:
            outcome = 0.0
        elif wl in _TIE:
            outcome = 0.5
        else:
            raise DataError(f"{where}: winner {w!r} is not model_a, model_b or tie")
        pid = row.get("prompt_id", row.get("id"))
        comps.append(Comparison(a, b, outcome, None if pid is None else str(pid)))
    return comps


@dataclass
class ModelRecord:
    model: str
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        """Ties count as half a win."""
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


def win_table(comps: Iterable[Comparison]) -> dict[str, ModelRecord]:
    table: dict[str, ModelRecord] = {}
    for c in comps:
        ra = table.setdefault(c.model_a, ModelRecord(c.model_a))
        rb = table.setdefault(c.model_b, ModelRecord(c.model_b))
        if c.outcome == 1.0:
            ra.wins += 1
            rb.losses += 1
        elif c.outcome == 0.0:
            rb.wins += 1
            ra.losses += 1
        else:
            ra.ties += 1
            rb.ties += 1
    return table


def head_to_head(comps: Iterable[Comparison]) -> dict[tuple[str, str], float]:
    """w[(i, j)] = number of (fractional) wins of i over j."""
    w: dict[tuple[str, str], float] = {}
    for c in comps:
        w[(c.model_a, c.model_b)] = w.get((c.model_a, c.model_b), 0.0) + c.outcome
        w[(c.model_b, c.model_a)] = w.get((c.model_b, c.model_a), 0.0) + (1.0 - c.outcome)
    return w


@dataclass
class BTResult:
    strengths: dict[str, float]  # normalised to sum to 1
    iterations: int
    converged: bool
    log_likelihood: float

    def prob(self, i: str, j: str) -> float:
        """Model probability that i beats j."""
        pi, pj = self.strengths[i], self.strengths[j]
        return pi / (pi + pj)

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.strengths.items(), key=lambda kv: (-kv[1], kv[0]))


def _reachable(start: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for nxt in edges.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def bradley_terry(
    comps: list[Comparison],
    prior: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> BTResult:
    """Fit Bradley-Terry strengths with the MM algorithm.

    ``prior`` adds that many virtual wins to *each* side of every pair that
    actually met (equivalent to 2*prior virtual ties), which regularises the
    fit and guarantees a finite MLE when the comparison graph is connected.
    With ``prior=0`` the MLE exists only if the "beats" graph is strongly
    connected; otherwise a DataError is raised.
    """
    if prior < 0:
        raise ValueError("prior must be >= 0")
    if not comps:
        raise DataError("no comparisons to fit")
    w = head_to_head(comps)
    for (i, j) in list(w):
        w[(i, j)] += prior
    models = sorted({m for pair in w for m in pair})
    if len(models) < 2:
        raise DataError("need at least two models")

    # Connectivity checks.
    undirected: dict[str, set[str]] = {}
    beats: dict[str, set[str]] = {}
    for (i, j), wins in w.items():
        undirected.setdefault(i, set()).add(j)
        if wins > 0:
            beats.setdefault(i, set()).add(j)
    if _reachable(models[0], undirected) != set(models):
        raise DataError("comparison graph is disconnected; strengths are not comparable")
    if prior == 0:
        reverse: dict[str, set[str]] = {}
        for i, outs in beats.items():
            for j in outs:
                reverse.setdefault(j, set()).add(i)
        if _reachable(models[0], beats) != set(models) or _reachable(models[0], reverse) != set(models):
            raise DataError(
                "Bradley-Terry MLE does not exist (some model never wins or never loses "
                "within a group); use prior > 0"
            )

    W = {m: 0.0 for m in models}
    n: dict[tuple[str, str], float] = {}
    for (i, j), wins in w.items():
        W[i] += wins
        n[(i, j)] = wins + w[(j, i)]
    opponents = {m: [j for (i, j) in n if i == m] for m in models}

    p = {m: 1.0 / len(models) for m in models}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        new = {}
        for i in models:
            denom = sum(n[(i, j)] / (p[i] + p[j]) for j in opponents[i])
            new[i] = W[i] / denom
        total = sum(new.values())
        new = {m: v / total for m, v in new.items()}
        delta = max(abs(new[m] - p[m]) for m in models)
        p = new
        if delta < tol:
            converged = True
            break

    ll = 0.0
    for (i, j), wins in w.items():
        if wins > 0:
            ll += wins * math.log(p[i] / (p[i] + p[j]))
    return BTResult(p, it, converged, ll)


def bt_to_elo(strengths: dict[str, float], base: float = 1000.0) -> dict[str, float]:
    """Express strengths on an Elo-like scale (400 * log10), centred so the geometric mean is ``base``."""
    logs = {m: math.log10(v) for m, v in strengths.items()}
    centre = sum(logs.values()) / len(logs)
    return {m: base + 400.0 * (v - centre) for m, v in logs.items()}
