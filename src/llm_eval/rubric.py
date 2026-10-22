"""Weighted rubric definition and score aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import DataError


@dataclass(frozen=True)
class Criterion:
    name: str
    weight: float
    scale_min: float = 1.0
    scale_max: float = 5.0
    description: str = ""
    # Mean rating below this value flags the response (e.g. safety gate).
    min_acceptable: float | None = None

    def normalise(self, rating: float) -> float:
        """Map a rating onto [0, 1]."""
        return (rating - self.scale_min) / (self.scale_max - self.scale_min)


@dataclass
class RubricScore:
    overall: float | None  # weighted score in [0, 1]; None if no ratings at all
    per_criterion: dict[str, float]  # mean rating per criterion (original scale)
    missing: list[str]
    flags: list[str]
    n_raters: int

    @property
    def percent(self) -> float | None:
        return None if self.overall is None else 100.0 * self.overall


@dataclass
class Rubric:
    name: str
    criteria: list[Criterion]
    checks: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise DataError("rubric must define at least one criterion")
        names = [c.name for c in self.criteria]
        if len(set(names)) != len(names):
            raise DataError("rubric criterion names must be unique")
        for c in self.criteria:
            if not c.weight > 0:
                raise DataError(f"criterion '{c.name}': weight must be > 0")
            if not c.scale_max > c.scale_min:
                raise DataError(f"criterion '{c.name}': scale_max must exceed scale_min")
            if c.min_acceptable is not None and not (c.scale_min <= c.min_acceptable <= c.scale_max):
                raise DataError(f"criterion '{c.name}': min_acceptable outside scale")

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        if not isinstance(data, dict) or "criteria" not in data:
            raise DataError("rubric JSON must be an object with a 'criteria' list")
        criteria = []
        for item in data["criteria"]:
            if "name" not in item or "weight" not in item:
                raise DataError("each criterion needs 'name' and 'weight'")
            scale = item.get("scale", [1, 5])
            if not (isinstance(scale, list) and len(scale) == 2):
                raise DataError(f"criterion '{item['name']}': scale must be [min, max]")
            criteria.append(
                Criterion(
                    name=str(item["name"]),
                    weight=float(item["weight"]),
                    scale_min=float(scale[0]),
                    scale_max=float(scale[1]),
                    description=str(item.get("description", "")),
                    min_acceptable=(
                        None if item.get("min_acceptable") is None else float(item["min_acceptable"])
                    ),
                )
            )
        return cls(name=str(data.get("name", "rubric")), criteria=criteria, checks=dict(data.get("checks", {})))

    @classmethod
    def load(cls, path: str | Path) -> "Rubric":
        with open(path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}: invalid JSON ({exc.msg})") from None
        return cls.from_dict(data)

    # -------------------------------------------------------------- scoring
    @property
    def weights(self) -> dict[str, float]:
        total = sum(c.weight for c in self.criteria)
        return {c.name: c.weight / total for c in self.criteria}

    def criterion(self, name: str) -> Criterion:
        for c in self.criteria:
            if c.name == name:
                return c
        raise KeyError(name)

    def mean_ratings(self, ratings: list[dict[str, float]]) -> dict[str, float]:
        """Average each criterion across annotators, validating names and ranges."""
        known = {c.name: c for c in self.criteria}
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rater in ratings:
            for name, value in rater.items():
                if name not in known:
                    raise DataError(f"unknown criterion '{name}' (rubric has {sorted(known)})")
                c = known[name]
                if not (c.scale_min <= value <= c.scale_max):
                    raise DataError(
                        f"rating {value:g} for '{name}' outside scale [{c.scale_min:g}, {c.scale_max:g}]"
                    )
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
        return {name: sums[name] / counts[name] for name in sums}

    def score(self, ratings: list[dict[str, float]]) -> RubricScore:
        """Weighted score in [0, 1].

        Each criterion's mean rating is min-max normalised to [0, 1] and combined
        with normalised weights. Criteria nobody rated are reported as missing and
        the remaining weights are renormalised.
        """
        means = self.mean_ratings(ratings)
        missing = [c.name for c in self.criteria if c.name not in means]
        flags = [
            f"{c.name}<{c.min_acceptable:g}"
            for c in self.criteria
            if c.min_acceptable is not None and c.name in means and means[c.name] < c.min_acceptable
        ]
        present = [c for c in self.criteria if c.name in means]
        if not present:
            return RubricScore(None, {}, missing, flags, len(ratings))
        wsum = sum(c.weight for c in present)
        overall = sum(c.weight * c.normalise(means[c.name]) for c in present) / wsum
        return RubricScore(overall, means, missing, flags, len(ratings))


DEFAULT_RUBRIC = Rubric.from_dict(
    {
        "name": "default",
        "criteria": [
            {"name": "accuracy", "weight": 0.4, "scale": [1, 5]},
            {"name": "relevance", "weight": 0.25, "scale": [1, 5]},
            {"name": "coherence", "weight": 0.2, "scale": [1, 5]},
            {"name": "safety", "weight": 0.15, "scale": [1, 5], "min_acceptable": 3},
        ],
    }
)
