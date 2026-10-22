"""Loading and validating JSONL response records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class DataError(ValueError):
    """Raised when input data is malformed."""


@dataclass
class ResponseRecord:
    id: str
    prompt: str
    response: str
    model: str
    # One dict per annotator: {criterion_name: rating}
    ratings: list[dict[str, float]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    language: str | None = None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file; blank lines are skipped, bad lines raise DataError with the line number."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise DataError(f"{path}:{lineno}: expected a JSON object")
            obj.setdefault("_line", lineno)
            rows.append(obj)
    return rows


def _normalise_ratings(raw: Any, where: str) -> list[dict[str, float]]:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise DataError(f"{where}: 'ratings' must be an object or a list of objects")
    out: list[dict[str, float]] = []
    for item in items:
        if not isinstance(item, dict):
            raise DataError(f"{where}: each ratings entry must be an object")
        clean: dict[str, float] = {}
        for key, value in item.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DataError(f"{where}: rating '{key}' must be a number, got {value!r}")
            clean[str(key)] = float(value)
        out.append(clean)
    return out


def parse_response_records(rows: Iterable[dict[str, Any]]) -> list[ResponseRecord]:
    records: list[ResponseRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        where = f"line {row.get('_line', '?')}"
        for key in ("id", "prompt", "response", "model"):
            if key not in row:
                raise DataError(f"{where}: missing required field '{key}'")
        if not isinstance(row["response"], str) or not isinstance(row["prompt"], str):
            raise DataError(f"{where}: 'prompt' and 'response' must be strings")
        rid, model = str(row["id"]), str(row["model"])
        if (rid, model) in seen:
            raise DataError(f"{where}: duplicate record for id={rid!r} model={model!r}")
        seen.add((rid, model))
        keywords = row.get("keywords") or []
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            raise DataError(f"{where}: 'keywords' must be a list of strings")
        language = row.get("language")
        if language is not None and not isinstance(language, str):
            raise DataError(f"{where}: 'language' must be a string")
        records.append(
            ResponseRecord(
                id=rid,
                prompt=row["prompt"],
                response=row["response"],
                model=model,
                ratings=_normalise_ratings(row.get("ratings"), where),
                keywords=list(keywords),
                language=language,
            )
        )
    return records
