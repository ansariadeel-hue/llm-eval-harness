"""Automatic heuristic checks on response text.

Checks: non-empty, refusal detection, word-length bounds, required-keyword
coverage and an expected-script check based on Unicode block ranges.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .records import ResponseRecord

# Unicode block ranges (inclusive) per script.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)],
}

# ISO 639-1 language code -> expected script.
LANGUAGE_SCRIPTS: dict[str, str] = {
    "en": "latin", "fr": "latin", "de": "latin", "es": "latin",
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "sa": "devanagari",
    "bn": "bengali", "pa": "gurmukhi", "gu": "gujarati", "ta": "tamil",
    "te": "telugu", "kn": "kannada", "ml": "malayalam",
    "ur": "arabic", "ar": "arabic", "fa": "arabic",
    "ru": "cyrillic", "zh": "han",
}

REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I (?:can(?:not|'t)|won't|am unable|'m unable|am not able)",
    r"\bI (?:can(?:not|'t)|won't|am unable to|'m unable to|am not able to) (?:help|assist|provide|comply|do that|fulfil)",
    r"\bas an AI(?: language)? model\b",
    r"\bI must (?:decline|refuse)\b",
    r"मदद नहीं कर सकत",  # Hindi: "cannot help"
    r"मदत करू शकत नाही",  # Marathi: "cannot help"
    r"مدد نہیں کر سکت",  # Urdu: "cannot help"
]
_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    value: float | None = None


@dataclass
class CheckConfig:
    min_words: int = 1
    max_words: int | None = None
    keyword_threshold: float = 1.0
    script_threshold: float = 0.6
    detect_refusal: bool = True
    extra_refusal_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CheckConfig":
        data = dict(data or {})
        cfg = cls(
            min_words=int(data.get("min_words", 1)),
            max_words=None if data.get("max_words") is None else int(data["max_words"]),
            keyword_threshold=float(data.get("keyword_threshold", 1.0)),
            script_threshold=float(data.get("script_threshold", 0.6)),
            detect_refusal=bool(data.get("detect_refusal", True)),
            extra_refusal_patterns=list(data.get("extra_refusal_patterns", [])),
        )
        if cfg.min_words < 0 or (cfg.max_words is not None and cfg.max_words < cfg.min_words):
            raise ValueError("invalid word bounds: need 0 <= min_words <= max_words")
        for name in ("keyword_threshold", "script_threshold"):
            if not 0.0 <= getattr(cfg, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return cfg


# --------------------------------------------------------------------- script
def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return None


def script_profile(text: str) -> dict[str, float]:
    """Fraction of letters (and combining marks) belonging to each script.

    Digits, punctuation and whitespace are ignored. A combining mark that is
    outside every known block inherits the script of the preceding base letter.
    Letters in unknown blocks are counted as "other".
    """
    counts: dict[str, int] = {}
    prev: str | None = None
    for ch in unicodedata.normalize("NFC", text):
        cat = unicodedata.category(ch)
        if not (cat.startswith("L") or cat.startswith("M")):
            prev = None
            continue
        script = script_of(ch)
        if script is None and cat.startswith("M") and prev is not None:
            script = prev
        script = script or "other"
        counts[script] = counts.get(script, 0) + 1
        prev = script
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def expected_script(language_or_script: str) -> str:
    key = language_or_script.strip().lower()
    if key in SCRIPT_RANGES:
        return key
    base = key.split("-")[0].split("_")[0]
    if base in LANGUAGE_SCRIPTS:
        return LANGUAGE_SCRIPTS[base]
    raise ValueError(f"unknown language/script '{language_or_script}'")


# --------------------------------------------------------------------- checks
def check_nonempty(text: str) -> CheckResult:
    ok = any(ch.isalnum() for ch in text)
    return CheckResult("nonempty", ok, "" if ok else "response is empty or has no letters/digits")


def is_refusal(text: str, extra_patterns: list[str] | None = None) -> bool:
    norm = text.replace("’", "'")
    patterns = _REFUSAL_RE + [re.compile(p, re.IGNORECASE) for p in (extra_patterns or [])]
    return any(p.search(norm) for p in patterns)


def check_refusal(text: str, extra_patterns: list[str] | None = None) -> CheckResult:
    refused = is_refusal(text, extra_patterns)
    return CheckResult("no_refusal", not refused, "refusal language detected" if refused else "")


def word_count(text: str) -> int:
    return len(text.split())


def check_length(text: str, min_words: int = 1, max_words: int | None = None) -> CheckResult:
    n = word_count(text)
    ok = n >= min_words and (max_words is None or n <= max_words)
    bound = f"[{min_words}, {'inf' if max_words is None else max_words}]"
    return CheckResult("length", ok, f"{n} words, bounds {bound}", float(n))


def _fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def keyword_coverage(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Return (coverage fraction, missing keywords). Case-insensitive, NFC-normalised."""
    if not keywords:
        return 1.0, []
    folded = _fold(text)
    missing = [k for k in keywords if _fold(k) not in folded]
    return (len(keywords) - len(missing)) / len(keywords), missing


def check_keywords(text: str, keywords: list[str], threshold: float = 1.0) -> CheckResult:
    cov, missing = keyword_coverage(text, keywords)
    detail = f"coverage {cov:.0%}" + (f", missing {missing}" if missing else "")
    return CheckResult("keywords", cov >= threshold - 1e-12, detail, cov)


def check_script(text: str, language_or_script: str, threshold: float = 0.6) -> CheckResult:
    target = expected_script(language_or_script)
    profile = script_profile(text)
    frac = profile.get(target, 0.0)
    dominant = next(iter(profile), "none")
    return CheckResult(
        "script", frac >= threshold, f"{frac:.0%} {target} (dominant: {dominant})", frac
    )


def run_checks(record: ResponseRecord, config: CheckConfig | None = None) -> list[CheckResult]:
    cfg = config or CheckConfig()
    text = record.response
    results = [check_nonempty(text)]
    if cfg.detect_refusal:
        results.append(check_refusal(text, cfg.extra_refusal_patterns))
    results.append(check_length(text, cfg.min_words, cfg.max_words))
    if record.keywords:
        results.append(check_keywords(text, record.keywords, cfg.keyword_threshold))
    if record.language:
        results.append(check_script(text, record.language, cfg.script_threshold))
    return results
