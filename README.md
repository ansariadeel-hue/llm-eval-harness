# llm-eval-harness

Rubric-based evaluation of LLM responses: weighted rubrics, automatic heuristic checks and pairwise Bradley–Terry rankings, in pure Python.

[![CI](https://github.com/ansariadeel-hue/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/ansariadeel-hue/llm-eval-harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ansariadeel-hue/llm-eval-harness/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/ansariadeel-hue/llm-eval-harness)

## Why

LLM evaluation work usually involves three kinds of signal: annotators scoring responses against a rubric (accuracy, relevance, coherence, safety…), cheap automatic checks that catch obvious failures (empty answers, refusals, a Hindi prompt answered in English), and side-by-side preference judgments between models. They usually end up in separate spreadsheets.

This harness puts them in one place. It produces a per-model leaderboard from rubric ratings, lists every response that failed an automatic check, and turns pairwise judgments into win rates and Bradley–Terry strengths. The heuristic checks handle multiple languages: the script check uses Unicode block ranges, so it can tell whether a Hindi or Marathi answer is really in Devanagari, an Urdu answer in Arabic script, or a Tamil answer in Tamil script.

## Features

- **Weighted rubric in JSON**: any number of criteria, each with a weight and its own scale (default 1–5). Weights are normalized. Each criterion is min–max normalized, so the overall score falls in 0–1 (shown as 0–100).
- **Several annotators per response**: ratings are averaged per criterion. If a criterion was never rated, it is reported as missing and the remaining weights are renormalized.
- **Gate criteria**: `min_acceptable` flags a response whose mean rating is below a threshold, for example safety < 3.
- **Heuristic checks**:
  - empty or no-content detection
  - refusal detection with English, Hindi, Marathi and Urdu patterns, plus your own regexes
  - word-count bounds
  - required-keyword coverage (case-insensitive and NFC-normalized)
  - expected-script check using Unicode ranges for Latin, Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Arabic, Cyrillic and Han. Combining marks (matras) count toward the script of the letter they attach to.
- **Pairwise aggregation**:
  - wins, losses and ties, with ties counted as half a win
  - win rates
  - a **Bradley–Terry** model fitted with Hunter's MM algorithm, reported with log-likelihood and an Elo-like scale
  - detection of cases where the MLE does not exist (a model that never loses), with an optional `--prior` for regularization
- **Per-model leaderboard**: a plain-text table, or `--json` for pipelines.
- Standard library only. No runtime dependencies.

## Install

```bash
pip install git+https://github.com/ansariadeel-hue/llm-eval-harness
```

## Quickstart

Score responses with a rubric and the heuristic checks:

```bash
llm-eval score examples/responses.jsonl --rubric examples/rubric.json
```

Real output on the bundled examples:

```text
Rubric 'general-qa-v1' (accuracy=0.40, relevance=0.25, coherence=0.20, safety=0.15)
15 responses from 3 models

rank  model    n  score  accuracy  relevance  coherence  safety  checks  refusal  empty  flagged
----  -------  -  -----  --------  ---------  ---------  ------  ------  -------  -----  -------
   1  model-a  5   98.9      5.00       4.90       4.90    5.00    100%        0      0        0
   2  model-b  5   71.9      3.25       4.62       4.00    4.12     40%        0      0        1
   3  model-c  5   61.8      3.20       3.20       3.20    5.00     40%        1      1        0

score = weighted rubric score (0-100); checks = share of responses passing all heuristic checks

Issues:
  q1 [model-b]: rubric flag safety<3
  q1 [model-c]: no_refusal (refusal language detected); keywords (coverage 0%, missing ['100'])
  q2 [model-b]: script (23% devanagari (dominant: latin))
  q2 [model-c]: nonempty (response is empty or has no letters/digits); length (0 words, bounds [3, 120]); keywords (coverage 0%, missing ['नई दिल्ली']); script (0% devanagari (dominant: none))
  q3 [model-b]: keywords (coverage 0%, missing ['چنبیلی'])
  q4 [model-c]: length (1 words, bounds [3, 120]); keywords (coverage 0%, missing ['சென்னை']); script (0% tamil (dominant: latin))
  q5 [model-b]: keywords (coverage 33%, missing ['carbon dioxide', 'glucose'])
```

Aggregate pairwise preferences:

```bash
llm-eval compare examples/pairwise.jsonl
```

```text
18 pairwise judgments, 3 models
Bradley-Terry (MM): 31 iterations, converged, log-likelihood -10.846

rank  model    games  W  L  T  win_rate  bt_strength   elo
----  -------  -----  -  -  -  --------  -----------  ----
   1  model-a     12  8  2  2     75.0%       0.5991  1128
   2  model-b     12  4  6  2     41.7%       0.2247   957
   3  model-c     12  3  7  2     33.3%       0.1762   915

P(model-a beats model-b) = 0.727
```

Other options:

- `--details` adds per-response scores.
- `--json` gives machine-readable output.
- `compare --prior 0.5` adds 0.5 virtual wins to each side of every pair that met. Use it when some model is undefeated.

You can also run `python -m llm_eval ...` from a checkout.

## How the scores are computed

- **Rubric score.** For criterion *c* with weight *w_c*, scale [lo, hi] and mean rating *r_c*, the score is `Σ w_c · (r_c − lo)/(hi − lo) / Σ w_c`. With weights 0.4/0.3/0.2/0.1 and ratings 5/4/3/5, the score is 0.825. That equals a weighted mean of 4.3 on the 1–5 scale, and the test suite checks it.
- **Bradley–Terry.** The model is P(i beats j) = p_i / (p_i + p_j). The MM update is `p_i ← W_i / Σ_j n_ij / (p_i + p_j)`, followed by normalization to Σp = 1. The tests check the closed form for two players (3–1 gives 0.75/0.25), equal strengths for a symmetric cycle, the MLE stationarity condition (observed wins = expected wins) and that the result is a local maximum of the likelihood.
- **Elo column.** `1000 + 400·log10(p_i / geometric-mean(p))`. It is only a familiar scale for the same fit.

## Data format

**Responses** are JSONL, one object per line:

```json
{"id": "q2", "model": "model-a", "prompt": "भारत की राजधानी क्या है?", "response": "भारत की राजधानी नई दिल्ली है।",
 "language": "hi", "keywords": ["नई दिल्ली"],
 "ratings": [{"accuracy": 5, "relevance": 5, "coherence": 5, "safety": 5},
             {"accuracy": 5, "relevance": 5, "coherence": 4, "safety": 5}]}
```

| field | required | meaning |
|---|---|---|
| `id`, `prompt`, `response`, `model` | yes | `(id, model)` must be unique |
| `ratings` | no | one object, or a list with one object per annotator: `{criterion: rating}` |
| `keywords` | no | required keywords for the coverage check |
| `language` | no | ISO code (`hi`, `mr`, `ur`, `ta`, `en`, …) or a script name (`devanagari`) for the script check |

**Rubric** is a JSON file:

```json
{
  "name": "general-qa-v1",
  "criteria": [
    {"name": "accuracy", "weight": 0.40, "scale": [1, 5]},
    {"name": "safety",   "weight": 0.15, "scale": [1, 5], "min_acceptable": 3}
  ],
  "checks": {"min_words": 3, "max_words": 120, "keyword_threshold": 1.0, "script_threshold": 0.6}
}
```

If you leave out `--rubric`, a built-in accuracy/relevance/coherence/safety rubric with weights 0.4/0.25/0.2/0.15 is used.

**Pairwise judgments** are JSONL: `{"prompt_id": "q1", "model_a": "model-a", "model_b": "model-b", "winner": "model-a"}`. `winner` can be either model name, `a`/`b` or `tie`.

## Project structure

```text
llm-eval-harness/
├── src/llm_eval/
│   ├── records.py    # JSONL loading + validation
│   ├── rubric.py     # weighted rubric, score aggregation
│   ├── checks.py     # empty/refusal/length/keyword/script checks
│   ├── pairwise.py   # win rates, Bradley–Terry (MM)
│   ├── report.py     # per-record evaluation, leaderboard, tables
│   └── cli.py        # `llm-eval score|compare`
├── tests/            # pytest suite (65 tests)
├── examples/         # responses.jsonl, rubric.json, pairwise.jsonl
└── .github/workflows/ci.yml
```

Run the tests from the project root with `python -m pytest -q`. `pythonpath` is set in `pyproject.toml`, so no install is needed.

## License

MIT © 2026 Ansari Adeel Ahmad. See [LICENSE](LICENSE).
