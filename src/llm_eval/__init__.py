"""llm_eval: rubric-based evaluation of LLM responses."""

from .records import ResponseRecord, load_jsonl, parse_response_records
from .rubric import Criterion, Rubric, RubricScore, DEFAULT_RUBRIC
from .checks import CheckConfig, CheckResult, run_checks, script_profile
from .pairwise import Comparison, parse_comparisons, win_table, bradley_terry

__version__ = "0.1.0"

__all__ = [
    "ResponseRecord",
    "load_jsonl",
    "parse_response_records",
    "Criterion",
    "Rubric",
    "RubricScore",
    "DEFAULT_RUBRIC",
    "CheckConfig",
    "CheckResult",
    "run_checks",
    "script_profile",
    "Comparison",
    "parse_comparisons",
    "win_table",
    "bradley_terry",
]
