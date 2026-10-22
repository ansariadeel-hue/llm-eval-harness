"""Command-line interface: ``llm-eval score`` and ``llm-eval compare``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .pairwise import bradley_terry, parse_comparisons, win_table
from .records import DataError, load_jsonl, parse_response_records
from .report import evaluate, leaderboard, render_failures, render_leaderboard, render_pairwise
from .rubric import DEFAULT_RUBRIC, Rubric


def _cmd_score(args: argparse.Namespace) -> int:
    rubric = Rubric.load(args.rubric) if args.rubric else DEFAULT_RUBRIC
    records = parse_response_records(load_jsonl(args.responses))
    if not records:
        raise DataError("no records found")
    results = evaluate(records, rubric)
    board = leaderboard(results)
    if args.json:
        out = {
            "rubric": rubric.name,
            "weights": rubric.weights,
            "leaderboard": [s.to_dict() for s in board],
        }
        if args.details:
            out["records"] = [r.to_dict() for r in results]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    weights = ", ".join(f"{k}={v:.2f}" for k, v in rubric.weights.items())
    print(f"Rubric '{rubric.name}' ({weights})")
    print(f"{len(records)} responses from {len(board)} models\n")
    print(render_leaderboard(board, rubric))
    print("\nscore = weighted rubric score (0-100); checks = share of responses passing all heuristic checks")
    print("\nIssues:")
    print(render_failures(results))
    if args.details:
        print("\nPer-response scores:")
        for r in results:
            sc = "-" if r.score.overall is None else f"{100 * r.score.overall:.1f}"
            print(f"  {r.record.id:<8} {r.record.model:<12} {sc:>5}  checks={'ok' if r.checks_passed else 'FAIL'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    comps = parse_comparisons(load_jsonl(args.pairs))
    table = win_table(comps)
    bt = bradley_terry(comps, prior=args.prior)
    if args.json:
        out = {
            "comparisons": len(comps),
            "prior": args.prior,
            "iterations": bt.iterations,
            "converged": bt.converged,
            "log_likelihood": round(bt.log_likelihood, 6),
            "models": [
                {
                    "model": m,
                    "wins": table[m].wins,
                    "losses": table[m].losses,
                    "ties": table[m].ties,
                    "win_rate": round(table[m].win_rate, 6),
                    "bt_strength": round(s, 6),
                }
                for m, s in bt.ranking()
            ],
        }
        print(json.dumps(out, indent=2))
        return 0
    print(f"{len(comps)} pairwise judgments, {len(table)} models")
    print(
        f"Bradley-Terry (MM): {bt.iterations} iterations, "
        f"{'converged' if bt.converged else 'NOT converged'}, log-likelihood {bt.log_likelihood:.3f}\n"
    )
    print(render_pairwise(table, bt))
    ranked = [m for m, _ in bt.ranking()]
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        print(f"\nP({a} beats {b}) = {bt.prob(a, b):.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-eval", description="Rubric-based evaluation of LLM responses.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score responses with a weighted rubric + heuristic checks")
    s.add_argument("responses", help="JSONL of {id, prompt, response, model, ratings?, keywords?, language?}")
    s.add_argument("--rubric", help="rubric JSON (default: built-in accuracy/relevance/coherence/safety)")
    s.add_argument("--details", action="store_true", help="include per-response results")
    s.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    s.set_defaults(func=_cmd_score)

    c = sub.add_parser("compare", help="aggregate pairwise preferences (win rates + Bradley-Terry)")
    c.add_argument("pairs", help="JSONL of {model_a, model_b, winner, prompt_id?}")
    c.add_argument("--prior", type=float, default=0.0, help="virtual wins per side per pair (regularisation)")
    c.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Avoid crashes on consoles that cannot encode Devanagari/Arabic text (e.g. cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    try:
        return args.func(args)
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
