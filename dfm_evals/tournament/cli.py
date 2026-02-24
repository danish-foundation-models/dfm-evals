import argparse
from typing import Any, Callable, Sequence, TypeVar

from ._cli_format import (
    add_models_result_payload,
    export_result_payload,
    format_add_models_result,
    format_export_result,
    format_generation_result,
    format_run_result,
    format_status,
    generation_result_payload,
    run_result_payload,
    status_payload,
    write_json_output,
)
from .exports import export_rankings
from .generation import run_generation
from .orchestrator import (
    add_models,
    resume_tournament,
    run_tournament,
    tournament_status,
)
from .store import initialize_tournament_store

ResultT = TypeVar("ResultT")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "init":
        db_path = initialize_tournament_store(args.config)
        print(db_path)
        return 0

    if args.command == "generate":
        models = args.models if args.models else None
        return _emit_result(
            run_generation(args.config, models=models),
            payload_fn=generation_result_payload,
            formatter_fn=format_generation_result,
            json_out=args.json_out,
        )

    if args.command == "run":
        return _emit_result(
            run_tournament(args.config, max_batches=args.max_batches),
            payload_fn=run_result_payload,
            formatter_fn=format_run_result,
            json_out=args.json_out,
        )

    if args.command == "resume":
        return _emit_result(
            resume_tournament(args.target, max_batches=args.max_batches),
            payload_fn=run_result_payload,
            formatter_fn=format_run_result,
            json_out=args.json_out,
        )

    if args.command == "add-model":
        return _emit_result(
            add_models(
                args.target,
                models=args.models,
                max_batches=args.max_batches,
            ),
            payload_fn=add_models_result_payload,
            formatter_fn=format_add_models_result,
            json_out=args.json_out,
        )

    if args.command == "status":
        return _emit_result(
            tournament_status(args.target),
            payload_fn=status_payload,
            formatter_fn=format_status,
            json_out=args.json_out,
        )

    if args.command == "export":
        return _emit_result(
            export_rankings(args.target, output_dir=args.output_dir),
            payload_fn=export_result_payload,
            formatter_fn=format_export_result,
            json_out=args.json_out,
        )

    parser.print_help()
    return 1


def _emit_result(
    result: ResultT,
    *,
    payload_fn: Callable[[ResultT], dict[str, Any]],
    formatter_fn: Callable[[ResultT], str],
    json_out: str | None,
) -> int:
    payload = payload_fn(result)
    output_path = write_json_output(payload, json_out)
    print(formatter_fn(result))
    if output_path is not None:
        print(f"\nSaved JSON output to {output_path.as_posix()}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dfm_evals.tournament.cli")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize tournament state")
    init_parser.add_argument("--config", required=True, help="Path to tournament config")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate contestant responses"
    )
    generate_parser.add_argument(
        "--config", required=True, help="Path to tournament config"
    )
    generate_parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of contestant models",
    )
    generate_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    run_parser = subparsers.add_parser("run", help="Run tournament from config")
    run_parser.add_argument("--config", required=True, help="Path to tournament config")
    run_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    run_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    resume_parser = subparsers.add_parser(
        "resume", help="Resume tournament from config path or state dir"
    )
    resume_parser.add_argument("target", help="Config file path or state dir path")
    resume_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    resume_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    add_model_parser = subparsers.add_parser(
        "add-model", help="Add one or more models to an existing tournament"
    )
    add_model_parser.add_argument(
        "target",
        help="Tournament state directory path (or config/state target)",
    )
    add_model_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model name to add (repeat for multiple models)",
    )
    add_model_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    add_model_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    status_parser = subparsers.add_parser("status", help="Show tournament status")
    status_parser.add_argument("target", help="Config file path or state dir path")
    status_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    export_parser = subparsers.add_parser("export", help="Export rankings artifacts")
    export_parser.add_argument("target", help="Config file path or state dir path")
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for export files",
    )
    export_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
