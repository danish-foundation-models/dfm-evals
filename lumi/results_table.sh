#!/bin/bash
# Print aggregate task/scorer/metric results from inspect .eval artifacts.
# Supports model-comparison pivot table across multiple runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EVAL_LOG_ROOT_HOST=${EVAL_LOG_ROOT_HOST:-$REPO_ROOT/logs/evals-logs}

SELECTOR="latest"
SELECTOR_SET=0
RUN_LABEL=""
LOG_DIR=""

COMPARE_MODELS=0
PRIMARY_ONLY=1
FORMAT=${FORMAT:-table}
COMPARE_ORIENTATION=${COMPARE_ORIENTATION:-model-rows}

usage() {
  cat <<'EOF'
Usage:
  ./lumi/results_table.sh [selector options] [view options]

Selector options:
  --latest             Use latest run dir under eval logs (default)
  --all-runs           Use all run dirs under eval logs
  --run-label <label>  Use logs/evals-logs/<label>
  --log-dir <path>     Use explicit run dir path

View options:
  --compare-models     Pivot table by model (columns are models, rows are tasks)
  --primary-only       For compare mode: use one primary metric per task (default)
  --all-metrics        For compare mode: include every scorer+metric row per task
  --model-rows         For compare mode: rows=models, columns=tasks (default)
  --task-rows          For compare mode: rows=tasks, columns=models
  --format <fmt>       Output format: table|csv|json (default: table)
  --help               Show help

Environment overrides:
  EVAL_LOG_ROOT_HOST   Host eval logs root (default: ./logs/evals-logs)

Examples:
  ./lumi/results_table.sh --latest
  ./lumi/results_table.sh --run-label external_suite_l100_hc
  ./lumi/results_table.sh --compare-models --all-runs
  ./lumi/results_table.sh --compare-models --all-runs --all-metrics --format csv
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

need_value() {
  local opt="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    die "missing value for $opt"
  fi
}

run_dir_for_latest() {
  [[ -d "$EVAL_LOG_ROOT_HOST" ]] || die "eval logs root not found: $EVAL_LOG_ROOT_HOST"
  local newest
  newest="$(
    find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d -printf '%T@|%f\n' \
      | sort -t'|' -k1,1nr \
      | head -n 1 \
      | cut -d'|' -f2
  )"
  [[ -n "$newest" ]] || die "no run directories found under $EVAL_LOG_ROOT_HOST"
  printf '%s' "$EVAL_LOG_ROOT_HOST/$newest"
}

all_run_dirs() {
  [[ -d "$EVAL_LOG_ROOT_HOST" ]] || die "eval logs root not found: $EVAL_LOG_ROOT_HOST"
  find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d | sort
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)
      SELECTOR="latest"
      SELECTOR_SET=1
      RUN_LABEL=""
      LOG_DIR=""
      shift
      ;;
    --all-runs)
      SELECTOR="all-runs"
      SELECTOR_SET=1
      RUN_LABEL=""
      LOG_DIR=""
      shift
      ;;
    --run-label)
      need_value "$1" "$#"
      SELECTOR="run-label"
      SELECTOR_SET=1
      RUN_LABEL="$2"
      LOG_DIR=""
      shift 2
      ;;
    --log-dir)
      need_value "$1" "$#"
      SELECTOR="log-dir"
      SELECTOR_SET=1
      LOG_DIR="$2"
      RUN_LABEL=""
      shift 2
      ;;
    --compare-models)
      COMPARE_MODELS=1
      shift
      ;;
    --primary-only)
      PRIMARY_ONLY=1
      shift
      ;;
    --all-metrics)
      PRIMARY_ONLY=0
      shift
      ;;
    --model-rows)
      COMPARE_ORIENTATION="model-rows"
      shift
      ;;
    --task-rows)
      COMPARE_ORIENTATION="task-rows"
      shift
      ;;
    --format)
      need_value "$1" "$#"
      FORMAT="$2"
      shift 2
      ;;
    --help|-h|help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

case "$FORMAT" in
  table|csv|json)
    ;;
  *)
    die "invalid --format: $FORMAT (expected table|csv|json)"
    ;;
esac

case "$COMPARE_ORIENTATION" in
  model-rows|task-rows)
    ;;
  *)
    die "invalid compare orientation: $COMPARE_ORIENTATION (expected model-rows|task-rows)"
    ;;
esac

# In comparison mode, default to all runs unless selector is explicitly set.
if [[ "$COMPARE_MODELS" == "1" && "$SELECTOR_SET" == "0" ]]; then
  SELECTOR="all-runs"
fi

RUN_DIRS=()
case "$SELECTOR" in
  latest)
    RUN_DIRS+=("$(run_dir_for_latest)")
    ;;
  all-runs)
    while IFS= read -r d; do
      [[ -n "$d" ]] || continue
      RUN_DIRS+=("$d")
    done < <(all_run_dirs)
    ;;
  run-label)
    [[ -n "$RUN_LABEL" ]] || die "--run-label requires a value"
    RUN_DIRS+=("$EVAL_LOG_ROOT_HOST/$RUN_LABEL")
    ;;
  log-dir)
    [[ -n "$LOG_DIR" ]] || die "--log-dir requires a value"
    if [[ "$LOG_DIR" == /* ]]; then
      RUN_DIRS+=("$LOG_DIR")
    elif [[ -d "$LOG_DIR" ]]; then
      RUN_DIRS+=("$LOG_DIR")
    else
      RUN_DIRS+=("$EVAL_LOG_ROOT_HOST/$LOG_DIR")
    fi
    ;;
  *)
    die "unknown selector: $SELECTOR"
    ;;
esac

[[ "${#RUN_DIRS[@]}" -gt 0 ]] || die "no run dirs resolved"
for d in "${RUN_DIRS[@]}"; do
  [[ -d "$d" ]] || die "run dir not found: $d"
done

{
  echo "Selector: $SELECTOR"
  echo "Compare models: $COMPARE_MODELS"
  if [[ "$COMPARE_MODELS" == "1" ]]; then
    echo "Primary only: $PRIMARY_ONLY"
    echo "Orientation: $COMPARE_ORIENTATION"
  fi
  echo "Format: $FORMAT"
  echo "Run dirs:"
  for d in "${RUN_DIRS[@]}"; do
    echo "  - $d"
  done
} >&2

RUN_DIRS_NL="$(printf '%s\n' "${RUN_DIRS[@]}")"

RUN_DIRS_NL="$RUN_DIRS_NL" FORMAT="$FORMAT" COMPARE_MODELS="$COMPARE_MODELS" PRIMARY_ONLY="$PRIMARY_ONLY" COMPARE_ORIENTATION="$COMPARE_ORIENTATION" python3 - <<'PY'
import csv
import glob
import json
import os
import re
import zipfile
from datetime import datetime

run_dirs = [d for d in os.environ["RUN_DIRS_NL"].splitlines() if d]
fmt = os.environ["FORMAT"]
compare_models = os.environ["COMPARE_MODELS"] == "1"
primary_only = os.environ["PRIMARY_ONLY"] == "1"
orientation = os.environ["COMPARE_ORIENTATION"]

preferred_metrics = [
    "accuracy",
    "final_acc",
    "mean",
    "correct",
    "f_score",
    "prompt_strict_acc",
    "inst_strict_acc",
]
preferred_rank = {name: idx for idx, name in enumerate(preferred_metrics)}


def model_from_run_label(run_label: str) -> str | None:
    label = (run_label or "").strip()
    if not label:
        return None

    # Default run labels often end with "__job-<id>"; drop that suffix so
    # reruns aggregate under the same model key.
    label = re.sub(r"__job-\d+$", "", label)

    # Convention: "<suite>__<model-like-label>".
    if "__" in label:
        _suite, _sep, rest = label.partition("__")
        rest = rest.strip()
        if rest:
            return rest
    return label


def parse_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def value_text(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def trim(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


rows = []
for run_dir in run_dirs:
    run_label = os.path.basename(os.path.normpath(run_dir))
    for path in sorted(glob.glob(os.path.join(run_dir, "*.eval"))):
        try:
            with zipfile.ZipFile(path) as zf:
                header = json.loads(zf.read("header.json"))
        except Exception:
            continue

        eval_obj = header.get("eval") or {}
        results = header.get("results") or {}
        stats = header.get("stats") or {}

        task = (eval_obj.get("task") or "").split("/")[-1] or "<unknown>"
        reported_model = eval_obj.get("model") or "-"
        model = model_from_run_label(run_label) or reported_model
        completed = results.get("completed_samples")
        total = results.get("total_samples")
        created = eval_obj.get("created") or stats.get("completed_at") or stats.get("started_at")
        ts = parse_ts(created)
        if ts <= 0:
            try:
                ts = os.path.getmtime(path)
            except Exception:
                ts = 0.0

        scores = results.get("scores") or []
        if not scores:
            rows.append(
                {
                    "run": run_label,
                    "path": path,
                    "ts": ts,
                    "task": task,
                    "scorer": "-",
                    "metric": "-",
                    "value": None,
                    "n": completed,
                    "total": total,
                    "model": model,
                    "reported_model": reported_model,
                }
            )
            continue

        for score in scores:
            scorer = score.get("scorer") or score.get("name") or "-"
            metrics = score.get("metrics") or {}
            if not metrics:
                rows.append(
                    {
                        "run": run_label,
                        "path": path,
                        "ts": ts,
                        "task": task,
                        "scorer": scorer,
                        "metric": "-",
                        "value": None,
                        "n": completed,
                        "total": total,
                        "model": model,
                        "reported_model": reported_model,
                    }
                )
                continue
            for metric_name, metric in metrics.items():
                value = metric.get("value") if isinstance(metric, dict) else None
                rows.append(
                    {
                        "run": run_label,
                        "path": path,
                        "ts": ts,
                        "task": task,
                        "scorer": scorer,
                        "metric": metric_name,
                        "value": value,
                        "n": completed,
                        "total": total,
                        "model": model,
                        "reported_model": reported_model,
                    }
                )

if not rows:
    print("No readable .eval files found.")
    raise SystemExit(0)

if not compare_models:
    rows.sort(key=lambda r: (r["run"], r["task"], r["scorer"], r["metric"], r["model"]))
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        raise SystemExit(0)
    if fmt == "csv":
        fieldnames = ["run", "task", "scorer", "metric", "value", "n", "total", "model"]
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
        raise SystemExit(0)

    columns = [
        ("run", "run", 24, "left"),
        ("task", "task", 22, "left"),
        ("scorer", "scorer", 16, "left"),
        ("metric", "metric", 20, "left"),
        ("value", "value", 10, "right"),
        ("n", "n", 6, "right"),
        ("total", "total", 6, "right"),
        ("model", "model", 28, "left"),
    ]
    widths = {}
    for key, title, cap, _align in columns:
        max_len = len(title)
        for row in rows:
            if key == "value":
                cell = value_text(row.get(key))
            else:
                cell = str(row.get(key) if row.get(key) is not None else "-")
            max_len = max(max_len, len(cell))
        widths[key] = min(max_len, cap)

    header = " ".join(title.ljust(widths[key]) for key, title, _cap, _align in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, _title, _cap, align in columns:
            if key == "value":
                cell = value_text(row.get(key))
            else:
                cell = str(row.get(key) if row.get(key) is not None else "-")
            cell = trim(cell, widths[key])
            if align == "right":
                parts.append(cell.rjust(widths[key]))
            else:
                parts.append(cell.ljust(widths[key]))
        print(" ".join(parts))
    raise SystemExit(0)

# compare_models path
# Keep latest result for each model+task+scorer+metric across included runs.
latest = {}
for row in rows:
    key = (row["model"], row["task"], row["scorer"], row["metric"])
    prev = latest.get(key)
    if prev is None or row["ts"] >= prev["ts"]:
        latest[key] = row

latest_rows = list(latest.values())
models = sorted({row["model"] for row in latest_rows})
tasks = sorted({row["task"] for row in latest_rows})

value_by_key = {
    (row["model"], row["task"], row["scorer"], row["metric"]): row.get("value")
    for row in latest_rows
}

combos_by_task = {}
for row in latest_rows:
    combos_by_task.setdefault(row["task"], set()).add((row["scorer"], row["metric"]))


def combo_rank(task: str, combo: tuple[str, str]):
    scorer, metric = combo
    metric_pri = preferred_rank.get(metric, len(preferred_rank) + 100)
    coverage = 0
    for model in models:
        value = value_by_key.get((model, task, scorer, metric))
        if value is not None:
            coverage += 1
    return (metric_pri, -coverage, scorer, metric)


def unique_model_labels(model_names: list[str]) -> dict[str, str]:
    parts = [m.split("/") for m in model_names]
    max_depth = max((len(p) for p in parts), default=1)
    labels: dict[str, str] = {}
    depth = 1
    while depth <= max_depth:
        seen = {}
        collision = False
        for model, p in zip(model_names, parts):
            label = "/".join(p[-depth:]) if len(p) >= depth else "/".join(p)
            labels[model] = label
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                collision = True
        if not collision:
            return labels
        depth += 1
    return {m: m for m in model_names}


table_rows = []
if primary_only:
    for task in tasks:
        combos = sorted(combos_by_task.get(task, []), key=lambda c: combo_rank(task, c))
        if not combos:
            continue
        scorer, metric = combos[0]
        row = {"task": task, "scorer": scorer, "metric": metric}
        for model in models:
            row[model] = value_by_key.get((model, task, scorer, metric))
        table_rows.append(row)
else:
    for task in tasks:
        combos = sorted(combos_by_task.get(task, []), key=lambda c: (c[0], c[1]))
        for scorer, metric in combos:
            row = {"task": task, "scorer": scorer, "metric": metric}
            for model in models:
                row[model] = value_by_key.get((model, task, scorer, metric))
            table_rows.append(row)

if orientation == "task-rows":
    if fmt == "json":
        print(json.dumps({"orientation": orientation, "models": models, "rows": table_rows}, indent=2))
        raise SystemExit(0)

    if fmt == "csv":
        fieldnames = ["task", "scorer", "metric"] + models
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            out = {"task": row["task"], "scorer": row["scorer"], "metric": row["metric"]}
            for model in models:
                out[model] = row.get(model)
            writer.writerow(out)
        raise SystemExit(0)

    columns = [("task", "task", 22, "left"), ("metric", "metric", 18, "left"), ("scorer", "scorer", 16, "left")]
    model_labels = unique_model_labels(models)
    for model in models:
        columns.append((model, model_labels[model], 24, "right"))

    widths = {}
    for key, title, cap, _align in columns:
        max_len = len(title)
        for row in table_rows:
            cell = value_text(row.get(key)) if key in models else str(row.get(key, "-"))
            max_len = max(max_len, len(cell))
        widths[key] = min(max_len, cap)

    header = " ".join(trim(title, widths[key]).ljust(widths[key]) for key, title, _cap, _align in columns)
    print(header)
    print("-" * len(header))
    for row in table_rows:
        parts = []
        for key, _title, _cap, align in columns:
            if key in models:
                cell = value_text(row.get(key))
            else:
                cell = str(row.get(key, "-"))
            cell = trim(cell, widths[key])
            if align == "right":
                parts.append(cell.rjust(widths[key]))
            else:
                parts.append(cell.ljust(widths[key]))
        print(" ".join(parts))
    raise SystemExit(0)

# model-rows orientation
col_defs = []
seen = set()
for row in table_rows:
    if primary_only:
        base = row["task"]
    else:
        base = f"{row['task']}|{row['scorer']}|{row['metric']}"
    col = base
    suffix = 2
    while col in seen:
        col = f"{base}#{suffix}"
        suffix += 1
    seen.add(col)
    col_defs.append((col, base, row))

model_rows = []
for model in models:
    out = {"model": model}
    for col_key, _base, source_row in col_defs:
        out[col_key] = source_row.get(model)
    model_rows.append(out)

if fmt == "json":
    print(json.dumps({"orientation": orientation, "columns": [c for c, _b, _r in col_defs], "rows": model_rows}, indent=2))
    raise SystemExit(0)

if fmt == "csv":
    fieldnames = ["model"] + [c for c, _b, _r in col_defs]
    writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in model_rows:
        writer.writerow(row)
    raise SystemExit(0)

model_labels = unique_model_labels(models)
columns = [("model", "model", 24, "left")]
for col_key, base, _source_row in col_defs:
    columns.append((col_key, base, 18, "right"))

widths = {}
for key, title, cap, _align in columns:
    max_len = len(title)
    for row in model_rows:
        if key == "model":
            cell = model_labels.get(row.get(key, "-"), row.get(key, "-"))
        else:
            cell = value_text(row.get(key))
        max_len = max(max_len, len(str(cell)))
    widths[key] = min(max_len, cap)

header = " ".join(trim(title, widths[key]).ljust(widths[key]) for key, title, _cap, _align in columns)
print(header)
print("-" * len(header))
for row in model_rows:
    parts = []
    for key, _title, _cap, align in columns:
        if key == "model":
            cell = model_labels.get(row.get(key, "-"), row.get(key, "-"))
        else:
            cell = value_text(row.get(key))
        cell = trim(str(cell), widths[key])
        if align == "right":
            parts.append(cell.rjust(widths[key]))
        else:
            parts.append(cell.ljust(widths[key]))
    print(" ".join(parts))
PY
