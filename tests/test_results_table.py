from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _write_eee_record(
    path: Path,
    *,
    model_id: str,
    model_name: str | None = None,
    gleu_score: float,
    exact_score: float,
    evaluation_timestamp: str = "1234.0",
    retrieved_timestamp: str = "1235.0",
    task_name: str = "gec_dala",
    benchmark_name: str = "demo",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"{task_name}/{model_id}/{evaluation_timestamp}",
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": retrieved_timestamp,
        "source_metadata": {
            "source_name": "inspect_ai",
            "source_type": "evaluation_run",
            "source_organization_name": "test",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {
            "name": "inspect_ai",
            "version": "0.1.0",
        },
        "model_info": {
            "name": model_name or model_id,
            "id": model_id,
        },
        "evaluation_results": [
            {
                "evaluation_name": f"{task_name}/gleu/mean",
                "source_data": {
                    "dataset_name": benchmark_name,
                    "source_type": "other",
                },
                "metric_config": {
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0,
                    "additional_details": {
                        "preferred_for_display": "true",
                    },
                },
                "score_details": {
                    "score": gleu_score,
                    "details": {
                        "task": task_name,
                        "scorer": "gleu",
                        "metric": "mean",
                    },
                },
            },
            {
                "evaluation_name": f"{task_name}/exact/mean",
                "source_data": {
                    "dataset_name": benchmark_name,
                    "source_type": "other",
                },
                "metric_config": {
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0,
                },
                "score_details": {
                    "score": exact_score,
                    "details": {
                        "task": task_name,
                        "scorer": "exact",
                        "metric": "mean",
                    },
                },
            },
        ],
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def test_results_table_prefers_marked_metric(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-a" / "record.json",
        model_id="org/model-a",
        gleu_score=0.42,
        exact_score=0.18,
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-b" / "record.json",
        model_id="org/model-b",
        gleu_score=0.55,
        exact_score=0.22,
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("task,scorer,metric,")
    assert "gec_dala,gleu,mean,0.42,0.55" in lines


def test_results_table_normalizes_legacy_local_path_model_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "demo-benchmark" / "unknown" / "flash" / "record.json",
        model_id="/flash",
        model_name="vllm//flash/project_465002183/trl-runs/hermes-4n-full-20260306-lr3e5-warmup50/final",
        gleu_score=0.42,
        exact_score=0.18,
        evaluation_timestamp="1234.0",
        retrieved_timestamp="1235.0",
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "unknown" / "pfs" / "record.json",
        model_id="/pfs",
        model_name="vllm//pfs/lustref1/flash/project_465002183/trl-runs/hermes-4n-full-20260306-lr3e5-warmup50/final",
        gleu_score=0.55,
        exact_score=0.22,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0] == "task,scorer,metric,local/hermes-4n-full-20260306-lr3e5-warmup50-final"
    assert "gec_dala,gleu,mean,0.55" in lines


def test_results_table_keeps_ruler_lengths_as_distinct_task_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "RULER-vt_8k" / "google" / "gemma-3-4b-it" / "vt-8k.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.75,
        exact_score=0.0,
        task_name="RULER-vt@8k",
        benchmark_name="RULER-vt@8k",
    )
    _write_eee_record(
        data_root / "RULER-vt_32k" / "google" / "gemma-3-4b-it" / "vt-32k.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.62,
        exact_score=0.0,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="RULER-vt@32k",
        benchmark_name="RULER-vt@32k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0] == "task,scorer,metric,google/gemma-3-4b-it"
    assert "RULER-vt@32k,gleu,mean,0.62" in lines
    assert "RULER-vt@8k,gleu,mean,0.75" in lines
