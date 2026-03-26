from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _write_eee_record(
    path: Path,
    *,
    model_id: str,
    gleu_score: float,
    exact_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"gec_dala/{model_id}/1234.0",
        "evaluation_timestamp": "1234.0",
        "retrieved_timestamp": "1235.0",
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
            "name": model_id,
            "id": model_id,
        },
        "evaluation_results": [
            {
                "evaluation_name": "gec_dala/gleu/mean",
                "source_data": {
                    "dataset_name": "demo",
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
                        "task": "gec_dala",
                        "scorer": "gleu",
                        "metric": "mean",
                    },
                },
            },
            {
                "evaluation_name": "gec_dala/exact/mean",
                "source_data": {
                    "dataset_name": "demo",
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
                        "task": "gec_dala",
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
