from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_submit_dry_run_uses_post_artifact_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "submit.sh"
    artifact_root = tmp_path / "artifacts"

    env = dict(os.environ)
    env["ENV_FILE"] = str(tmp_path / "missing.env")
    env["POST_ARTIFACT_ROOT"] = str(artifact_root)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--model",
            "google/gemma-3-4b-it",
            "--suite",
            "fundamentals",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    assert f"DFM_EVALS_RUN_ROOT={artifact_root}/evals/runs" in stdout
    assert f"DFM_EVALS_LOG_ROOT={artifact_root}/evals/logs" in stdout
    assert f"DFM_EVALS_EEE_OUTPUT_DIR={artifact_root}/evals/eee/data" in stdout
    assert f"--output {artifact_root}/evals/slurm/" in stdout
    assert f"--error {artifact_root}/evals/slurm/" in stdout
