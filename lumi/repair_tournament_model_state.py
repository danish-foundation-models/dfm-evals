#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RepairSummary:
    model_id: str
    match_count: int
    rated_match_count: int
    judgment_count: int
    batch_ids: list[str]
    pending_batch_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove pending tournament matches/judgments for one model without "
            "touching ratings or responses."
        )
    )
    parser.add_argument(
        "target",
        help="Tournament DB path, state dir, or run dir",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact contestant model name to repair",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without mutating the DB",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped DB backup before applying changes",
    )
    return parser.parse_args()


def resolve_db_path(target: str | Path) -> Path:
    path = Path(target)
    if path.is_file():
        if path.suffix == ".db":
            return path
        raise FileNotFoundError(f"Expected a .db file, got: {path}")
    if path.name == "state":
        return path / "tournament.db"
    state_db = path / "state" / "tournament.db"
    if state_db.is_file():
        return state_db
    return path / "tournament.db"


def load_summary(conn: sqlite3.Connection, model_name: str) -> RepairSummary:
    row = conn.execute(
        "SELECT model_id FROM models WHERE model_name = ?",
        (model_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Model not found in tournament DB: {model_name}")
    model_id = str(row["model_id"])

    match_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM matches
            WHERE model_a = ?
               OR model_b = ?
            """,
            (model_id, model_id),
        ).fetchone()[0]
    )
    rated_match_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM matches
            WHERE (model_a = ? OR model_b = ?)
              AND status = 'rated'
            """,
            (model_id, model_id),
        ).fetchone()[0]
    )
    judgment_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM judgments
            WHERE match_id IN (
              SELECT match_id
              FROM matches
              WHERE model_a = ?
                 OR model_b = ?
            )
            """,
            (model_id, model_id),
        ).fetchone()[0]
    )
    batch_ids = sorted(
        str(row["batch_id"])
        for row in conn.execute(
            """
            SELECT DISTINCT batch_id
            FROM matches
            WHERE model_a = ?
               OR model_b = ?
            ORDER BY batch_id
            """,
            (model_id, model_id),
        ).fetchall()
    )
    pending_row = conn.execute(
        "SELECT value FROM run_state WHERE key = 'pending_batch_id'"
    ).fetchone()
    pending_batch_id = None
    if pending_row is not None:
        value = str(pending_row["value"]).strip()
        pending_batch_id = value if value else None

    return RepairSummary(
        model_id=model_id,
        match_count=match_count,
        rated_match_count=rated_match_count,
        judgment_count=judgment_count,
        batch_ids=batch_ids,
        pending_batch_id=pending_batch_id,
    )


def backup_db(db_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.stem}.repair-backup.{timestamp}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def apply_repair(conn: sqlite3.Connection, summary: RepairSummary) -> None:
    if summary.rated_match_count > 0:
        raise RuntimeError(
            "Refusing to repair model state because rated matches already exist for "
            f"{summary.model_id}."
        )

    with conn:
        conn.execute(
            """
            DELETE FROM judgments
            WHERE match_id IN (
              SELECT match_id
              FROM matches
              WHERE model_a = ?
                 OR model_b = ?
            )
            """,
            (summary.model_id, summary.model_id),
        )
        conn.execute(
            """
            DELETE FROM matches
            WHERE model_a = ?
               OR model_b = ?
            """,
            (summary.model_id, summary.model_id),
        )
        if summary.pending_batch_id is not None and summary.pending_batch_id in summary.batch_ids:
            conn.execute(
                """
                UPDATE run_state
                SET value = '', updated_at = CURRENT_TIMESTAMP
                WHERE key = 'pending_batch_id'
                """
            )
            conn.execute(
                """
                UPDATE run_state
                SET value = '0', updated_at = CURRENT_TIMESTAMP
                WHERE key = 'pending_round_index'
                """
            )
        conn.execute(
            """
            UPDATE run_state
            SET value = 'running', updated_at = CURRENT_TIMESTAMP
            WHERE key = 'run_status'
            """
        )
        conn.execute(
            """
            UPDATE run_state
            SET value = '[]', updated_at = CURRENT_TIMESTAMP
            WHERE key = 'stop_reasons'
            """
        )
        conn.execute(
            """
            UPDATE run_state
            SET value = '0', updated_at = CURRENT_TIMESTAMP
            WHERE key = 'converged'
            """
        )
        conn.execute(
            """
            UPDATE run_state
            SET value = '0', updated_at = CURRENT_TIMESTAMP
            WHERE key = 'stable_batches'
            """
        )


def main() -> int:
    args = parse_args()
    db_path = resolve_db_path(args.target)
    if not db_path.is_file():
        raise FileNotFoundError(f"Tournament DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        summary = load_summary(conn, args.model)
        print(f"db_path={db_path}")
        print(f"model_id={summary.model_id}")
        print(f"match_count={summary.match_count}")
        print(f"rated_match_count={summary.rated_match_count}")
        print(f"judgment_count={summary.judgment_count}")
        print(f"batch_ids={','.join(summary.batch_ids)}")
        print(f"pending_batch_id={summary.pending_batch_id or ''}")

        if args.dry_run or summary.match_count == 0:
            print("dry_run=1" if args.dry_run else "dry_run=0")
            print("applied=0")
            return 0

        if not args.no_backup:
            backup_path = backup_db(db_path)
            print(f"backup_path={backup_path}")

        apply_repair(conn, summary)
        print("applied=1")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
