import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .config import TournamentConfig, load_tournament_config


def resolve_tournament_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Resolve config from config object/file or persisted tournament state."""
    if isinstance(config_or_state, TournamentConfig):
        return config_or_state
    if isinstance(config_or_state, Mapping):
        return TournamentConfig.model_validate(dict(config_or_state))

    path = Path(config_or_state)
    try:
        return load_tournament_config(path)
    except Exception as load_error:
        if not _supports_state_fallback(path):
            raise

        config_json = _state_config_json(path)
        if config_json is None or config_json.strip() == "":
            raise ValueError(
                f"Could not load tournament config from {config_or_state!r} "
                "and no config_json was found in run_state."
            ) from load_error

        try:
            return TournamentConfig.model_validate_json(config_json)
        except Exception as state_error:
            raise ValueError(
                f"Found config_json in run_state for {config_or_state!r}, "
                "but it is invalid."
            ) from state_error


def _supports_state_fallback(path: Path) -> bool:
    if path.suffix == ".db":
        return True
    if path.exists() and path.is_dir():
        return True
    return False


def _state_config_json(path: Path) -> str | None:
    db_path = path if path.suffix == ".db" else path / "tournament.db"
    if not db_path.exists() or not db_path.is_file():
        return None

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM run_state WHERE key = ?",
            ("config_json",),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if row is None:
        return None
    return str(row[0])
