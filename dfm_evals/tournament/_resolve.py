import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .config import TournamentConfig, load_tournament_config


def resolve_tournament_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Resolve config from config object/file or persisted tournament state.

    Resolution is explicit:
    - config mappings / TournamentConfig instances are validated directly
    - state paths (`*.db` or existing directories) load persisted `config_json`
    - regular files load via `load_tournament_config`
    """
    if isinstance(config_or_state, TournamentConfig):
        return config_or_state
    if isinstance(config_or_state, Mapping):
        return TournamentConfig.model_validate(dict(config_or_state))

    path = Path(config_or_state)
    if path.suffix == ".db" or (path.exists() and path.is_dir()):
        return _load_state_config(path)

    if path.exists() and path.is_file():
        return load_tournament_config(path)

    raise FileNotFoundError(f"Tournament config/state path not found: {path}")


def _load_state_config(path: Path) -> TournamentConfig:
    config_json = _state_config_json(path)
    if config_json is None or config_json.strip() == "":
        raise ValueError(
            f"No persisted config_json found in tournament state at: {path}"
        )
    try:
        return TournamentConfig.model_validate_json(config_json)
    except Exception as state_error:
        raise ValueError(
            f"Persisted config_json in tournament state is invalid at: {path}"
        ) from state_error


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
