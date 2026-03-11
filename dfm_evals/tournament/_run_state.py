import json
from dataclasses import dataclass

from .store import TournamentStore

KEY_CONFIG_JSON = "config_json"
KEY_RUN_STATUS = "run_status"
KEY_STABLE_BATCHES = "stable_batches"
KEY_NEXT_ROUND_INDEX = "next_round_index"
KEY_PENDING_BATCH_ID = "pending_batch_id"
KEY_PENDING_ROUND_INDEX = "pending_round_index"
KEY_CONVERGED = "converged"
KEY_STOP_REASONS = "stop_reasons"
KEY_LAST_MIN_PROBABILITY = "last_min_probability"
KEY_LAST_MIN_MARGIN = "last_min_margin"
KEY_LAST_BATCH_ID = "last_batch_id"
KEY_PROJECT_ID = "project_id"


@dataclass(frozen=True)
class RunState:
    project_id: str | None
    run_status: str
    stable_batches: int
    next_round_index: int
    pending_batch_id: str | None
    pending_round_index: int
    converged: bool
    stop_reasons: list[str]
    last_min_probability: float | None
    last_min_margin: float | None


def ensure_defaults(store: TournamentStore, *, config_json: str, run_status: str) -> None:
    defaults: dict[str, str] = {
        KEY_CONFIG_JSON: config_json,
        KEY_RUN_STATUS: run_status,
        KEY_STABLE_BATCHES: "0",
        KEY_NEXT_ROUND_INDEX: "1",
        KEY_PENDING_BATCH_ID: "",
        KEY_PENDING_ROUND_INDEX: "0",
        KEY_CONVERGED: "0",
        KEY_STOP_REASONS: "[]",
        KEY_LAST_MIN_PROBABILITY: "",
        KEY_LAST_MIN_MARGIN: "",
    }
    with store.transaction():
        for key, value in defaults.items():
            current = store.get_run_state(key)
            if current is None:
                store.set_run_state(key, value, commit=False)


def load(store: TournamentStore, *, run_status_default: str) -> RunState:
    next_round_index = _get_int(store, KEY_NEXT_ROUND_INDEX, default=1)
    pending_batch_id = _get_optional_str(store, KEY_PENDING_BATCH_ID)
    pending_round_index = _get_int(
        store,
        KEY_PENDING_ROUND_INDEX,
        default=next_round_index,
    )
    return RunState(
        project_id=_get_optional_str(store, KEY_PROJECT_ID),
        run_status=_get_str(store, KEY_RUN_STATUS, default=run_status_default),
        stable_batches=_get_int(store, KEY_STABLE_BATCHES, default=0),
        next_round_index=next_round_index,
        pending_batch_id=pending_batch_id,
        pending_round_index=pending_round_index,
        converged=_get_bool(store, KEY_CONVERGED, default=False),
        stop_reasons=_get_str_list(store, KEY_STOP_REASONS),
        last_min_probability=_get_optional_float(store, KEY_LAST_MIN_PROBABILITY),
        last_min_margin=_get_optional_float(store, KEY_LAST_MIN_MARGIN),
    )


def set_config_json(store: TournamentStore, config_json: str, *, commit: bool = True) -> None:
    store.set_run_state(KEY_CONFIG_JSON, config_json, commit=commit)


def set_pending_batch(
    store: TournamentStore,
    *,
    batch_id: str,
    round_index: int,
    commit: bool = True,
) -> None:
    store.set_run_state(KEY_PENDING_BATCH_ID, batch_id, commit=False)
    store.set_run_state(KEY_PENDING_ROUND_INDEX, str(round_index), commit=commit)


def clear_pending_batch(store: TournamentStore, *, commit: bool = True) -> None:
    store.set_run_state(KEY_PENDING_BATCH_ID, "", commit=False)
    store.set_run_state(KEY_PENDING_ROUND_INDEX, "0", commit=commit)


def set_run_status(store: TournamentStore, run_status: str, *, commit: bool = True) -> None:
    store.set_run_state(KEY_RUN_STATUS, run_status, commit=commit)


def set_stop_reasons(
    store: TournamentStore,
    reasons: list[str],
    *,
    commit: bool = True,
) -> None:
    store.set_run_state(KEY_STOP_REASONS, json.dumps(reasons, sort_keys=True), commit=commit)


def set_stable_batches(store: TournamentStore, stable_batches: int, *, commit: bool = True) -> None:
    store.set_run_state(KEY_STABLE_BATCHES, str(stable_batches), commit=commit)


def set_converged(store: TournamentStore, converged: bool, *, commit: bool = True) -> None:
    store.set_run_state(KEY_CONVERGED, "1" if converged else "0", commit=commit)


def set_next_round_index(
    store: TournamentStore,
    next_round_index: int,
    *,
    commit: bool = True,
) -> None:
    store.set_run_state(KEY_NEXT_ROUND_INDEX, str(next_round_index), commit=commit)


def set_last_batch_id(store: TournamentStore, batch_id: str, *, commit: bool = True) -> None:
    store.set_run_state(KEY_LAST_BATCH_ID, batch_id, commit=commit)


def set_last_min_probability(
    store: TournamentStore,
    min_probability: float | None,
    *,
    commit: bool = True,
) -> None:
    value = "" if min_probability is None else str(min_probability)
    store.set_run_state(KEY_LAST_MIN_PROBABILITY, value, commit=commit)


def set_last_min_margin(
    store: TournamentStore,
    min_margin: float | None,
    *,
    commit: bool = True,
) -> None:
    value = "" if min_margin is None else str(min_margin)
    store.set_run_state(KEY_LAST_MIN_MARGIN, value, commit=commit)


def _get_str(store: TournamentStore, key: str, *, default: str) -> str:
    value = store.get_run_state(key)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped != "" else default


def _get_optional_str(store: TournamentStore, key: str) -> str | None:
    value = store.get_run_state(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped != "" else None


def _get_int(store: TournamentStore, key: str, *, default: int) -> int:
    value = store.get_run_state(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_bool(store: TournamentStore, key: str, *, default: bool) -> bool:
    value = store.get_run_state(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _get_optional_float(store: TournamentStore, key: str) -> float | None:
    value = store.get_run_state(key)
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _get_str_list(store: TournamentStore, key: str) -> list[str]:
    value = store.get_run_state(key)
    if value is None or value.strip() == "":
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]
