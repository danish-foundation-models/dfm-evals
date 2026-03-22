from __future__ import annotations

from dfm_evals import _registry
from dfm_evals._exports import (
    REGISTRY_MODULES,
    SCORER_EXPORTS,
    SCORER_PACKAGE_EXPORTS,
    TASK_EXPORTS,
)
from dfm_evals.scorers import __all__ as scorer_exports
from dfm_evals.tasks import __all__ as task_exports


def test_load_registry_imports_manifest_modules(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        _registry,
        "import_module",
        lambda name: calls.append(name) or object(),
    )
    monkeypatch.setattr(
        _registry,
        "install_hf_eval_yaml_extensions",
        lambda: calls.append("install"),
    )

    _registry._load_registry()

    assert calls == [*REGISTRY_MODULES, "install"]


def test_task_exports_follow_manifest() -> None:
    assert task_exports == list(TASK_EXPORTS)


def test_scorer_exports_follow_manifest() -> None:
    assert scorer_exports[: len(SCORER_EXPORTS)] == list(SCORER_EXPORTS)
    assert scorer_exports == list(SCORER_PACKAGE_EXPORTS)
