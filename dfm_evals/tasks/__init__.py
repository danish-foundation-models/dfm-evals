"""Task implementations for dfm_evals."""

from __future__ import annotations

from .._exports import REGISTRY_EXPORTS, TASK_EXPORTS, load_export

__all__ = list(TASK_EXPORTS)

_TASK_LAZY_EXPORTS = {name: REGISTRY_EXPORTS[name] for name in TASK_EXPORTS}


def __getattr__(name: str):
    return load_export(
        name=name,
        exports=_TASK_LAZY_EXPORTS,
        namespace=globals(),
        module_name="dfm_evals.tasks",
    )
