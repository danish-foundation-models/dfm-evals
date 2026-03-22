"""CLI and task extensions for inspect_ai."""

from __future__ import annotations

from ._exports import REGISTRY_EXPORTS, TOP_LEVEL_EXPORTS, load_export

__all__ = list(TOP_LEVEL_EXPORTS)

_TOP_LEVEL_LAZY_EXPORTS = {
    name: REGISTRY_EXPORTS[name] for name in TOP_LEVEL_EXPORTS
}


def __getattr__(name: str):
    return load_export(
        name=name,
        exports=_TOP_LEVEL_LAZY_EXPORTS,
        namespace=globals(),
        module_name="dfm_evals",
    )
