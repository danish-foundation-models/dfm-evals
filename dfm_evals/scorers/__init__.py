from __future__ import annotations

from .._exports import SCORER_PACKAGE_EXPORTS, load_export

__all__ = list(SCORER_PACKAGE_EXPORTS)


def __getattr__(name: str):
    return load_export(
        name=name,
        exports=SCORER_PACKAGE_EXPORTS,
        namespace=globals(),
        module_name="dfm_evals.scorers",
    )
