"""Sandbox providers for dfm_evals."""

from __future__ import annotations

from .._exports import SANDBOX_EXPORTS, load_export

__all__ = list(SANDBOX_EXPORTS)


def __getattr__(name: str):
    return load_export(
        name=name,
        exports=SANDBOX_EXPORTS,
        namespace=globals(),
        module_name="dfm_evals.sandboxes",
    )
