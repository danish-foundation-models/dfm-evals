"""Import registry modules for dfm_evals."""

from __future__ import annotations

from importlib import import_module

from ._exports import REGISTRY_MODULES
from .hf_eval_yaml import install_hf_eval_yaml_extensions


def _load_registry() -> None:
    for module_name in REGISTRY_MODULES:
        import_module(module_name)
    install_hf_eval_yaml_extensions()


_load_registry()
