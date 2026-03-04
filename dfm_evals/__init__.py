"""CLI and task extensions for inspect_ai."""

from __future__ import annotations

from importlib import import_module

__all__ = ["multi_wiki_qa", "bfcl", "bfcl_da", "ifeval_da", "gleu", "comet"]

_LAZY_EXPORTS = {
    "multi_wiki_qa": "dfm_evals.tasks.multi_wiki_qa:multi_wiki_qa",
    "bfcl": "dfm_evals.tasks.bfcl.bfcl:bfcl",
    "bfcl_da": "dfm_evals.tasks.bfcl.bfcl:bfcl_da",
    "ifeval_da": "dfm_evals.tasks.ifeval_da:ifeval_da",
    "gleu": "dfm_evals.scorers.gleu:gleu",
    "comet": "dfm_evals.scorers.comet:comet",
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'dfm_evals' has no attribute '{name}'")

    module_name, attribute = target.split(":", 1)
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
