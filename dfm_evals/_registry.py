"""Inspect registry imports for dfm_evals."""

from __future__ import annotations

from .hf_eval_yaml import install_hf_eval_yaml_extensions
from .scorers import comet, gleu
from .tasks import bfcl, bfcl_da, multi_wiki_qa
from .tournament.scorer import decision_valid_rate, pairwise_judge

__all__ = [
    "multi_wiki_qa",
    "bfcl",
    "bfcl_da",
    "gleu",
    "comet",
    "pairwise_judge",
    "decision_valid_rate",
]


# Ensure hf/... task specs use the dfm_evals extended eval.yaml loader.
install_hf_eval_yaml_extensions()
