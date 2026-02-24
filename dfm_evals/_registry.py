"""Inspect registry imports for dfm_evals."""

from __future__ import annotations

from .hf_eval_yaml import install_hf_eval_yaml_extensions
from .scorers import comet, gleu
from .tasks import multi_wiki_qa

__all__ = ["multi_wiki_qa", "gleu", "comet"]


# Ensure hf/... task specs use the dfm_evals extended eval.yaml loader.
install_hf_eval_yaml_extensions()
