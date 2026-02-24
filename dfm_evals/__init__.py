"""CLI and task extensions for inspect_ai."""

from .scorers import comet, gleu
from .tasks import bfcl, bfcl_da, multi_wiki_qa

__all__ = ["multi_wiki_qa", "bfcl", "bfcl_da", "gleu", "comet"]
