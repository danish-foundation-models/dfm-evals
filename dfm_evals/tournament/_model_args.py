import os
import re
from functools import lru_cache
from typing import Any, Mapping

VLLM_BASE_URL = "VLLM_BASE_URL"


def apply_default_vllm_model_args(
    model_name: str,
    model_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply default local vLLM args for tournament runs.

    If a model is `vllm/...` and no explicit device/tensor-parallel settings are
    provided, default to all visible CUDA devices so local tournament runs make
    full use of available GPUs.
    """
    resolved = dict(model_args)
    if not model_name.startswith("vllm/"):
        return resolved

    if _has_explicit_vllm_device_args(resolved):
        return resolved

    if _uses_existing_vllm_server(resolved):
        return resolved

    visible_devices = _visible_cuda_devices()
    if len(visible_devices) > 0:
        resolved["device"] = visible_devices

    return resolved


def _has_explicit_vllm_device_args(model_args: Mapping[str, Any]) -> bool:
    return any(
        key in model_args for key in ("device", "devices", "tensor_parallel_size")
    )


def _uses_existing_vllm_server(model_args: Mapping[str, Any]) -> bool:
    return ("base_url" in model_args) or (VLLM_BASE_URL in os.environ)


def _visible_cuda_devices() -> list[int | str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return _parse_cuda_visible_devices(visible_devices)

    device_count = _cuda_device_count()
    if device_count <= 0:
        return []
    return list(range(device_count))


def _parse_cuda_visible_devices(value: str) -> list[int | str]:
    tokens = [token.strip() for token in value.split(",") if token.strip() != ""]
    if len(tokens) == 0:
        return []
    if len(tokens) == 1 and tokens[0] in ("-1", "none", "None"):
        return []
    if all(re.fullmatch(r"\d+", token) is not None for token in tokens):
        return [int(token) for token in tokens]
    return tokens


@lru_cache(maxsize=1)
def _cuda_device_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0
