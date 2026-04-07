from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import wraps
from typing import Any

_INSTANCE_LOCK_ATTR = "_dfm_evals_thread_safe_lock"
_PATCH_LOCK = threading.Lock()
_SHARED_LOCKS: dict[int, threading.RLock] = {}


def apply_instance_method_rlock_patch(
    target_cls: type,
    method_names: Sequence[str],
) -> None:
    patch_flag = "_dfm_evals_instance_method_rlock_patch"
    if getattr(target_cls, patch_flag, False):
        return

    def get_lock_target(instance: Any) -> Any:
        backend = getattr(instance, "_tokenizer", None)
        if backend is not None:
            return backend
        return instance

    def get_instance_lock(instance: Any) -> threading.RLock:
        lock_target = get_lock_target(instance)
        if lock_target is instance:
            lock = getattr(instance, _INSTANCE_LOCK_ATTR, None)
            if lock is not None:
                return lock

        lock = _SHARED_LOCKS.get(id(lock_target))
        if lock is not None:
            return lock

        with _PATCH_LOCK:
            lock = _SHARED_LOCKS.get(id(lock_target))
            if lock is None:
                lock = threading.RLock()
                _SHARED_LOCKS[id(lock_target)] = lock

            if lock_target is instance:
                setattr(instance, _INSTANCE_LOCK_ATTR, lock)
            return lock

    for method_name in method_names:
        original = getattr(target_cls, method_name, None)
        if original is None or not callable(original):
            continue

        @wraps(original)
        def wrapped(
            self: Any,
            *args: Any,
            __original: Any = original,
            **kwargs: Any,
        ) -> Any:
            with get_instance_lock(self):
                return __original(self, *args, **kwargs)

        setattr(target_cls, method_name, wrapped)

    setattr(target_cls, patch_flag, True)


def apply_fast_tokenizer_thread_safety_patch(fast_tokenizer_cls: type) -> None:
    apply_instance_method_rlock_patch(
        fast_tokenizer_cls,
        method_names=(
            "__call__",
            "encode",
            "encode_plus",
            "batch_encode_plus",
            "_batch_encode_plus",
            "set_truncation_and_padding",
            "decode",
            "batch_decode",
        ),
    )


def monkey_patch_fast_tokenizer_thread_safety() -> None:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    apply_fast_tokenizer_thread_safety_patch(PreTrainedTokenizerFast)


def apply_runtime_thread_safety_patches() -> list[str]:
    monkey_patch_fast_tokenizer_thread_safety()
    return [
        "transformers-fast-tokenizer",
    ]
