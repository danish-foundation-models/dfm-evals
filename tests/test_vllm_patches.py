import threading
import time

from dfm_evals.vllm_patches import (
    apply_hermes_tool_parser_thread_safety_patch,
    apply_instance_method_rlock_patch,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self._borrow_lock = threading.Lock()
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not self._borrow_lock.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            self.encode_calls += 1
            time.sleep(0.01)
            return [len(text)]
        finally:
            self._borrow_lock.release()

    def decode(self, token_ids: list[int]) -> str:
        if not self._borrow_lock.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            self.decode_calls += 1
            time.sleep(0.01)
            return str(token_ids[0])
        finally:
            self._borrow_lock.release()


class FakeToolParser:
    def __init__(self, tokenizer) -> None:
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.model_tokenizer = tokenizer


class FakeMistralTokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer


class FakeHermesParser(FakeToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        if isinstance(tokenizer, FakeMistralTokenizer):
            self.model_tokenizer = tokenizer.tokenizer

        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )
        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]
        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]
        self.buffered_delta_text = ""


class FakeFastTokenizer:
    def __init__(self, backend=None) -> None:
        self._tokenizer = backend or self
        self._borrow_lock = getattr(self._tokenizer, "_borrow_lock", threading.Lock())
        self.call_calls = 0
        self.encode_calls = 0
        self.encode_plus_calls = 0
        self.batch_encode_plus_calls = 0
        self.set_truncation_and_padding_calls = 0
        self.decode_calls = 0
        self.batch_decode_calls = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return self._borrow("call_calls", {"input_ids": [len(text)]})

    def encode(self, text: str, **_: object) -> list[int]:
        return self._borrow("encode_calls", [len(text)])

    def encode_plus(self, text: str, **_: object) -> dict[str, list[int]]:
        return self._borrow("encode_plus_calls", {"input_ids": [len(text)]})

    def batch_encode_plus(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return self._borrow(
            "batch_encode_plus_calls",
            {"input_ids": [[len(text)] for text in texts]},
        )

    def _batch_encode_plus(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return self.batch_encode_plus(texts, **_)

    def set_truncation_and_padding(self, **_: object) -> None:
        self._borrow("set_truncation_and_padding_calls", None)

    def decode(self, token_ids: list[int]) -> str:
        return self._borrow("decode_calls", str(token_ids[0]))

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        return self._borrow("batch_decode_calls", [str(ids[0]) for ids in token_ids])

    def _borrow(self, counter_name: str, result):
        if not self._borrow_lock.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            setattr(self, counter_name, getattr(self, counter_name) + 1)
            time.sleep(0.01)
            return result
        finally:
            self._borrow_lock.release()


class FakeFastTokenizerBackend:
    def __init__(self) -> None:
        self._borrow_lock = threading.Lock()


def test_hermes_patch_avoids_concurrent_tokenizer_borrows() -> None:
    apply_hermes_tool_parser_thread_safety_patch(
        hermes_parser_cls=FakeHermesParser,
        tool_parser_cls=FakeToolParser,
        mistral_tokenizer_cls=FakeMistralTokenizer,
    )

    tokenizer = FakeTokenizer()
    wrapped = FakeMistralTokenizer(tokenizer)
    barrier = threading.Barrier(8)
    parsers = []
    errors = []

    def worker() -> None:
        try:
            barrier.wait()
            parsers.append(FakeHermesParser(wrapped))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(parsers) == 8
    assert tokenizer.encode_calls == 2
    assert tokenizer.decode_calls == 2

    for parser in parsers:
        assert parser.tool_call_start_token_ids == [11]
        assert parser.tool_call_end_token_ids == [12]
        assert parser.tool_call_start_token_array == ["11"]
        assert parser.tool_call_end_token_array == ["12"]


def test_instance_method_rlock_patch_serializes_fast_tokenizer_calls() -> None:
    apply_instance_method_rlock_patch(
        FakeFastTokenizer,
        (
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

    shared_backend = FakeFastTokenizerBackend()
    tokenizers = [FakeFastTokenizer(shared_backend) for _ in range(2)]
    barrier = threading.Barrier(7)
    errors = []
    results = []

    def run(worker) -> None:
        try:
            barrier.wait()
            results.append(worker())
        except Exception as exc:
            errors.append(exc)

    workers = [
        lambda: tokenizers[0]("hello", truncation=True, max_length=4),
        lambda: tokenizers[0].encode("<tool_call>", add_special_tokens=False),
        lambda: tokenizers[0].encode_plus("question", truncation=True, max_length=8),
        lambda: tokenizers[0].batch_encode_plus(["a", "bb"], truncation=True, max_length=4),
        lambda: tokenizers[1].set_truncation_and_padding(max_length=4),
        lambda: tokenizers[1].decode([7]),
        lambda: tokenizers[1].batch_decode([[1], [2]]),
    ]
    threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == len(workers)
    assert tokenizers[0].call_calls == 1
    assert tokenizers[0].encode_calls == 1
    assert tokenizers[0].encode_plus_calls == 1
    assert tokenizers[0].batch_encode_plus_calls == 1
    assert tokenizers[1].set_truncation_and_padding_calls == 1
    assert tokenizers[1].decode_calls == 1
    assert tokenizers[1].batch_decode_calls == 1
