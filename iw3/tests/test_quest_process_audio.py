import ctypes
import importlib
import threading
from collections import deque
from typing import Any

import pytest


class NativeAudio:
    def __init__(self) -> None:
        self.chunks = deque([b"\x01\x00\x02\x00" * 480])
        self.stopped = threading.Event()
        self.created: list[tuple[int, int]] = []
        self.freed: list[int] = []
        self.failure = 0
        self.start_failure = 0
        self.stop_failure = 0

    def pa_create(self, pid: int, include: int, handle: Any) -> int:
        self.created.append((pid, include))
        handle._obj.value = 123
        return 0

    def pa_start(self, handle: Any) -> int:
        return self.start_failure

    def pa_read(self, handle: Any, buffer: Any, size: int, timeout: int, written: Any) -> int:
        if self.failure:
            return self.failure
        if self.chunks:
            data = self.chunks.popleft()
            assert len(data) <= size
            ctypes.memmove(buffer, data, len(data))
            written._obj.value = len(data)
            return 0
        self.stopped.wait(0.01)
        written._obj.value = 0
        return 1

    def pa_stop(self, handle: Any) -> int:
        self.stopped.set()
        return self.stop_failure

    def pa_free(self, handle: Any) -> None:
        self.freed.append(handle.value)


@pytest.fixture
def capture(monkeypatch):
    module = importlib.import_module("iw3.desktop.quest_process_audio")
    native = NativeAudio()
    monkeypatch.setattr(module, "_load_library", lambda: native)
    return module.ProcessAudio(), native


def test_process_capture_excludes_own_process_and_preserves_pcm(capture) -> None:
    import os
    import time

    audio, native = capture
    audio.start()
    try:
        deadline = time.monotonic() + 1
        chunks: list[bytes] = []
        while not chunks and time.monotonic() < deadline:
            chunks = audio.get_chunks()
            threading.Event().wait(0.01)
        assert native.created == [(os.getpid(), 0)]
        assert chunks == [b"\x01\x00\x02\x00" * 480]
        audio.check_error()
    finally:
        audio.stop()
    audio.stop()
    assert native.freed == [123]


def test_native_start_failure_frees_without_starting_reader(capture) -> None:
    audio, native = capture
    native.start_failure = -1
    with pytest.raises(RuntimeError, match="HRESULT"):
        audio.start()
    assert native.freed == [123]
    audio.stop()
    assert native.freed == [123]


def test_native_stop_failure_still_joins_and_frees_capture(capture) -> None:
    audio, native = capture
    audio.start()
    reader = audio._thread
    native.stop_failure = -1
    with pytest.raises(RuntimeError, match="HRESULT=0xFFFFFFFF"):
        audio.stop()
    assert not reader.is_alive()
    assert audio._thread is None
    assert not audio._handle.value
    assert native.freed == [123]
    assert audio.get_chunks() == []
    audio.stop()
    assert native.freed == [123]


def test_reader_start_failure_still_frees_native_capture(capture, monkeypatch) -> None:
    audio, native = capture

    def fail_start(self) -> None:
        raise RuntimeError("reader thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="reader thread unavailable"):
        audio.start()
    assert audio._thread is None
    assert not audio._handle.value
    assert native.freed == [123]
    audio.stop()
    assert native.freed == [123]


def test_native_read_error_reaches_caller_and_is_cleaned(capture) -> None:
    import time

    audio, native = capture
    native.failure = -1
    audio.start()
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                audio.check_error()
            except RuntimeError as error:
                assert "HRESULT" in str(error.__cause__)
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("read failure was lost")
    finally:
        audio.stop()
    assert native.freed == [123]


def test_malformed_native_pcm_is_rejected(capture) -> None:
    import time

    audio, native = capture
    native.chunks = deque([b"x"])
    audio.start()
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                audio.check_error()
            except RuntimeError as error:
                assert "PCM" in str(error.__cause__)
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("invalid PCM was accepted")
    finally:
        audio.stop()
