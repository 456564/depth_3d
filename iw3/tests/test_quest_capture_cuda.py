import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from iw3.desktop.screenshot_thread_cuda import ScreenshotThreadWCCUDA


@pytest.fixture
def native_capture(monkeypatch: pytest.MonkeyPatch) -> Any:
    instances = []
    constructing, allow_construct = threading.Event(), threading.Event()
    entering_start, allow_start = threading.Event(), threading.Event()
    allow_construct.set()
    allow_start.set()

    class Capture:
        def __init__(self, **kwargs: Any) -> None:
            self.running = False
            self.start_calls = self.stop_calls = 0
            self.started, self.stop_called, self.wake = threading.Event(), threading.Event(), threading.Event()
            instances.append(self)
            constructing.set()
            assert allow_construct.wait(2)

        def event(self, handler: Any) -> Any:
            return handler

        def start(self) -> None:
            self.start_calls += 1
            entering_start.set()
            assert allow_start.wait(2)
            # 模拟 wc_cuda：start 会覆盖原生启动前收到的 stop。
            self.running = True
            self.wake.clear()
            self.started.set()
            while self.running:
                self.wake.wait(0.1)

        def stop(self) -> None:
            self.stop_calls += 1
            self.running = False
            self.wake.set()
            self.stop_called.set()

    monkeypatch.setitem(sys.modules, "wc_cuda", SimpleNamespace(WindowsCapture=Capture))
    thread = ScreenshotThreadWCCUDA(30, 10, 8, 0, "Browser", torch.device("cuda:0"))
    yield SimpleNamespace(
        thread=thread,
        instances=instances,
        constructing=constructing,
        allow_construct=allow_construct,
        entering_start=entering_start,
        allow_start=allow_start,
    )
    allow_construct.set()
    allow_start.set()
    for instance in instances:
        if instance.start_calls:
            assert instance.started.wait(1)
        instance.stop()
    if thread.ident is not None:
        thread.join(2)
        assert not thread.is_alive()


def test_cuda_capture_stop_terminates_native_without_any_frame(native_capture: Any) -> None:
    state = native_capture
    state.thread.start()
    assert state.constructing.wait(1)
    native = state.instances[0]
    assert native.started.wait(1)
    state.thread.stop()
    assert native.stop_called.is_set()
    assert not state.thread.is_alive() and state.thread.frame_set_event.is_set()


def test_cuda_capture_stop_before_thread_start_prevents_native_start(native_capture: Any) -> None:
    state = native_capture
    state.thread.stop()
    state.thread.start()
    state.thread.join(0.5)
    assert not state.thread.is_alive()
    assert all(instance.start_calls == 0 for instance in state.instances)


def test_cuda_capture_stop_during_construction_does_not_start_late(native_capture: Any) -> None:
    state = native_capture
    state.allow_construct.clear()
    state.thread.start()
    assert state.constructing.wait(1)
    finished = threading.Event()

    def stop() -> None:
        state.thread.stop()
        finished.set()

    worker = threading.Thread(target=stop)
    worker.start()
    try:
        assert state.thread.stop_event.wait(1)
        state.allow_construct.set()
        assert finished.wait(0.5)
        assert not state.thread.is_alive() and state.instances[0].start_calls == 0
    finally:
        state.allow_construct.set()
        state.instances[0].stop()
        worker.join(2)


def test_cuda_capture_stop_during_start_is_not_overwritten_by_native_start(native_capture: Any) -> None:
    state = native_capture
    state.allow_start.clear()
    state.thread.start()
    assert state.entering_start.wait(1)
    native = state.instances[0]
    finished = threading.Event()

    def stop() -> None:
        state.thread.stop()
        finished.set()

    worker = threading.Thread(target=stop)
    worker.start()
    try:
        assert native.stop_called.wait(0.5)
        state.allow_start.set()
        assert finished.wait(0.5)
        assert not state.thread.is_alive() and native.stop_calls >= 2
    finally:
        state.allow_start.set()
        assert native.started.wait(1)
        native.stop()
        worker.join(2)
