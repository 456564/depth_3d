import ctypes
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from iw3.desktop import screenshot_process as capture

from .test_quest_capture_cuda import native_capture as native_capture


@pytest.fixture
def cpu_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Any]]:
    states = []

    def make(width: int = 2560, height: int = 1440, out_width: int = 1920, out_height: int = 1080) -> Any:
        thread = capture.ScreenshotProcess(
            30, out_width, out_height, 0, "Synthetic", torch.device("cpu"), draw_cursor_enabled=False
        )
        raw = bytearray(width * height * 4)
        wake = threading.Event()
        process = SimpleNamespace(pid=123, alive=True, join=lambda **_: None)
        process.is_alive = lambda: process.alive
        process.terminate = lambda: setattr(process, "alive", False)
        thread.screen_width, thread.screen_height = width, height
        thread.process = process
        thread.process_frame_buffer = SimpleNamespace(buf=raw, close=lambda: None, unlink=lambda: None)
        thread.process_frame_mapping = [0] * 22
        thread.process_frame_pos = [0, 0]
        thread.process_frame_lock = threading.Lock()
        thread.process_stop_event = threading.Event()

        def wait(timeout: float) -> bool:
            ready = wake.wait(timeout)
            if ready:
                wake.clear()
            return ready

        thread.process_frame_event = SimpleNamespace(clear=lambda: None, wait=wait)
        monkeypatch.setattr(thread, "start_process", lambda: None)

        def emit(rect: tuple[int, int, int, int]) -> None:
            left, top, right, bottom = rect
            source = np.zeros((height, width, 4), dtype=np.uint8)
            source[top:bottom, left:right] = (25, 50, 100, 255)
            with thread.process_frame_lock:
                raw[:] = source.tobytes()
                thread.process_frame_mapping[16:22] = [*rect, width, height]
            wake.set()
            assert thread.frame_set_event.wait(3)

        state = SimpleNamespace(thread=thread, emit=emit, wake=wake)
        states.append(state)
        thread.start()
        return state

    yield make
    for state in states:
        state.thread.stop_event.set()
        state.wake.set()
        state.thread.stop()
        assert not state.thread.is_alive()


def test_cpu_roi_is_scaled_from_1440_buffer_to_consumed_1080_frame(cpu_capture: Any) -> None:
    state = cpu_capture()
    thread = state.thread
    assert thread.get_content_rect() is None
    state.emit((320, 240, 2240, 1200))
    assert thread.get_content_rect() is None
    frame = thread.get_frame()
    assert frame.shape == (3, 1080, 1920)
    assert thread.get_content_rect() == (240, 180, 1680, 900)
    assert thread.get_frame_mapping() is None
    torch.testing.assert_close(frame[:, 400, 800], torch.tensor([100, 50, 25]) / 255)


def test_cpu_pending_roi_cannot_replace_the_consumed_frame_roi(cpu_capture: Any) -> None:
    state = cpu_capture(width=20, height=16, out_width=10, out_height=8)
    thread = state.thread
    state.emit((2, 2, 18, 14))
    thread.get_frame()
    assert thread.get_content_rect() == (1, 1, 9, 7)
    state.emit((4, 4, 16, 12))
    assert thread.get_content_rect() == (1, 1, 9, 7)
    thread.get_frame()
    assert thread.get_content_rect() == (2, 2, 8, 6)
    thread.stop_event.set()
    state.wake.set()
    thread.stop()
    assert thread.get_content_rect() is None


def test_cuda_non_control_letterbox_roi_matches_consumed_pixels(native_capture: Any) -> None:
    state = native_capture
    thread = state.thread
    assert thread.get_content_rect() is None
    thread.start()
    assert state.constructing.wait(1) and state.instances[0].started.wait(1)
    native = state.instances[0]
    native.emit(torch.full((4, 4, 4), 255, dtype=torch.uint8))
    assert thread.get_content_rect() is None
    frame = thread.get_frame()
    assert frame.shape == (3, 8, 10)
    assert thread.get_content_rect() == (1, 0, 9, 8)
    assert torch.count_nonzero(frame[:, :, 0]) == torch.count_nonzero(frame[:, :, -1]) == 0
    assert thread.get_frame_mapping() is None
    native.emit(torch.full((8, 16, 4), 255, dtype=torch.uint8))
    assert thread.get_content_rect() == (1, 0, 9, 8)
    thread.get_frame()
    assert thread.get_content_rect() == (0, 1, 10, 6)
    thread.stop()
    assert thread.get_content_rect() is None


def test_cuda_uncropped_monitor_returns_full_frame_roi(native_capture: Any) -> None:
    state = native_capture
    thread = state.make_thread(window_name=None)
    thread.start()
    assert state.constructing.wait(1) and state.instances[0].started.wait(1)
    state.instances[0].emit(torch.full((16, 20, 4), 255, dtype=torch.uint8))
    thread.get_frame()
    assert thread.get_content_rect() == (0, 0, 10, 8)


@pytest.mark.parametrize("source_size", [(10, 8), (6, 4)])
def test_cpu_copy_publishes_roi_without_input_mapping(
    monkeypatch: pytest.MonkeyPatch, source_size: tuple[int, int]
) -> None:
    width, height = source_size
    buffer = bytearray(10 * 8 * 4)
    mapping = [0] * 22
    observed = []

    class Native:
        on_frame_arrived: Any

        def __init__(self, **_: Any) -> None:
            pass

        def event(self, handler: Any) -> None:
            setattr(self, handler.__name__, handler)

        def start(self) -> None:
            self.on_frame_arrived(
                SimpleNamespace(frame_buffer=np.full((height, width, 4), 255, dtype=np.uint8)), SimpleNamespace()
            )
            observed.append(mapping[:])

    monkeypatch.setattr(capture, "WindowsCaptureMSS", Native)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        (8, 10, 4),
        0,
        "Synthetic",
        SimpleNamespace(buf=buffer),
        [0, 0],
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        "mss",
        frame_mapping=mapping,
    )
    assert observed[0][0] == 0
    assert observed[0][16:22] == [0, 0, width, height, 10, 8]


def test_cpu_fitted_roi_survives_invalid_input_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = (99, (0, 0, 4, 4), (0, 0, 4, 4))
    current = [geometry]
    monkeypatch.setattr(capture, "_window_geometry", lambda *_: current[0])
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_: SimpleNamespace(SetProcessDpiAwareness=lambda _: 0), raising=False)
    mapping, observed = [0] * 22, []

    class Native:
        on_frame_arrived: Any

        def __init__(self, **_: Any) -> None:
            pass

        def event(self, handler: Any) -> None:
            setattr(self, handler.__name__, handler)

        def start(self) -> None:
            current[0] = (99, (1, 0, 5, 4), (1, 0, 5, 4))
            self.on_frame_arrived(
                SimpleNamespace(frame_buffer=np.full((4, 4, 4), 255, dtype=np.uint8)),
                SimpleNamespace(frame_geometry=geometry),
            )
            observed.append(mapping[:])

    monkeypatch.setattr(capture, "WindowsCaptureMSS", Native)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        (8, 10, 4),
        0,
        "Synthetic",
        SimpleNamespace(buf=bytearray(10 * 8 * 4)),
        [0, 0],
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        "mss",
        window_handle=7,
        fit_window=True,
        frame_mapping=mapping,
    )
    assert observed[0][0] == 0
    assert observed[0][16:22] == [1, 0, 9, 8, 10, 8]
