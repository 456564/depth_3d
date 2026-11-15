import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from iw3.desktop import screenshot_process as capture_source
from iw3.desktop.screenshot_thread_cuda import ScreenshotThreadWCCUDA


@pytest.fixture
def native_capture(monkeypatch: pytest.MonkeyPatch) -> Any:
    instances = []
    constructing, allow_construct = threading.Event(), threading.Event()
    entering_start, allow_start = threading.Event(), threading.Event()
    allow_construct.set()
    allow_start.set()

    class Capture:
        on_frame_arrived: Any

        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs
            self.running = False
            self.start_calls = self.stop_calls = 0
            self.started, self.stop_called, self.wake = threading.Event(), threading.Event(), threading.Event()
            instances.append(self)
            constructing.set()
            assert allow_construct.wait(2)

        def event(self, handler: Any) -> Any:
            setattr(self, handler.__name__, handler)
            return handler

        def emit(self, tensor: torch.Tensor) -> None:
            self.on_frame_arrived(SimpleNamespace(frame_buffer=tensor), self)

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
    threads = []

    def make_thread(**options: Any) -> ScreenshotThreadWCCUDA:
        arguments = dict(
            fps=30, frame_width=10, frame_height=8, monitor_index=0, window_name="Browser", device=torch.device("cpu")
        )
        arguments.update(options)
        thread = ScreenshotThreadWCCUDA(**arguments)
        threads.append(thread)
        return thread

    thread = make_thread()
    yield SimpleNamespace(
        thread=thread,
        instances=instances,
        constructing=constructing,
        allow_construct=allow_construct,
        entering_start=entering_start,
        allow_start=allow_start,
        make_thread=make_thread,
    )
    allow_construct.set()
    allow_start.set()
    for instance in instances:
        if instance.start_calls:
            assert instance.started.wait(1)
        instance.stop()
    for thread in threads:
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


@pytest.fixture
def mapped_capture(native_capture: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = native_capture
    geometry = SimpleNamespace(window=(99, (99, 100, 113, 109), (100, 100, 112, 108)))
    monkeypatch.setattr(capture_source, "_window_geometry", lambda hwnd, backend: geometry.window)
    thread = state.make_thread(frame_width=6, frame_height=4, window_handle=7, fit_window=True)
    thread.start()
    assert state.constructing.wait(1)
    native = state.instances[0]
    assert native.started.wait(1)
    return SimpleNamespace(
        thread=thread, native=native, geometry=geometry, make_thread=state.make_thread, instances=state.instances
    )


def four_corners(width: int = 12, height: int = 8, device: str = "cpu") -> torch.Tensor:
    # BGRA：四象限依次红、绿、蓝、黄，直接检查缩小后四边仍保留。
    tensor = torch.zeros((height, width, 4), dtype=torch.uint8, device=device)
    tensor[: height // 2, : width // 2] = torch.tensor([0, 0, 255, 255], device=device)
    tensor[: height // 2, width // 2 :] = torch.tensor([0, 255, 0, 255], device=device)
    tensor[height // 2 :, : width // 2] = torch.tensor([255, 0, 0, 255], device=device)
    tensor[height // 2 :, width // 2 :] = torch.tensor([0, 255, 255, 255], device=device)
    return tensor


def test_cuda_pins_native_window_by_handle_without_title_priority(mapped_capture: Any) -> None:
    assert mapped_capture.native.options["window_name"] is None
    assert mapped_capture.native.options["window_hwnd"] == 7
    assert mapped_capture.native.options["monitor_index"] is None


def test_cuda_complete_window_resize_retains_four_corners_and_consumed_mapping(mapped_capture: Any) -> None:
    thread, native = mapped_capture.thread, mapped_capture.native
    native.emit(four_corners())
    assert thread.get_frame_mapping() is None
    frame = thread.get_frame()
    assert frame.shape == (3, 4, 6) and frame.dtype == torch.float32
    assert torch.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 1
    torch.testing.assert_close(frame[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(frame[:, 0, -1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(frame[:, -1, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(frame[:, -1, -1], torch.tensor([1.0, 1.0, 0.0]))
    assert thread.get_frame_mapping() == {
        "hwnd": 7,
        "pid": 99,
        "backend": "wc_cuda",
        "window_rect": (99, 100, 113, 109),
        "capture_rect": (100, 100, 112, 108),
        "crops": (0, 0, 0, 0),
        "content_rect": (0, 0, 6, 4),
        "buffer_size": (6, 4),
    }


def test_cuda_crop_letterbox_preserves_content_and_reports_black_bars(mapped_capture: Any) -> None:
    state = mapped_capture
    state.thread.crop_left = 4
    state.thread.crop_right = 4
    state.native.emit(torch.full((8, 12, 4), 255, dtype=torch.uint8))
    frame = state.thread.get_frame()
    assert not frame[:, :, :2].any() and not frame[:, :, 4:].any()
    assert torch.all(frame[:, :, 2:4] == 1)
    mapping = state.thread.get_frame_mapping()
    assert mapping["content_rect"] == (2, 0, 4, 4)
    assert mapping["crops"] == (0, 4, 4, 0)


@pytest.mark.parametrize("transition", ["move", "pid", "raw_size"])
def test_cuda_transition_never_reuses_previous_frame_mapping(mapped_capture: Any, transition: str) -> None:
    state = mapped_capture
    state.native.emit(four_corners())
    state.thread.get_frame()
    assert state.thread.get_frame_mapping() is not None
    if transition == "move":
        state.geometry.window = (99, (199, 200, 213, 209), (200, 200, 212, 208))
    elif transition == "pid":
        state.geometry.window = (100, (99, 100, 113, 109), (100, 100, 112, 108))
    source = four_corners(width=14) if transition == "raw_size" else four_corners()
    state.native.emit(source)
    assert state.thread.get_frame_mapping() is None
    state.thread.get_frame()
    assert (state.thread.get_frame_mapping() is not None) == (transition == "move")


def test_cuda_fullscreen_mapping_only_appears_after_matching_frame_is_consumed(mapped_capture: Any) -> None:
    state = mapped_capture
    state.native.emit(four_corners())
    state.thread.get_frame()
    state.geometry.window = (99, (0, 0, 24, 16), (0, 0, 24, 16))
    assert state.thread.get_frame_mapping() is None
    state.native.emit(four_corners(24, 16))
    assert state.thread.get_frame_mapping() is None
    state.thread.get_frame()
    assert state.thread.get_frame_mapping()["capture_rect"] == (0, 0, 24, 16)


def test_cuda_geometry_change_during_resize_rejects_mapping(
    mapped_capture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iw3.desktop import screenshot_thread_cuda as capture_cuda

    state = mapped_capture
    original_resize = capture_cuda.resize_frame

    def moving_resize(frame: torch.Tensor, size: Any) -> torch.Tensor:
        result = original_resize(frame, size)
        state.geometry.window = (99, (199, 200, 213, 209), (200, 200, 212, 208))
        return result

    monkeypatch.setattr(capture_cuda, "resize_frame", moving_resize)
    state.native.emit(four_corners())
    state.thread.get_frame()
    assert state.thread.get_frame_mapping() is None


@pytest.mark.parametrize("transition", ["handle", "device", "rect", "index", "raw_size"])
def test_cuda_monitor_mapping_requires_same_monitor_identity_and_frame_size(
    native_capture: Any, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    target = {"monitor_index": 0, "monitor_handle": 71, "monitor_device": "DISPLAY2", "monitor_rect": (-12, 0, 0, 8)}
    current = dict(target)
    monkeypatch.setattr(capture_source, "_monitor_geometry", lambda index: dict(current), raising=False)
    thread = native_capture.make_thread(
        frame_width=6, frame_height=4, window_name=None, fit_window=True, monitor_target=target
    )
    thread.start()
    assert native_capture.constructing.wait(1)
    native = native_capture.instances[0]
    assert native.started.wait(1)
    assert native.options["monitor_index"] == 1
    native.emit(four_corners())
    thread.get_frame()
    assert thread.get_frame_mapping() == {
        "target": "monitor",
        **target,
        "backend": "wc_cuda",
        "capture_rect": (-12, 0, 0, 8),
        "crops": (0, 0, 0, 0),
        "content_rect": (0, 0, 6, 4),
        "buffer_size": (6, 4),
    }
    if transition == "handle":
        current["monitor_handle"] = 72
    elif transition == "device":
        current["monitor_device"] = "DISPLAY1"
    elif transition == "rect":
        current["monitor_rect"] = (-24, 0, 0, 16)
    elif transition == "index":
        current["monitor_index"] = 1
    else:
        native.emit(four_corners(14, 8))
    assert thread.get_frame_mapping() is None
    native.emit(four_corners())
    thread.get_frame()
    assert (thread.get_frame_mapping() is not None) == (transition == "raw_size")


def test_cuda_stop_clears_consumed_and_pending_mapping(mapped_capture: Any) -> None:
    state = mapped_capture
    state.native.emit(four_corners())
    state.thread.get_frame()
    state.native.emit(four_corners())
    state.thread.stop()
    assert state.thread.get_frame_mapping() is None and state.thread.frame is None
    with pytest.raises(RuntimeError, match="dead"):
        state.thread.get_frame()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for stream handoff")
def test_cuda_callback_hands_ready_frame_to_different_consumer_stream(mapped_capture: Any) -> None:
    state = mapped_capture
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        # 延迟生产，消费流必须通过 Event 等待；不捕获真实窗口。
        torch.cuda._sleep(20_000_000)
        state.native.emit(four_corners(device="cuda:0"))
    frame = state.thread.get_frame()
    assert frame.is_cuda
    torch.testing.assert_close(frame[:, -1, -1].cpu(), torch.tensor([1.0, 1.0, 0.0]))
