import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from iw3.desktop import screenshot_process as capture


def capture_one_frame(
    monkeypatch: pytest.MonkeyPatch, window_name: str | None, crops: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    source: np.ndarray = np.arange(8 * 10 * 4, dtype=np.uint8).reshape(8, 10, 4)
    buffer = bytearray(source.nbytes)
    frame_pos = [0, 0]

    class FakeMSS:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def event(self, handler: Any) -> None:
            setattr(self, handler.__name__, handler)

        def start(self) -> None:
            self.on_frame_arrived(  # type: ignore[attr-defined]
                SimpleNamespace(frame_buffer=source), SimpleNamespace(frame_pos=[100, 200])
            )

    monkeypatch.setattr(capture, "WindowsCaptureMSS", FakeMSS)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        source.shape,
        0,
        window_name,
        SimpleNamespace(buf=buffer),
        frame_pos,
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        "mss",
        *crops,
    )
    return source, np.ndarray(source.shape, dtype=np.uint8, buffer=buffer), frame_pos


def test_mss_cropped_window_publishes_cropped_cursor_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    source, frame, origin = capture_one_frame(monkeypatch, "Browser", (2, 3, 1, 1))
    assert origin == [102, 203]
    np.testing.assert_array_equal(frame[:5, :6], source[2:7, 3:9])
    assert not frame[5:].any()
    assert not frame[:, 6:].any()


@pytest.mark.parametrize(
    "window_name,crops", [("Browser", (0, 0, 0, 0)), (None, (2, 3, 1, 1)), ("Browser", (-2, -3, 0, 0))]
)
def test_mss_without_window_crop_keeps_original_frame_and_cursor_origin(
    monkeypatch: pytest.MonkeyPatch, window_name: str | None, crops: tuple[int, int, int, int]
) -> None:
    source, frame, origin = capture_one_frame(monkeypatch, window_name, crops)
    assert origin == [100, 200]
    np.testing.assert_array_equal(frame, source)


@pytest.mark.parametrize("backend", ["mss", "windows_capture"])
@pytest.mark.parametrize("enabled", [False, True])
def test_capture_cursor_switch_reaches_native_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, enabled: bool
) -> None:
    options = []
    source = np.full((8, 10, 4), [25, 50, 100, 255], dtype=np.uint8)
    buffer = bytearray(source.nbytes)

    class NativeCapture:
        def __init__(self, **kwargs: Any) -> None:
            options.append(kwargs)

        def event(self, handler: Any) -> None:
            setattr(self, handler.__name__, handler)

        def start(self) -> None:
            self.on_frame_arrived(SimpleNamespace(frame_buffer=source), SimpleNamespace())  # type: ignore[attr-defined]

    monkeypatch.setattr(capture, "WindowsCaptureMSS", NativeCapture)
    monkeypatch.setitem(sys.modules, "windows_capture", SimpleNamespace(WindowsCapture=NativeCapture))
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        source.shape,
        0,
        None,
        SimpleNamespace(buf=buffer),
        [0, 0],
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        backend,
        draw_cursor_enabled=enabled,
    )
    assert options[0]["cursor_capture"] is (enabled if backend == "mss" else None if enabled else False)
    np.testing.assert_array_equal(np.ndarray(source.shape, dtype=np.uint8, buffer=buffer), source)


@pytest.mark.parametrize("enabled", [False, True])
def test_mss_cursor_switch_controls_the_actual_grab(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    options = []
    frames = []

    class MSS:
        monitors = [None, {"left": 0, "top": 0, "width": 10, "height": 8}]

        def __init__(self, **kwargs: Any) -> None:
            options.append(kwargs)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def grab(self, position: Any) -> np.ndarray:
            return np.full((8, 10, 4), 37, dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=MSS))
    native = capture.WindowsCaptureMSS(cursor_capture=enabled)

    @native.event
    def on_frame_arrived(frame: Any, control: Any) -> None:
        frames.append(frame.frame_buffer)
        control.stop()

    @native.event
    def on_closed() -> None:
        pass

    native.start()
    assert options == [{"with_cursor": enabled}]
    assert frames[0].shape == (8, 10, 4)
    assert (frames[0] == 37).all()


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_thread_sends_its_cursor_switch_to_the_child(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    import torch

    processes = []

    def process(**kwargs: Any) -> Any:
        processes.append(kwargs)
        return SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(capture, "get_screen_size", lambda _: (10, 8))
    monkeypatch.setattr(capture.mp, "Process", process)
    thread = capture.ScreenshotProcess(30, 10, 8, 0, None, torch.device("cpu"), draw_cursor_enabled=enabled)
    try:
        thread.start_process()
        assert processes[0]["target"] is capture.capture_process
        assert processes[0]["args"][-1] is enabled
    finally:
        if thread.process_frame_buffer is not None:
            thread.process_frame_buffer.close()
            thread.process_frame_buffer.unlink()


class CaptureWindows:
    def __init__(self) -> None:
        self.rect = (100, 100, 110, 108)
        self.capture_rect = (101, 100, 109, 107)
        self.pid = 99
        self.title = "Browser"
        self.after_rect: tuple[int, int, int, int] | None = None
        self.rect_reads = 0
        self.capture_pid: int | None = None
        self.mss_grab_moved = False

    def get_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        assert hwnd == 7
        self.rect_reads += 1
        if self.after_rect is not None and self.rect_reads % 2 == 0:
            return self.after_rect
        return self.rect

    def dwm_rect(self, hwnd: int, attribute: int, output: Any, size: int) -> int:
        assert hwnd == 7 and attribute == 9
        for name, value in zip(("left", "top", "right", "bottom"), self.capture_rect):
            setattr(output._obj, name, value)
        return 0


def fitted_frames(
    monkeypatch: pytest.MonkeyPatch,
    windows: CaptureWindows,
    sources: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray]],
    crops: tuple[int, int, int, int] = (0, 0, 0, 0),
    backend: str = "windows_capture",
) -> list[tuple[np.ndarray, Any]]:
    import ctypes

    gui = SimpleNamespace(
        error=OSError,
        IsWindow=lambda hwnd: hwnd == 7,
        GetWindowRect=windows.get_rect,
        FindWindow=lambda cls, title: 7 if title == windows.title else 0,
        GetWindowText=lambda hwnd: windows.title,
        EnumWindows=lambda callback, arg: callback(7, arg),
    )
    monkeypatch.setitem(sys.modules, "win32gui", gui)
    monkeypatch.setitem(
        sys.modules, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (1, windows.pid))
    )

    def dwm_function(*values: Any) -> int:
        return windows.dwm_rect(*values)

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(
            DwmGetWindowAttribute=dwm_function, SetProcessDpiAwareness=lambda value: 0
        ),
    )
    frame_size = (8, 10, 4)
    buffer = bytearray(8 * 10 * 4)
    shared_mapping = [0] * 22
    frame_event = threading.Event()
    result: list[tuple[np.ndarray, Any]] = []

    class FakeWGC:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["window_name"] == "Browser"

        def event(self, handler: Any) -> None:
            setattr(self, handler.__name__, handler)

        def start(self) -> None:
            for rect, capture_rect, source in sources:
                if windows.capture_pid is not None:
                    windows.pid = windows.capture_pid
                windows.rect, windows.capture_rect = rect, capture_rect
                windows.rect_reads = 0
                control = SimpleNamespace(frame_geometry=(windows.pid, rect, rect))
                if windows.mss_grab_moved:
                    windows.rect = (rect[0] + 1, rect[1], rect[2] + 1, rect[3])
                self.on_frame_arrived(SimpleNamespace(frame_buffer=source), control)  # type: ignore[attr-defined]
                result.append(
                    (
                        np.ndarray(frame_size, dtype=np.uint8, buffer=buffer).copy(),
                        capture._decode_frame_mapping(shared_mapping),
                    )
                )
                frame_event.clear()
                windows.title = "Renamed browser after first frame"

    monkeypatch.setitem(sys.modules, "windows_capture", SimpleNamespace(WindowsCapture=FakeWGC))
    if backend == "mss":
        monkeypatch.setattr(capture, "WindowsCaptureMSS", FakeWGC)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        frame_size,
        0,
        "Browser",
        SimpleNamespace(buf=buffer),
        [0, 0],
        threading.Lock(),
        frame_event,
        threading.Event(),
        backend,
        *crops,
        window_handle=7,
        fit_window=True,
        frame_mapping=shared_mapping,
    )
    return result


def test_fitted_wgc_fullscreen_and_restore_keep_entire_source(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    large: np.ndarray = np.full((16, 20, 4), 37, dtype=np.uint8)
    large[0, 0], large[-1, -1] = 81, 193
    small: np.ndarray = np.full((4, 5, 4), 53, dtype=np.uint8)
    frames = fitted_frames(
        monkeypatch,
        windows,
        [((0, 0, 20, 16), (0, 0, 20, 16), large), ((100, 100, 105, 104), (100, 100, 105, 104), small)],
    )
    assert frames[0][0][-1, -1, 0] > 37
    assert frames[0][1] == {
        "hwnd": 7,
        "pid": 99,
        "backend": "windows_capture",
        "window_rect": (0, 0, 20, 16),
        "capture_rect": (0, 0, 20, 16),
        "crops": (0, 0, 0, 0),
        "content_rect": (0, 0, 10, 8),
        "buffer_size": (10, 8),
    }
    assert np.all(frames[1][0] == 53)
    assert frames[1][1]["capture_rect"] == (100, 100, 105, 104)


@pytest.mark.parametrize("transparent", [False, True])
def test_fitted_resize_preserves_raw_bgra_channels_without_opencv(
    monkeypatch: pytest.MonkeyPatch, transparent: bool
) -> None:
    block = np.array([[[0, 40, 80, 0], [100, 60, 20, 255]], [[20, 80, 100, 0], [80, 20, 40, 255]]], dtype=np.uint8)
    if transparent:
        block[:, :, 3] = 0
    source = np.tile(block, (8, 10, 1))
    monkeypatch.setitem(sys.modules, "cv2", None)
    frames = fitted_frames(monkeypatch, CaptureWindows(), [((0, 0, 20, 16), (0, 0, 20, 16), source)])
    expected = np.tile(np.array([50, 50, 60, 0 if transparent else 128], dtype=np.uint8), (8, 10, 1))
    np.testing.assert_array_equal(frames[0][0], expected)


def test_fitted_wgc_letterbox_crops_and_clears_old_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    full: np.ndarray = np.full((8, 10, 4), 21, dtype=np.uint8)
    narrow: np.ndarray = np.full((8, 6, 4), 91, dtype=np.uint8)
    frames = fitted_frames(
        monkeypatch,
        windows,
        [((0, 0, 10, 8), (0, 0, 10, 8), full), ((0, 0, 6, 8), (0, 0, 6, 8), narrow)],
        crops=(1, 1, 1, 1),
    )
    frame, mapping = frames[-1]
    assert mapping["crops"] == (1, 1, 1, 1)
    assert mapping["content_rect"] == (2, 0, 7, 8)
    assert not frame[:, :2].any() and not frame[:, 7:].any()
    assert np.all(frame[:, 2:7] == 91)


@pytest.mark.parametrize("mismatch", ["frame_size", "geometry"])
def test_fitted_wgc_transition_frame_invalidates_old_mapping(monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    windows = CaptureWindows()
    if mismatch == "geometry":
        windows.after_rect = (1, 0, 11, 8)
    source: np.ndarray = np.ones((8, 10, 4), dtype=np.uint8)
    result = fitted_frames(
        monkeypatch,
        windows,
        [((0, 0, 10, 8), (0, 0, 9 if mismatch == "frame_size" else 10, 8), source)],
    )
    assert result[0][1] is None


def test_fitted_wgc_changed_process_identity_is_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    windows.capture_pid = 100
    source: np.ndarray = np.ones((8, 10, 4), dtype=np.uint8)
    result = fitted_frames(monkeypatch, windows, [((0, 0, 10, 8), (0, 0, 10, 8), source)])
    assert result[0][1] is None
    assert not result[0][0].any()


def test_fitted_mss_maps_full_window_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    source: np.ndarray = np.full((8, 10, 4), 87, dtype=np.uint8)
    frame, mapping = fitted_frames(
        monkeypatch, windows, [((100, 100, 110, 108), (101, 100, 109, 107), source)], backend="mss"
    )[0]
    assert np.all(frame == 87)
    assert mapping["backend"] == "mss"
    assert mapping["capture_rect"] == (100, 100, 110, 108)


def test_mss_move_during_grab_invalidates_mapping_even_without_size_change(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    windows.mss_grab_moved = True
    source: np.ndarray = np.ones((8, 10, 4), dtype=np.uint8)
    result = fitted_frames(monkeypatch, windows, [((100, 100, 110, 108), (100, 100, 110, 108), source)], backend="mss")
    assert result[0][1] is None


def test_fitted_wgc_start_rejects_title_rebound_to_a_different_window(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    windows.title = "Another window now owns old title"
    with pytest.raises(RuntimeError, match="title"):
        fitted_frames(monkeypatch, windows, [])


def test_mss_pinned_handle_ignores_title_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    gui = SimpleNamespace(error=OSError, IsWindow=lambda hwnd: hwnd == 7, GetWindowRect=windows.get_rect)
    monkeypatch.setitem(sys.modules, "win32gui", gui)
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (1, 99)))
    positions = []

    class MSS:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def grab(self, position: Any) -> np.ndarray:
            positions.append(position)
            return np.ones((8, 10, 4), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda **kwargs: MSS()))
    native = capture.WindowsCaptureMSS(window_name="Title already changed", window_handle=7)
    frames = []

    @native.event
    def on_frame_arrived(frame: Any, control: Any) -> None:
        frames.append(frame.frame_buffer.shape)
        control.stop()

    @native.event
    def on_closed() -> None:
        pass

    native.start()
    assert positions == [{"left": 100, "top": 100, "width": 10, "height": 8}]
    assert frames == [(8, 10, 4)]


def test_mss_records_geometry_before_grab_moves_same_sized_window(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = CaptureWindows()
    gui = SimpleNamespace(error=OSError, IsWindow=lambda hwnd: hwnd == 7, GetWindowRect=windows.get_rect)
    monkeypatch.setitem(sys.modules, "win32gui", gui)
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (1, 99)))

    class MSS:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def grab(self, position: Any) -> np.ndarray:
            windows.rect = (110, 100, 120, 108)
            return np.ones((8, 10, 4), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda **kwargs: MSS()))
    native = capture.WindowsCaptureMSS(window_handle=7)
    snapshots = []

    @native.event
    def on_frame_arrived(frame: Any, control: Any) -> None:
        snapshots.append((control.frame_geometry, capture._window_geometry(7, "mss")))
        control.stop()

    @native.event
    def on_closed() -> None:
        pass

    native.start()
    assert snapshots == [
        ((99, (100, 100, 110, 108), (100, 100, 110, 108)), (99, (110, 100, 120, 108), (110, 100, 120, 108)))
    ]


def test_frame_mapping_getter_returns_isolated_same_frame_metadata() -> None:
    import torch

    thread = capture.ScreenshotProcess(30, 10, 8, 0, "Browser", torch.device("cpu"), window_handle=7, fit_window=True)
    assert thread.get_frame_mapping() is None
    thread.frame_mapping = {"hwnd": 7, "content_rect": (2, 0, 7, 8)}
    mapping = thread.get_frame_mapping()
    assert mapping == {"hwnd": 7, "content_rect": (2, 0, 7, 8)}
    assert mapping is not None
    mapping["hwnd"] = 8
    current = thread.get_frame_mapping()
    assert current is not None and current["hwnd"] == 7


def test_fitted_cursor_uses_letterbox_mapping_and_skips_outside_source() -> None:
    import torch

    mapping = {
        "capture_rect": (100, 200, 180, 240),
        "crops": (0, 0, 0, 0),
        "content_rect": (0, 6, 40, 26),
        "buffer_size": (40, 32),
    }
    image = torch.zeros(3, 32, 40)
    capture.draw_cursor(image, (140, 220), size=4, mapping=mapping)
    assert image[:, 14:18, 18:22].sum() > 0
    assert not image[:, :6].any() and not image[:, 26:].any()
    untouched = torch.zeros_like(image)
    capture.draw_cursor(untouched, (90, 220), size=4, mapping=mapping)
    assert not untouched.any()


def test_stopped_or_dead_capture_never_exposes_stale_mapping() -> None:
    import torch

    thread = capture.ScreenshotProcess(30, 10, 8, 0, "Browser", torch.device("cpu"), window_handle=7, fit_window=True)
    thread.frame_mapping = {"hwnd": 7}
    thread.process = SimpleNamespace(is_alive=lambda: False)
    assert thread.get_frame_mapping() is None
    thread.process = None
    thread.stop_event.set()
    assert thread.get_frame_mapping() is None


@pytest.mark.parametrize("transition", ["closed", "new_geometry"])
def test_live_capture_rejects_cached_mapping_after_shared_transition(transition: str) -> None:
    import torch

    thread = capture.ScreenshotProcess(30, 10, 8, 0, "Browser", torch.device("cpu"), window_handle=7, fit_window=True)
    thread.process = SimpleNamespace(is_alive=lambda: True)
    thread.process_frame_lock = threading.Lock()
    thread.process_frame_mapping = [1, 7, 99, 1, 100, 100, 110, 108, 100, 100, 110, 108, 0, 0, 0, 0, 0, 0, 10, 8, 10, 8]
    thread.frame_mapping = capture._decode_frame_mapping(thread.process_frame_mapping)
    assert thread.get_frame_mapping() == thread.frame_mapping
    if transition == "closed":
        thread.process_frame_mapping[0] = 0
    else:
        thread.process_frame_mapping[4] += 1
    assert thread.get_frame_mapping() is None


def test_capture_thread_publishes_matching_pixels_and_mapping_then_clears_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    thread = capture.ScreenshotProcess(
        30,
        10,
        8,
        0,
        "Browser",
        torch.device("cpu"),
        window_handle=7,
        fit_window=True,
        draw_cursor_enabled=False,
    )
    source: np.ndarray = np.full((8, 10, 4), [25, 50, 100, 255], dtype=np.uint8)
    thread.screen_width, thread.screen_height = 10, 8
    thread.process_frame_buffer = SimpleNamespace(
        buf=bytearray(source.tobytes()), close=lambda: None, unlink=lambda: None
    )
    thread.process_frame_mapping = [
        1,
        7,
        99,
        1,
        100,
        100,
        110,
        108,
        100,
        100,
        110,
        108,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        8,
        10,
        8,
    ]
    thread.process_frame_lock = threading.Lock()
    thread.process_frame_event = SimpleNamespace(clear=lambda: None, wait=lambda timeout: True)
    thread.process_stop_event = threading.Event()
    thread.process = SimpleNamespace(pid=123, is_alive=lambda: False, join=lambda **kwargs: None)
    published = []
    state: Any = thread

    def on_frame() -> None:
        if not thread.stop_event.is_set():
            published.append((state.frame.clone(), dict(state.frame_mapping)))
            thread.stop_event.set()

    monkeypatch.setattr(thread, "frame_set_event", SimpleNamespace(set=on_frame))
    monkeypatch.setattr(thread, "start_process", lambda: None)
    thread.run()
    frame, mapping = published[0]
    assert frame.shape == (3, 8, 10)
    torch.testing.assert_close(frame[:, 0, 0], torch.tensor([100, 50, 25]) / 255)
    assert 0 <= frame.min() <= frame.max() <= 1
    assert mapping["capture_rect"] == (100, 100, 110, 108)
    assert thread.get_frame_mapping() is None


@pytest.mark.parametrize("failure_stage", ["window_closed", "shared_state", "process_start"])
def test_capture_start_failure_wakes_reader_and_releases_partial_resources(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    import torch

    thread = capture.ScreenshotProcess(30, 10, 8, 0, "Browser", torch.device("cpu"), window_handle=7, fit_window=True)
    buffers = []
    create_buffer = capture.shared_memory.SharedMemory

    def tracked_buffer(*args: Any, **kwargs: Any) -> Any:
        buffer = create_buffer(*args, **kwargs)
        buffers.append(buffer)
        return buffer

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Capture startup failed")

    geometry = None if failure_stage == "window_closed" else (99, (0, 0, 10, 8), (0, 0, 10, 8))
    monkeypatch.setattr(capture, "_window_geometry", lambda *_: geometry)
    monkeypatch.setattr(capture.shared_memory, "SharedMemory", tracked_buffer)
    if failure_stage == "shared_state":
        monkeypatch.setattr(capture.mp, "Array", fail)
    elif failure_stage == "process_start":
        monkeypatch.setattr(capture.mp.Process, "start", fail)
    try:
        with pytest.raises(RuntimeError, match="Capture (window is no longer available|startup failed)"):
            thread.run()
        assert thread.stop_event.is_set()
        assert thread.frame_set_event.is_set()
        assert thread.get_frame_mapping() is None
        assert all(buffer.buf is None for buffer in buffers)
        with pytest.raises(RuntimeError, match="thread is already dead"):
            thread.get_frame()
        thread.stop()
    finally:
        for buffer in buffers:
            buffer.close()
            buffer.unlink()
