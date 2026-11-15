import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from iw3.desktop import screenshot_process as capture


def target():
    return {"monitor_index": 1, "monitor_handle": 22, "monitor_device": "DISPLAY2", "monitor_rect": (-20, 0, 0, 16)}


def test_monitor_geometry_preserves_negative_origin_and_device_identity(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "win32api",
        SimpleNamespace(
            EnumDisplayMonitors=lambda: [(11, None, (0, 0, 20, 16)), (22, None, (-20, 0, 0, 16))],
            GetMonitorInfo=lambda handle: {"Device": "DISPLAY2", "Monitor": (-20, 0, 0, 16)},
        ),
    )
    assert capture._monitor_geometry(1) == target()
    assert capture._monitor_geometry(-1) is None
    assert capture._monitor_geometry(2) is None


@pytest.mark.parametrize("backend", ["mss", "windows_capture"])
@pytest.mark.parametrize("changed", [None, "monitor_handle", "monitor_device", "monitor_rect", "frame_size"])
def test_monitor_capture_publishes_fitted_pixels_and_rejects_changed_identity(monkeypatch, backend, changed):
    current = target()
    monkeypatch.setattr(capture, "_monitor_geometry", lambda index: dict(current), raising=False)
    buffer = bytearray(8 * 10 * 4)
    mapping = [0] * 22
    observed = []
    source = np.full((16, 20, 4), 30, dtype=np.uint8)
    source[:4, :4] = 90
    source[-4:, -4:] = 210

    class Native:
        def __init__(self, **kwargs):
            assert kwargs["monitor_index"] == (1 if backend == "mss" else 2)
            assert kwargs.get("window_name") is None

        def event(self, callback):
            setattr(self, callback.__name__, callback)

        def start(self):
            if changed == "monitor_handle":
                current[changed] = 33
            elif changed == "monitor_device":
                current[changed] = "DISPLAY3"
            elif changed == "monitor_rect":
                current[changed] = (0, 0, 20, 16)
            frame = source[:-1] if changed == "frame_size" else source
            self.on_frame_arrived(SimpleNamespace(frame_buffer=frame), SimpleNamespace())
            observed.append(capture._decode_frame_mapping(mapping, target()))

    monkeypatch.setitem(sys.modules, "windows_capture", SimpleNamespace(WindowsCapture=Native))
    monkeypatch.setattr(capture, "WindowsCaptureMSS", Native)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.capture_process(
        (8, 10, 4),
        1,
        None,
        SimpleNamespace(buf=buffer),
        [0, 0],
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        backend,
        fit_window=True,
        frame_mapping=mapping,
        monitor_target=target(),
    )
    frame = np.ndarray((8, 10, 4), dtype=np.uint8, buffer=buffer)
    result = observed[0]
    assert capture._decode_frame_mapping(mapping, target()) is None
    if changed:
        assert result is None and not frame.any()
    else:
        assert frame[0, 0, 0] == 90 and frame[-1, -1, 0] == 210
        assert result == {
            "target": "monitor",
            **target(),
            "backend": backend,
            "capture_rect": (-20, 0, 0, 16),
            "crops": (0, 0, 0, 0),
            "content_rect": (0, 0, 10, 8),
            "buffer_size": (10, 8),
        }


def test_monitor_mapping_read_invalidates_after_display_topology_changes(monkeypatch):
    current = target()
    monkeypatch.setattr(capture, "_monitor_geometry", lambda index: dict(current), raising=False)
    thread = capture.ScreenshotProcess(
        30, 10, 8, 1, None, torch.device("cpu"), fit_window=True, monitor_target=target()
    )
    thread.process_frame_mapping = [1, 22, 1, 4, -20, 0, 0, 16, -20, 0, 0, 16, 0, 0, 0, 0, 0, 0, 10, 8, 10, 8]
    thread.process_frame_lock = threading.Lock()
    thread.process = SimpleNamespace(is_alive=lambda: True)
    thread.frame_mapping = capture._decode_frame_mapping(thread.process_frame_mapping, target())
    assert thread.get_frame_mapping() is not None
    current["monitor_handle"] = 33
    assert thread.get_frame_mapping() is None
