import ctypes
import importlib
import sys
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest


class WindowError(Exception):
    pass


class WinFunction:
    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


class WindowsAPI:
    def __init__(self) -> None:
        self.hwnd, self.pid = 7, 99
        self.window_rect = (100, 100, 520, 420)
        self.dwm_rect = (109, 100, 511, 411)
        self.visible, self.iconic, self.exists = True, False, True
        self.foreground, self.point_root = 7, 7
        self.cursor = (0, 0)
        self.points: list[tuple[int, int]] = []
        self.batches: list[list[tuple[int, int, int]]] = []
        self.send_counts: list[int] = []
        self.cursor_result = 1
        self.dwm_result = 0
        self.window_error = False
        self.title_hwnds = [7]
        self.gui = SimpleNamespace(
            error=WindowError,
            FindWindow=lambda cls, title: self.hwnd if title == "Source Browser" else 0,
            IsWindow=lambda hwnd: self.exists,
            IsWindowVisible=lambda hwnd: self.visible,
            IsIconic=lambda hwnd: self.iconic,
            GetWindowRect=self.get_window_rect,
            GetForegroundWindow=lambda: self.foreground,
            WindowFromPoint=lambda point: 8,
            GetAncestor=lambda hwnd, flags: self.point_root,
            EnumWindows=lambda callback, arg: [callback(hwnd, arg) for hwnd in self.title_hwnds],
            GetWindowText=lambda hwnd: "Source Browser",
        )
        self.process = SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (123, self.pid))
        self.user32 = SimpleNamespace(
            SetPhysicalCursorPos=WinFunction(self.set_cursor),
            GetPhysicalCursorPos=WinFunction(self.get_cursor),
            SendInput=WinFunction(self.send_input),
        )
        self.dwm = SimpleNamespace(DwmGetWindowAttribute=WinFunction(self.get_dwm_rect))

    def get_window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        if self.window_error:
            raise WindowError("window closed")
        return self.window_rect

    def set_cursor(self, x: int, y: int) -> int:
        self.points.append((x, y))
        if self.cursor_result:
            self.cursor = (x, y)
        return self.cursor_result

    def get_cursor(self, point: Any) -> int:
        point._obj.x, point._obj.y = self.cursor
        return 1

    def get_dwm_rect(self, hwnd: int, attribute: int, rect: Any, size: int) -> int:
        assert attribute == 9
        for name, value in zip(("left", "top", "right", "bottom"), self.dwm_rect):
            setattr(rect._obj, name, value)
        return self.dwm_result

    def send_input(self, count: int, inputs: Any, size: int) -> int:
        assert size == ctypes.sizeof(inputs[0])
        events = []
        for item in inputs:
            if item.type == 0:
                events.append((0, item.mi.dwFlags, item.mi.mouseData))
            else:
                events.append((1, item.ki.dwFlags, item.ki.wVk))
        assert count == len(events)
        self.batches.append(events)
        return self.send_counts.pop(0) if self.send_counts else count


def patch_windows_api(monkeypatch: pytest.MonkeyPatch) -> WindowsAPI:
    api = WindowsAPI()
    monkeypatch.setitem(sys.modules, "win32gui", api.gui)
    monkeypatch.setitem(sys.modules, "win32process", api.process)
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, **kwargs: api.user32 if name == "user32" else api.dwm, raising=False
    )
    monkeypatch.setattr(sys, "platform", "win32")
    return api


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> WindowsAPI:
    return patch_windows_api(monkeypatch)


def backend(screenshot: str = "wc_mp", crops: tuple[int, int, int, int] = (0, 0, 0, 0)) -> Any:
    module = importlib.import_module("iw3.desktop.quest_windows")
    return module.WindowsInput("Source Browser", screenshot, crops)


def frame_mapping(api: WindowsAPI, **changes: Any) -> dict[str, Any]:
    return {
        "hwnd": 7,
        "pid": 99,
        "backend": "windows_capture",
        "window_rect": api.window_rect,
        "capture_rect": api.dwm_rect,
        "crops": (0, 0, 0, 0),
        "content_rect": (0, 0, 420, 320),
        "buffer_size": (420, 320),
        **changes,
    }


def mapped_backend(mapping: Sequence[dict[str, Any] | None], screenshot: str = "wc_mp") -> Any:
    module = importlib.import_module("iw3.desktop.quest_windows")
    return module.WindowsInput("Source Browser", screenshot, (0, 0, 0, 0), frame_mapping=lambda: mapping[0])


def test_mapping_waits_for_a_captured_frame_and_exposes_pinned_hwnd(api: WindowsAPI) -> None:
    control = mapped_backend([None])
    assert control.hwnd == 7
    assert not control.move(0.5, 0.5)
    assert not api.points


@pytest.mark.parametrize("screenshot", ["wc_mp", "mss", "wc_cuda"])
def test_fullscreen_mapping_follows_same_hwnd_then_windowed_again(api: WindowsAPI, screenshot: str) -> None:
    mapping: list[dict[str, Any] | None] = [None]
    control = mapped_backend(mapping, screenshot)
    api.window_rect = api.dwm_rect = (0, 0, 2560, 1440)
    mapping[0] = frame_mapping(
        api, backend="windows_capture" if screenshot == "wc_mp" else screenshot, content_rect=(0, 42, 420, 278)
    )
    assert control.move(0.5, 0.5)
    assert api.points[-1] == (1280, 720)
    assert control.move(0.25, 0.5)
    assert api.points[-1] == (640, 720)
    api.window_rect = api.dwm_rect = (100, 100, 520, 420)
    assert not control.move(0.5, 0.5)
    mapping[0] = frame_mapping(api, backend="windows_capture" if screenshot == "wc_mp" else screenshot)
    assert control.move(0.5, 0.5)
    assert api.points[-1] == (310, 260)


@pytest.mark.parametrize("u,v", [(0.5, 0), (0.5, 0.1), (0.5, 0.9), (0.5, 1)])
def test_letterbox_bars_do_not_move_the_pointer(api: WindowsAPI, u: float, v: float) -> None:
    mapping: list[dict[str, Any] | None] = [frame_mapping(api, content_rect=(0, 42, 420, 278))]
    control = mapped_backend(mapping)
    assert not control.move(u, v)
    assert not api.points


def test_mapping_converts_crop_and_letterbox_instead_of_initial_buffer_pixels(api: WindowsAPI) -> None:
    module = importlib.import_module("iw3.desktop.quest_windows")
    mapping: list[dict[str, Any] | None] = [None]
    control = module.WindowsInput("Source Browser", "wc_mp", (10, 20, 30, 40), frame_mapping=lambda: mapping[0])
    api.window_rect = api.dwm_rect = (100, 200, 2100, 1250)
    mapping[0] = frame_mapping(api, crops=(10, 20, 30, 40), content_rect=(0, 52, 420, 267))
    assert control.move(0.5, 0.5)
    assert api.points[-1] == (1095, 712)


def test_mapped_full_buffer_last_pixel_reaches_source_edge(api: WindowsAPI) -> None:
    mapping: list[dict[str, Any] | None] = [None]
    control = mapped_backend(mapping)
    api.window_rect = api.dwm_rect = (0, 0, 840, 640)
    mapping[0] = frame_mapping(api)
    assert control.move(1, 1)
    assert api.points[-1] == (839, 639)


@pytest.mark.parametrize("field,value", [("hwnd", 8), ("pid", 100), ("backend", "mss")])
def test_frame_mapping_cannot_change_window_identity_or_backend(api: WindowsAPI, field: str, value: Any) -> None:
    control = mapped_backend([frame_mapping(api, **{field: value})])
    assert not control.move(0.5, 0.5)
    assert not api.points


@pytest.mark.parametrize(
    "changes",
    [
        {"content_rect": (0, 0, 421, 320)},
        {"content_rect": (0, 0, 0, 320)},
        {"buffer_size": (420, 0)},
        {"capture_rect": (109, 100, float("nan"), 411)},
        {"crops": (0, 1, 0, 0)},
        {"window_rect": None},
    ],
)
def test_bad_mapping_is_rejected_without_input(api: WindowsAPI, changes: dict[str, Any]) -> None:
    control = mapped_backend([frame_mapping(api, **changes)])
    assert not control.move(0.5, 0.5)
    assert not api.points


def test_old_or_transition_mapping_blocks_click_until_fresh_move(api: WindowsAPI) -> None:
    mapping: list[dict[str, Any] | None] = [frame_mapping(api)]
    control = mapped_backend(mapping)
    assert control.move(0.5, 0.5)
    mapping[0] = None
    assert not control.mouse_down()
    assert not control.tap_key("down")
    api.window_rect = api.dwm_rect = (0, 0, 2560, 1440)
    mapping[0] = frame_mapping(api, content_rect=(0, 42, 420, 278))
    assert not control.mouse_down()
    assert control.move(0.5, 0.5)
    api.point_root = 9
    assert not control.mouse_down()
    assert not api.batches


@pytest.mark.parametrize(
    "field,value",
    [("exists", False), ("visible", False), ("iconic", True), ("pid", 100), ("foreground", 9), ("point_root", 9)],
)
def test_dynamic_mapping_preserves_target_guards(api: WindowsAPI, field: str, value: Any) -> None:
    mapping: list[dict[str, Any] | None] = [frame_mapping(api)]
    control = mapped_backend(mapping)
    setattr(api, field, value)
    assert not control.move(0.5, 0.5)
    assert not control.mouse_down()
    assert not control.tap_key("down")
    assert not api.points and not api.batches


def test_mapping_change_after_move_still_allows_owned_button_cleanup(api: WindowsAPI) -> None:
    mapping: list[dict[str, Any] | None] = [frame_mapping(api)]
    control = mapped_backend(mapping)
    assert control.move(0.5, 0.5) and control.mouse_down()
    api.window_rect = api.dwm_rect = (0, 0, 2560, 1440)
    assert not control.move(0.5, 0.5)
    control.mouse_up()
    assert api.batches == [[(0, 2, 0)], [(0, 4, 0)]]


def test_cuda_mapping_uses_dwm_origin_and_rejects_other_backend_metadata(api: WindowsAPI) -> None:
    mapping = [frame_mapping(api, backend="wc_cuda", buffer_size=(402, 311), content_rect=(0, 0, 402, 311))]
    control = mapped_backend(mapping, "wc_cuda")
    assert control.move(0, 0) and api.points[-1] == (109, 100)
    assert control.move(1, 1) and api.points[-1] == (510, 410)
    mapping[0]["backend"] = "windows_capture"
    assert not control.move(0.5, 0.5) and not control.mouse_down()


@pytest.fixture
def monitor_control(api: WindowsAPI, monkeypatch: pytest.MonkeyPatch) -> Any:
    from iw3.desktop import screenshot_process

    target = {
        "monitor_index": 1,
        "monitor_handle": 31,
        "monitor_device": "DISPLAY_TEST",
        "monitor_rect": (-1920, -100, 0, 980),
    }
    current = [dict(target)]
    monkeypatch.setattr(
        screenshot_process, "_monitor_geometry", lambda index: current[0] if index == 1 else None, raising=False
    )
    mapping = [
        {
            "target": "monitor",
            **target,
            "backend": "wc_cuda",
            "capture_rect": target["monitor_rect"],
            "crops": (0, 0, 0, 0),
            "content_rect": (0, 0, 960, 540),
            "buffer_size": (960, 540),
        }
    ]
    module = importlib.import_module("iw3.desktop.quest_windows")
    control = module.WindowsInput(None, "wc_cuda", (0, 0, 0, 0), frame_mapping=lambda: mapping[0], monitor_index=1)
    return SimpleNamespace(control=control, mapping=mapping, target=target, current=current)


def test_monitor_control_maps_negative_origin_and_can_click_new_windows(api: WindowsAPI, monitor_control: Any) -> None:
    state = monitor_control
    assert state.control.hwnd is None and state.control.monitor_target == state.target
    copied = state.control.monitor_target
    copied["monitor_handle"] = 99
    assert state.control.monitor_target == state.target
    api.foreground, api.point_root = 9, 12
    assert state.control.move(0, 0) and api.points[-1] == (-1920, -100)
    assert state.control.mouse_down()
    state.control.mouse_up()
    assert state.control.move(1, 1) and api.points[-1] == (-1, 979)
    assert all(-1920 <= x < 0 and -100 <= y < 980 for x, y in api.points)


def test_monitor_keys_require_pointed_window_to_be_foreground(api: WindowsAPI, monitor_control: Any) -> None:
    control = monitor_control.control
    assert control.move(0.5, 0.5)
    api.foreground, api.point_root = 9, 12
    assert not control.tap_key("down") and not api.batches
    api.foreground = 12
    assert control.tap_key("down")
    assert api.batches == [[(1, 0, 0x28), (1, 2, 0x28)]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("monitor_handle", 32),
        ("monitor_device", "OTHER_DISPLAY"),
        ("monitor_index", 0),
        ("monitor_rect", (-1920, 0, 0, 1080)),
    ],
)
def test_monitor_topology_change_blocks_input_and_keeps_owned_release(
    api: WindowsAPI, monitor_control: Any, field: str, value: Any
) -> None:
    state = monitor_control
    assert state.control.move(0.5, 0.5) and state.control.mouse_down()
    state.current[0] = {**state.current[0], field: value}
    assert not state.control.move(0.5, 0.5) and not state.control.tap_key("down")
    state.control.mouse_up()
    assert api.batches == [[(0, 2, 0)], [(0, 4, 0)]]


@pytest.mark.parametrize(
    "changes",
    [
        {"target": "window"},
        {"monitor_handle": 32},
        {"monitor_device": "OTHER_DISPLAY"},
        {"backend": "windows_capture"},
        {"capture_rect": (0, 0, 1920, 1080)},
        {"crops": (0, 1, 0, 0)},
        {"content_rect": (0, 0, 961, 540)},
    ],
)
def test_monitor_mapping_cannot_change_capture_identity_or_bounds(
    api: WindowsAPI, monitor_control: Any, changes: dict[str, Any]
) -> None:
    state = monitor_control
    state.mapping[0] = {**state.mapping[0], **changes}
    assert not state.control.move(0.5, 0.5)
    assert not api.points and not api.batches


def test_monitor_mapping_black_bars_and_stale_frames_are_not_clickable(api: WindowsAPI, monitor_control: Any) -> None:
    state = monitor_control
    state.mapping[0]["content_rect"] = (0, 100, 960, 440)
    assert not state.control.move(0.5, 0)
    assert state.control.move(0.5, 0.5)
    state.mapping[0] = None
    assert not state.control.mouse_down() and not state.control.tap_key("down")
    assert not api.batches


@pytest.mark.parametrize("crops,mapping", [((0, 0, 0, 0), None), ((1, 0, 0, 0), lambda: {})])
def test_monitor_control_requires_uncropped_frame_mapping(api: WindowsAPI, crops: Any, mapping: Any) -> None:
    module = importlib.import_module("iw3.desktop.quest_windows")
    with pytest.raises(ValueError):
        module.WindowsInput(None, "wc_cuda", crops, frame_mapping=mapping)


def test_capture_metadata_drives_real_backend_inverse_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from iw3.tests.test_quest_capture import CaptureWindows, fitted_frames

    api = patch_windows_api(monkeypatch)
    with pytest.MonkeyPatch.context() as capture_patch:
        frames = fitted_frames(
            capture_patch,
            CaptureWindows(),
            [((0, 0, 20, 10), (0, 0, 20, 10), np.ones((10, 20, 4), dtype=np.uint8))],
        )
    mapping: list[dict[str, Any] | None] = [frames[0][1]]
    control = mapped_backend(mapping)
    api.window_rect = api.dwm_rect = (0, 0, 20, 10)
    assert not control.move(0.5, 0)
    assert control.move(0.5, 3.5 / 8)
    assert api.points[-1] == (10, 5)
    assert control.move(1, 3.5 / 8)
    assert api.points[-1] == (19, 5)


@pytest.mark.parametrize("screenshot,expected", [("wc_mp", (319, 260)), ("mss", (310, 260))])
def test_uv_uses_verified_capture_origin_and_full_buffer(
    api: WindowsAPI, screenshot: str, expected: tuple[int, int]
) -> None:
    controller = backend(screenshot)
    assert controller.move(0.5, 0.5)
    assert api.points == [expected]


@pytest.mark.parametrize("screenshot,expected", [("wc_mp", (339, 270)), ("mss", (330, 270))])
def test_crops_add_source_offset_without_shrinking_shared_buffer(
    api: WindowsAPI, screenshot: str, expected: tuple[int, int]
) -> None:
    controller = backend(screenshot, (10, 20, 30, 40))
    assert controller.move(0.5, 0.5)
    assert api.points == [expected]


@pytest.mark.parametrize("u,v", [(0.96, 0.5), (0.5, 0.98), (1, 1)])
def test_wgc_trailing_padding_is_not_clickable(api: WindowsAPI, u: float, v: float) -> None:
    assert not backend().move(u, v)
    assert not api.points


def test_mss_last_pixel_is_clickable_without_crop(api: WindowsAPI) -> None:
    assert backend("mss").move(1, 1)
    assert api.points == [(519, 419)]


@pytest.mark.parametrize("u,v", [(-0.01, 0.5), (0.5, 1.01), (float("nan"), 0.5), (0.5, float("inf")), ("x", 0.5)])
def test_invalid_uv_never_moves(api: WindowsAPI, u: Any, v: Any) -> None:
    assert not backend().move(u, v)
    assert not api.points


@pytest.mark.parametrize("u,v", [(True, 0.5), (0.5, False), (10**1000, 0.5)], ids=["bool-u", "bool-v", "huge-int"])
def test_non_coordinate_numbers_never_move(api: WindowsAPI, u: Any, v: Any) -> None:
    assert not backend("mss").move(u, v)
    assert not api.points


def test_failed_cursor_move_cannot_become_a_button_target(api: WindowsAPI) -> None:
    controller = backend()
    api.cursor_result = 0
    assert not controller.move(0.5, 0.5)
    assert not controller.mouse_down()
    assert not api.batches


def test_failed_dwm_snapshot_aborts_backend(api: WindowsAPI) -> None:
    api.dwm_result = -1
    with pytest.raises(OSError):
        backend()


def test_duplicate_titles_are_rejected_before_capture_can_bind_another_window(api: WindowsAPI) -> None:
    api.title_hwnds = [7, 11]
    with pytest.raises(ValueError):
        backend()


@pytest.mark.parametrize(
    "field,value",
    [
        ("exists", False),
        ("visible", False),
        ("iconic", True),
        ("pid", 100),
        ("foreground", 9),
        ("point_root", 9),
        ("window_rect", (101, 100, 521, 420)),
        ("dwm_rect", (110, 100, 512, 411)),
        ("window_error", True),
    ],
)
def test_changed_or_unsafe_target_never_receives_input(api: WindowsAPI, field: str, value: Any) -> None:
    controller = backend()
    setattr(api, field, value)
    assert not controller.move(0.5, 0.5)
    assert not controller.mouse_down()
    assert not controller.tap_key("down")
    assert not api.points and not api.batches


def test_button_rechecks_focus_and_cannot_reuse_stale_target(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.foreground = 9
    assert not controller.mouse_down()
    api.foreground = 7
    assert not controller.mouse_down()
    assert not api.batches


def test_local_cursor_movement_blocks_cached_button_and_key(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.cursor = (900, 900)
    assert not controller.mouse_down()
    assert not controller.tap_key("down")
    assert not api.batches


def test_left_button_edges_and_owned_release_ignore_lost_focus(api: WindowsAPI) -> None:
    controller = backend()
    controller.mouse_up()
    assert controller.move(0.5, 0.5)
    assert controller.mouse_down()
    assert controller.mouse_down()
    api.foreground = 9
    controller.mouse_up()
    controller.mouse_up()
    assert api.batches == [[(0, 2, 0)], [(0, 4, 0)]]


@pytest.mark.parametrize("direction,key", [("up", 0x26), ("down", 0x28)])
def test_arrow_tap_batches_press_and_release(api: WindowsAPI, direction: str, key: int) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    assert controller.tap_key(direction)
    assert len(api.batches) == 1
    assert [(kind, flags & 2, vk) for kind, flags, vk in api.batches[0]] == [(1, 0, key), (1, 2, key)]


def test_keyboard_rechecks_occlusion(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.point_root = 9
    assert not controller.tap_key("down")
    assert not api.batches


@pytest.mark.parametrize("count", [0, 1])
def test_failed_key_tap_releases_before_raising(api: WindowsAPI, count: int) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.send_counts = [count]
    with pytest.raises(OSError):
        controller.tap_key("down")
    if count == 1:
        assert len(api.batches) == 2
        assert api.batches[1][0][1] & 2 and api.batches[1][0][2] == 0x28


def test_partial_key_cleanup_remains_retryable_after_focus_loss(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.send_counts = [1, 0, 0, 1]
    with pytest.raises(OSError):
        controller.tap_key("down")
    api.foreground = 9
    with pytest.raises(OSError):
        controller.release_keys()
    controller.release_keys()
    controller.release_keys()
    assert api.batches == [
        [(1, 0, 0x28), (1, 2, 0x28)],
        [(1, 2, 0x28)],
        [(1, 2, 0x28)],
        [(1, 2, 0x28)],
    ]


@pytest.mark.parametrize("count", [0, 2])
def test_release_keys_ignores_keys_not_owned_by_backend(api: WindowsAPI, count: int) -> None:
    controller = backend()
    controller.release_keys()
    assert controller.move(0.5, 0.5)
    api.send_counts = [count]
    if count == 0:
        with pytest.raises(OSError):
            controller.tap_key("up")
    else:
        assert controller.tap_key("up")
    controller.release_keys()
    assert api.batches == [[(1, 0, 0x26), (1, 2, 0x26)]]


def test_new_tap_cannot_overwrite_an_unreleased_key(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.send_counts = [1, 0, 0]
    with pytest.raises(OSError):
        controller.tap_key("down")
    with pytest.raises(OSError):
        controller.tap_key("up")
    controller.release_keys()
    assert api.batches == [
        [(1, 0, 0x28), (1, 2, 0x28)],
        [(1, 2, 0x28)],
        [(1, 2, 0x28)],
        [(1, 2, 0x28)],
    ]


def test_real_controller_reset_retries_partial_key_without_a_mouse_press(api: WindowsAPI) -> None:
    from iw3.desktop.quest_input import QuestController
    from iw3.tests.test_quest_input import entrada, quad

    control = QuestController(backend())
    control.update(entrada(), quad())
    api.send_counts = [1, 0]
    with pytest.raises(OSError):
        control.update(entrada(analogico=[0, -1]), quad())
    assert not control._pressed
    api.foreground = 9
    control.reset()
    control.reset()
    assert api.batches == [
        [(1, 0, 0x28), (1, 2, 0x28)],
        [(1, 2, 0x28)],
        [(1, 2, 0x28)],
    ]


def test_failed_mouse_release_keeps_cleanup_retryable(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5) and controller.mouse_down()
    api.send_counts = [0]
    with pytest.raises(OSError):
        controller.mouse_up()
    controller.mouse_up()
    assert api.batches == [[(0, 2, 0)], [(0, 4, 0)], [(0, 4, 0)]]


def test_failed_mouse_down_does_not_claim_button(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    api.send_counts = [0]
    with pytest.raises(OSError):
        controller.mouse_down()
    controller.mouse_up()
    assert api.batches == [[(0, 2, 0)]]


def test_invalid_move_discards_previous_point(api: WindowsAPI) -> None:
    controller = backend()
    assert controller.move(0.5, 0.5)
    assert not controller.move(float("nan"), 0.5)
    assert not controller.mouse_down() and not controller.tap_key("down")
    assert not api.batches


@pytest.mark.parametrize(
    "screenshot,crops", [("wc_cuda", (0, 0, 0, 0)), ("wc_mp", (0, -1, 0, 0)), ("wc_mp", (0, 402, 0, 0))]
)
def test_invalid_capture_configuration_rejected(
    api: WindowsAPI, screenshot: str, crops: tuple[int, int, int, int]
) -> None:
    with pytest.raises(ValueError):
        backend(screenshot, crops)


def test_unknown_key_is_rejected(api: WindowsAPI) -> None:
    with pytest.raises(ValueError):
        backend().tap_key("delete")
    assert not api.batches


def test_non_windows_module_import_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    module = importlib.import_module("iw3.desktop.quest_windows")
    with pytest.raises(RuntimeError, match="Windows"):
        module.WindowsInput("Source Browser", "wc_mp", (0, 0, 0, 0))


def test_input_abi_matches_windows_pointer_size(api: WindowsAPI) -> None:
    module = importlib.import_module("iw3.desktop.quest_windows")
    controller = backend()
    assert ctypes.sizeof(module.INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
    assert api.user32.SendInput.argtypes[1]._type_ is module.INPUT
    assert api.user32.SendInput.restype is ctypes.c_uint
    assert controller.move(0.5, 0.5)
