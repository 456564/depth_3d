import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from iw3.desktop import utils as desktop
from iw3.desktop.quest_session import QuestSession


def test_cli_auto_browser_is_opt_in_and_preserves_selected_desktop_backend():
    parser = desktop.create_parser()
    assert parser.parse_args([]).quest_auto_browser is False
    args = parser.parse_args(["--quest", "--quest-auto-browser", "--screenshot", "mss"])
    desktop.prepare_quest_args(args)
    assert args.quest_auto_browser and args.screenshot == "mss"


@pytest.mark.parametrize("option", [[], ["--pad-mode", "16:9"], ["--ipd-offset", "1"], ["--rotate-left"]])
def test_auto_browser_rejects_nonquest_or_transforms_that_reintroduce_black_canvas(option):
    argv = ["--quest-auto-browser", *option]
    if option:
        argv.append("--quest")
    args = desktop.create_parser().parse_args(argv)
    with pytest.raises(ValueError, match="自动|全屏"):
        desktop.prepare_quest_args(args)


def test_auto_browser_keeps_legal_window_crops_for_desktop_only():
    args = desktop.create_parser().parse_args(
        ["--quest", "--quest-auto-browser", "--screenshot", "mss", "--window-name", "Browser", "--crop-left", "5"]
    )
    desktop.prepare_quest_args(args)
    assert args.crop_left == 5


def test_session_forwards_real_source_kind_to_fps_sink_only_when_changed():
    args = desktop.create_parser().parse_args(["--quest"])
    seen = []
    args.state = {"fps_event": SimpleNamespace(set_source_kind=seen.append)}
    session = QuestSession(args)
    session.set_stats(source_kind="desktop", decoder="屏幕捕获")
    session.set_stats(source_kind="desktop", screenshot_fps=30)
    session.set_stats(source_kind="browser", decoder="NVDEC")
    assert seen == ["desktop", "browser"]


def test_session_desktop_status_is_not_mislabeled_as_nvdec():
    args = desktop.create_parser().parse_args(["--quest"])
    args.state = {}
    session = QuestSession(args)
    event = threading.Event()
    event.set()
    polls = []
    capture = SimpleNamespace(
        poll_route=lambda: polls.append(True),
        frame_set_event=event,
        get_frame=lambda: None,
        get_status=lambda: dict(
            source_kind="desktop",
            status="running",
            input_source="桌面屏幕捕获",
            decoder="屏幕捕获",
            hardware_active=False,
            audio_path="Windows 独立回环采音",
        ),
    )
    assert session.read_frame(capture) is None
    stats = session.snapshot()["values"]
    assert polls and stats["source_kind"] == "desktop" and stats["decoder"] == "屏幕捕获"
    assert stats["input_source"] == "桌面屏幕捕获"


@pytest.mark.parametrize("invalidation", ["ai", "ambient"])
@pytest.mark.parametrize("target", ["plain", "window", "monitor", "obs"])
def test_auto_runtime_shares_ai_server_and_rejects_frames_on_fullscreen_exit(monkeypatch, invalidation, target):
    from iw3.desktop import quest_server

    backend_name = "wc_cuda" if target == "monitor" else "obs" if target == "obs" else "mss"
    argv = [
        "--quest",
        "--quest-auto-browser",
        "--screenshot",
        backend_name,
        "--bind-addr",
        "127.0.0.1",
        "--stream-height",
        "320",
    ]
    if target in {"window", "monitor"}:
        argv.append("--quest-control")
    if target == "window":
        argv += ["--window-name", "Browser", "--crop-left", "5", "--crop-right", "5"]
    args = desktop.create_parser().parse_args(argv)
    desktop.prepare_quest_args(args)
    stop = threading.Event()
    state = dict(fresh=True, fullscreen=False, generation="video-1")
    log, processed, backgrounds, sent, configurations, references, depth_resets = [], [], [], [], [], [], []
    sources = []
    inputs = []
    depth = SimpleNamespace(
        loaded=lambda: True,
        load=lambda **kw: pytest.fail("route switches must not reload AI weights"),
        enable_ema=lambda *a, **kw: None,
        is_metric=lambda: False,
        clear_compiled_model=lambda: None,
        reset=lambda: depth_resets.append(True),
    )
    args.state = dict(
        depth_model=depth, convergence_model=None, args_lock=threading.Lock(), stop_event=stop, fps_event=None
    )

    class Capture:
        def __init__(self, kind, **kwargs):
            self.kind, self.kwargs, self.alive = kind, kwargs, False
            self.frame_set_event = threading.Event()
            self.frame_set_event.set()
            self.generation = state["generation"]
            sources.append(self)

        def start(self):
            assert not any(source.alive for source in sources)
            self.alive = True
            log.append(("start", self.kind))

        def stop(self):
            self.alive = False
            log.append(("stop", self.kind))

        def is_alive(self):
            return self.alive

        def get_frame(self):
            self.generation = state["generation"]
            return torch.ones((3, 640, 360) if self.kind == "browser" else (3, 320, 568))

        def init_capture(self, **kw):
            log.append(("init_capture", self.kind))
            return 1920, 1080

        def set_frame_size(self, width, height):
            self.kwargs.update(frame_width=width, frame_height=height)
            log.append(("set_frame_size", (width, height)))

        def get_frame_generation(self):
            return self.generation

        def is_current_generation(self, generation):
            return generation == state["generation"] and state["fresh"]

        def get_content_rect(self):
            return (0, 0, 568, 320)

        def get_frame_mapping(self):
            return {"capture": id(self)}

        def get_fps(self):
            return 30

        def get_status(self):
            return dict(status="running", decoder="NVDEC", hardware_active=True, source_size=(360, 640))

    class Bridge:
        def start(self):
            log.append(("bridge", "start"))

        def stop(self):
            log.append(("bridge", "stop"))

        def snapshot(self):
            return None

        def status(self):
            return dict(state)

    class WindowsInput:
        def __init__(self, window_name, screenshot, crops, frame_mapping, **kw):
            self.hwnd = 42 if window_name else None
            self.monitor_target = None if window_name else {"monitor_index": 0, "monitor_rect": (0, 0, 1920, 1080)}
            self.mapping = frame_mapping
            inputs.append(self)

        def mouse_up(self):
            log.append(("mouse_up", True))

        def release_keys(self):
            log.append(("release_keys", True))

    def process(frame, *a, **kw):
        kind = "browser" if frame.shape[-1] == 360 else "desktop"
        processed.append((kind, tuple(frame.shape)))
        if kind == "browser" and state["generation"] == "video-2" and invalidation == "ai":
            state["fullscreen"] = False
        return torch.cat((frame, frame), dim=2)

    def background(*a, **kw):
        backgrounds.append(kw)
        log.append(("ambient", state["generation"]))
        if state["generation"] == "video-2" and invalidation == "ambient":
            state["fullscreen"] = False

    def send(data):
        frame = data[0]
        if not frame.any():
            return
        if inputs and frame.shape[-1] == 1136:
            assert inputs[0].mapping() == {"capture": id(sources[-1])}
        sent.append(tuple(frame.shape))
        if len(sent) == 1:
            state["fullscreen"] = True
        elif len(sent) == 2:
            state["generation"] = "video-2"
        elif len(sent) == 3:
            stop.set()

    active_controller = None

    def set_controller(factory):
        nonlocal active_controller
        if active_controller is not None:
            active_controller.reset()
        active_controller = factory() if factory else None
        if active_controller is not None and isinstance(active_controller.backend, WindowsInput):
            active_controller._pressed = True
        log.append(("controller", bool(factory)))

    def create_server(**kw):
        set_controller(kw["controller_factory"])
        return server

    server = SimpleNamespace(
        start=lambda: log.append(("server", "start")),
        stop=lambda: log.append(("server", "stop")),
        check_error=lambda: None,
        set_audio_gain=lambda gain: None,
        set_controller_factory=set_controller,
        discard_video_frames=lambda: log.append(("discard", True)),
        set_reference_aspect=references.append,
        refresh_ambient_frame=lambda: None,
        set_ambient_frame=background,
        configure_video=lambda w, h, fps, bitrate: configurations.append((w, h)),
        set_frame_data=send,
        get_fps=lambda: 0,
        actual_cylinder=False,
    )
    monkeypatch.setitem(sys.modules, "iw3.desktop.browser_video_bridge", SimpleNamespace(BrowserVideoBridge=Bridge))
    monkeypatch.setitem(sys.modules, "iw3.desktop.quest_windows", SimpleNamespace(WindowsInput=WindowsInput))
    monkeypatch.setitem(
        sys.modules,
        "iw3.desktop.media_video_source",
        SimpleNamespace(MediaVideoSource=lambda *a, **kw: Capture("browser", **kw)),
    )
    monkeypatch.setattr(desktop, "ScreenshotProcess", lambda **kw: Capture("desktop", **kw))
    monkeypatch.setattr(desktop, "ScreenshotThreadWCCUDA", lambda **kw: Capture("desktop", **kw))
    monkeypatch.setattr(desktop, "ScreenshotProcessOBS", lambda **kw: Capture("desktop", **kw))
    monkeypatch.setattr(desktop, "obs_unsupported_reason", lambda: None)
    monkeypatch.setattr(desktop, "create_device", lambda ids: torch.device("cpu"))
    monkeypatch.setattr(desktop, "get_monitor_size_list", lambda: [(1920, 1080)])
    monkeypatch.setattr(desktop, "get_window_rect_by_title", lambda name: dict(width=1930, height=1080))
    monkeypatch.setattr(desktop, "test_output_size", lambda *a: (320, 1136))
    monkeypatch.setattr(desktop.IW3U, "create_stereo_model", lambda *a, **kw: None)
    monkeypatch.setattr(desktop.IW3U, "process_image", process)
    monkeypatch.setattr(quest_server, "QuestServer", create_server)
    desktop._iw3_desktop_main_once(args, init_wxapp=False)
    assert processed == [
        ("desktop", (3, 320, 568)),
        ("browser", (3, 640, 360)),
        ("browser", (3, 640, 360)),
        ("desktop", (3, 320, 568)),
    ]
    assert sent == [(3, 320, 1136), (3, 640, 720), (3, 320, 1136)]
    assert len(backgrounds) == (1 if invalidation == "ai" else 2)
    assert backgrounds[0]["content_rect"] == (0, 0, 360, 640)
    if invalidation == "ambient":
        assert log[log.index(("ambient", "video-2")) + 1] == ("discard", True)
    assert configurations == [(720, 640), (1136, 320)]
    assert references == [None, 16 / 9, None]
    assert log.count(("server", "start")) == log.count(("bridge", "start")) == 1
    assert [kind for action, kind in log if action == "start"] == ["desktop", "browser", "desktop"]
    assert all(not source.alive for source in sources)
    assert len(depth_resets) == 5
    if inputs:
        assert len(inputs) == 1
        for source in (sources[0], sources[-1]):
            assert source.kwargs["window_handle"] == inputs[0].hwnd
            assert source.kwargs["monitor_target"] == inputs[0].monitor_target
            assert source.kwargs["fit_window"] and not source.kwargs["draw_cursor_enabled"]
        assert log.count(("mouse_up", True)) == 2 and log.count(("release_keys", True)) == 2
    assert "crop_left" not in sources[1].kwargs and "window_handle" not in sources[1].kwargs
    if target == "window":
        assert sources[0].kwargs["crop_left"] == sources[-1].kwargs["crop_left"] == 5
    if target == "obs":
        assert log.count(("init_capture", "desktop")) == log.count(("set_frame_size", (568, 320))) == 2
