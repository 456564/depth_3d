import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from iw3.desktop import utils as desktop
from iw3.desktop.quest_session import QuestSession
from iw3.desktop.quest_settings import validate_patch


def test_browser_media_choice_is_allowed_with_semantic_quest_control():
    args = desktop.create_parser().parse_args(["--quest", "--quest-control", "--screenshot", "browser_video"])
    desktop.prepare_quest_args(args)
    assert args.screenshot == "browser_video" and args.full_sbs
    assert validate_patch({"screenshot": "browser_video", "quest_control": True, "pad_mode": None})


def test_browser_media_rejects_padding_that_reintroduces_web_canvas_black_bars():
    args = desktop.create_parser().parse_args(["--quest", "--screenshot", "browser_video", "--pad-mode", "16:9"])
    with pytest.raises(ValueError, match="padding|补边"):
        desktop.prepare_quest_args(args)


def test_browser_session_wait_updates_visible_source_status_and_keeps_control_alive():
    args = desktop.create_parser().parse_args(["--quest"])
    args.state = {"stop_event": threading.Event()}
    session = QuestSession(args)
    source = SimpleNamespace(
        frame_set_event=threading.Event(),
        is_alive=lambda: True,
        get_status=lambda: {"status": "waiting", "message": "等待网页视频", "decoder": "NVDEC", "source_size": None},
    )
    source.frame_set_event.set()
    source.get_frame = lambda: None
    assert session.read_frame(source) is None
    snapshot = session.snapshot()
    assert snapshot["status"] == "waiting"
    assert snapshot["message"] == "等待网页视频"
    assert snapshot["values"]["decoder"] == "NVDEC"


def test_browser_source_error_replaces_running_status_and_stale_fps():
    args = desktop.create_parser().parse_args(["--quest"])
    args.state = {"stop_event": threading.Event()}
    session = QuestSession(args)
    session._committed = True
    session.status("running", "previous clip")
    session.set_stats(screenshot_fps=30, streaming_fps=30)
    session.server = SimpleNamespace(check_error=lambda: None, get_fps=lambda: 0)
    source = SimpleNamespace(
        frame_set_event=threading.Event(),
        is_alive=lambda: True,
        get_status=lambda: {"status": "error", "message": "NVDEC 解码失败", "hardware_active": False},
        get_fps=lambda: 0,
        get_frame=lambda: None,
    )
    source.frame_set_event.set()
    assert session.read_frame(source) is None
    snapshot = session.snapshot()
    assert snapshot["status"] == "error" and snapshot["message"] == "NVDEC 解码失败"
    assert snapshot["values"]["screenshot_fps"] == snapshot["values"]["streaming_fps"] == 0


def test_depth_temporal_reset_retains_loaded_weights_and_ema_configuration():
    from iw3.depth_models.depth_anything_model import DepthAnythingModel

    depth = DepthAnythingModel("Any_V2_S")
    weights = object()
    depth.model = weights
    depth.enable_ema(0.937, buffer_size=1)
    depth.scaler(torch.tensor([[[1.0, 3.0]]]))
    assert depth.scaler.min_value is not None
    depth.reset()
    assert depth.model is weights and depth.loaded()
    assert depth.get_ema_state() == (0.937, 1)
    assert depth.scaler.min_value is None and depth.scaler.max_value is None


@pytest.mark.parametrize("odd", [False, True])
def test_browser_media_runtime_uses_real_shapes_and_discards_stale_generation(monkeypatch, odd):
    from iw3.desktop import quest_server

    args = desktop.create_parser().parse_args(["--quest", "--bind-addr", "127.0.0.1", "--stream-height", "320"])
    args.screenshot = "browser_video"
    desktop.prepare_quest_args(args)
    cleaned, configured, sent, processed, barriers, refreshes, backgrounds = [], [], [], [], [], [], []
    source = None
    depth = SimpleNamespace(
        loaded=lambda: True,
        enable_ema=lambda *a, **kw: None,
        is_metric=lambda: False,
        clear_compiled_model=lambda: cleaned.append("depth-clear"),
        reset=lambda: cleaned.append("depth-reset"),
    )
    convergence_resets = []
    convergence = SimpleNamespace(reset=lambda **kw: convergence_resets.append(kw))
    args.state = dict(
        depth_model=depth, convergence_model=convergence, args_lock=threading.Lock(), stop_event=None, fps_event=None
    )

    class SourceEvent:
        def __init__(self):
            self.calls = 0
            self.ready = True

        def set(self):
            pass

        def is_set(self):
            return self.ready

        def clear(self):
            self.ready = False

        def wait(self, timeout):
            self.calls += 1
            self.ready = self.calls != 2
            return self.ready

    class Source:
        def __init__(self, snapshot, device, fps, quality="auto", state=None):
            nonlocal source
            source = self
            self.index = 0
            self.generation = "first"
            self.current = True
            self.frame_set_event = SourceEvent()
            self.frame_set_event.set()
            self.frames = [
                torch.ones(3, 640 + odd, 360 + odd),
                torch.ones(3, 360 + odd, 640 + odd),
                torch.ones(3, 360 + odd, 640 + odd),
            ]

        def start(self):
            pass

        def stop(self):
            cleaned.append("source")

        def is_alive(self):
            return self.index <= len(self.frames)

        def get_frame(self):
            assert self.frame_set_event.ready, "paused source must not be consumed without a new frame"
            self.index += 1
            self.generation = "first" if self.index == 1 else "second"
            return self.frames[self.index - 1] if self.index <= len(self.frames) else None

        def get_frame_generation(self):
            return self.generation

        def is_current_generation(self, generation):
            return self.current and generation == self.generation

        def get_fps(self):
            return 30

        def get_status(self):
            return {"status": "running", "decoder": "NVDEC", "source_size": (640, 360)}

    class Bridge:
        def start(self):
            pass

        def stop(self):
            cleaned.append("bridge")

        def snapshot(self):
            return None

        def status(self):
            return {"fresh": True, "generation": "first"}

    def process(frame, *a, **kw):
        expected_resets = 1 if source.index == 1 else 2
        assert cleaned.count("depth-reset") == expected_resets
        assert convergence_resets == [{"enable_ema": True, "decay": 0.98}] * (expected_resets + 1)
        processed.append(tuple(frame.shape))
        if source.index == 3:
            source.current = False
        return torch.cat((frame, frame), dim=2)

    def send(data):
        frame = data[0]
        if source is not None and source.index == 1 and not source.frame_set_event.ready:
            assert frame.any(), "pause keeps the decoded frame visible"
        sent.append(frame.clone())

    def background(*a, **kw):
        assert sent and torch.equal(sent[-1], kw["sbs"]), "valid video must precede ambient pixel work"
        assert kw["wake"] is True, "a paused first frame must wake the ambient sender"
        backgrounds.append(kw)

    server = SimpleNamespace(
        start=lambda: None,
        stop=lambda: cleaned.append("server"),
        check_error=lambda: None,
        set_audio_gain=lambda gain: None,
        set_controller_factory=lambda value: None,
        set_reference_aspect=lambda value: None,
        discard_video_frames=lambda: barriers.append(True),
        refresh_ambient_frame=lambda **kw: refreshes.append(kw["still_current"]()),
        set_ambient_frame=background,
        configure_video=lambda w, h, fps, bitrate: configured.append((w, h)),
        set_frame_data=send,
        get_fps=lambda: 0,
        actual_cylinder=False,
    )
    monkeypatch.setitem(sys.modules, "iw3.desktop.media_video_source", SimpleNamespace(MediaVideoSource=Source))
    monkeypatch.setitem(sys.modules, "iw3.desktop.browser_video_bridge", SimpleNamespace(BrowserVideoBridge=Bridge))
    monkeypatch.setattr(desktop, "create_device", lambda ids: torch.device("cpu"))
    monkeypatch.setattr(desktop, "test_output_size", lambda *a: pytest.fail("media path must not run desktop sizing"))
    monkeypatch.setattr(desktop, "get_monitor_size_list", lambda: pytest.fail("media source is not a desktop"))
    monkeypatch.setattr(desktop.IW3U, "create_stereo_model", lambda *a, **kw: None)
    monkeypatch.setattr(desktop.IW3U, "process_image", process)
    monkeypatch.setattr(quest_server, "QuestServer", lambda **kw: server)
    desktop._iw3_desktop_main_once(args, init_wxapp=False)
    height, width = 640 + 2 * odd, 360 + 2 * odd
    assert processed == [(3, height, width), (3, width, height), (3, width, height)]
    assert configured == [(width * 2, height), (height * 2, width)]
    assert len(barriers) == 3 and len(refreshes) == 1
    assert [value["content_rect"] for value in backgrounds] == [(0, 0, width, height), (0, 0, height, width)]
    assert [tuple(frame.shape) for frame in sent if frame.any()] == [(3, height, width * 2), (3, width, height * 2)]
    assert "bridge" in cleaned and "source" in cleaned and "depth-clear" in cleaned
