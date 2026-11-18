import threading
import time
from argparse import Namespace

import pytest
import torch

from iw3.desktop import utils
from iw3.desktop.quest_menu import QuestMenu
from iw3.desktop.quest_restart import load_resume, write_resume
from iw3.desktop.quest_session import QuestSession
from iw3.desktop.quest_settings import FIELD_MAP, PAGES, validate_patch

from .test_quest_menu import click, opened


def session() -> QuestSession:
    args = utils.create_parser().parse_args(["--quest", "--quest-ambient"])
    args.state = {"stop_event": threading.Event()}
    return QuestSession(args)


def test_ambient_settings_default_ranges_and_cli() -> None:
    args = utils.create_parser().parse_args([])
    assert not args.quest_ambient
    assert args.quest_ambient_brightness == 0.25 and args.quest_ambient_spread == 2
    assert all(
        FIELD_MAP[key].immediate for key in ("quest_ambient", "quest_ambient_brightness", "quest_ambient_spread")
    )
    assert "环境光" in PAGES
    for patch in (
        {"quest_ambient": 1},
        {"quest_ambient_brightness": -0.1},
        {"quest_ambient_brightness": float("nan")},
        {"quest_ambient_spread": 1},
        {"quest_ambient_spread": 4.1},
    ):
        with pytest.raises(ValueError):
            validate_patch(patch)
    args = utils.create_parser().parse_args(["--quest", "--quest-ambient-brightness", "2"])
    with pytest.raises(ValueError):
        utils.prepare_quest_args(args)


def test_ambient_live_update_applies_all_current_values_without_media_restart() -> None:
    owner = session()
    applied = []
    owner.server = Namespace(configure_ambient=lambda *values, **options: applied.append(values))
    owner.submit("set", {"quest_ambient_brightness": 0.4, "quest_ambient_spread": 3.0})
    owner.apply_live()
    assert applied == [(True, 0.4, 3.0, True, 0.03, "edge")]
    assert owner.snapshot()["values"]["quest_ambient_brightness"] == 0.4
    assert not owner.run_stop.is_set()
    owner.submit("set", {"quest_ambient": False})
    owner.apply_live()
    assert applied[-1] == (False, 0.4, 3.0, True, 0.03, "edge")


def test_ambient_menu_controls_and_restart_values(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    values = {"quest_ambient": False, "quest_ambient_brightness": 0.25, "quest_ambient_spread": 2.0}
    commands = []
    menu = QuestMenu(
        lambda: {"values": values, "choices": {}, "status": "running"},
        lambda command, patch: commands.append((command, patch)),
    )
    opened(menu)
    click(menu, "page", PAGES.index("环境光"))
    now[0] += 0.2
    menu.texture()
    click(menu, "toggle", "quest_ambient")
    assert commands[-1] == ("set", {"quest_ambient": True})
    click(menu, "adjust", ("quest_ambient_brightness", 1, "x"))
    assert commands[-1] == ("set", {"quest_ambient_brightness": 0.3})
    resume = write_resume(values, gui=False)
    assert load_resume(resume, gui=False) == values


@pytest.mark.parametrize("reference_aspect", [None, 16 / 9])
def test_media_reconfiguration_passes_ambient_values(reference_aspect) -> None:
    owner = session()
    applied, layouts = [], []
    owner.server = Namespace(
        pause_media=lambda: None,
        set_reference_aspect=layouts.append,
        configure_video=lambda *a: None,
        configure_audio=lambda *a: None,
        set_screen_curvature=lambda *a: None,
        set_controller_factory=lambda *a: None,
        configure_ambient=lambda *values, **options: applied.append(values),
    )
    owner.configure(
        width=128,
        height=64,
        fps=30,
        bitrate=35,
        audio=False,
        audio_source="process",
        audio_gain=1,
        ambient_enabled=True,
        ambient_brightness=0.3,
        ambient_spread=2.2,
        ambient_fill_borders=False,
        ambient_feather=0.07,
        reference_aspect=reference_aspect,
    )
    assert applied == [(True, 0.3, 2.2, False, 0.07, "edge")]
    assert layouts == [reference_aspect]


@pytest.mark.parametrize("has_content_rect", [True, False])
def test_capture_roi_and_final_sbs_reach_ambient_together_without_args_lock(monkeypatch, has_content_rect) -> None:
    from iw3.desktop import quest_server

    args = utils.create_parser().parse_args(["--quest", "--quest-ambient", "--bind-addr", "127.0.0.1"])
    args.quest_ambient_style = "halo"
    utils.prepare_quest_args(args)
    depth = Namespace(
        loaded=lambda: True,
        enable_ema=lambda *a, **kw: None,
        is_metric=lambda: False,
        clear_compiled_model=lambda: None,
        reset=lambda: None,
    )
    lock = threading.Lock()
    args.state = dict(depth_model=depth, convergence_model=None, args_lock=lock, stop_event=None, fps_event=None)
    frame = torch.zeros(3, 1080, 1920)
    final_sbs = torch.ones(3, 1080, 3840)
    content_rect = (0, 135, 1920, 945)
    calls = []

    class Capture:
        def __init__(self, **kwargs):
            self.count = 0
            if has_content_rect:
                self.get_content_rect = lambda: content_rect

        def start(self):
            pass

        def stop(self):
            pass

        def get_frame_mapping(self):
            raise AssertionError("input mapping uses original buffer coordinates, not final frame pixels")

        def get_frame(self):
            self.count += 1
            return frame if self.count == 1 else None

    def ambient(image, *, content_rect, sbs, wake, still_current):
        assert wake and still_current is None
        assert image is frame
        assert sbs is final_sbs
        assert not lock.locked()
        assert content_rect == ((0, 135, 1920, 945) if has_content_rect else None)
        calls.append("ambient")

    def stereo(image, *args, **kwargs):
        calls.append("stereo")
        return final_sbs

    server = Namespace(
        start=lambda: None,
        stop=lambda: None,
        check_error=lambda: None,
        set_audio_gain=lambda gain: None,
        set_ambient_frame=ambient,
        set_frame_data=lambda data: calls.append("video"),
    )
    monkeypatch.setattr(utils, "create_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "ScreenshotThreadPIL", Capture)
    monkeypatch.setattr(utils, "get_monitor_size_list", lambda: [(2560, 1440)])
    monkeypatch.setattr(utils, "test_output_size", lambda *a: (1080, 3840))
    monkeypatch.setattr(utils.IW3U, "create_stereo_model", lambda *a, **kw: None)
    monkeypatch.setattr(utils.IW3U, "process_image", stereo)

    def create_server(**settings):
        assert settings["ambient_style"] == "halo"
        return server

    monkeypatch.setattr(quest_server, "QuestServer", create_server)
    utils._iw3_desktop_main_once(args, init_wxapp=False)
    assert calls == ["stereo", "video", "ambient"]
