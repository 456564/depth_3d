import sys
import threading

import pytest

from iw3.desktop import utils as desktop


def test_quest_selects_full_sbs_without_changing_web_defaults() -> None:
    web = desktop.create_parser().parse_args([])
    assert not getattr(web, "quest", False)
    assert not web.quest_control
    assert web.quest_audio_source == "process"
    assert web.quest_audio_gain == 1.0
    assert not web.full_sbs
    args = desktop.create_parser().parse_args(["--quest", "--half-sbs", "--tb", "--quest-bitrate", "40"])
    desktop.prepare_quest_args(args)
    assert args.full_sbs and not args.half_sbs and not args.tb
    assert args.quest_bitrate == 40 and not args.quest_no_audio


@pytest.mark.parametrize("capture", ["wc_mp", "mss"])
def test_quest_control_accepts_explicit_windows_capture_target(monkeypatch, capture: str) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    args = desktop.create_parser().parse_args(
        ["--quest", "--quest-control", "--window-name", "Browser", "--screenshot", capture]
    )
    desktop.prepare_quest_args(args)
    assert args.quest_control and args.full_sbs


@pytest.mark.parametrize(
    "changes",
    [
        {"quest": False},
        {"window_name": None},
        {"window_name": "   "},
        {"screenshot": "pil"},
        {"screenshot": "wc_cuda"},
        {"screenshot": "obs"},
        {"screenshot": "kwcapture"},
        {"pad": 0.1},
        {"pad_mode": "16:9"},
        {"pad_mode": "1:1"},
        {"pad_mode": "tb"},
        {"ipd_offset": 1},
        {"rotate_left": True},
        {"rotate_right": True},
    ],
)
def test_quest_control_rejects_uncalibrated_options(monkeypatch, changes) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    args = desktop.create_parser().parse_args(["--quest", "--window-name", "Browser", "--screenshot", "mss"])
    args.quest_control = True
    for name, value in changes.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match="Quest.*control"):
        desktop.prepare_quest_args(args)


def test_quest_control_requires_windows(monkeypatch) -> None:
    args = desktop.create_parser().parse_args(["--quest", "--window-name", "Browser", "--screenshot", "mss"])
    args.quest_control = True
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    with pytest.raises(ValueError, match="Windows"):
        desktop.prepare_quest_args(args)


def test_quest_and_local_viewer_cannot_run_together() -> None:
    with pytest.raises(SystemExit):
        desktop.create_parser().parse_args(["--quest", "--local-viewer"])


@pytest.mark.parametrize("gain", ["-0.1", "4.1", "nan", "inf"])
def test_quest_rejects_invalid_audio_gain(gain: str) -> None:
    args = desktop.create_parser().parse_args(["--quest", "--quest-audio-gain", gain])
    with pytest.raises(ValueError, match="gain"):
        desktop.prepare_quest_args(args)


def test_quest_audio_can_select_speaker_fallback_and_independent_gain() -> None:
    args = desktop.create_parser().parse_args(
        ["--quest", "--quest-audio-source", "speaker", "--quest-audio-gain", "2.0"]
    )
    desktop.prepare_quest_args(args)
    assert args.quest_audio_source == "speaker" and args.quest_audio_gain == 2.0


@pytest.mark.parametrize("extra", [["--quest-bitrate", "0"], ["--gpu", "-1"], ["--stream-fps", "0"]])
def test_quest_rejects_unsupported_settings(extra: list[str]) -> None:
    args = desktop.create_parser().parse_args(["--quest", *extra])
    with pytest.raises(ValueError):
        desktop.prepare_quest_args(args)


def test_gui_switches_native_web_and_local_without_changing_saved_web_settings(monkeypatch) -> None:
    import wx

    from iw3.desktop import gui

    monkeypatch.setattr(sys, "argv", ["iw3-desktop"])
    monkeypatch.setattr(gui.MainFrame, "load_preset", lambda self: None)
    monkeypatch.setattr(gui, "set_state_args", lambda args, **kwargs: None)
    app = wx.GetApp() or wx.App()
    frame = gui.MainFrame()
    try:
        assert frame.cbo_view_mode.FindString("Quest / Pop3D") >= 0
        assert not frame.chk_quest_control.GetValue()
        assert not frame.chk_quest_control.IsShown()
        assert frame.cbo_quest_audio_source.GetValue() == "独立采音"
        assert not frame.cbo_quest_audio_source.IsShown()
        assert not frame.lbl_quest_screen_help.IsShown()
        frame.cbo_stream_quality.SetValue("95")
        frame.cbo_stereo_format.SetValue("Half SBS")
        frame.cbo_view_mode.SetValue("Quest / Pop3D")
        frame.update_view_mode()
        assert not frame.cbo_stream_quality.IsShown()
        assert frame.cbo_quest_bitrate.IsShown()
        assert frame.chk_quest_control.IsShown()
        assert frame.cbo_quest_audio_source.IsShown()
        assert frame.lbl_quest_screen_help.IsShown()
        assert "任一手" in frame.lbl_quest_screen_help.GetLabel()
        assert not frame.cbo_stereo_format.IsShown()
        validation_errors = []
        monkeypatch.setattr(frame, "show_validation_error_message", lambda *args: validation_errors.append(args))
        frame.cbo_quest_bitrate.SetValue("35.5")
        assert frame.parse_args() is None
        assert validation_errors[0][0] == "HEVC (Mbps)"
        frame.cbo_quest_bitrate.SetValue("35")
        args = frame.parse_args()
        assert args.quest and args.full_sbs and not args.local_viewer
        assert args.quest_bitrate == 35 and not args.quest_no_audio
        assert not args.quest_control
        assert args.quest_audio_source == "process" and args.quest_audio_gain == 1.0
        frame.sld_quest_audio_gain.SetValue(175)
        args = frame.parse_args()
        assert args.quest_audio_gain == 1.75
        frame.args = args
        frame.sld_quest_audio_gain.SetValue(225)
        frame.on_quest_audio_gain(None)
        assert frame.args.quest_audio_gain == 2.25
        frame.chk_quest_control.SetValue(True)
        frame.cbo_screenshot.SetValue("mss")
        frame.cbo_window_name.SetValue("Browser")
        args = frame.parse_args()
        assert args.quest_control
        errors = []
        monkeypatch.setattr(gui, "set_state_args", lambda args, **kwargs: desktop.prepare_quest_args(args))
        monkeypatch.setattr(gui.wx, "MessageBox", lambda message, *args: errors.append(message))
        frame.cbo_window_name.SetValue("")
        assert frame.parse_args() is None
        assert "window name" in errors[0]
        frame.cbo_window_name.SetValue("Browser")
        frame.cbo_view_mode.SetValue("Web Streaming")
        frame.update_view_mode()
        assert frame.cbo_stream_quality.IsShown()
        assert not frame.cbo_quest_bitrate.IsShown()
        assert not frame.chk_quest_control.IsShown()
        assert not frame.cbo_quest_audio_source.IsShown()
        assert frame.cbo_stereo_format.GetValue() == "Half SBS"
        assert frame.cbo_stream_quality.GetValue() == "95"
        args = frame.parse_args()
        assert not args.quest and not args.full_sbs and args.stream_quality == 95
        assert not args.quest_control
        frame.cbo_view_mode.SetValue("Local Viewer")
        frame.update_view_mode()
        args = frame.parse_args()
        assert args.local_viewer and not args.quest
        assert not args.quest_control
        assert frame.chk_fit_window.IsShown()
        assert not frame.cbo_quest_bitrate.IsShown()
    finally:
        frame.Destroy()
        app.ProcessPendingEvents()


@pytest.mark.parametrize(
    "control,binding_fails,capture_kind",
    [(False, False, "pil"), (True, False, "mss"), (True, False, "wc_mp"), (True, True, "mss")],
)
def test_failed_startup_or_stop_still_cleans_capture_and_depth(
    monkeypatch, control: bool, binding_fails: bool, capture_kind: str
) -> None:
    import torch

    from iw3.desktop import quest_input, quest_server, quest_windows

    cleaned = []
    bound = []
    controllers = []
    mapping_getters = []
    captures = []

    class InputBackend:
        def __init__(self, window_name, screenshot, crops, frame_mapping=None):
            assert control
            assert callable(frame_mapping)
            assert frame_mapping() is None
            mapping_getters.append(frame_mapping)
            self.hwnd = 7
            bound.append((window_name, screenshot, crops))
            if binding_fails:
                raise ValueError("simulated input binding failure")

    class Depth:
        def loaded(self):
            return True

        def enable_ema(self, *args, **kwargs):
            pass

        def is_metric(self):
            return False

        def clear_compiled_model(self):
            cleaned.append("clear")

        def reset(self):
            cleaned.append("reset")

    class Capture:
        def __init__(self, **kwargs):
            captures.append(self)
            self.started = False
            self.kwargs = kwargs

        def start(self):
            self.started = True
            if control:
                assert self.kwargs["window_handle"] == 7 and self.kwargs["fit_window"] is True
                assert mapping_getters[0]() == {"hwnd": 7}
            else:
                assert "window_handle" not in self.kwargs and "fit_window" not in self.kwargs

        def get_frame_mapping(self):
            return {"hwnd": 7} if self.started else None

        def get_frame(self):
            return None

        def stop(self):
            self.started = False
            cleaned.append("capture")

    class FailedStop:
        def __init__(self, **kwargs):
            factory = kwargs.get("controller_factory")
            assert (factory is not None) == control
            if factory is not None:
                controllers.append(factory())

        def start(self):
            pass

        def check_error(self):
            pass

        def set_audio_gain(self, gain):
            pass

        def stop(self):
            raise RuntimeError("simulated stop failure")

    args = desktop.create_parser().parse_args(["--quest", "--bind-addr", "127.0.0.1"])
    if control:
        args.quest_control = True
        args.window_name = "Browser"
        args.screenshot = capture_kind
        args.crop_top, args.crop_left, args.crop_right, args.crop_bottom = 1, 2, 3, 4
    desktop.prepare_quest_args(args)
    args.state = dict(
        depth_model=Depth(), convergence_model=None, args_lock=threading.Lock(), stop_event=None, fps_event=None
    )
    monkeypatch.setattr(desktop, "create_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(desktop, "ScreenshotThreadPIL", Capture)
    monkeypatch.setattr(desktop, "ScreenshotProcess", Capture)
    monkeypatch.setattr(desktop, "get_monitor_size_list", lambda: [(64, 64)])
    monkeypatch.setattr(desktop, "get_window_rect_by_title", lambda title: {"width": 64, "height": 64})
    monkeypatch.setattr(desktop, "test_output_size", lambda *args: (64, 128))
    monkeypatch.setattr(desktop.IW3U, "create_stereo_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(quest_server, "QuestServer", FailedStop)
    monkeypatch.setattr(quest_windows, "WindowsInput", InputBackend)
    error, message = (
        (ValueError, "simulated input binding failure") if binding_fails else (RuntimeError, "simulated stop failure")
    )
    with pytest.raises(error, match=message):
        desktop.iw3_desktop_main(args, init_wxapp=False)
    assert cleaned == (["clear", "reset"] if binding_fails else ["capture", "clear", "reset"])
    assert bound == ([("Browser", capture_kind, (1, 2, 3, 4))] if control else [])
    assert len(controllers) == int(control and not binding_fails)
    assert len(captures) == int(not binding_fails)
    if mapping_getters:
        assert mapping_getters[0]() is None
    if controllers:
        assert isinstance(controllers[0], quest_input.QuestController)
        assert isinstance(controllers[0].backend, InputBackend)


def test_loaded_audio_gain_label_matches_restored_slider(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import wx

    from iw3.desktop import gui

    monkeypatch.setattr(sys, "argv", ["iw3-desktop"])
    monkeypatch.setattr(gui.persist.PersistenceManager, "Get", lambda: MagicMock())
    monkeypatch.setattr(gui, "persistent_manager_register_all", lambda *args: None)
    monkeypatch.setattr(gui, "persistent_manager_register", lambda *args: None)
    monkeypatch.setattr(gui, "persistent_manager_unregister_all", lambda *args: None)
    monkeypatch.setattr(gui, "persistent_manager_restore_all", lambda *args: None)
    app = wx.GetApp() or wx.App()
    frame = gui.MainFrame()
    try:
        monkeypatch.setattr(
            gui, "persistent_manager_restore_all", lambda *args: frame.sld_quest_audio_gain.SetValue(225)
        )
        frame.load_preset()
        assert frame.sld_quest_audio_gain.GetValue() == 225
        assert frame.lbl_quest_audio_gain.GetLabel() == "头显音量 225%"
    finally:
        frame.Destroy()
        app.ProcessPendingEvents()
