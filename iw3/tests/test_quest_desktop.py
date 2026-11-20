import sys
import threading

import pytest

from iw3.desktop import utils as desktop


@pytest.mark.parametrize("method,canonical", [("row_flow", "row_flow_v3"), ("row_flow_sym", "row_flow_v3_sym")])
def test_quest_cli_normalizes_legacy_settings_for_menu_and_resume(method, canonical):
    from iw3.desktop.quest_settings import validate_patch

    args = desktop.create_parser().parse_args(["--quest", "--method", method, "--edge-dilation", "2"])
    desktop.prepare_quest_args(args)
    assert args.method == canonical and args.edge_dilation == 2
    validate_patch({"method": args.method, "edge_dilation": args.edge_dilation})


@pytest.mark.parametrize("old,new", [(2.0, 2.1), (3.5, 3.6)])
def test_single_eye_mlbw_switch_uses_effective_divergence(monkeypatch, old, new):
    from argparse import Namespace

    calls = []
    args = Namespace(method="mlbw_l2", synthetic_view="right", gpu=[0], compile=False)
    monkeypatch.setattr(desktop.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(desktop.IW3U, "create_stereo_model", lambda *a, **kw: calls.append(kw) or "new")
    assert desktop.try_switch_mlbw_model(old, new, "old", args) == "new"
    assert calls[0]["divergence"] == new * 2


def test_new_divergence_model_is_selected_before_first_changed_frame(monkeypatch):
    from argparse import Namespace

    import torch

    from iw3.desktop import quest_server

    args = desktop.create_parser().parse_args(
        ["--quest", "--bind-addr", "127.0.0.1", "--method", "mlbw_l2", "--synthetic-view", "right"]
    )
    desktop.prepare_quest_args(args)
    depth = Namespace(
        loaded=lambda: True,
        enable_ema=lambda *a, **kw: None,
        is_metric=lambda: False,
        clear_compiled_model=lambda: None,
        reset=lambda: None,
    )
    args.state = dict(
        depth_model=depth, convergence_model=None, args_lock=threading.Lock(), stop_event=None, fps_event=None
    )
    made, inferred = [], []

    class Capture:
        def __init__(self, **kwargs):
            self.count = 0

        def start(self):
            pass

        def stop(self):
            pass

        def get_frame(self):
            self.count += 1
            args.divergence = 3.0
            return torch.zeros(3, 16, 16) if self.count == 1 else None

    server = Namespace(
        start=lambda: None,
        stop=lambda: None,
        check_error=lambda: None,
        set_audio_gain=lambda gain: None,
        set_frame_data=lambda data: None,
    )
    monkeypatch.setattr(desktop, "create_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(desktop.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(desktop, "ScreenshotThreadPIL", Capture)
    monkeypatch.setattr(desktop, "get_monitor_size_list", lambda: [(16, 16)])
    monkeypatch.setattr(desktop, "test_output_size", lambda *a: (16, 32))
    monkeypatch.setattr(desktop.IW3U, "create_stereo_model", lambda *a, **kw: made.append(kw["divergence"]) or made[-1])
    monkeypatch.setattr(
        desktop.IW3U, "process_image", lambda frame, args, depth, side, **kw: inferred.append(side) or frame
    )
    monkeypatch.setattr(quest_server, "QuestServer", lambda **kw: server)
    desktop._iw3_desktop_main_once(args, init_wxapp=False)
    assert made == [2.0, 6.0] and inferred == [6.0]


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


@pytest.mark.hwaccel
@pytest.mark.skipif(not desktop.torch.cuda.is_available(), reason="requires CUDA")
def test_quest_4k_writes_clamped_eyes_without_full_size_intermediate() -> None:
    import torch

    desktop.pyav_init_cuda_primary_context(max_devices=1)
    args = desktop.create_parser().parse_args(["--quest"])
    desktop.prepare_quest_args(args)
    with torch.inference_mode():
        left = torch.full((3, 2160, 3840), 0.2, device="cuda", dtype=torch.float32)
        right = torch.full_like(left, 0.7)
        left[:, :, 0] = -0.1
        right[:, 0] = 1.1
        expected = torch.cat((left, right), dim=-1).clamp_(0, 1)
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = desktop.IW3U.postprocess_image(left, right, args)
        added = torch.cuda.max_memory_allocated() - before
        assert output.is_contiguous() and torch.equal(output, expected)
        assert added < output.nbytes * 1.5, (added, output.nbytes)


@pytest.mark.hwaccel
@pytest.mark.skipif(not desktop.torch.cuda.is_available(), reason="requires CUDA")
def test_quest_4k_defers_symmetric_warp_clamp_until_full_sbs(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch

    from iw3 import backward_warp

    class SymmetricFlow(torch.nn.Module):
        symmetric = delta_output = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat((x[:, :1] * 6, torch.zeros_like(x[:, :1])), dim=1)

    desktop.pyav_init_cuda_primary_context(max_devices=1)
    args = desktop.create_parser().parse_args(
        ["--quest", "--screenshot", "browser_video", "--method", "row_flow_v3_sym"]
    )
    desktop.prepare_quest_args(args)
    args.mapper, args.state = "none", {"convergence_model": None}
    gpu = torch.device("cuda:0")
    rng = torch.Generator(device=gpu).manual_seed(392)
    frame = torch.rand((3, 2160, 3840), device=gpu, generator=rng).mul_(2).sub_(0.5)
    depth = torch.rand((1, 392, 686), device=gpu, generator=rng)
    depth_model = SimpleNamespace(
        get_ema_buffer_size=lambda: 1,
        is_matting=lambda: False,
        infer=lambda *args, **kwargs: depth,
        minmax_normalize_chw=lambda d: d,
    )
    original = backward_warp.backward_warp
    clamp_calls, outside = [], []

    def capture(*args, **kwargs):
        clamp_output = kwargs.get("clamp_output", True)
        clamp_calls.append(clamp_output)
        warped = original(*args, **kwargs)
        if not clamp_output:
            outside.append(warped.amin().item() < 0)
        return warped

    monkeypatch.setattr(backward_warp, "backward_warp", capture)
    flow = SymmetricFlow().eval()
    monkeypatch.setenv("IW3_DISABLE_QUEST_FUSED_STEREO", "1")
    args.quest = False
    expected = desktop.IW3U.process_image(frame, args, depth_model, flow)
    args.quest = True
    result = desktop.IW3U.process_image(frame, args, depth_model, flow)
    assert result.shape == (3, 2160, 7680) and torch.equal(expected, result)
    assert clamp_calls == [True, True, False, False] and outside == [True, True]
    args.half_sbs, args.full_sbs = True, False
    desktop.IW3U.process_image(frame, args, depth_model, flow)
    assert clamp_calls[-2:] == [True, True]

    from iw3.desktop.quest_nv12 import _compatible

    if _compatible(gpu):
        monkeypatch.delenv("IW3_DISABLE_QUEST_FUSED_STEREO")
        args.half_sbs, args.full_sbs = False, True
        warp_calls = len(clamp_calls)
        result = desktop.IW3U.process_image(frame, args, depth_model, flow)
        assert len(clamp_calls) == warp_calls
        assert torch.equal(result.view(torch.int32), expected.view(torch.int32))
        args.ipd_offset = 1.0
        desktop.IW3U.process_image(frame, args, depth_model, flow)
        assert len(clamp_calls) == warp_calls + 2


@pytest.mark.hwaccel
@pytest.mark.skipif(not desktop.torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("height,width,gh,gw", [(2160, 3840, 392, 686), (3840, 2160, 686, 392)])
def test_fused_quest_warp_keeps_both_4k_eyes_bitwise(height, width, gh, gw) -> None:
    from iw3.backward_warp import backward_warp, make_grid
    from iw3.desktop.quest_nv12 import _compatible, try_warp_sbs

    torch = desktop.torch
    desktop.pyav_init_cuda_primary_context(max_devices=1)
    device = torch.device("cuda:0")
    if not _compatible(device):
        pytest.skip("requires validated CUDA runtime")
    rng = torch.Generator(device=device).manual_seed(392)
    with torch.inference_mode():
        rgb = torch.rand((1, 3, height, width), device=device, generator=rng).mul_(1.2).sub_(0.1)
        grid = make_grid(1, gw, gh, device)
        dx = torch.randn((1, 1, gh, gw), device=device, generator=rng)
        dx[..., :2].sub_(gw)
        dx[..., -2:].add_(gw)
        delta = torch.cat((dx, torch.zeros_like(dx)), dim=1)
        scale = 1 / (gw // 2 - 1)
        expected = torch.cat((backward_warp(rgb, grid, delta, scale), backward_warp(rgb, grid, -delta, scale)), dim=-1)
        actual = try_warp_sbs(rgb, grid, delta, scale)
        assert actual is not None and actual.shape == (3, height, width * 2) and actual.is_contiguous()
        assert torch.equal(expected[0].view(torch.int32), actual.view(torch.int32))


@pytest.mark.parametrize("capture", ["wc_mp", "mss", "wc_cuda"])
def test_quest_control_accepts_explicit_windows_capture_target(monkeypatch, capture: str) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    args = desktop.create_parser().parse_args(
        ["--quest", "--quest-control", "--window-name", "Browser", "--screenshot", capture]
    )
    desktop.prepare_quest_args(args)
    assert args.quest_control and args.full_sbs


@pytest.mark.parametrize("capture", ["wc_mp", "mss", "wc_cuda"])
@pytest.mark.parametrize("window_name", [None, "", "   "])
def test_quest_control_accepts_uncropped_monitor_target(monkeypatch, capture, window_name):
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    args = desktop.create_parser().parse_args(["--quest", "--quest-control", "--screenshot", capture])
    args.window_name = window_name
    desktop.prepare_quest_args(args)
    assert args.window_name is None


def test_quest_monitor_control_rejects_silently_ignored_crops(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    args = desktop.create_parser().parse_args(
        ["--quest", "--quest-control", "--screenshot", "wc_cuda", "--crop-left", "3"]
    )
    with pytest.raises(ValueError, match="crop"):
        desktop.prepare_quest_args(args)


@pytest.mark.parametrize(
    "changes",
    [
        {"quest": False},
        {"screenshot": "pil"},
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
        assert frame.parse_args().window_name is None
        assert not errors
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
    "control,binding_fails,capture_kind,monitor",
    [
        (False, False, "pil", False),
        (True, False, "mss", False),
        (True, False, "wc_mp", False),
        (True, False, "wc_cuda", False),
        (True, True, "mss", False),
        (True, False, "mss", True),
        (True, False, "wc_mp", True),
        (True, False, "wc_cuda", True),
    ],
)
@pytest.mark.parametrize("disable_cursor", [False, True])
def test_failed_startup_or_stop_still_cleans_capture_and_depth(
    monkeypatch, control: bool, binding_fails: bool, capture_kind: str, monitor: bool, disable_cursor: bool
) -> None:
    import torch

    from iw3.desktop import quest_input, quest_server, quest_windows

    cleaned = []
    bound = []
    controllers = []
    mapping_getters = []
    captures = []
    monitor_target = {
        "monitor_index": 0,
        "monitor_handle": 31,
        "monitor_device": "DISPLAY_TEST",
        "monitor_rect": (0, 0, 64, 64),
    }
    captured_mapping = {"target": "monitor"} if monitor else {"hwnd": 7}

    class InputBackend:
        def __init__(self, window_name, screenshot, crops, frame_mapping=None, monitor_index=0):
            assert control
            assert callable(frame_mapping)
            assert frame_mapping() is None
            assert monitor_index == 0
            mapping_getters.append(frame_mapping)
            self.hwnd = None if monitor else 7
            self.monitor_target = dict(monitor_target) if monitor else None
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
                assert self.kwargs["window_handle"] == (None if monitor else 7)
                assert self.kwargs["monitor_target"] == (monitor_target if monitor else None)
                assert self.kwargs["fit_window"] is True
                assert mapping_getters[0]() == captured_mapping
            else:
                assert "window_handle" not in self.kwargs and "fit_window" not in self.kwargs

        def get_frame_mapping(self):
            return captured_mapping if self.started else None

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
    args.disable_draw_cursor = disable_cursor
    if control:
        args.quest_control = True
        args.window_name = None if monitor else "Browser"
        args.screenshot = capture_kind
        if not monitor:
            args.crop_top, args.crop_left, args.crop_right, args.crop_bottom = 1, 2, 3, 4
    desktop.prepare_quest_args(args)
    args.state = dict(
        depth_model=Depth(), convergence_model=None, args_lock=threading.Lock(), stop_event=None, fps_event=None
    )
    monkeypatch.setattr(desktop, "create_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(desktop, "ScreenshotThreadPIL", Capture)
    monkeypatch.setattr(desktop, "ScreenshotProcess", Capture)
    monkeypatch.setattr(desktop, "ScreenshotThreadWCCUDA", Capture)
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
        desktop._iw3_desktop_main_once(args, init_wxapp=False)
    assert cleaned == (["clear", "reset"] if binding_fails else ["capture", "clear", "reset"])
    expected_binding = (None, capture_kind, (0, 0, 0, 0)) if monitor else ("Browser", capture_kind, (1, 2, 3, 4))
    assert bound == ([expected_binding] if control else [])
    assert len(controllers) == int(control and not binding_fails)
    assert len(captures) == int(not binding_fails)
    if captures:
        assert captures[0].kwargs["draw_cursor_enabled"] is (not control and not disable_cursor)
    if mapping_getters:
        assert mapping_getters[0]() is None
    if controllers:
        assert isinstance(controllers[0], quest_input.QuestController)
        assert isinstance(controllers[0].backend, InputBackend)
    assert args.disable_draw_cursor is disable_cursor


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
