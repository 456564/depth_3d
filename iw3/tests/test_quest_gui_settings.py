import sys
from argparse import Namespace

import pytest


@pytest.fixture
def frame(monkeypatch):
    import wx

    from iw3.desktop import gui

    monkeypatch.setattr(sys, "argv", ["iw3-desktop"])
    monkeypatch.setattr(gui.MainFrame, "load_preset", lambda self: None)
    app = wx.GetApp() or wx.App()
    window = gui.MainFrame()
    yield window
    window.Destroy()
    app.ProcessPendingEvents()


def test_applied_snapshot_updates_startup_and_adjustment_without_event_feedback(frame) -> None:
    import wx

    from iw3.desktop.gui import FPSGUI

    submitted = []
    sink = FPSGUI(frame)
    frame._fps_event = sink
    frame.args = Namespace(state={"fps_event": sink, "quest_session": Namespace(submit=lambda *v: submitted.append(v))})
    values = {
        "divergence": 24.125,
        "convergence": -5.125,
        "foreground_scale": 1.25,
        "edge_dilation": [18, 0],
        "quest_audio_gain": 1.7,
        "quest_no_audio": True,
        "depth_model": "Any_V2_N_B",
        "stream_fps": 42,
        "quest_bitrate": 60,
        "window_name": "Browser",
        "crop_top": 3,
        "pad_mode": "16:9",
    }
    sink.settings_applied(values)
    values["divergence"] = 1.0
    wx.GetApp().ProcessPendingEvents()
    wx.GetApp().ProcessPendingEvents()
    assert frame.cbo_divergence.GetValue() == "24.125"
    assert frame.sld_adj_divergence.GetValue() == 24.125
    assert frame.sld_adj_convergence.GetValue() == -5.125
    assert frame.sld_adj_edge_dilation.GetValue() == 18
    assert frame.cbo_adj_edge_dilation_y.GetValue() == "0"
    assert frame.cbo_edge_dilation_y.GetValue() == "0"
    assert frame.cbo_depth_model.GetValue() == "Any_V2_N_B"
    assert not frame.chk_small_model_only.IsChecked()
    assert not frame.chk_quest_audio.IsChecked()
    assert frame.sld_quest_audio_gain.GetValue() == 170
    assert frame.cbo_stream_fps.GetValue() == "42"
    assert frame.cbo_quest_bitrate.GetValue() == "60"
    assert frame.cbo_window_name.GetValue() == "Browser"
    assert frame.txt_crop_top.GetValue() == 3 and frame.chk_pad_16_9.IsChecked()
    assert submitted == []


def test_old_session_snapshot_cannot_write_new_session_controls(frame) -> None:
    import wx

    from iw3.desktop.gui import FPSGUI

    old, current = FPSGUI(frame), FPSGUI(frame)
    frame._fps_event = current
    frame.args = Namespace(state={"fps_event": current})
    old.settings_applied({"divergence": 9.0})
    current.settings_applied({"divergence": 4.25})
    wx.GetApp().ProcessPendingEvents()
    assert frame.cbo_divergence.GetValue() == "4.25"
    old.settings_applied({"divergence": 8.0})
    wx.GetApp().ProcessPendingEvents()
    assert frame.cbo_divergence.GetValue() == "4.25"


def test_gui_live_adjustments_submit_to_session_owner(frame) -> None:
    import wx

    submitted = []
    frame.args = Namespace(state={"quest_session": Namespace(submit=lambda *v: submitted.append(v))})
    frame.sld_adj_divergence.SetValue(3.25)
    frame.cbo_adj_edge_dilation_y.ChangeValue("0")
    event = wx.CommandEvent()
    event.SetEventObject(frame.sld_adj_divergence)
    frame.update_args_adjustment(event)
    event.SetEventObject(frame.cbo_adj_edge_dilation_y)
    frame.update_args_adjustment(event)
    frame.sld_quest_audio_gain.SetValue(180)
    frame.on_quest_audio_gain(None)
    assert submitted == [
        ("set", {"divergence": 3.25}),
        ("set", {"edge_dilation": [0, 0]}),
        ("set", {"quest_audio_gain": 1.8}),
    ]
    assert not hasattr(frame.args, "divergence")


def test_gui_edit_does_not_overwrite_remote_values_before_snapshot_arrives(frame) -> None:
    import wx

    from iw3.desktop.gui import FPSGUI

    submitted = []
    sink = FPSGUI(frame)
    frame._fps_event = sink
    frame.args = Namespace(state={"quest_session": Namespace(submit=lambda *v: submitted.append(v))})
    frame.sld_adj_divergence.SetValue(1.0)
    sink.settings_applied({"divergence": 8.0})
    frame.sld_adj_foreground_scale.SetValue(0.2)
    event = wx.CommandEvent()
    event.SetEventObject(frame.sld_adj_foreground_scale)
    frame.update_args_adjustment(event)
    assert submitted == [("set", {"foreground_scale": 0.2})]


def test_exit_uses_final_owner_values_when_snapshot_event_has_not_arrived(frame, monkeypatch) -> None:
    from iw3.desktop import gui

    monkeypatch.setattr(gui, "gc_collect", lambda: None)
    args = Namespace(
        divergence=7.5125,
        convergence=0.3,
        foreground_scale=1.0,
        edge_dilation=[4, 0],
        quest=True,
        depth_model="Any_V2_S",
        gpu=[0],
        resolution=None,
        state={"depth_model": None},
    )
    frame.args = args
    frame.on_exit_worker(Namespace(get=lambda: args))
    assert frame.cbo_divergence.GetValue() == "7.5125"
    assert frame.cbo_edge_dilation_y.GetValue() == "0"
    assert frame.depth_model is None and frame.args is None


def test_gui_resume_starts_quest_worker_with_restored_settings_and_no_old_model_cache(frame, monkeypatch) -> None:
    from iw3.desktop import gui

    def setup(args, **kwargs):
        args.state = {"depth_model": object(), "fps_event": kwargs["fps_event"]}

    monkeypatch.setattr(gui, "set_state_args", setup)
    monkeypatch.setattr(gui.BaseDepthModel, "has_model_checkpoint_file", lambda name: True)
    workers = []
    monkeypatch.setattr(gui, "startWorker", lambda callback, worker, wargs: workers.append(wargs[0]))
    frame.resume_quest({"gpu": [0], "stream_fps": 42, "quest_bitrate": 50, "window_name": "Browser"})
    assert len(workers) == 1 and workers[0].quest
    assert workers[0].stream_fps == 42 and workers[0].quest_bitrate == 50
    assert workers[0].window_name == "Browser"
    assert frame.depth_model is None
    assert frame.args.state["depth_model"] is not None


def test_gui_restart_waits_for_worker_cleanup_then_closes_mainloop(frame, monkeypatch) -> None:
    import wx

    from iw3.desktop import gui

    sink = gui.FPSGUI(frame)
    frame._fps_event = sink
    frame.processing = True
    closed = []
    monkeypatch.setattr(frame, "Close", lambda: closed.append(True))
    monkeypatch.setattr(frame, "save_preset", lambda: None)
    monkeypatch.setattr(gui, "gc_collect", lambda: None)
    args = Namespace(
        quest=True,
        depth_model="Any_V2_S",
        gpu=[0],
        resolution=None,
        state={"depth_model": None, "quest_restart_path": "generated-resume-path"},
    )
    frame.args = args
    sink.restart_requested("generated-resume-path")
    wx.GetApp().ProcessPendingEvents()
    assert closed == []
    frame.on_exit_worker(Namespace(get=lambda: args))
    assert closed == [True]
    assert getattr(wx.GetApp(), "quest_restart_path") == "generated-resume-path"


def test_gui_entry_releases_single_instance_lock_before_process_replacement(monkeypatch) -> None:
    from iw3.desktop import gui, quest_restart

    resume = quest_restart.write_resume({"gpu": [0], "stream_fps": 42}, gui=True)
    monkeypatch.setattr(sys, "argv", ["iw3.desktop.gui", "--quest-resume", str(resume)])
    created = []

    class App:
        def __init__(self, resume_values):
            self.values = resume_values
            self.instance = object()
            self.quest_restart_path = str(resume)
            created.append(self)

        def MainLoop(self):
            pass

    replaced = []

    def replace(path, gui):
        assert not hasattr(created[0], "instance")
        replaced.append((path, gui))

    monkeypatch.setattr(gui, "IW3DesktopApp", App)
    monkeypatch.setattr(gui, "restart_process", replace)
    gui.main()
    assert created[0].values == {"gpu": [0], "stream_fps": 42}
    assert replaced == [(str(resume), True)]
