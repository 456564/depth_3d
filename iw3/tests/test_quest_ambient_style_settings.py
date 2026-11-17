from argparse import Namespace

import pytest

from iw3.desktop import utils
from iw3.desktop.quest_menu import QuestMenu
from iw3.desktop.quest_restart import apply_resume, load_resume, write_resume
from iw3.desktop.quest_session import QuestSession
from iw3.desktop.quest_settings import FIELD_MAP, validate_patch

from .test_quest_ambient_settings import session
from .test_quest_gui_settings import frame as frame
from .test_quest_menu import click, opened

STYLES = ("edge", "blur", "halo", "cinema")
LABELS = ("方向柔光", "画面模糊延展", "四边光晕", "影院暗光")


def test_legacy_settings_and_resume_default_to_directional_light() -> None:
    args = utils.create_parser().parse_args([])
    assert args.quest_ambient_style == "edge"
    apply_resume(args, write_resume({"quest_ambient": True}, gui=False), gui=False)
    assert args.quest_ambient_style == "edge"
    args.state = {}
    assert QuestSession(args).snapshot()["values"]["quest_ambient_style"] == "edge"


@pytest.mark.parametrize("style", STYLES)
def test_style_cli_validation_and_restart_roundtrip(style: str) -> None:
    args = utils.create_parser().parse_args(["--quest", "--quest-ambient-style", style])
    utils.prepare_quest_args(args)
    assert args.quest_ambient_style == style
    assert validate_patch({"quest_ambient_style": style}) == {"quest_ambient_style": style}
    assert load_resume(write_resume({"quest_ambient_style": style}, gui=True), gui=True) == {
        "quest_ambient_style": style
    }
    assert FIELD_MAP["quest_ambient_style"].immediate
    assert FIELD_MAP["quest_ambient_style"].choices == STYLES


@pytest.mark.parametrize("value", [None, True, 0, "", "unknown", [], {}])
def test_invalid_style_is_rejected_at_settings_boundary(value) -> None:
    with pytest.raises(ValueError):
        validate_patch({"quest_ambient_style": value})


def test_invalid_cli_and_programmatic_style_are_rejected() -> None:
    with pytest.raises(SystemExit):
        utils.create_parser().parse_args(["--quest-ambient-style", "unknown"])
    args = utils.create_parser().parse_args(["--quest"])
    args.quest_ambient_style = "unknown"
    with pytest.raises(ValueError):
        utils.prepare_quest_args(args)


@pytest.mark.parametrize("style", STYLES)
def test_style_live_patch_and_media_reconfiguration_reach_server(style: str) -> None:
    owner = session()
    applied = []
    owner.server = Namespace(
        configure_ambient=lambda *values: applied.append(values),
        pause_media=lambda: None,
        set_reference_aspect=lambda value: None,
        configure_video=lambda *values: None,
        configure_audio=lambda *values: None,
        set_screen_curvature=lambda value: None,
        set_controller_factory=lambda value: None,
    )
    owner.submit("set", {"quest_ambient_style": style})
    owner.apply_live()
    assert applied == [(True, 0.25, 2.0, True, 0.03, style)]
    assert owner.snapshot()["values"]["quest_ambient_style"] == style
    assert not owner.run_stop.is_set()
    owner.configure(
        width=128, height=64, fps=30, bitrate=35, audio=False, audio_source="process", audio_gain=1, ambient_style=style
    )
    assert applied[-1][-1] == style
    owner.configure(width=128, height=64, fps=30, bitrate=35, audio=False, audio_source="process", audio_gain=1)
    assert applied[-1][-1] == "edge"


def test_initial_session_server_receives_selected_style(monkeypatch) -> None:
    from iw3.desktop import quest_server

    owner = session()
    owner.args.quest_ambient_style = "cinema"
    owner.args.bind_addr = "127.0.0.1"
    owner = QuestSession(owner.args)
    created = []

    def create_server(*args, **settings):
        created.append(settings)
        return Namespace(
            set_menu=lambda menu: None,
            start=lambda: None,
            stop=lambda: None,
            pause_media=lambda: None,
            set_controller_factory=lambda factory: None,
        )

    monkeypatch.setattr(quest_server, "QuestServer", create_server)
    monkeypatch.setattr(owner, "refresh_choices", lambda: None)
    monkeypatch.setattr(utils, "set_state_args", lambda *args, **settings: None)
    monkeypatch.setattr("nunif.initializer.gc_collect", lambda: None)
    owner.run(lambda args, **settings: owner.stop_event.set(), False)
    assert created[0]["ambient_style"] == "cinema"


def test_headset_style_picker_is_chinese_explained_and_immediate() -> None:
    commands = []
    owner = QuestMenu(
        lambda: {"values": {"quest_ambient_style": "edge"}, "choices": {}, "status": "running"},
        lambda command, patch: commands.append((command, patch)),
    )
    opened(owner)
    click(owner, "page", 5)
    owner._render()
    click(owner, "choices", "quest_ambient_style")
    owner._render()
    assert owner._available("quest_ambient_style") == list(STYLES)
    assert tuple(owner._value("quest_ambient_style", style) for style in STYLES) == LABELS
    assert all(label in FIELD_MAP["quest_ambient_style"].help for label in LABELS)
    click(owner, "choose", ("quest_ambient_style", "cinema"))
    assert commands[-1] == ("set", {"quest_ambient_style": "cinema"})
    assert owner._draft == {}


def test_real_hidden_gui_style_roundtrip_live_event_visibility_and_preset(frame, monkeypatch, tmp_path) -> None:
    import wx

    from iw3.desktop import gui

    assert not frame.IsShown()
    frame.cbo_view_mode.SetSelection(2)
    frame.update_view_mode()
    assert frame.parse_args().quest_ambient_style == "edge"
    assert frame.cbo_quest_ambient_style.GetStrings() == list(LABELS)
    assert frame.cbo_quest_ambient_style.IsShown()
    frame.GetSizer().Layout()
    frame.pnl_process.Layout()
    assert frame.cbo_quest_ambient_style.GetSize().width >= frame.cbo_quest_ambient_style.GetBestSize().width
    assert frame.cbo_quest_ambient_style.GetRect().GetRight() <= frame.pnl_process.GetClientSize().width
    submitted = []
    frame.args = Namespace(state={"quest_session": Namespace(submit=lambda *values: submitted.append(values))})
    frame._apply_quest_settings({"quest_ambient_style": "halo"})
    assert frame.parse_args().quest_ambient_style == "halo"
    assert submitted == []
    frame.cbo_quest_ambient_style.SetSelection(1)
    event = wx.CommandEvent(wx.EVT_CHOICE.typeId)
    event.SetEventObject(frame.cbo_quest_ambient_style)
    frame.on_quest_ambient(event)
    assert submitted == [("set", {"quest_ambient_style": "blur"})]
    monkeypatch.setattr(gui, "CONFIG_PATH", str(tmp_path / "ambient-style.cfg"))
    frame.save_preset()
    manager = gui.persist.PersistenceManager.Get()
    manager.SetPersistenceFile(gui.CONFIG_PATH)
    gui.persistent_manager_register_all(manager, frame)
    frame.cbo_quest_ambient_style.SetSelection(0)
    gui.persistent_manager_restore_all(manager, {})
    gui.persistent_manager_unregister_all(manager)
    assert frame.parse_args().quest_ambient_style == "blur"
    frame.cbo_view_mode.SetSelection(0)
    frame.update_view_mode()
    assert not frame.cbo_quest_ambient_style.IsShown()
