from argparse import Namespace

import pytest

from iw3.desktop.quest_restart import load_resume, write_resume
from iw3.desktop.quest_settings import validate_patch

from .test_quest_gui_settings import frame as frame
from .test_quest_menu import click, opened
from .test_quest_menu import menu as menu


@pytest.mark.parametrize("feather", [0.0, 0.03, 0.1])
def test_feather_and_fill_settings_survive_restart_without_losing_native_types(feather: float) -> None:
    values = {"quest_ambient_fill_borders": False, "quest_ambient_feather": feather}
    assert validate_patch(values) == values
    assert load_resume(write_resume(values, gui=True), gui=True) == values


@pytest.mark.parametrize(
    "patch",
    [
        {"quest_ambient_fill_borders": 1},
        {"quest_ambient_fill_borders": "false"},
        {"quest_ambient_feather": True},
        {"quest_ambient_feather": -0.001},
        {"quest_ambient_feather": 0.101},
        {"quest_ambient_feather": float("nan")},
    ],
)
def test_fill_and_feather_reject_invalid_remote_values(patch) -> None:
    with pytest.raises(ValueError):
        validate_patch(patch)


def test_headset_fill_toggle_and_feather_adjust_submit_immediately_as_percent(menu) -> None:
    panel, snapshot, commands = menu
    snapshot["values"].update({"quest_ambient_fill_borders": True, "quest_ambient_feather": 0.03})
    opened(panel)
    click(panel, "page", 5)
    panel._render()
    click(panel, "toggle", "quest_ambient_fill_borders")
    assert commands[-1] == ("set", {"quest_ambient_fill_borders": False})
    click(panel, "adjust", ("quest_ambient_feather", 1, "x"))
    assert commands[-1] == ("set", {"quest_ambient_feather": 0.035})
    assert panel._value("quest_ambient_feather", 0.035) == "3.5%"


def test_gui_fill_and_feather_roundtrip_and_live_edits_do_not_overwrite_other_values(frame) -> None:
    import wx

    frame.cbo_view_mode.SetSelection(2)
    frame.update_view_mode()
    args = frame.parse_args()
    assert args.quest_ambient_fill_borders is True and args.quest_ambient_feather == 0.03
    submitted = []
    frame.args = Namespace(state={"quest_session": Namespace(submit=lambda *values: submitted.append(values))})
    frame._apply_quest_settings({"quest_ambient_fill_borders": False, "quest_ambient_feather": 0.035})
    assert frame.parse_args().quest_ambient_fill_borders is False
    assert frame.parse_args().quest_ambient_feather == 0.035
    assert "3.5%" in frame.lbl_quest_ambient_feather.GetLabel()
    assert submitted == []
    frame.chk_quest_ambient_fill_borders.SetValue(True)
    event = wx.CommandEvent()
    event.SetEventObject(frame.chk_quest_ambient_fill_borders)
    frame.on_quest_ambient(event)
    assert submitted == [("set", {"quest_ambient_fill_borders": True})]
    frame.sld_quest_ambient_feather.SetValue(0)
    event.SetEventObject(frame.sld_quest_ambient_feather)
    frame.on_quest_ambient(event)
    assert submitted[-1] == ("set", {"quest_ambient_feather": 0})
    assert frame.parse_args().quest_ambient_fill_borders is True
    assert frame.parse_args().quest_ambient_feather == 0
    frame.cbo_view_mode.SetSelection(0)
    frame.update_view_mode()
    assert not frame.chk_quest_ambient_fill_borders.IsShown() and not frame.sld_quest_ambient_feather.IsShown()
