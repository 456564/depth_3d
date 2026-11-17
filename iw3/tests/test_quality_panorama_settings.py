from argparse import Namespace

import pytest

from iw3.desktop import utils
from iw3.desktop.quest_menu import QuestMenu
from iw3.desktop.quest_restart import load_resume, write_resume
from iw3.desktop.quest_settings import FIELD_MAP, validate_patch

from .test_quest_ambient_settings import session
from .test_quest_gui_settings import frame as frame


def test_quality_preference_is_saved_and_requires_reload_without_changing_desktop_height():
    args = utils.create_parser().parse_args(["--quest", "--browser-quality", "2160"])
    utils.prepare_quest_args(args)
    assert args.browser_quality == "2160" and args.stream_height == 1080
    field = FIELD_MAP["browser_quality"]
    assert not field.immediate and "失败" in field.help and "卡顿" in field.help
    assert QuestMenu._value("browser_quality", "2160") == "4K 上限"
    saved = {"browser_quality": "2160", "quest_ambient_360": True, "quest_ambient_motion": False}
    assert load_resume(write_resume(saved, gui=True), gui=True) == saved


@pytest.mark.parametrize("value", [None, True, 2160, "4k", [], {}])
def test_invalid_quality_values_are_rejected(value):
    with pytest.raises(ValueError):
        validate_patch({"browser_quality": value})


def test_panorama_and_motion_default_and_live_updates_reach_server():
    owner = session()
    assert owner.args.quest_ambient_360 and owner.args.quest_ambient_motion
    received = []
    owner.server = Namespace(configure_ambient=lambda *args, **kwargs: received.append((args, kwargs)))
    owner.submit("set", {"quest_ambient_360": False, "quest_ambient_motion": False})
    owner.apply_live()
    assert received[-1][1] == {"panoramic": False, "motion": False}
    assert not owner.run_stop.is_set()


def test_gui_quality_and_panorama_roundtrip_and_visibility(frame):
    frame.cbo_view_mode.SetSelection(2)
    frame.update_view_mode()
    frame._apply_quest_settings({"browser_quality": "1440", "quest_ambient_360": False, "quest_ambient_motion": False})
    args = frame.parse_args()
    assert args.browser_quality == "1440"
    assert not args.quest_ambient_360 and not args.quest_ambient_motion
    assert frame.cbo_browser_quality.IsShown() and frame.chk_quest_ambient_360.IsShown()
    frame.cbo_view_mode.SetSelection(0)
    frame.update_view_mode()
    assert not frame.cbo_browser_quality.IsShown() and not frame.chk_quest_ambient_motion.IsShown()


def test_gui_quest_controls_fit_available_desktop_without_clipping(frame):
    import wx

    from iw3.desktop import gui

    frame.cbo_view_mode.SetSelection(2)
    frame.update_view_mode()
    frame._apply_quest_settings({"browser_quality": "1440", "quest_ambient_360": True, "quest_ambient_motion": True})
    gui.refresh_layouts(frame)
    workarea = wx.GetClientDisplayRect()
    assert frame.GetSize().height <= workarea.height
    assert frame.GetSize().width <= workarea.width
    for control in (frame.cbo_browser_quality, frame.chk_quest_ambient_360, frame.chk_quest_ambient_motion):
        assert control.IsShown()
        assert control.GetSize().width >= control.GetBestSize().width
        assert control.GetRect().GetRight() <= control.GetParent().GetClientSize().width
        assert control.GetRect().GetBottom() <= control.GetParent().GetClientSize().height
