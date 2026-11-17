from argparse import Namespace

import pytest

from iw3.desktop.quest_settings import FIELD_MAP, validate_patch

from .test_quest_gui_settings import frame as frame


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_auto_browser_setting_rejects_non_boolean_values(value):
    with pytest.raises(ValueError):
        validate_patch({"quest_auto_browser": value})


def test_auto_browser_is_an_explained_reload_setting():
    field = FIELD_MAP["quest_auto_browser"]
    assert field.kind == "bool" and not field.immediate
    assert validate_patch({"quest_auto_browser": True}) == {"quest_auto_browser": True}
    assert "全屏" in field.help and "桌面" in field.help and "等待" in field.help


def test_auto_browser_rejects_padding_but_keeps_desktop_crop_configuration():
    with pytest.raises(ValueError, match="补边"):
        validate_patch({"quest_auto_browser": True}, {"screenshot": "wc_cuda", "pad_mode": "16:9"})
    assert validate_patch(
        {"quest_auto_browser": True}, {"screenshot": "wc_cuda", "crop_left": 20, "window_name": "Browser"}
    ) == {"quest_auto_browser": True}


def test_gui_enables_auto_browser_by_default_only_for_quest_and_restores_setting(frame):
    assert not frame.parse_args().quest_auto_browser
    frame.cbo_view_mode.SetSelection(2)
    frame.update_view_mode()
    assert frame.parse_args().quest_auto_browser
    assert frame.chk_quest_auto_browser.IsShown()
    frame._apply_quest_settings({"quest_auto_browser": False})
    assert not frame.parse_args().quest_auto_browser
    frame._apply_quest_settings({"quest_auto_browser": True})
    assert frame.parse_args().quest_auto_browser
    frame.cbo_view_mode.SetSelection(0)
    frame.update_view_mode()
    assert not frame.chk_quest_auto_browser.IsShown()


def test_gui_fps_uses_effective_route_instead_of_configured_desktop_backend(frame):
    from iw3.desktop.gui import FPSGUI, FPSEvent, myEVT_FPS

    frame.args = Namespace(screenshot="wc_cuda")
    frame._fps_event = FPSGUI(frame)
    for kind, expected in (("browser", "NVDEC"), ("desktop", "Screenshot")):
        frame.on_fps(
            FPSEvent(
                myEVT_FPS,
                -1,
                30,
                30,
                30,
                ((1920, 1080), (1920, 1080)),
                session=frame._fps_event,
                source_kind=kind,
            )
        )
        assert f"{expected} FPS" in frame.GetStatusBar().GetStatusText()


def test_gui_auto_browser_choice_roundtrips_through_native_preset(frame, monkeypatch, tmp_path):
    from iw3.desktop import gui

    monkeypatch.setattr(gui, "CONFIG_PATH", str(tmp_path / "auto-browser.cfg"))
    frame.chk_quest_auto_browser.SetValue(False)
    frame.save_preset()
    manager = gui.persist.PersistenceManager.Get()
    manager.SetPersistenceFile(gui.CONFIG_PATH)
    gui.persistent_manager_register_all(manager, frame)
    frame.chk_quest_auto_browser.SetValue(True)
    gui.persistent_manager_restore_all(manager, {})
    gui.persistent_manager_unregister_all(manager)
    assert not frame.chk_quest_auto_browser.GetValue()
