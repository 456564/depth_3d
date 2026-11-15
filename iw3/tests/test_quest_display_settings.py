import time
from argparse import Namespace

import pytest

from iw3.desktop import utils
from iw3.desktop.quest_menu import QuestMenu
from iw3.desktop.quest_settings import FIELD_MAP, PAGES, validate_patch

from .test_quest_menu import click, opened
from .test_quest_session import make_session


def test_curvature_is_live_and_validated_in_cli_schema() -> None:
    args = utils.create_parser().parse_args(["--quest", "--quest-screen-curvature", "60"])
    utils.prepare_quest_args(args)
    assert args.quest_screen_curvature == 60
    field = FIELD_MAP["quest_screen_curvature"]
    assert field.immediate and field.page == PAGES[4]
    assert validate_patch({"quest_screen_curvature": 0}) == {"quest_screen_curvature": 0}
    for value in (True, -1, 121, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_patch({"quest_screen_curvature": value})
        args.quest_screen_curvature = value
        with pytest.raises(ValueError):
            utils.prepare_quest_args(args)


def test_curvature_reaches_the_live_server_without_capture_frames() -> None:
    session = make_session()
    session.args.quest_screen_curvature = 0
    changes: list[float] = []
    session.server = Namespace(set_screen_curvature=changes.append)
    session.submit("set", {"quest_screen_curvature": 90})
    session.apply_live()
    assert changes == [90]
    assert session.args.quest_screen_curvature == 90
    assert session.snapshot()["values"]["quest_screen_curvature"] == 90
    assert not session.run_stop.is_set()


def test_screen_page_adjusts_curvature_and_names_the_desktop_target(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    commands = []
    menu = QuestMenu(
        lambda: {"values": {"quest_screen_curvature": 0}, "choices": {}, "status": "running"},
        lambda command, patch: commands.append((command, patch)),
    )
    opened(menu)
    click(menu, "page", 4)
    now[0] += 0.2
    menu.texture()
    click(menu, "adjust", ("quest_screen_curvature", 1, "x"))
    assert commands == [("set", {"quest_screen_curvature": 10})]
    assert "平面" in menu._value("quest_screen_curvature", 0)
    assert "60" in menu._value("quest_screen_curvature", 60)
    assert "桌面" in menu._value("window_name", None)
