import importlib
import math
from typing import Any

import pytest


def quad(**changes: Any) -> dict[str, Any]:
    return {"pos": [0, 0, -3], "rot": [0, 0, 0, 1], "tam": [4, 2], **changes}


def hand(**changes: Any) -> dict[str, Any]:
    return {
        "pos": [0, 0, 0],
        "rot": [0, 0, 0, 1],
        "gatilho": False,
        "aperto": False,
        "a": False,
        "b": False,
        "analogico": [0, 0],
        **changes,
    }


def entrada(**changes: Any) -> dict[str, Any]:
    return {
        "estado": "focado",
        "cabeca": {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
        "maos": {"esq": hand(), "dir": hand(**changes)},
        "recentrou": False,
    }


class Backend:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.allowed = True
        self.down_allowed = True
        self.key_allowed = True
        self.down_error = False
        self.key_releases = 0

    def move(self, u: float, v: float) -> bool:
        self.events.append(("move", u, v))
        return self.allowed

    def mouse_down(self) -> bool:
        self.events.append("down")
        if self.down_error:
            raise OSError("partial input failure")
        return self.down_allowed

    def mouse_up(self) -> None:
        self.events.append("up")

    def tap_key(self, direction: str) -> bool:
        self.events.append("key_" + direction)
        return self.key_allowed

    def release_keys(self) -> None:
        self.key_releases += 1


def controller(backend: Backend) -> Any:
    return importlib.import_module("iw3.desktop.quest_input").QuestController(backend)


def test_valid_pc_move_returns_a_native_pointer_on_the_same_surface() -> None:
    backend = Backend()
    laser = controller(backend).update(entrada(pos=[1, 0.5, 0]), quad(rot=[0, 0, 0, 7]))
    assert backend.events == [("move", 0.75, 0.25)]
    assert laser == {"tipo": "laser", "mao": "dir", "ponto": [1.0, 0.5, -3.0]}


def test_native_pointer_uses_the_world_point_of_a_rotated_surface() -> None:
    rotation = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    laser = controller(Backend()).update(entrada(pos=[0, 0.5, -1], rot=rotation), quad(pos=[-3, 0, 0], rot=rotation))
    assert laser["ponto"] == pytest.approx([-3, 0.5, -1])


@pytest.mark.parametrize("blocked", ["focus", "miss", "backend"])
def test_invalid_pc_move_does_not_return_a_native_pointer(blocked: str) -> None:
    backend = Backend()
    data = entrada()
    if blocked == "focus":
        data["estado"] = "visivel"
    elif blocked == "miss":
        data["maos"]["dir"]["pos"] = [10, 0, 0]
    else:
        backend.allowed = False
    assert controller(backend).update(data, quad()) is None


def test_reset_cleans_pending_keys_even_without_a_mouse_press() -> None:
    backend = Backend()
    control = controller(backend)
    control.reset()
    assert backend.key_releases == 1


def test_exit_fullscreen_releases_owned_input_then_targets_capture_center() -> None:
    backend = Backend()
    control = controller(backend)
    control.update(entrada(), quad())
    control.update(entrada(gatilho=True), quad())
    backend.events.clear()
    assert control.exit_fullscreen()
    assert backend.events == ["up", ("move", 0.5, 0.5), "key_escape"]
    assert control._last_valid is None and not control._trigger_armed and not control._stick_armed


@pytest.mark.parametrize("move_allowed,key_allowed", [(False, True), (True, False)])
def test_exit_fullscreen_does_not_bypass_target_rejection(move_allowed: bool, key_allowed: bool) -> None:
    backend = Backend()
    backend.allowed, backend.key_allowed = move_allowed, key_allowed
    assert not controller(backend).exit_fullscreen()
    assert backend.events == [("move", 0.5, 0.5)] + (["key_escape"] if move_allowed else [])


def test_expiry_reports_remaining_wait_without_extending_deadline(monkeypatch) -> None:
    module = importlib.import_module("iw3.desktop.quest_input")
    now = [1.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    backend = Backend()
    control = module.QuestController(backend)
    control.update(entrada(), quad())
    now[0] = 1.4
    assert control.expire() == pytest.approx(0.1)
    now[0] = 1.5
    assert control.expire() == 0.5
    assert backend.key_releases == 1


def project(pointing: Any, screen: Any) -> Any:
    return importlib.import_module("iw3.desktop.quest_input").ray_to_uv(pointing, screen)


@pytest.mark.parametrize(
    "position,expected",
    [([0, 0, 0], (0.5, 0.5)), ([-2, 1, 0], (0, 0)), ([2, -1, 0], (1, 1)), ([1, 0.5, 0], (0.75, 0.25))],
)
def test_quad_uv_uses_single_eye_size_and_downward_v(position: list[float], expected: tuple[float, float]) -> None:
    assert project(hand(pos=position), quad()) == pytest.approx(expected)


def test_aim_orientation_sets_ray_instead_of_head_direction() -> None:
    assert project(hand(rot=[0, -math.sin(math.pi / 8), 0, math.cos(math.pi / 8)]), quad(tam=[12, 2])) == pytest.approx(
        (0.75, 0.5)
    )


def test_rotated_quad_uses_local_axes() -> None:
    # 平面绕 Y 轴转 90 度后，世界 -Z 是平面向右，瞄准方向为世界 -X。
    rotation = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    assert project(hand(pos=[0, 0.5, -1], rot=rotation), quad(pos=[-3, 0, 0], rot=rotation)) == pytest.approx(
        (0.75, 0.25)
    )


def test_normalizes_both_quaternions() -> None:
    assert project(hand(rot=[0, 0, 0, 2]), quad(rot=[0, 0, 0, 7])) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(
    "changes",
    [
        {"pos": [0, 0, -4]},
        {"pos": [2.01, 0, 0]},
        {"pos": [0, -1.01, 0]},
        {"rot": [0, 1, 0, 0]},
        {"rot": [0, math.sqrt(0.5), 0, math.sqrt(0.5)]},
        {"rot": [0, 0, 0, 0]},
        {"rot": [0, 0, 0, 1e-20]},
        {"pos": None},
        {"rot": [0, 0, 1]},
        {"pos": [True, 0, 0]},
        {"pos": [float("nan"), 0, 0]},
        {"rot": [0, 0, 0, float("inf")]},
        {"pos": [10**1000, 0, 0]},
    ],
)
def test_rejects_invalid_or_missed_aim(changes: dict[str, Any]) -> None:
    assert project(hand(**changes), quad()) is None


@pytest.mark.parametrize(
    "screen",
    [
        None,
        {},
        quad(tam=[0, 2]),
        quad(tam=[4, -1]),
        quad(tam=[4, float("nan")]),
        quad(rot=[0, 0, 0, 0]),
        quad(pos=[0, 0, 10**1000]),
        quad(tam=[True, 2]),
    ],
)
def test_rejects_invalid_quad(screen: Any) -> None:
    assert project(hand(), screen) is None


def test_trigger_waits_for_release_then_drags_without_repeated_down() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(gatilho=True), quad())
    assert backend.events == [("move", 0.5, 0.5)]
    state.update(entrada(), quad())
    state.update(entrada(gatilho=True), quad())
    state.update(entrada(gatilho=True, pos=[1, 0, 0]), quad())
    state.update(entrada(), quad())
    assert backend.events == [
        ("move", 0.5, 0.5),
        ("move", 0.5, 0.5),
        ("move", 0.5, 0.5),
        "down",
        ("move", 0.75, 0.5),
        ("move", 0.5, 0.5),
        "up",
    ]
    assert backend.events.count("down") == 1
    assert backend.events.count("up") == 1
    state.reset()
    assert backend.events.count("up") == 1


def test_a_clicks_once_on_press_without_drag_or_repeat() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(a=True), quad())
    assert "down" not in backend.events
    state.update(entrada(), quad())
    backend.events.clear()
    state.update(entrada(a=True, pos=[1, 0, 0]), quad())
    assert backend.events == [("move", 0.75, 0.5), "down", "up"]
    state.update(entrada(a=True), quad())
    state.update(entrada(), quad())
    state.update(entrada(a=True), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up", "down", "up"]


@pytest.mark.parametrize("fault", ["focus", "miss", "backend", "timeout"])
def test_a_requires_release_after_losing_valid_input(monkeypatch, fault: str) -> None:
    module = importlib.import_module("iw3.desktop.quest_input")
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    data = entrada(a=True)
    if fault == "focus":
        data["estado"] = "visivel"
    elif fault == "miss":
        data["maos"]["dir"]["pos"] = [100, 0, 0]
    elif fault == "backend":
        backend.allowed = False
    else:
        now[0] += 0.5
    state.update(data, quad())
    backend.allowed = True
    state.update(entrada(a=True), quad())
    assert "down" not in backend.events
    state.update(entrada(), quad())
    state.update(entrada(a=True), quad())
    assert backend.events[-2:] == ["down", "up"]


def test_a_is_consumed_during_trigger_drag() -> None:
    backend = Backend()
    state = controller(backend)
    for changes in ({}, {"gatilho": True}, {"gatilho": True, "a": True}, {"a": True}, {"a": True}):
        state.update(entrada(**changes), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]
    state.update(entrada(), quad())
    state.update(entrada(a=True), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up", "down", "up"]


@pytest.mark.parametrize("value", [1, None, "true"])
def test_invalid_a_disarms_click(value: Any) -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    assert state.update(entrada(a=value), quad()) is None
    state.update(entrada(a=True), quad())
    assert "down" not in backend.events


def test_a_partial_down_error_keeps_ownership_for_cleanup() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    backend.down_error = True
    with pytest.raises(OSError, match="partial input failure"):
        state.update(entrada(a=True), quad())
    state.reset()
    assert backend.events[-2:] == ["down", "up"]


def test_stick_requires_center_and_one_deflection_per_video() -> None:
    backend = Backend()
    state = controller(backend)
    for y in [1, -1, 0, 0.69, 0.7, 1, -1, -0.31, -0.3, -0.7, -1, 0, 1]:
        state.update(entrada(analogico=[0, y]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["key_up", "key_down", "key_up"]


def test_trigger_disarms_stick_until_center_after_release() -> None:
    backend = Backend()
    state = controller(backend)
    for trigger, y in [(False, 0), (True, 0), (True, 1), (False, 1), (False, -1), (False, 0), (False, -1)]:
        state.update(entrada(gatilho=trigger, analogico=[0, y]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up", "key_down"]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"estado": "visivel"},
        {**entrada(), "estado": "oculto"},
        {**entrada(), "maos": []},
        {**entrada(), "maos": {"dir": None}},
        entrada(pos=None),
        entrada(pos=[3, 0, 0]),
        entrada(gatilho=1),
        entrada(analogico=[0]),
        entrada(analogico=[0, 1.1]),
        entrada(analogico=[False, 0]),
        entrada(analogico=[0, float("nan")]),
        entrada(analogico=[0, 10**1000]),
    ],
)
def test_invalid_input_releases_and_requires_fresh_release(invalid: Any) -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    state.update(entrada(gatilho=True), quad())
    state.update(invalid, quad())
    state.update(entrada(gatilho=True, analogico=[0, 1]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]


def test_failed_move_releases_and_blocks_click_and_key() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    state.update(entrada(gatilho=True), quad())
    backend.allowed = False
    state.update(entrada(gatilho=True, analogico=[0, 1]), quad())
    backend.allowed = True
    state.update(entrada(gatilho=True, analogico=[0, 1]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]


def test_failed_mouse_down_disarms_and_attempts_release() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    backend.down_allowed = False
    state.update(entrada(gatilho=True), quad())
    state.update(entrada(gatilho=True), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]


def test_partial_backend_error_propagates_and_reset_can_release() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    backend.down_error = True
    with pytest.raises(OSError, match="partial input failure"):
        state.update(entrada(gatilho=True), quad())
    state.reset()
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]


def test_failed_key_disarms_until_another_center() -> None:
    backend = Backend()
    state = controller(backend)
    state.update(entrada(), quad())
    backend.key_allowed = False
    state.update(entrada(analogico=[0, -1]), quad())
    backend.key_allowed = True
    state.update(entrada(analogico=[0, 1]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["key_down"]
    state.update(entrada(), quad())
    state.update(entrada(analogico=[0, 1]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["key_down", "key_up"]


def test_late_held_packet_cannot_continue_expired_drag(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("iw3.desktop.quest_input")
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    backend = Backend()
    state = module.QuestController(backend)
    state.update(entrada(), quad())
    state.update(entrada(gatilho=True), quad())
    clock[0] = 100.6
    state.update(entrada(gatilho=True), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]


def test_expiry_after_500ms_releases_and_disarms(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("iw3.desktop.quest_input")
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    backend = Backend()
    state = module.QuestController(backend)
    state.expire()
    state.update(entrada(), quad())
    state.update(entrada(gatilho=True), quad())
    clock[0] = 100.499
    state.expire()
    assert backend.events[-1] == "down"
    clock[0] = 100.5
    state.expire()
    state.expire()
    state.update(entrada(gatilho=True, analogico=[0, 1]), quad())
    assert [event for event in backend.events if isinstance(event, str)] == ["down", "up"]
