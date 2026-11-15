import importlib
import math
import time
from typing import Any

import pytest


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


def packet(side: str = "dir", **changes: Any) -> dict[str, Any]:
    hands = {"esq": hand(), "dir": hand()}
    hands[side] = hand(**changes)
    return {"estado": "focado", "maos": hands, "recentrou": False}


def screen(aspect: float = 0.5) -> Any:
    return importlib.import_module("iw3.desktop.quest_screen").QuestScreen(aspect)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def test_default_geometry_is_a_fresh_snapshot() -> None:
    state = screen()
    assert state.quad == {"pos": [0, 0, -3], "rot": [0, 0, 0, 1], "tam": [3.6, 1.8]}
    snapshot = state.quad
    for value in snapshot.values():
        value[0] = 999
    assert state.quad == {"pos": [0, 0, -3], "rot": [0, 0, 0, 1], "tam": [3.6, 1.8]}


@pytest.mark.parametrize("aspect", [0, -1, True, float("nan"), float("inf"), 10**1000, 3e307])
def test_invalid_aspect_cannot_create_or_scale_to_nonfinite_geometry(aspect: Any) -> None:
    with pytest.raises(ValueError, match="aspect"):
        screen(aspect)


def test_active_distinguishes_grabbing_from_suppressed_release_frame() -> None:
    state = screen()
    assert not state.active
    state.update(packet())
    state.update(packet(aperto=True))
    assert state.active
    assert state.update(packet())
    assert not state.active


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_each_hand_requires_release_and_aim_then_moves_without_a_press_jump(side: str) -> None:
    state = screen()
    assert state.update(packet(side, aperto=True))
    assert state.update(packet(side, aperto=True, pos=[0.5, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]
    assert state.update(packet())
    assert not state.update(packet())
    assert state.update(packet(side, aperto=True, pos=[0.5, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]
    state.update(packet(side, aperto=True, pos=[0.7, 0.3, -0.4]))
    assert state.quad["pos"] == pytest.approx([0.2, 0.3, -3.4])
    assert state.quad["rot"] == [0, 0, 0, 1]
    assert state.update(packet())
    assert not state.update(packet())


def test_missed_grip_never_grabs_later_without_another_release() -> None:
    state = screen()
    state.update(packet())
    assert state.update(packet(aperto=True, pos=[4, 0, 0]))
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[0.4, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]


@pytest.mark.parametrize("first,second", [("esq", "dir"), ("dir", "esq")])
def test_first_hand_keeps_control_until_both_grips_release(first: str, second: str) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(first, aperto=True))
    both = packet(first, aperto=True, pos=[0.2, 0, 0])
    both["maos"][second] = hand(aperto=True, pos=[-0.4, 0, 0], analogico=[1, 1])
    state.update(both)
    assert state.quad["pos"] == pytest.approx([0.2, 0, -3])
    state.update(packet(second, aperto=True))
    state.update(packet(second, aperto=True, pos=[0.4, 0, 0]))
    assert state.quad["pos"] == pytest.approx([0.2, 0, -3])
    state.update(packet())
    state.update(packet(second, aperto=True))
    state.update(packet(second, aperto=True, pos=[0.3, 0, 0]))
    assert state.quad["pos"] == pytest.approx([0.5, 0, -3])


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_same_hand_stick_changes_distance_and_uniform_scale(side: str, clock: list[float]) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(side, aperto=True, analogico=[1, 1]))
    assert state.quad["pos"] == [0, 0, -3]
    clock[0] += 0.1
    state.update(packet(side, aperto=True, analogico=[1, 1]))
    assert state.quad["pos"] == pytest.approx([0, 0, -3.15])
    assert state.quad["tam"] == pytest.approx([3.8998334436298504, 1.9499167218149252])
    clock[0] += 0.1
    state.update(packet(side, aperto=True, analogico=[-1, -1]))
    assert state.quad["pos"] == pytest.approx([0, 0, -3])
    assert state.quad["tam"] == pytest.approx([3.6, 1.8])


@pytest.mark.parametrize("axis", [0, 1])
def test_stick_deadzone_is_remapped_linearly(axis: int, clock: list[float]) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    clock[0] += 0.1
    state.update(packet(aperto=True, analogico=[0.2, -0.2]))
    assert state.quad["pos"] == [0, 0, -3]
    assert state.quad["tam"] == [3.6, 1.8]
    stick = [0.0, 0.0]
    stick[axis] = 0.6
    clock[0] += 0.1
    state.update(packet(aperto=True, analogico=stick))
    if axis == 0:
        assert state.quad["tam"] == pytest.approx([3.746918786735183, 1.8734593933675915])
    else:
        assert state.quad["pos"] == pytest.approx([0, 0, -3.075])


def test_long_valid_frame_integrates_at_most_a_tenth_second(clock: list[float]) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    clock[0] += 0.4
    state.update(packet(aperto=True, analogico=[0, 1]))
    assert state.quad["pos"] == pytest.approx([0, 0, -3.15])


def test_position_and_scale_limits_reverse_immediately_after_clamping(clock: list[float]) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    for i in range(1, 101):
        clock[0] += 0.1
        state.update(packet(aperto=True, pos=[i * 0.5, i * 0.5, 0], analogico=[1, 1]))
    assert state.quad["pos"] == pytest.approx([5, 5, -10])
    assert state.quad["tam"] == [8, 4]
    clock[0] += 0.1
    state.update(packet(aperto=True, pos=[49.8, 49.8, 0], analogico=[-1, -1]))
    assert state.quad["pos"] == pytest.approx([4.8, 4.8, -9.85])
    for _ in range(100):
        clock[0] += 0.1
        state.update(packet(aperto=True, pos=[49.8, 49.8, 0], analogico=[-1, -1]))
    assert state.quad["pos"][2] == -1
    assert state.quad["tam"] == [0.8, 0.4]


def test_recenter_anchors_bounds_and_stick_to_the_horizontal_head_frame(clock: list[float]) -> None:
    state = screen()
    yaw = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    state.recenter(([-1, 1, 4], yaw))
    assert state.quad["pos"] == pytest.approx([-1, 1, 4])
    assert state.quad["rot"] == pytest.approx(yaw)
    data = packet(aperto=True, pos=[2, 1, 4], rot=yaw)
    state.update(data)
    state.update(packet())
    state.update(data)
    clock[0] += 0.1
    state.update(packet(aperto=True, pos=[2, 1, 3.5], rot=yaw, analogico=[0, 1]))
    assert state.quad["pos"] == pytest.approx([-1.15, 1, 3.5])
    for i in range(1, 21):
        clock[0] += 0.1
        state.update(packet(aperto=True, pos=[2, 1, 3.5 - i * 0.5], rot=yaw))
    assert state.quad["pos"] == pytest.approx([-1.15, 1, -1])
    state.recenter(([-1, 1, 4], yaw))
    assert state.quad["pos"] == pytest.approx([-1, 1, 4])
    assert state.quad["tam"] == [3.6, 1.8]
    state.update(packet(aperto=True, pos=[2, 1, 4], rot=yaw))
    state.update(packet(aperto=True, pos=[2, 1, 3.5], rot=yaw))
    assert state.quad["pos"] == pytest.approx([-1, 1, 4])


def test_reset_cancels_and_disarms_without_changing_geometry() -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[0.4, 0, 0]))
    before = state.quad
    state.reset()
    state.update(packet(aperto=True, pos=[0.8, 0, 0]))
    assert state.quad == before


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"estado": "visivel"},
        {**packet(), "estado": "oculto"},
        {**packet(), "maos": []},
        {**packet(), "maos": {"dir": hand()}},
        packet(aperto=1),
        packet(aperto=True, pos=None),
        packet(aperto=True, rot=None),
        packet(aperto=True, rot=[0, 0, 0, 0]),
        packet(aperto=True, rot=[0, 0, 0, float("inf")]),
        packet(aperto=True, pos=[float("nan"), 0, 0]),
        packet(aperto=True, pos=[1e308, 0, 0]),
        packet(aperto=True, pos=[10**1000, 0, 0]),
        packet(aperto=True, pos=[True, 0, 0]),
        packet(aperto=True, analogico=None),
        packet(aperto=True, analogico=[0]),
        packet(aperto=True, analogico=[0, 1.01]),
        packet(aperto=True, analogico=[False, 0]),
        packet(aperto=True, analogico=[float("nan"), 0]),
        packet(aperto=True, analogico=[10**1000, 0]),
    ],
)
def test_invalid_packet_disarms_and_cannot_resume_a_held_drag(invalid: Any) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    assert state.update(invalid)
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[0.4, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]
    state.update(packet())
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[0.3, 0, 0]))
    assert state.quad["pos"] == pytest.approx([0.3, 0, -3])


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_untracked_inactive_hand_does_not_block_release_or_other_hand(side: str) -> None:
    state = screen()
    other = "dir" if side == "esq" else "esq"
    data = packet()
    data["maos"][other] = hand(pos=None, rot=None, analogico=None)
    assert not state.update(data)
    data["maos"][side] = hand(aperto=True)
    assert state.update(data)
    data["maos"][side] = hand(aperto=True, pos=[0.3, 0, 0])
    state.update(data)
    assert state.quad["pos"] == pytest.approx([0.3, 0, -3])


def test_tracking_jump_cancels_instead_of_launching_the_screen() -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[1.01, 0, 0]))
    state.update(packet(aperto=True, pos=[0.3, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]


def test_stale_packet_requires_release_before_drag_can_restart(clock: list[float]) -> None:
    state = screen()
    state.update(packet())
    state.update(packet(aperto=True))
    clock[0] += 0.6
    assert state.update(packet(aperto=True, analogico=[1, 1]))
    state.update(packet(aperto=True, pos=[0.3, 0, 0]))
    assert state.quad["pos"] == [0, 0, -3]
    assert state.quad["tam"] == [3.6, 1.8]
    state.update(packet())
    state.update(packet(aperto=True))
    state.update(packet(aperto=True, pos=[0.4, 0, 0]))
    assert state.quad["pos"] == pytest.approx([0.4, 0, -3])
