import json
import math
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from iw3.desktop.quest_input import QuestController, ray_to_uv
from iw3.desktop.quest_screen import QuestScreen
from iw3.desktop.quest_server import QuestServer

from .test_quest_control_server import RecordedInput, entrada
from .test_quest_input import hand
from .test_quest_screen import packet
from .test_quest_server import EncoderDouble, connect, initial_messages, read_message, send_json


def cylinder(**changes: Any) -> dict[str, Any]:
    return {
        "tipo": "cilindro",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0, 1],
        "raio": 2,
        "angulo": 2 * math.pi / 3,
        "aspecto": 2,
        **changes,
    }


def read_layers(client: Any) -> list[dict[str, Any]]:
    kind, payload = read_message(client)
    assert kind == 0x11
    return json.loads(payload)["camadas"]


@pytest.fixture
def curved_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[QuestServer, RecordedInput]]:
    from iw3.desktop import quest_server

    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    backend = RecordedInput()
    server = QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        controller_factory=lambda: QuestController(backend),
        screen_curvature=90,
    )
    server.start()
    try:
        yield server, backend
    finally:
        server.stop()


@pytest.mark.parametrize(
    "position,expected",
    [
        ([0, 0, 0], (0.5, 0.5)),
        ([1, 0, 0], (0.75, 0.5)),
        ([-1, math.pi / 3, 0], (0.25, 0)),
        ([1, -math.pi / 3, 0], (0.75, 1)),
    ],
)
def test_cylinder_aim_uses_arc_coordinates(position: list[float], expected: tuple[float, float]) -> None:
    assert ray_to_uv(hand(pos=position), cylinder()) == pytest.approx(expected)


def test_cylinder_aim_uses_axis_pose_and_normalizes_rotation() -> None:
    # 轴心在(1,2,3)，圆柱转90°；世界-X射线落在本地x=1、y=0。
    rotation = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    assert ray_to_uv(
        hand(pos=[1, 2, 2], rot=rotation), cylinder(pos=[1, 2, 3], rot=[v * 3 for v in rotation])
    ) == pytest.approx((0.75, 0.5))


@pytest.mark.parametrize("degrees", [0.1, 5, 120])
@pytest.mark.parametrize("width", [0.8, 8])
def test_aim_remains_accurate_across_curvature_and_screen_size_limits(degrees: float, width: float) -> None:
    angle = math.radians(degrees)
    radius, height = width / angle, width / 2
    # 用规范的UV→弧面点生成固定(0.1,0.7)，手柄从中点前方3m瞄准该点。
    target = [radius * math.sin(-0.4 * angle), -0.2 * height, radius * (1 - math.cos(-0.4 * angle))]
    direction = [target[0], target[1], target[2] - 3]
    length = math.hypot(*direction)
    dx, dy, dz = [value / length for value in direction]
    rotation = [dy, -dx, 0, 1 - dz]
    screen = cylinder(pos=[0, 0, radius], raio=radius, angulo=angle)
    assert ray_to_uv(hand(pos=[0, 0, 3], rot=rotation), screen) == pytest.approx((0.1, 0.7), abs=1e-7)


@pytest.mark.parametrize(
    "pointing,screen",
    [
        (hand(pos=[1.9, 0, 0]), cylinder()),
        (hand(pos=[0, 2, 0]), cylinder()),
        (hand(pos=[0, 0, -3]), cylinder()),
        (hand(rot=[1, 0, 0, 0]), cylinder()),
        (hand(rot=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), cylinder()),
        (hand(), cylinder(raio=0)),
        (hand(), cylinder(angulo=0)),
        (hand(), cylinder(aspecto=float("nan"))),
        (hand(), cylinder(raio=True)),
        (hand(), cylinder(angulo=7)),
        (hand(), cylinder(tipo="unknown")),
    ],
)
def test_cylinder_miss_backface_and_invalid_geometry_do_not_control_pc(pointing: Any, screen: Any) -> None:
    assert ray_to_uv(pointing, screen) is None


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_grab_uses_sent_curved_surface_instead_of_flat_approximation(side: str) -> None:
    screen = QuestScreen(0.5)
    # 弧面实际最右x=1.732m；手柄x=1.75m虽在平面宽度内，但在曲面外。
    screen.update(packet(side))
    screen.update(packet(side, aperto=True, pos=[1.75, 0, 0]), cylinder())
    assert not screen.active
    screen.update(packet(side))
    screen.update(packet(side, aperto=True, pos=[1, 0, 0]), cylinder())
    assert screen.active


@pytest.mark.parametrize("value", [True, -1, 121, float("nan"), float("inf"), 10**1000])
def test_invalid_curvature_does_not_start_server_or_replace_geometry(value: Any) -> None:
    with pytest.raises(ValueError, match="curvature"):
        QuestServer("127.0.0.1", 128, 64, 30, 35, screen_curvature=value)
    server = QuestServer("127.0.0.1", 128, 64, 30, 35)
    with pytest.raises(ValueError, match="curvature"):
        server.set_screen_curvature(value)
    assert not server.actual_cylinder


def test_capability_enables_identical_eye_cylinders_without_moving_surface(curved_server) -> None:
    server, backend = curved_server
    with connect(server) as client:
        assert initial_messages(client)["camadas"][0]["tipo"] == "quad"
        assert server.cylinder_supported is None and not server.actual_cylinder
        send_json(client, 0x84, {"versao": 3, "cilindro": True})
        left, right = read_layers(client)
        assert server.cylinder_supported is True and server.actual_cylinder
        assert left["tipo"] == right["tipo"] == "cilindro"
        assert left["rect"] == [0, 0, 64, 64] and right["rect"] == [64, 0, 64, 64]
        assert left["olho"] == "esq" and right["olho"] == "dir"
        assert all(left[key] == right[key] for key in ("pos", "rot", "raio", "angulo", "aspecto"))
        assert left["angulo"] == pytest.approx(math.pi / 2)
        assert left["raio"] == pytest.approx(7.2 / math.pi)
        assert left["aspecto"] == 1
        assert left["pos"] == pytest.approx([0, 0, -3 + 7.2 / math.pi])
        send_json(client, 0x83, entrada())
        read_layers(client)
        backend.wait((0.5, 0.5))
        server.set_screen_curvature(0)
        flat = read_layers(client)[0]
        assert flat["tipo"] == "quad" and flat["pos"] == [0, 0, -3]
        assert flat["tam"] == [3.6, 3.6] and not server.actual_cylinder
        server.set_screen_curvature(60)
        curved = read_layers(client)[0]
        assert curved["raio"] == pytest.approx(10.8 / math.pi)
        assert curved["pos"] == pytest.approx([0, 0, -3 + 10.8 / math.pi])


@pytest.mark.parametrize("capability", [False, 1, "true", None, {}, []])
def test_unsupported_or_malformed_capability_keeps_flat_control(curved_server, capability: Any) -> None:
    server, backend = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"versao": 3, "cilindro": capability})
        if capability is False:
            assert read_layers(client)[0]["tipo"] == "quad"
        send_json(client, 0x83, entrada())
        assert read_layers(client)[0]["tipo"] == "quad"
        backend.wait((0.5, 0.5))
        assert not server.actual_cylinder
        assert server.cylinder_supported is (False if capability is False else None)
        server.check_error()


def test_capability_and_axis_pose_are_not_replayed_after_reconnect(curved_server) -> None:
    server, _ = curved_server
    with connect(server) as first:
        initial_messages(first)
        send_json(first, 0x84, {"cilindro": True})
        read_layers(first)
    with server._condition:
        assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
    with connect(server) as second:
        left = initial_messages(second)["camadas"][0]
        assert left["tipo"] == "quad" and left["pos"] == [0, 0, -3]
        assert server.cylinder_supported is None and not server.actual_cylinder


def test_video_reconfiguration_preserves_curved_surface_midpoint(curved_server) -> None:
    server, _ = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        before = read_layers(client)[0]
        server.configure_video(256, 96, 30, 35)
        assert read_message(client)[0] == 0x01
        after = read_layers(client)[0]
        assert after["pos"] == before["pos"]
        assert after["raio"] == before["raio"] and after["aspecto"] == pytest.approx(4 / 3)
        assert after["rect"] == [0, 0, 128, 96]
        server.set_screen_curvature(0)
        flat = read_layers(client)[0]
        assert flat["pos"] == [0, 0, -3] and flat["tam"] == pytest.approx([3.6, 2.7])


def test_subthreshold_curvature_stays_flat_without_nonfinite_radius(curved_server) -> None:
    server, _ = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        read_layers(client)
        server.set_screen_curvature(1e-300)
        flat = read_layers(client)[0]
        assert flat["tipo"] == "quad" and flat["pos"] == [0, 0, -3]
        assert not server.actual_cylinder


def test_controller_hits_curved_arc_and_capability_downgrade_releases_drag(curved_server) -> None:
    server, backend = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        read_layers(client)
        data = entrada()
        data["maos"]["dir"]["pos"] = [7.2 / math.pi * math.sin(math.pi / 8), 0, 0]
        send_json(client, 0x83, data)
        read_layers(client)
        with backend.condition:
            assert backend.condition.wait_for(
                lambda: any(event == pytest.approx((0.75, 0.5)) for event in backend.events), timeout=1
            )
        data["maos"]["dir"]["gatilho"] = True
        send_json(client, 0x83, data)
        backend.wait("down")
        send_json(client, 0x84, {"cilindro": False})
        assert read_layers(client)[0]["tipo"] == "quad"
        backend.wait("up")
        assert server.cylinder_supported is False and not server.actual_cylinder


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_curved_screen_drag_scale_distance_and_recenter_keep_midpoint_semantics(curved_server, side: str) -> None:
    server, _ = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        read_layers(client)
        data = entrada()
        send_json(client, 0x83, data)
        read_layers(client)
        data["maos"][side]["aperto"] = True
        send_json(client, 0x83, data)
        threading.Event().wait(0.05)
        data["maos"][side].update(pos=[0.2, 0.1, 0], analogico=[0.9, 0.9])
        send_json(client, 0x83, data)
        curved = read_layers(client)[0]
        assert curved["raio"] > 7.2 / math.pi
        assert curved["pos"][:2] == pytest.approx([0.2, 0.1])
        surface_z = curved["pos"][2] - curved["raio"]
        assert surface_z < -3
        server.set_screen_curvature(0)
        flat = read_layers(client)[0]
        assert flat["pos"] == pytest.approx([0.2, 0.1, surface_z])
        assert flat["tam"][0] > 3.6
        data = entrada()
        data["recentrou"] = True
        send_json(client, 0x83, data)
        flat = read_layers(client)[0]
        assert flat["pos"] == [0, 0, -3] and flat["tam"] == [3.6, 3.6]


def test_stopped_server_does_not_claim_an_active_cylinder(curved_server) -> None:
    server, _ = curved_server
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        read_layers(client)
        assert server.actual_cylinder
        server.stop()
        assert not server.actual_cylinder and server.cylinder_supported is None


def test_invalid_capability_json_does_not_break_media_or_controller(curved_server) -> None:
    server, backend = curved_server
    with connect(server) as client:
        initial_messages(client)
        client.sendall(b"\x84\x00\x00\x00\x01{")
        send_json(client, 0x83, entrada())
        assert read_layers(client)[0]["tipo"] == "quad"
        backend.wait((0.5, 0.5))
        assert server.cylinder_supported is None and not server.actual_cylinder
        server.check_error()
