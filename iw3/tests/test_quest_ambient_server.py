import json
import math
import socket
import struct
import threading
import zlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from iw3.desktop import quest_server
from iw3.desktop.quest_input import QuestController

from .test_quest_control_server import RecordedInput, entrada
from .test_quest_server import EncoderDouble, connect, initial_messages, read_message, send_json


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, ambient_enabled=True
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def read_layers(client: socket.socket) -> list[dict[str, Any]]:
    kind, payload = read_message(client)
    assert kind == 0x11
    return json.loads(payload)["camadas"]


def main_video(layers: list[dict[str, Any]]) -> dict[str, Any]:
    return next(layer for layer in layers if layer.get("tex") == "video" and layer.get("olho") == "esq")


def publish(server: Any, client: socket.socket) -> tuple[bytes, list[dict[str, Any]]]:
    with server._condition:
        assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
    server.set_ambient_frame(torch.ones(3, 64, 64))
    server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
    assert read_message(client)[0] == 0x02
    kind, payload = read_message(client)
    assert kind == 0x10
    return payload, read_layers(client)


@pytest.mark.parametrize(
    "changes",
    [
        {"ambient_enabled": 1},
        {"ambient_brightness": True},
        {"ambient_brightness": -0.1},
        {"ambient_brightness": 1.1},
        {"ambient_brightness": float("nan")},
        {"ambient_spread": True},
        {"ambient_spread": 1.1},
        {"ambient_spread": 4.1},
        {"ambient_spread": float("inf")},
    ],
)
def test_reject_invalid_ambient_configuration(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="ambient"):
        quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35, **changes)
    server = quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35)
    settings = {"ambient_enabled": False, "ambient_brightness": 0.25, "ambient_spread": 2.0, **changes}
    with pytest.raises(ValueError, match="ambient"):
        server.configure_ambient(*settings.values())


def test_video_precedes_texture_and_feather_layer(server: Any) -> None:
    with connect(server) as client:
        assert len(initial_messages(client)["camadas"]) == 2
        payload, layers = publish(server, client)
        name_length = payload[0]
        assert payload[1 : 1 + name_length] == b"iw3_ambient"
        assert struct.unpack("!HHB", payload[1 + name_length : 6 + name_length]) == (320, 180, 1)
        assert len(zlib.decompress(payload[6 + name_length :])) == 320 * 180 * 4
        left, right, background = layers
        assert background["tex"] == "iw3_ambient" and background["olho"] == "ambos"
        assert background["alpha"] is True and background["rect"] == [0, 0, 320, 180]
        assert left["tex"] == right["tex"] == "video" and not left["alpha"]
        assert background["tipo"] == "quad" and background["tam"] == [7.2, 7.2]
        assert background["rot"] == left["rot"]
        assert background["pos"] == left["pos"] == [0, 0, -3]
        server.check_error()


def test_same_texture_id_updates_without_resending_layers(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        first, _ = publish(server, client)
        background = server._ambient_background
        background.reset()
        server.set_ambient_frame(torch.zeros(3, 64, 64))
        server.set_frame_data((torch.ones(3, 64, 128), 2.0))
        assert read_message(client)[0] == 0x02
        kind, second = read_message(client)
        assert kind == 0x10 and second[:18] == first[:18]
        assert second != first
        client.settimeout(0.1)
        with pytest.raises(socket.timeout):
            read_message(client)


def test_latest_background_replaces_pending_while_video_encodes(server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()

    class SlowEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> Any:
            entered.set()
            assert release.wait(2)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(quest_server, "QuestEncoder", SlowEncoder)
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            server.set_ambient_frame(torch.ones(3, 64, 64))
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            assert entered.wait(1)
            server._ambient_background.reset()
            server.set_ambient_frame(torch.zeros(3, 64, 64))
            latest = server._ambient_pending[0]
            release.set()
            assert read_message(client)[0] == 0x02
            assert read_message(client) == (0x10, latest)
            assert len(read_layers(client)) == 3
            client.settimeout(0.1)
            with pytest.raises(socket.timeout):
                read_message(client)
    finally:
        release.set()


def test_feather_follows_both_eyes_and_precedes_menu(server: Any) -> None:
    class Menu:
        def reset(self) -> None:
            pass

        def texture(self) -> None:
            return None

        def layers(self) -> list[dict[str, Any]]:
            return [{"tipo": "quad", "tex": "test_menu", "olho": "ambos"}, {"tipo": "laser"}]

    server.set_menu(Menu())
    with connect(server) as client:
        initial_messages(client)
        _, layers = publish(server, client)
        assert [layer.get("tex") for layer in layers] == ["video", "video", "iw3_ambient", "test_menu", None]


def test_ambient_tracks_canonical_pose_size_and_cylinder_limit(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        send_json(client, 0x84, {"cilindro": True})
        layers = read_layers(client)
        assert len(layers) == 2 and not server.actual_cylinder
        _, layers = publish(server, client)
        assert layers[-1]["tipo"] == "quad"
        server.set_screen_curvature(120)
        layers = read_layers(client)
        assert len(layers) == 2
        assert server.actual_cylinder and main_video(layers)["angulo"] == pytest.approx(math.radians(120))
        publish(server, client)
        canonical: dict[str, Any] = {"pos": [1, 2, -4], "rot": [0, 0, 0, 1], "tam": [5, 2.5]}
        server._send_layers(server._connection, canonical)
        left, _, background = read_layers(client)
        assert background["rot"] == left["rot"] == canonical["rot"]
        assert background["pos"] == left["pos"]
        assert background["raio"] == left["raio"]
        assert background["angulo"] == pytest.approx(math.radians(170))
        assert background["pos"][2] - background["raio"] == pytest.approx(-4)
        assert background["raio"] * background["angulo"] / background["aspecto"] == pytest.approx(5)
        send_json(client, 0x84, {"cilindro": False})
        assert len(read_layers(client)) == 2
        _, layers = publish(server, client)
        left, _, background = layers
        assert background["tipo"] == left["tipo"] == "quad"
        assert background["tam"] == [10, 5] and background["pos"] == [1, 2, -4]


def test_background_does_not_change_pc_mapping_or_grab_target(server: Any) -> None:
    backend = RecordedInput()
    server.set_controller_factory(lambda: QuestController(backend))
    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        data = entrada()
        data["maos"]["dir"]["pos"] = [0.9, 0, 0]
        send_json(client, 0x83, data)
        read_layers(client)
        backend.wait((0.75, 0.5))
        data["maos"]["dir"]["gatilho"] = True
        send_json(client, 0x83, data)
        backend.wait("down")
        data["maos"]["dir"]["aperto"] = True
        send_json(client, 0x83, data)
        backend.wait("up")
        # 射线命中背景但错过主视频，不能开始新的抓取。
        data = entrada()
        send_json(client, 0x83, data)
        data["maos"]["dir"].update(pos=[2.5, 0, 0], aperto=True)
        send_json(client, 0x83, data)
        with server._layer_lock:
            assert server._screen is not None and not server._screen.active


def test_disabling_removes_background_immediately_and_clears_history(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        previous = server._ambient_background
        server.configure_ambient(False, 0.25, 2.0)
        assert len(read_layers(client)) == 2
        assert server._ambient_pending is None and server._ambient_background is not previous
        server.configure_ambient(True, 0.25, 2.0)
        assert len(read_layers(client)) == 2
        publish(server, client)


def test_reconfigure_and_reconnect_do_not_reuse_background(server: Any) -> None:
    with connect(server) as first:
        initial_messages(first)
        publish(server, first)
        previous = server._ambient_background
        server.configure_video(256, 96, 30, 35)
        messages = [read_message(first) for _ in range(3)]
        assert sorted(kind for kind, _ in messages) == [0x01, 0x11, 0x11]
        assert all(len(json.loads(payload)["camadas"]) == 2 for kind, payload in messages if kind == 0x11)
        assert server._ambient_background is not previous
    with server._condition:
        assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
    with connect(server) as second:
        kind, payload = read_message(second)
        assert kind == 0x01 and json.loads(payload)["largura"] == 256
        assert len(read_layers(second)) == 2
        assert server._ambient_pending is None


def test_pause_removes_visible_background_without_a_new_video_frame(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        server.pause_media()
        assert len(read_layers(client)) == 2
        assert server._ambient_background is None and server._ambient_pending is None


def test_failed_video_reconfigure_removes_old_background(server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEncoder:
        def __init__(self, *args: Any) -> None:
            raise RuntimeError("injected reconfigure failure")

    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        monkeypatch.setattr(quest_server, "QuestEncoder", BrokenEncoder)
        with pytest.raises(RuntimeError, match="injected reconfigure failure"):
            server.configure_video(256, 96, 30, 35)
        assert len(read_layers(client)) == 2
        assert server._ambient_background is None and server._ambient_pending is None


def test_timed_out_video_reconfigure_removes_background_before_owner_finishes(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class BlockedEncoder(EncoderDouble):
        def __init__(self, *args: Any) -> None:
            entered.set()
            assert release.wait(8)
            finished.set()

    try:
        with connect(server) as client:
            initial_messages(client)
            publish(server, client)
            monkeypatch.setattr(quest_server, "QuestEncoder", BlockedEncoder)
            with pytest.raises(RuntimeError, match="执行超时"):
                server.configure_video(256, 96, 30, 35)
            assert entered.is_set() and not finished.is_set()
            assert len(read_layers(client)) == 2
            assert len(server._sent_layers["camadas"]) == 2
            assert server._ambient_background is None and server._ambient_pending is None
            release.set()
            assert finished.wait(1)
    finally:
        release.set()


def test_stale_connection_does_not_consume_new_connection_background(server: Any) -> None:
    with connect(server) as first:
        initial_messages(first)
        with server._condition:
            assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            previous = server._connection
    with server._condition:
        assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
    with connect(server) as second:
        initial_messages(second)
        with server._condition:
            assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
        server.set_ambient_frame(torch.ones(3, 64, 64))
        pending = server._ambient_pending
        server._send_ambient(previous)
        assert server._ambient_pending is pending and pending is not None
        server.set_frame_data((torch.zeros(3, 64, 128), 1))
        assert read_message(second)[0] == 0x02
        assert read_message(second) == (0x10, pending[0])
        assert len(read_layers(second)) == 3


def test_stopped_disconnected_disabled_and_paused_sources_skip_pixel_work(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("pixel work ran without an active source and consumer")

    monkeypatch.setattr(quest_server.AmbientBackground, "update", forbidden)
    frame = torch.zeros(3, 64, 64)
    server.set_ambient_frame(frame)
    with connect(server) as client:
        initial_messages(client)
        server.configure_ambient(False, 0.25, 2)
        read_layers(client)
        server.set_ambient_frame(frame)
        server.configure_ambient(True, 0.25, 2)
        read_layers(client)
        server.pause_media()
        server.set_ambient_frame(frame)
    server.stop()
    server.set_ambient_frame(frame)


@pytest.mark.parametrize("action", ["disable", "reconfigure", "pause", "disconnect", "stop", "curvature", "capability"])
def test_inflight_pixel_work_cannot_revive_old_background(
    server: Any, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    entered, release = threading.Event(), threading.Event()
    original = quest_server.AmbientBackground.update

    def slow(background: Any, *args: Any, **kwargs: Any) -> bytes | None:
        # Torch与压缩执行期间两把server锁必须可用。
        assert not server._layer_lock._is_owned()
        assert not server._condition._is_owned()
        entered.set()
        assert release.wait(2)
        return original(background, *args, **kwargs)

    monkeypatch.setattr(quest_server.AmbientBackground, "update", slow)
    worker = threading.Thread(target=lambda: server.set_ambient_frame(torch.ones(3, 64, 64)))
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            worker.start()
            assert entered.wait(1)
            if action == "disable":
                server.configure_ambient(False, 0.25, 2)
                assert len(read_layers(client)) == 2
            elif action == "reconfigure":
                server.configure_video(128, 64, 30, 35)
                assert read_message(client)[0] == 0x01
                assert len(read_layers(client)) == 2
            elif action == "pause":
                server.pause_media()
            elif action == "disconnect":
                server._disconnect(server._connection)
            elif action == "curvature":
                server.set_screen_curvature(120)
                assert len(read_layers(client)) == 2
            elif action == "capability":
                send_json(client, 0x84, {"cilindro": True})
                assert len(read_layers(client)) == 2
            else:
                server.stop()
            release.set()
            worker.join(2)
            assert not worker.is_alive() and server._ambient_pending is None
            if action in {"disable", "reconfigure"}:
                server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
                assert read_message(client)[0] == 0x02
                client.settimeout(0.1)
                with pytest.raises(socket.timeout):
                    read_message(client)
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)


@pytest.mark.hwaccel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVENC GPU")
def test_cuda_synthetic_color_survives_hevc_and_ambient_upload() -> None:
    import av

    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context()
    server = quest_server.QuestServer(
        "127.0.0.1", 640, 180, 30, 35, audio=False, port=0, discovery_port=0, ambient_enabled=True
    )
    server.start()
    try:
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                mono = torch.zeros(3, 180, 320, device="cuda")
                mono[0] = 1
                stereo = torch.cat((mono, mono), dim=2)
                server.set_ambient_frame(mono, sbs=stereo)
                server.set_frame_data((stereo, 1.0))
                stereo.zero_()
            kind, encoded = read_message(client)
            assert kind == 0x02 and encoded[8] == 1
            decoder = av.CodecContext.create("hevc", "r")
            pictures = decoder.decode(av.Packet(encoded[9:])) + decoder.decode(None)
            assert len(pictures) == 1
            decoded = torch.from_numpy(pictures[0].to_ndarray(format="rgb24"))
            assert torch.allclose(decoded[90, 160].float(), torch.tensor([255.0, 0, 0]), atol=10)
            assert torch.allclose(decoded[90, 480].float(), torch.tensor([255.0, 0, 0]), atol=10)
            kind, texture = read_message(client)
            assert kind == 0x10
            rgba = torch.frombuffer(bytearray(zlib.decompress(texture[6 + texture[0] :])), dtype=torch.uint8)
            rgba = rgba.reshape(180, 320, 4)
            assert rgba[90, 160].tolist() == [64, 0, 0, 0]
            assert rgba[90, 40, :3].tolist() == [64, 0, 0]
            assert 0 < rgba[90, 40, 3] <= 255
            assert len(read_layers(client)) == 3
            server.check_error()
    finally:
        server.stop()
