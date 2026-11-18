import struct
import zlib
from types import SimpleNamespace

import torch

from iw3.desktop import quest_ambient, quest_server
from iw3.desktop.quest_ambient import AmbientBackground
from iw3.desktop.quest_restart import load_resume, write_resume
from iw3.desktop.quest_settings import AMBIENT_STYLE_LABELS, validate_patch
from iw3.desktop.utils import create_parser

from .test_quest_ambient_server import read_layers
from .test_quest_server import EncoderDouble, connect, read_message


def texture_info(payload):
    size = payload[0]
    return payload[1 : size + 1].decode(), struct.unpack("!HHB", payload[size + 1 : size + 6])


def test_night_setting_reaches_cli_restart_and_chinese_menu():
    patch = {"quest_ambient_style": "night"}
    assert validate_patch(patch) == patch
    assert create_parser().parse_args(["--quest-ambient-style", "night"]).quest_ambient_style == "night"
    assert load_resume(write_resume(patch, gui=True), gui=True) == patch
    assert AMBIENT_STYLE_LABELS["night"] == "自然夜景＋视频柔光"


def test_night_dynamic_payload_stays_small_and_follows_video():
    payloads = []
    for channel in (0, 2):
        frame = torch.zeros(3, 64, 36)
        frame[channel] = 1
        payload = AmbientBackground().update(frame, 0.6, tick=0, style="night", panoramic=True)
        assert texture_info(payload) == ("iw3_ambient", (320, 180, 1))
        offset = payload[0] + 6
        rgba = torch.frombuffer(bytearray(zlib.decompress(payload[offset:])), dtype=torch.uint8).reshape(180, 320, 4)
        assert rgba[:, :, channel].sum() > 0
        assert rgba[90, 160, 3] == 0
        payloads.append(payload)
    assert payloads[0] != payloads[1]


def test_zero_night_glow_leaves_static_scenery_visible():
    payload = AmbientBackground().update(torch.ones(3, 64, 36), 0, tick=0, style="night", panoramic=True)
    rgba = torch.frombuffer(bytearray(zlib.decompress(payload[payload[0] + 6 :])), dtype=torch.uint8).reshape(
        180, 320, 4
    )
    assert not torch.count_nonzero(rgba[:, :, 3])


def test_night_geometry_uses_high_resolution_rects_with_a_front_only_option():
    server = quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35, ambient_style="night", ambient_360=True)
    server._ambient_visible = server._night_sent = True
    layers = server._layers()["camadas"]
    assert len(layers) == 9
    for index, layer in enumerate(layers[:6]):
        assert layer["tex"] == "iw3_night"
        assert layer["rect"] == [index % 3 * 516 + 2, index // 3 * 516 + 2, 512, 512]
    assert layers[-1]["tex"] == "iw3_ambient"
    server._ambient_360 = False
    assert len(server._layers()["camadas"]) == 4


def test_night_texture_is_sent_once_per_connection_and_before_its_layers(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(quest_ambient, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        ambient_enabled=True,
        ambient_style="night",
        ambient_360=True,
    )
    server.start()

    def publish(tick):
        clock[0] = tick
        server.set_ambient_frame(torch.ones(3, 64, 64))
        server.set_frame_data((torch.ones(3, 64, 128), tick + 1))

    try:
        for attempt in range(2):
            with connect(server) as client:
                assert read_message(client)[0] == 0x01
                assert len(read_layers(client)) == 2
                with server._condition:
                    assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
                publish(0)
                assert read_message(client)[0] == 0x02
                kind, payload = read_message(client)
                assert kind == 0x10 and texture_info(payload) == ("iw3_night", (1548, 1032, 1))
                kind, payload = read_message(client)
                assert kind == 0x10 and texture_info(payload) == ("iw3_ambient", (320, 180, 1))
                assert len(read_layers(client)) == 9
                server.discard_video_frames()
                assert len(read_layers(client)) == 2
                publish(2)
                assert read_message(client)[0] == 0x02
                kind, payload = read_message(client)
                assert kind == 0x10 and texture_info(payload)[0] == "iw3_ambient"
                assert len(read_layers(client)) == 9
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
        server.check_error()
    finally:
        server.stop()
