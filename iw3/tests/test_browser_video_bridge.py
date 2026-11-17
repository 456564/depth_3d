"""浏览器压缩视频桥：真实HTTP边界、代次和有界输入。"""

import http.client
import importlib
import importlib.util
import json
import time

import pytest

ORIGIN = "chrome-extension://" + "a" * 32


def test_bridge_module_exists():
    assert importlib.util.find_spec("iw3.desktop.browser_video_bridge") is not None


@pytest.fixture
def bridge():
    module_name = "iw3.desktop.browser_video_bridge"
    assert importlib.util.find_spec(module_name) is not None, "缺少真实媒体桥模块"
    module = importlib.import_module(module_name)
    bridge = module.BrowserVideoBridge(port=0)
    bridge.start()
    yield bridge
    bridge.stop()


def request(bridge, path, payload=b"", *, origin=ORIGIN, token=None, headers=None):
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
        content_type = "application/json"
    else:
        content_type = "application/octet-stream"
    request_headers = {"Content-Type": content_type, **(headers or {})}
    if origin is not None:
        request_headers["Origin"] = origin
    if token is not None:
        request_headers["X-IW3-Token"] = token
    connection = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=3)
    try:
        connection.request("POST", path, payload, request_headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        return response.status, data, dict(response.headers)
    finally:
        connection.close()


def pair(bridge):
    code, body, _ = request(bridge, "/hello", {})
    assert code == 200
    return body["token"]


def state(serial=0, generation="tab:video:track", **overrides):
    return {
        "session": "tab",
        "generation": generation,
        "serial": serial,
        "media_time": 2.5,
        "paused": False,
        "playback_rate": 1.0,
        "duration": 20.0,
        "width": 1080,
        "height": 1920,
        **overrides,
    }


def chunk(bridge, token, payload=b"ftyp", *, generation="tab:video:track", sequence=0):
    return request(
        bridge,
        "/chunk",
        payload,
        token=token,
        headers={
            "X-IW3-Session": "tab",
            "X-IW3-Generation": generation,
            "X-IW3-Sequence": str(sequence),
        },
    )


@pytest.mark.parametrize("origin", [None, "null", "https://www.douyin.com", "http://127.0.0.1"])
def test_pairing_rejects_web_page_origins_without_cors_access(bridge, origin):
    code, body, headers = request(bridge, "/hello", {}, origin=origin)
    assert code == 403
    assert "token" not in body
    assert "Access-Control-Allow-Origin" not in headers
    assert bridge.snapshot() is None


def test_pairing_binds_token_to_one_extension_and_rejects_wrong_token(bridge):
    token = pair(bridge)
    assert len(token) >= 32
    assert request(bridge, "/state", state(), token="wrong")[0] == 403
    assert request(bridge, "/state", state(), token=token, origin="chrome-extension://" + "b" * 32)[0] == 403
    assert request(bridge, "/hello", {}, origin="chrome-extension://" + "b" * 32)[0] == 403
    assert request(bridge, "/state", state(), token=token)[0] == 200


def test_state_then_ordered_compressed_bytes_produce_immutable_snapshot(bridge):
    token = pair(bridge)
    code, body, _ = request(bridge, "/state", state(), token=token)
    assert code == 200 and body["need_media"]
    assert bridge.snapshot() is None
    assert chunk(bridge, token)[0] == 200
    first = bridge.snapshot()
    assert first.data == b"ftyp"
    assert (first.media_time, first.paused, first.playback_rate) == (2.5, False, 1.0)
    assert chunk(bridge, token, b"moov", sequence=1)[0] == 200
    second = bridge.snapshot()
    assert first.data == b"ftyp" and second.data == b"ftypmoov"
    assert second.revision > first.revision
    assert second.generation == "tab:video:track"
    assert bridge.status()["width"] == 1080
    assert bridge.status()["height"] == 1920


def test_old_generation_and_out_of_order_chunks_do_not_mix_videos(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    assert chunk(bridge, token, b"old")[0] == 200
    assert chunk(bridge, token, b"skip", sequence=2)[0] == 409
    assert bridge.snapshot().data == b"old"
    request(bridge, "/state", state(serial=1, generation="new"), token=token)
    assert bridge.snapshot() is None
    assert chunk(bridge, token, b"late", sequence=1)[0] == 409
    assert chunk(bridge, token, b"new", generation="new")[0] == 200
    assert bridge.snapshot().data == b"new"
    assert request(bridge, "/state", state(serial=0), token=token)[0] == 409
    assert bridge.snapshot().generation == "new"


def test_exact_state_retry_keeps_bytes_but_different_same_serial_is_rejected(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    chunk(bridge, token)
    assert request(bridge, "/state", state(), token=token)[0] == 200
    assert bridge.snapshot().data == b"ftyp"
    assert request(bridge, "/state", state(generation="other"), token=token)[0] == 409
    assert bridge.snapshot().generation == "tab:video:track"


@pytest.mark.parametrize(
    "overrides",
    [
        {"media_time": float("nan")},
        {"media_time": -1},
        {"playback_rate": 0},
        {"playback_rate": float("inf")},
        {"paused": "false"},
        {"width": True},
        {"height": 20000},
        {"generation": "x" * 257},
        {"serial": -1},
        {"duration": -1},
    ],
)
def test_metadata_rejects_nonfinite_or_wrong_types_without_changing_source(bridge, overrides):
    token = pair(bridge)
    assert request(bridge, "/state", state(**overrides), token=token)[0] == 400
    assert bridge.snapshot() is None


def test_semantic_commands_require_fresh_source_and_never_cross_generation(bridge):
    token = pair(bridge)
    assert not bridge.queue_command("next")
    request(bridge, "/state", state(), token=token)
    assert bridge.queue_command("pause_toggle")
    assert not bridge.queue_command("arbitrary")
    body = request(bridge, "/state", state(serial=1), token=token)[1]
    assert body["commands"] == [{"name": "pause_toggle", "generation": "tab:video:track"}]
    assert request(bridge, "/state", state(serial=2), token=token)[1]["commands"] == []
    assert bridge.queue_command("next")
    body = request(bridge, "/state", state(serial=3, generation="new"), token=token)[1]
    assert body["commands"] == []


def test_stale_metadata_does_not_keep_old_video_or_deliver_old_command(bridge, monkeypatch):
    module = importlib.import_module("iw3.desktop.browser_video_bridge")
    clock = [time.monotonic()]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    chunk(bridge, token)
    assert bridge.queue_command("next")
    clock[0] += 2
    assert bridge.snapshot() is None
    assert not bridge.queue_command("next")
    assert bridge.status()["status"] == "waiting"
    assert request(bridge, "/state", state(serial=1), token=token)[1]["commands"] == []


def test_binary_limits_reject_oversized_body_and_total_before_append(bridge, monkeypatch):
    module = importlib.import_module("iw3.desktop.browser_video_bridge")
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    monkeypatch.setattr(module, "MAX_CHUNK_BYTES", 8)
    monkeypatch.setattr(module, "MAX_MEDIA_BYTES", 10)
    assert chunk(bridge, token, b"123456789")[0] == 413
    assert bridge.snapshot() is None
    assert chunk(bridge, token, b"12345678")[0] == 200
    assert chunk(bridge, token, b"abc", sequence=1)[0] == 413
    assert bridge.snapshot().data == b"12345678"


def test_unknown_path_and_missing_media_headers_are_explicit_errors(bridge):
    token = pair(bridge)
    assert request(bridge, "/unknown", {}, token=token)[0] == 404
    assert request(bridge, "/chunk", b"data", token=token)[0] == 400


def test_stop_releases_port_and_clears_source(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    chunk(bridge, token)
    bridge.stop()
    assert bridge.snapshot() is None
    assert not bridge.queue_command("next")


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Content-Length": "-1"}, 400),
        ({"Content-Length": "9" * 5000}, 413),
        ({"Transfer-Encoding": "chunked"}, 400),
    ],
)
def test_request_length_is_bounded_before_reading_body(bridge, headers, expected):
    token = pair(bridge)
    assert request(bridge, "/state", {}, token=token, headers=headers)[0] == expected


def test_invalid_nonascii_token_is_rejected_without_crashing_request(bridge):
    pair(bridge)
    assert request(bridge, "/state", state(), token="é")[0] == 403


def test_media_sequence_header_is_bounded_before_integer_conversion(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    assert (
        request(
            bridge,
            "/chunk",
            b"payload",
            token=token,
            headers={"X-IW3-Session": "tab", "X-IW3-Generation": "tab:video:track", "X-IW3-Sequence": "9" * 5000},
        )[0]
        == 400
    )


def test_explicit_command_generation_must_match_current_state(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    assert not bridge.queue_command("next", generation="old")
    assert bridge.queue_command("next", generation="tab:video:track")


def test_full_media_replaces_only_current_generation_with_a_new_revision(bridge):
    token = pair(bridge)
    request(bridge, "/state", state(), token=token)
    code, body, _ = request(
        bridge,
        "/media",
        b"complete-mp4",
        token=token,
        headers={"X-IW3-Session": "tab", "X-IW3-Generation": "tab:video:track", "X-IW3-Sequence": "0"},
    )
    assert code == 200 and body["next_sequence"] == 1
    assert bridge.snapshot().data == b"complete-mp4"


def test_preflight_allows_extension_only_and_expected_headers(bridge):
    connection = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=3)
    try:
        connection.request("OPTIONS", "/hello", headers={"Origin": ORIGIN})
        response = connection.getresponse()
        assert response.status == 204
        assert response.getheader("Access-Control-Allow-Origin") == ORIGIN
        assert "X-IW3-Token" in response.getheader("Access-Control-Allow-Headers")
        assert response.read() == b""
    finally:
        connection.close()


@pytest.mark.parametrize("width,height", [(0, 0), (540, 960)])
def test_unavailable_media_heartbeat_keeps_navigation_without_requesting_bytes(bridge, width, height):
    token = pair(bridge)
    playback = state(generation="live-target", media_available=False, width=width, height=height)
    code, body, _ = request(bridge, "/state", playback, token=token)
    assert code == 200
    assert body == {"commands": [], "need_media": False, "next_sequence": 0}
    status = bridge.status()
    assert status["fresh"] and status["status"] == "waiting"
    assert status.get("media_available") is False
    assert bridge.snapshot() is None
    for command in ("next", "previous", "exit_fullscreen"):
        assert bridge.queue_command(command, "live-target")
    assert not bridge.queue_command("pause_toggle", "live-target")
    playback["serial"] = 1
    reply = request(bridge, "/state", playback, token=token)[1]
    assert reply["commands"] == [
        {"name": "next", "generation": "live-target"},
        {"name": "previous", "generation": "live-target"},
        {"name": "exit_fullscreen", "generation": "live-target"},
    ]
    playback["serial"] = 2
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == []


@pytest.mark.parametrize("available", [None, 0, 1, "false", [], {}])
def test_media_availability_rejects_non_boolean_values(bridge, available):
    token = pair(bridge)
    assert request(bridge, "/state", state(media_available=available), token=token)[0] == 400
    assert not bridge.status()["fresh"]


@pytest.mark.parametrize("width,height", [(0, 960), (540, 0), (-1, 960), (True, 0), (0, False), (8193, 960)])
def test_unavailable_media_dimensions_require_both_zero_or_valid_positive_sizes(bridge, width, height):
    token = pair(bridge)
    playback = state(media_available=False, width=width, height=height)
    assert request(bridge, "/state", playback, token=token)[0] == 400
    assert not bridge.status()["fresh"]


@pytest.mark.parametrize("endpoint", ["/chunk", "/media"])
def test_unavailable_media_rejects_bytes_even_with_current_generation(bridge, endpoint):
    token = pair(bridge)
    assert request(bridge, "/state", state(media_available=False), token=token)[0] == 200
    code, _, _ = request(
        bridge,
        endpoint,
        b"stale-init",
        token=token,
        headers={"X-IW3-Session": "tab", "X-IW3-Generation": "tab:video:track", "X-IW3-Sequence": "0"},
    )
    assert code == 409
    assert bridge.status()["bytes"] == 0
    assert bridge.snapshot() is None


def test_availability_changes_clear_old_media_and_commands_even_with_same_generation(bridge):
    token = pair(bridge)
    assert request(bridge, "/state", state(), token=token)[0] == 200
    assert bridge.status().get("media_available") is True
    assert chunk(bridge, token, b"old-init")[0] == 200
    old = bridge.snapshot()
    assert bridge.queue_command("pause_toggle") and bridge.queue_command("next")
    unavailable = state(serial=1, media_available=False)
    code, reply, _ = request(bridge, "/state", unavailable, token=token)
    assert code == 200 and reply == {"commands": [], "need_media": False, "next_sequence": 0}
    assert bridge.snapshot() is None
    assert bridge.status()["bytes"] == 0 and bridge.status()["revision"] > old.revision
    assert chunk(bridge, token, b"late", sequence=1)[0] == 409
    assert chunk(bridge, token, b"forged-init")[0] == 409
    assert request(bridge, "/state", state(), token=token)[0] == 409
    assert request(bridge, "/state", state(serial=1), token=token)[0] == 409
    assert bridge.queue_command("next")
    code, reply, _ = request(bridge, "/state", state(serial=2), token=token)
    assert code == 200 and reply == {"commands": [], "need_media": True, "next_sequence": 0}
    assert bridge.snapshot() is None
    assert chunk(bridge, token, b"new-init")[0] == 200
    assert bridge.snapshot().data == b"new-init"
    assert old.data == b"old-init"


def test_expired_state_rejects_late_media_before_heartbeat_can_revive_it(bridge, monkeypatch):
    module = importlib.import_module("iw3.desktop.browser_video_bridge")
    clock = [time.monotonic()]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    token = pair(bridge)
    assert request(bridge, "/state", state(), token=token)[0] == 200
    assert chunk(bridge, token, b"current")[0] == 200
    clock[0] += 2
    assert chunk(bridge, token, b"late", sequence=1)[0] == 409
    assert request(bridge, "/state", state(serial=1), token=token)[0] == 200
    assert bridge.snapshot().data == b"current"


def test_unavailable_media_navigation_expires_and_never_crosses_target(bridge, monkeypatch):
    module = importlib.import_module("iw3.desktop.browser_video_bridge")
    clock = [time.monotonic()]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    token = pair(bridge)
    playback = state(generation="live-one", media_available=False)
    assert request(bridge, "/state", playback, token=token)[0] == 200
    assert bridge.queue_command("next", "live-one")
    clock[0] += 2
    assert not bridge.status()["fresh"]
    assert not bridge.queue_command("next", "live-one")
    playback["serial"] = 1
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == []
    assert bridge.queue_command("next", "live-one")
    playback.update(serial=2, generation="live-two")
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == []
    assert not bridge.queue_command("next", "live-one")


def test_real_quest_controller_can_navigate_unavailable_live_media_once_without_losing_ray(bridge):
    token = pair(bridge)
    playback = state(generation="live-target", media_available=False, width=540, height=960)
    assert request(bridge, "/state", playback, token=token)[0] == 200
    BrowserInput = importlib.import_module("iw3.desktop.browser_input").BrowserInput
    QuestController = importlib.import_module("iw3.desktop.quest_input").QuestController
    backend = BrowserInput(bridge)
    controller = QuestController(backend)
    screen = {"pos": [0, 0, -3], "rot": [0, 0, 0, 1], "tam": [4, 2]}
    hand = {"pos": [0, 0, 0], "rot": [0, 0, 0, 1], "analogico": [0, 0], "gatilho": False, "a": False}
    data = {"estado": "focado", "maos": {"dir": hand}}
    assert controller.update(data, screen)["tipo"] == "laser"
    hand["analogico"] = [0, -1]
    for _ in range(4):
        assert controller.update(data, screen)["tipo"] == "laser"
    playback["serial"] = 1
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == [
        {"name": "next", "generation": "live-target"}
    ]
    playback["serial"] = 2
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == []
    assert not backend.mouse_down()
    hand["analogico"] = [0, 0]
    assert controller.update(data, screen)["tipo"] == "laser"
    assert controller.exit_fullscreen()
    playback["serial"] = 3
    assert request(bridge, "/state", playback, token=token)[1]["commands"] == [
        {"name": "exit_fullscreen", "generation": "live-target"}
    ]
