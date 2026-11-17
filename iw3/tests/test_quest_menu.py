import importlib
import struct
import time
import zlib
from typing import Any

import pytest

VIDEO = {"pos": [0, 0, -3], "rot": [0, 0, 0, 1], "tam": [3.6, 2]}


def packet(side: str = "dir", **changes: Any) -> dict[str, Any]:
    neutral = {
        "pos": [0, 0, 0],
        "rot": [0, 0, 0, 1],
        "b": False,
        "a": False,
        "gatilho": False,
        "aperto": False,
        "analogico": [0, 0],
    }
    return {"estado": "focado", "maos": {"esq": dict(neutral), "dir": dict(neutral), side: {**neutral, **changes}}}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def menu() -> tuple[Any, dict[str, Any], list[Any]]:
    snapshot = {
        "values": {
            "divergence": 1.0,
            "convergence": 0.5,
            "convergence_mode": "constant",
            "foreground_scale": 0.0,
            "edge_dilation": 2,
            "synthetic_view": "both",
            "depth_model": "Any_V2_S",
            "method": "row_flow_v3_sym",
            "gpu": [0],
            "stream_fps": 30,
            "stream_height": 1080,
            "quest_no_audio": False,
        },
        "choices": {"depth_model": ["Any_V2_S", "Any_L"], "depth_model_small": ["Any_V2_S"]},
        "status": "ready",
        "message": "电脑端已应用",
    }
    commands: list[Any] = []
    state = importlib.import_module("iw3.desktop.quest_menu").QuestMenu(
        lambda: snapshot, lambda command, patch: commands.append((command, patch))
    )
    return state, snapshot, commands


def opened(menu: Any) -> None:
    menu.update(packet(), VIDEO)
    menu.update(packet(b=True), VIDEO)
    assert menu.visible
    menu.texture()
    menu.update(packet(), VIDEO)


def point(menu: Any, x: float, y: float) -> list[float]:
    quad = menu.layers()[0]
    return [
        quad["pos"][0] + (x / menu.WIDTH - 0.5) * quad["tam"][0],
        quad["pos"][1] + (0.5 - y / menu.HEIGHT) * quad["tam"][1],
        0,
    ]


def click(menu: Any, action: str, payload: Any = None, side: str = "dir") -> list[float]:
    rect = next(rect for rect, name, value, _ in menu._buttons if name == action and value == payload)
    pos = point(menu, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    menu.update(packet(side, pos=pos), VIDEO)
    menu.update(packet(side, pos=pos, gatilho=True), VIDEO)
    menu.update(packet(side, pos=pos), VIDEO)
    return pos


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_each_hand_toggle_requires_release(menu: Any, side: str) -> None:
    state, _, _ = menu
    assert state.update(packet(side, b=True), VIDEO)
    assert not state.visible
    state.update(packet(), VIDEO)
    assert state.update(packet(side, b=True), VIDEO)
    assert state.visible
    for _ in range(4):
        assert state.update(packet(side, b=True), VIDEO)
        assert state.visible
    state.update(packet(), VIDEO)
    state.update(packet(side, b=True), VIDEO)
    assert not state.visible


def test_open_close_release_gate_prevents_pc_click_or_grab(menu: Any) -> None:
    state, _, commands = menu
    state.update(packet(), VIDEO)
    state.update(packet(b=True, gatilho=True), VIDEO)
    state.texture()
    state.update(packet(gatilho=True), VIDEO)
    assert commands == []
    state.update(packet(b=True, gatilho=True), VIDEO)
    assert not state.visible
    assert state.update(packet(gatilho=True, aperto=True), VIDEO)
    assert state.update(packet(analogico=[0, 0.8]), VIDEO)
    assert state.update(packet(), VIDEO)
    assert not state.update(packet(), VIDEO)


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_pointer_click_uses_actual_panel_and_native_laser(menu: Any, side: str) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "adjust", ("divergence", 1, "x"), side)
    assert commands == [("set", {"divergence": 1.25})]
    layers = state.layers()
    assert len(layers) == 2
    assert layers[1]["tipo"] == "laser" and layers[1]["mao"] == side
    assert layers[1]["ponto"][2] == pytest.approx(layers[0]["pos"][2])


def test_missed_press_does_not_click_when_aim_enters_later(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    state.update(packet(gatilho=True, pos=[100, 0, 0]), VIDEO)
    state.update(packet(gatilho=True), VIDEO)
    assert commands == []


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_a_or_x_clicks_menu_once_without_hold_repeat(menu: Any, clock: list[float], side: str) -> None:
    state, _, commands = menu
    opened(state)
    rect = next(
        rect for rect, action, value, _ in state._buttons if action == "adjust" and value == ("divergence", 1, "x")
    )
    pos = point(state, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    state.update(packet(side, pos=pos), VIDEO)
    for _ in range(5):
        clock[0] += 0.2
        state.update(packet(side, pos=pos, a=True), VIDEO)
    assert commands == [("set", {"divergence": 1.25})]
    assert state.layers()[1]["mao"] == side
    state.update(packet(side, pos=pos), VIDEO)
    state.update(packet(side, pos=pos, a=True), VIDEO)
    assert len(commands) == 2


def test_menu_close_waits_for_a_release_before_pc_control(menu: Any) -> None:
    state, _, _ = menu
    opened(state)
    state.update(packet(b=True, a=True), VIDEO)
    assert not state.visible
    assert state.update(packet(a=True), VIDEO)
    assert state.update(packet(), VIDEO)
    assert not state.update(packet(), VIDEO)


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_a_cannot_continue_trigger_repeat_after_trigger_release(menu: Any, clock: list[float], side: str) -> None:
    state, _, commands = menu
    opened(state)
    rect = next(
        rect for rect, action, value, _ in state._buttons if action == "adjust" and value == ("divergence", 1, "x")
    )
    pos = point(state, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    state.update(packet(side, pos=pos), VIDEO)
    state.update(packet(side, pos=pos, gatilho=True), VIDEO)
    state.update(packet(side, pos=pos, gatilho=True, a=True), VIDEO)
    for _ in range(5):
        clock[0] += 0.2
        state.update(packet(side, pos=pos, a=True), VIDEO)
    assert commands == [("set", {"divergence": 1.25})]


def test_texture_is_single_fixed_id_rgba_and_cached_across_pointer_motion(menu: Any, clock: list[float]) -> None:
    state, _, _ = menu
    state.update(packet(), VIDEO)
    state.update(packet(b=True), VIDEO)
    payload = state.texture()
    length = payload[0]
    assert payload[1 : 1 + length] == b"iw3_settings"
    width, height, fmt = struct.unpack("!HHB", payload[1 + length : 6 + length])
    assert (width, height, fmt) == (state.WIDTH, state.HEIGHT, 1)
    assert len(zlib.decompress(payload[6 + length :])) == width * height * 4
    state.update(packet(pos=[0.1, 0, 0]), VIDEO)
    clock[0] += 0.11
    assert state.texture() is None


def test_texture_changes_for_actual_snapshot_and_coalesces_changes(menu: Any, clock: list[float]) -> None:
    state, snapshot, _ = menu
    opened(state)
    snapshot["values"]["divergence"] = 2.0
    assert state.texture() is None
    clock[0] += 0.11
    assert state.texture() is not None
    assert state.texture() is None


def test_staged_enum_does_not_claim_applied_and_apply_sends_patch(menu: Any, clock: list[float]) -> None:
    state, snapshot, commands = menu
    opened(state)
    click(state, "choices", "synthetic_view")
    clock[0] += 0.11
    state.texture()
    click(state, "choose", ("synthetic_view", "right"))
    assert commands == [] and snapshot["values"]["synthetic_view"] == "both"
    clock[0] += 0.11
    state.texture()
    click(state, "command", "apply")
    assert commands == [("apply", {"synthetic_view": "right"})]


def test_restart_app_requires_two_distinct_clicks(menu: Any, clock: list[float]) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "command", "restart_app")
    assert commands == []
    clock[0] += 0.11
    state.texture()
    click(state, "command", "restart_app")
    assert commands == [("restart_app", {})]


@pytest.mark.parametrize("side", ["esq", "dir"])
def test_exit_fullscreen_button_sends_once_per_click_and_preserves_menu_draft(menu: Any, clock, side: str) -> None:
    state, _, commands = menu
    opened(state)
    state._draft = {"stream_fps": 60}
    click(state, "page", 4, side=side)
    state._render()
    rect = next(rect for rect, action, value, _ in state._buttons if action == "command" and value == "exit_fullscreen")
    pos = point(state, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    state.update(packet(side, pos=pos), VIDEO)
    state.update(packet(side, pos=pos, gatilho=True), VIDEO)
    for _ in range(5):
        clock[0] += 0.2
        state.update(packet(side, pos=pos, gatilho=True), VIDEO)
    assert commands == [("exit_fullscreen", {})]
    assert state.visible and state._draft == {"stream_fps": 60}
    assert "已发送 Esc" in state._message and "确认" in state._message


def test_exit_fullscreen_failure_stays_in_menu_instead_of_restarting_media(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "page", 4)
    state._render()

    def reject(command, patch):
        raise RuntimeError("退出未发送：捕获窗口不在前台")

    state._submit = reject
    click(state, "command", "exit_fullscreen")
    assert state.visible and "退出未发送" in state._message
    assert commands == []


def test_restart_confirmation_expires_and_stream_restart_is_independent(menu: Any, clock: list[float]) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "command", "restart_app")
    clock[0] += 6
    state.texture()
    state.update(packet(), VIDEO)
    opened(state)
    click(state, "command", "restart_app")
    assert commands == []
    click(state, "command", "restart_stream")
    assert commands == [("restart_stream", {})]


def test_hold_repeats_numeric_only_after_delay_and_stops_on_release(menu: Any, clock: list[float]) -> None:
    state, _, commands = menu
    opened(state)
    rect = next(
        rect for rect, action, value, _ in state._buttons if action == "adjust" and value == ("divergence", 1, "x")
    )
    pos = point(state, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    state.update(packet(gatilho=True, pos=pos), VIDEO)
    clock[0] += 0.3
    state.update(packet(gatilho=True, pos=pos), VIDEO)
    assert len(commands) == 1
    clock[0] += 0.2
    state.update(packet(gatilho=True, pos=pos), VIDEO)
    assert commands[-1] == ("set", {"divergence": 1.5})
    state.update(packet(pos=pos), VIDEO)
    clock[0] += 0.2
    state.update(packet(pos=pos), VIDEO)
    assert len(commands) == 2


@pytest.mark.parametrize("fault", ["focus", "timeout", "pose"])
def test_invalid_input_clears_press_and_cannot_click_until_release(menu: Any, clock: list[float], fault: str) -> None:
    state, _, commands = menu
    opened(state)
    bad = packet(gatilho=True)
    if fault == "focus":
        bad["estado"] = "visivel"
    elif fault == "timeout":
        clock[0] += 0.6
    else:
        bad["maos"]["dir"]["pos"] = None
    state.update(bad, VIDEO)
    assert commands == []
    state.update(packet(gatilho=True), VIDEO)
    assert commands == []


def test_restore_clears_draft_and_reset_hides_layers(menu: Any, clock: list[float]) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "command", "restore")
    assert commands == [("restore", {})]
    state.reset()
    assert not state.visible and state.layers() == [] and state.texture() is None


def test_joystick_scroll_and_page_tabs_make_all_fields_reachable(menu: Any, clock: list[float]) -> None:
    state, _, _ = menu
    opened(state)
    state.update(packet(analogico=[0, -1]), VIDEO)
    clock[0] += 0.11
    state.texture()
    assert state._offset == 1
    for page in range(5):
        click(state, "page", page)
        clock[0] += 0.11
        assert state.texture() is not None
        assert state._page == page


def test_panel_bounds_outside_pixels_do_not_activate(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    for x in (-1, state.WIDTH + 1):
        pos = point(state, x, 500)
        state.update(packet(pos=pos), VIDEO)
        state.update(packet(pos=pos, gatilho=True), VIDEO)
    assert commands == []


def test_opening_requires_grips_and_sticks_to_release_before_buttons(menu: Any) -> None:
    state, _, commands = menu
    state.update(packet(), VIDEO)
    state.update(packet(b=True, aperto=True, analogico=[0, 1]), VIDEO)
    state.texture()
    rect = next(
        rect for rect, action, value, _ in state._buttons if action == "adjust" and value == ("divergence", 1, "x")
    )
    pos = point(state, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    state.update(packet(aperto=True, pos=pos), VIDEO)
    state.update(packet(aperto=True, pos=pos, gatilho=True), VIDEO)
    assert commands == []


def test_enabling_control_can_be_staged_before_related_capture_edits(menu: Any) -> None:
    state, _, _ = menu
    opened(state)
    state._act("toggle", "quest_control")
    state._edit("quest_control", True)
    assert state._draft == {"quest_control": True}


def test_unrecognized_extra_hand_key_is_ignored(menu: Any) -> None:
    state, _, commands = menu
    data = packet()
    data["maos"]["unrecognized"] = "invalid"
    assert state.update(data, VIDEO)
    assert not state.update(data, VIDEO)
    assert commands == []


def test_restart_confirmation_expires_while_inputs_keep_menu_alive(menu: Any, clock: list[float]) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "command", "restart_app")
    for _ in range(25):
        clock[0] += 0.21
        state.update(packet(), VIDEO)
    state.texture()
    assert state.visible and state._confirm_until == 0
    assert "过期" in state._message
    click(state, "command", "restart_app")
    assert commands == []


def test_laser_follows_last_hand_after_its_click_releases(menu: Any) -> None:
    state, _, _ = menu
    opened(state)
    click(state, "adjust", ("divergence", 1, "x"), "esq")
    state.update(packet(), VIDEO)
    assert state.layers()[1]["mao"] == "esq"


def test_related_invalid_draft_shows_error_without_losing_menu(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    state._edit("quest_control", True)
    click(state, "command", "apply")
    assert commands == [] and state.visible and state._message


def test_edge_follow_and_distinct_y_keep_native_values(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "adjust", ("edge_dilation", -1, "y"))
    assert commands[-1] == ("set", {"edge_dilation": [2, 1]})
    click(state, "adjust", ("edge_dilation", 1, "x"))
    assert commands[-1] == ("set", {"edge_dilation": [3, 1]})
    click(state, "follow")
    assert commands[-1] == ("set", {"edge_dilation": 3})


def test_small_model_filter_uses_factory_candidates(menu: Any) -> None:
    state, _, _ = menu
    opened(state)
    state._act("small", None)
    assert state._available("depth_model") == ["Any_V2_S"]
    state._act("small", None)
    assert state._available("depth_model") == ["Any_V2_S", "Any_L"]


@pytest.mark.parametrize(
    "available, selected, source_size",
    [([2160, 1440, 1080, 720], "1440p", (1440, 2560)), ([], None, None), ([], "1088p", (1088, 1920))],
)
def test_status_keeps_reported_quality_and_actual_decode_visible(
    menu: Any, monkeypatch, available, selected, source_size
):
    state, snapshot, _ = menu
    snapshot["values"].update(
        source_kind="browser",
        hardware_active=selected is not None,
        available_quality=available,
        selected_quality=selected,
        source_size=source_size,
        codec="hevc",
        fallback_reason="媒体 HTTP 403",
        output_size=(2560, 1440),
        estimated_fps=27,
        screenshot_fps=30,
        streaming_fps=28,
        screen_surface="平面",
    )
    rendered = []
    monkeypatch.setattr(state, "_text", lambda draw, xy, text, *args, **kwargs: rendered.append((xy, str(text))))
    opened(state)
    click(state, "page", 4)
    rendered.clear()
    state._render()
    visible = "".join(text for (_, y), text in rendered if 418 <= y < 796)
    assert "可用档位" in visible
    if available:
        assert "来源标注，未逐档验证" in visible
        assert "2160p / 1440p / 1080p / 720p" in visible
    else:
        assert "来源画质表未取得或尺寸未标注" in visible
        assert "实际以 NVDEC 为准" in visible
    if selected is not None:
        assert f"实际档位：{selected}；编码：hevc" in visible
    assert "电脑 FPS：处理 27 / 解码 30 / 发送 28" in visible


def test_desktop_status_keeps_processing_capture_and_send_fps_visible(menu: Any, monkeypatch):
    state, snapshot, _ = menu
    snapshot["values"].update(source_kind="desktop", estimated_fps=27, screenshot_fps=30, streaming_fps=28)
    rendered = []
    monkeypatch.setattr(state, "_text", lambda draw, xy, text, *args, **kwargs: rendered.append((xy, str(text))))
    opened(state)
    click(state, "page", 4)
    rendered.clear()
    state._render()
    visible = "".join(text for (_, y), text in rendered if 418 <= y < 796)
    assert "电脑 FPS：处理 27 / 捕获 30 / 发送 28" in visible


def test_fine_step_updates_real_patch(menu: Any) -> None:
    state, _, commands = menu
    opened(state)
    click(state, "fine")
    click(state, "adjust", ("divergence", 1, "x"))
    assert commands == [("set", {"divergence": 1.005})]


def test_integer_fine_adjustment_reaches_every_pixel(menu: Any) -> None:
    state, _, _ = menu
    opened(state)
    state._act("fine", None)
    state._act("adjust", ("stream_height", -1, "x"))
    assert state._draft == {"stream_height": 1079}


def test_snapshot_ack_clears_pending_without_changing_actual_early(menu: Any) -> None:
    state, snapshot, commands = menu
    opened(state)
    click(state, "adjust", ("divergence", 1, "x"))
    assert state._waiting == {"divergence": 1.25} and snapshot["values"]["divergence"] == 1.0
    snapshot["values"]["divergence"] = 1.25
    state.update(packet(), VIDEO)
    assert state._waiting == {} and commands == [("set", {"divergence": 1.25})]
