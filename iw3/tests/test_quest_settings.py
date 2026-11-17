import importlib
from typing import Any

import pytest


def schema() -> Any:
    return importlib.import_module("iw3.desktop.quest_settings")


def test_schema_covers_quest_fields_with_chinese_help() -> None:
    fields = schema().get_fields()
    keys = {field.key for field in fields}
    assert keys == {
        "divergence",
        "convergence",
        "convergence_mode",
        "foreground_scale",
        "edge_dilation",
        "synthetic_view",
        "preserve_screen_border",
        "ema_normalize",
        "ema_decay",
        "depth_model",
        "resolution",
        "method",
        "gpu",
        "compile",
        "stream_height",
        "stream_fps",
        "quest_bitrate",
        "quest_no_audio",
        "quest_audio_source",
        "quest_audio_gain",
        "screenshot",
        "monitor_index",
        "window_name",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
        "autocrop",
        "pad_mode",
        "quest_control",
        "quest_screen_curvature",
        "quest_ambient",
        "quest_ambient_style",
        "quest_ambient_brightness",
        "quest_ambient_spread",
        "quest_ambient_fill_borders",
        "quest_ambient_feather",
    }
    assert all(field.label and len(field.help) >= 25 for field in fields)
    assert {field.page for field in fields} <= set(schema().PAGES)
    assert schema().get_fields(schema().PAGES[0])


@pytest.mark.parametrize(
    "patch",
    [
        {"stream_fps": True},
        {"stream_fps": 30.5},
        {"stream_fps": 0},
        {"stream_height": 4321},
        {"quest_bitrate": 101},
        {"divergence": float("nan")},
        {"convergence": float("inf")},
        {"quest_audio_gain": 10**1000},
        {"ema_decay": 1},
        {"compile": 1},
        {"method": "unknown"},
        {"gpu": 0},
        {"gpu": [-1]},
        {"gpu": [True]},
        {"edge_dilation": [2]},
        {"edge_dilation": [2, None]},
        {"edge_dilation": [2, -1]},
        {"crop_top": "3"},
        {"window_name": 3},
        {"unknown": 1},
        {"__class__": "anything"},
        {"quest_audio_source": "mic"},
        {"resolution": 191},
    ],
)
def test_invalid_values_rejected(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        schema().validate_patch(patch)


def test_native_shapes_and_y_follow_semantics_are_preserved() -> None:
    original = {"edge_dilation": [2, 0], "gpu": [0], "resolution": None, "window_name": None}
    validated = schema().validate_patch(original)
    assert validated == original
    assert schema().validate_patch({"edge_dilation": 2}) == {"edge_dilation": 2}
    validated["gpu"].append(1)
    assert original["gpu"] == [0]


def test_dynamic_choices_validate_models_gpu_and_monitor() -> None:
    choices = {"depth_model": ["Any_V2_S"], "gpu": [[0]], "monitor_index": [0, 1]}
    assert schema().validate_patch({"gpu": [0]}, choices=choices) == {"gpu": [0]}
    for patch in ({"depth_model": "Fake"}, {"gpu": [1]}, {"monitor_index": 2}):
        with pytest.raises(ValueError):
            schema().validate_patch(patch, choices=choices)


@pytest.mark.parametrize(
    "patch,current",
    [
        ({"quest_control": True}, {"screenshot": "pil", "window_name": "Browser", "pad_mode": None}),
        ({"pad_mode": "16:9"}, {"quest_control": True, "screenshot": "mss", "window_name": "Browser"}),
        ({"window_name": None}, {"quest_control": True, "screenshot": "wc_mp", "crop_top": 1}),
        ({"resolution": 512}, {"depth_model": "DepthPro"}),
        ({"method": "forward_fill"}, {"preserve_screen_border": True}),
    ],
)
def test_cross_field_constraints_use_current_values(patch: dict[str, Any], current: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        schema().validate_patch(patch, current)


def test_related_edits_can_be_validated_atomically() -> None:
    patch = {"quest_control": True, "screenshot": "wc_mp", "window_name": "Browser", "pad_mode": None}
    assert schema().validate_patch(patch) == patch
    assert schema().COMMANDS == {
        "set",
        "apply",
        "restart_stream",
        "restart_app",
        "restore",
        "refresh_windows",
        "recenter",
        "exit_fullscreen",
    }


@pytest.mark.parametrize("capture", ["wc_mp", "mss", "wc_cuda"])
@pytest.mark.parametrize("window_name", [None, "", "Browser"])
def test_control_supports_window_or_uncropped_monitor(capture: str, window_name: str | None) -> None:
    patch = {"quest_control": True, "screenshot": capture, "window_name": window_name, "pad_mode": None}
    assert schema().validate_patch(patch) == patch


def test_native_cli_padding_default_can_be_restored() -> None:
    assert schema().validate_patch({"pad_mode": "tblr"}) == {"pad_mode": "tblr"}


def test_draft_validation_checks_types_before_cross_field_apply() -> None:
    assert schema().validate_patch({"quest_control": True}, check_constraints=False) == {"quest_control": True}
    with pytest.raises(ValueError):
        schema().validate_patch({"quest_control": 1}, check_constraints=False)
