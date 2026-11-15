import json
import sys
from pathlib import Path

import pytest


def test_resume_round_trip_applies_quest_defaults_and_preserves_values() -> None:
    from iw3.desktop import quest_restart, utils

    values = {
        "depth_model": "Any_V2_S",
        "gpu": [0],
        "divergence": 3.5,
        "edge_dilation": [2, 0],
        "quest_audio_gain": 1.7,
        "bind_addr": "127.0.0.1",
    }
    resume = quest_restart.write_resume(values, gui=False)
    assert resume.parent == quest_restart.RESUME_DIR
    assert quest_restart.load_resume(resume, gui=False) == values
    args = utils.create_parser().parse_args([])
    quest_restart.apply_resume(args, resume, gui=False)
    assert args.quest and not args.local_viewer and args.full_sbs
    assert args.divergence == 3.5 and args.edge_dilation == [2, 0]
    assert args.quest_audio_gain == 1.7 and args.bind_addr == "127.0.0.1"


@pytest.mark.parametrize(
    "values",
    [
        {"password": "secret"},
        {"command": "calc.exe"},
        {"stream_fps": True},
        {"gpu": "0"},
        {"quest_audio_gain": float("nan")},
        {"bind_addr": "example.com"},
        {"bind_addr": "127.0.0.1 & calc.exe"},
        {"bind_addr": "::1"},
        {"bind_addr": "8.8.8.8"},
        {"depth_model": "not-a-model"},
    ],
)
def test_resume_rejects_unknown_fields_and_invalid_native_values(values) -> None:
    from iw3.desktop import quest_restart

    with pytest.raises(ValueError):
        quest_restart.write_resume(values, gui=True)


def test_resume_cannot_read_an_arbitrary_path_or_wrong_launch_mode() -> None:
    from iw3.desktop import quest_restart

    with pytest.raises(ValueError):
        quest_restart.load_resume(Path(__file__), gui=True)
    resume = quest_restart.write_resume({"gpu": [0]}, gui=True)
    with pytest.raises(ValueError):
        quest_restart.load_resume(resume, gui=False)
    payload = json.loads(resume.read_text(encoding="utf-8"))
    payload["command"] = "calc.exe"
    resume.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        quest_restart.load_resume(resume, gui=True)


def test_resume_supports_large_models_without_the_gui_small_filter() -> None:
    from iw3.desktop import quest_restart

    values = {"depth_model": "Any_L", "gpu": [0]}
    resume = quest_restart.write_resume(values, gui=True)
    assert quest_restart.load_resume(resume, gui=True) == values


@pytest.mark.parametrize("gui,module", [(False, "iw3.desktop"), (True, "iw3.desktop.gui")])
def test_process_replacement_uses_the_same_interpreter_and_fixed_argv(monkeypatch, gui, module) -> None:
    from iw3.desktop import quest_restart

    resume = quest_restart.write_resume({"gpu": [0]}, gui=gui)
    called = []
    monkeypatch.setattr(quest_restart.os, "execv", lambda executable, argv: called.append((executable, argv)))
    quest_restart.restart_process(resume, gui=gui)
    assert called == [(sys.executable, [sys.executable, "-m", module, "--quest-resume", str(resume)])]


def test_cli_resume_starts_original_quest_stream(monkeypatch) -> None:
    from iw3.desktop import cli, quest_restart

    resume = quest_restart.write_resume({"gpu": [0], "stream_fps": 42, "window_name": "Browser"}, gui=False)
    monkeypatch.setattr(sys, "argv", ["iw3.desktop", "--quest-resume", str(resume)])
    monkeypatch.setattr(cli, "init_win32", lambda: None)
    monkeypatch.setattr(cli.os, "system", lambda command: None)
    monkeypatch.setattr(cli, "set_state_args", lambda args: None)
    started = []
    monkeypatch.setattr(cli, "iw3_desktop_main", lambda args, **kwargs: started.append(args))
    cli.cli_main()
    assert len(started) == 1 and started[0].quest
    assert started[0].stream_fps == 42 and started[0].window_name == "Browser"
