"""Resume only validated Quest settings through fixed Python module entry points."""

import ipaddress
import json
import os
import re
import sys
import uuid
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quest_settings import validate_patch

RESUME_DIR = Path(__file__).resolve().parents[2] / "tmp"
_RESUME_NAME = re.compile(r"quest-resume-[0-9a-f]{32}\.json")
_MAX_RESUME_BYTES = 65536


def _validate_values(values: Mapping[str, Any]) -> dict[str, Any]:
    from ..base_depth_model import BaseDepthModel

    if not isinstance(values, Mapping):
        raise ValueError("Quest resume settings must be an object")
    settings = dict(values)
    address = settings.pop("bind_addr", None)
    if address is not None:
        if type(address) is not str:
            raise ValueError("Quest bind address must be an explicit private LAN IPv4 address")
        try:
            parsed_address = ipaddress.IPv4Address(address)
            networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
            if not any(parsed_address in ipaddress.IPv4Network(network) for network in networks):
                raise ValueError("Address is outside private LAN ranges")
        except ValueError as error:
            raise ValueError("Quest bind address must be an explicit private LAN IPv4 address") from error
    validated = validate_patch(
        settings, choices={"depth_model": BaseDepthModel.get_available_models(desktop=True, small_only=False)}
    )
    if "bind_addr" in values:
        validated["bind_addr"] = address
    return validated


def _resume_path(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if (
        RESUME_DIR.is_symlink()
        or candidate.parent.resolve() != RESUME_DIR
        or _RESUME_NAME.fullmatch(candidate.name) is None
        or candidate.is_symlink()
    ):
        raise ValueError("Quest resume must be a generated file inside the project tmp directory")
    return candidate


def write_resume(values: Mapping[str, Any], gui: bool) -> Path:
    if type(gui) is not bool:
        raise ValueError("Quest resume launch mode must be boolean")
    payload = {"version": 1, "gui": gui, "values": _validate_values(values)}
    RESUME_DIR.mkdir(exist_ok=True)
    resume = _resume_path(RESUME_DIR / f"quest-resume-{uuid.uuid4().hex}.json")
    temporary = resume.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, resume)
    return resume


def load_resume(path: str | Path, gui: bool) -> dict[str, Any]:
    if type(gui) is not bool:
        raise ValueError("Quest resume launch mode must be boolean")
    resume = _resume_path(path)
    with resume.open("rb") as stream:
        data = stream.read(_MAX_RESUME_BYTES + 1)
    if len(data) > _MAX_RESUME_BYTES:
        raise ValueError("Quest resume file is too large")
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Quest resume file is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "gui", "values"}
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or type(payload["gui"]) is not bool
        or payload["gui"] is not gui
    ):
        raise ValueError("Quest resume format or launch mode is invalid")
    return _validate_values(payload["values"])


def add_resume_argument(parser: ArgumentParser) -> None:
    parser.add_argument("--quest-resume", metavar="PATH", help="Restore a generated Quest restart configuration")


def apply_resume(args: Namespace, path: str | Path, gui: bool) -> None:
    for key, value in load_resume(path, gui=gui).items():
        setattr(args, key, value)
    args.quest = True
    args.local_viewer = False
    args.full_sbs = True


def restart_process(resume_path: str | Path, gui: bool) -> None:
    load_resume(resume_path, gui=gui)
    module = "iw3.desktop.gui" if gui else "iw3.desktop"
    os.execv(sys.executable, [sys.executable, "-m", module, "--quest-resume", str(_resume_path(resume_path))])
