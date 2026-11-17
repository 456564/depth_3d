"""头显操作只投递给当前网页目标，不向 Windows 注入坐标。"""

import math
from typing import Any


class BrowserInput:
    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self._generation: str | None = None
        self._pressed = False

    def move(self, u: float, v: float) -> bool:
        self._generation = None
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in (u, v)):
            return False
        state = self.bridge.status()
        generation = state.get("generation")
        if not state.get("fresh") or not isinstance(generation, str) or not generation:
            return False
        self._generation = generation
        return True

    def _current(self) -> bool:
        state = self.bridge.status()
        return bool(self._generation is not None and state.get("fresh") and state.get("generation") == self._generation)

    def mouse_down(self) -> bool:
        if not self._current() or not self.bridge.status().get("media_available", True):
            self._pressed = False
            return False
        if not self._pressed:
            if not self.bridge.queue_command("pause_toggle", self._generation):
                return False
            self._pressed = True
        return True

    def mouse_up(self) -> None:
        self._pressed = False

    def tap_key(self, direction: str) -> bool:
        commands = {"up": "previous", "down": "next", "escape": "exit_fullscreen"}
        if direction not in commands:
            raise ValueError("Only up/down navigation and escape are allowed")
        return self._current() and self.bridge.queue_command(commands[direction], self._generation)

    def release_keys(self) -> None:
        self._generation = None
        self._pressed = False
