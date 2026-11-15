import math
import time
from typing import Any

from .quest_input import _rotate, _vector, ray_to_uv


class QuestScreen:
    def __init__(self, aspect: float) -> None:
        value = _vector([aspect], 1)
        if value is None or value[0] <= 0 or not math.isfinite(8 * value[0]):
            raise ValueError("Screen aspect must be finite and positive")
        self._aspect = value[0]
        self._origin = [0.0, 0.0, 0.0]
        self._rotation = [0.0, 0.0, 0.0, 1.0]
        self._position = [0.0, 0.0, -3.0]
        self._width = 3.6
        self._held = False
        self.reset()

    @property
    def quad(self) -> dict[str, list[float]]:
        position = _rotate(self._rotation, self._position)
        return {
            "pos": [a + b for a, b in zip(self._origin, position)],
            "rot": self._rotation.copy(),
            "tam": [self._width, self._width * self._aspect],
        }

    @property
    def active(self) -> bool:
        return self._hand is not None

    def set_aspect(self, aspect: float) -> None:
        value = _vector([aspect], 1)
        if value is None or value[0] <= 0 or not math.isfinite(8 * value[0]):
            raise ValueError("Screen aspect must be finite and positive")
        self._aspect = value[0]

    def recenter(self, pose: tuple[list[float], list[float]]) -> None:
        position, rotation = pose
        self._rotation = rotation.copy()
        offset = _rotate(rotation, [0.0, 0.0, 3.0])
        self._origin = [a + b for a, b in zip(position, offset)]
        self._position = [0.0, 0.0, -3.0]
        self._width = 3.6
        self.reset()

    def reset(self) -> None:
        self._hand: str | None = None
        self._last_position: list[float] | None = None
        self._last_time: float | None = None
        self._armed = False

    def update(self, data: Any, geometry: dict[str, Any] | None = None) -> bool:
        # 0x83 dict → 两手 pos[3]/rot[4]/analogico[2] → quad pos[3]/rot[4]/tam[2]。
        hands = data.get("maos") if isinstance(data, dict) and data.get("estado") == "focado" else None
        if not isinstance(hands, dict) or any(
            not isinstance(hands.get(side), dict) or type(hands[side].get("aperto")) is not bool
            for side in ("esq", "dir")
        ):
            self.reset()
            return True
        now = time.monotonic()
        elapsed = now - self._last_time if self._last_time is not None else 0.0
        if elapsed >= 0.5:
            self.reset()
        self._last_time = now
        if not any(hands[side]["aperto"] for side in ("esq", "dir")):
            suppressed = self._held
            self._held = False
            self._hand = None
            self._last_position = None
            self._armed = True
            return suppressed
        self._held = True
        armed, self._armed = self._armed, False
        if self._hand is not None and not hands[self._hand]["aperto"]:
            self._hand = None
            self._last_position = None
        if self._hand is None:
            if armed:
                for side in ("esq", "dir"):
                    if (
                        hands[side]["aperto"]
                        and ray_to_uv(hands[side], self.quad if geometry is None else geometry) is not None
                    ):
                        self._hand = side
                        break
            if self._hand is None:
                return True
            elapsed = 0.0
        hand = hands[self._hand]
        position, rotation = _vector(hand.get("pos"), 3), _vector(hand.get("rot"), 4)
        stick = _vector(hand.get("analogico"), 2)
        if position is None or rotation is None or stick is None:
            self.reset()
            return True
        norm = math.hypot(*rotation)
        if (
            norm < 1e-6
            or not math.isfinite(norm)
            or any(abs(value) > 1 for value in stick)
            or max(map(abs, position)) > 10_000
        ):
            self.reset()
            return True
        if self._last_position is None:
            self._last_position = position
            return True
        delta = [a - b for a, b in zip(position, self._last_position)]
        if math.hypot(*delta) > 1.0:
            # 单包超过一米视为追踪跳变；松开握柄后才能重新抓取。
            self.reset()
            return True
        self._last_position = position
        inverse = [-value for value in self._rotation[:3]] + [self._rotation[3]]
        delta = _rotate(inverse, delta)
        dt = max(0.0, min(elapsed, 0.1))
        x, y = (math.copysign(max(0.0, (abs(value) - 0.2) / 0.8), value) for value in stick)
        delta[2] -= 1.5 * y * dt
        self._position = [
            max(-5.0, min(5.0, self._position[0] + delta[0])),
            max(-5.0, min(5.0, self._position[1] + delta[1])),
            max(-10.0, min(-1.0, self._position[2] + delta[2])),
        ]
        self._width = max(0.8, min(8.0, self._width * math.exp(0.8 * x * dt)))
        return True
