import math
import time
from typing import Any


def _vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size or any(type(v) not in (int, float) for v in value):
        return None
    try:
        result = [float(v) for v in value]
    except OverflowError:
        return None
    return result if all(math.isfinite(v) for v in result) else None


def _rotate(q: list[float], v: list[float]) -> list[float]:
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx]


def ray_to_uv(hand: Any, quad: Any) -> tuple[float, float] | None:
    if not isinstance(hand, dict) or not isinstance(quad, dict):
        return None
    origin, aim = _vector(hand.get("pos"), 3), _vector(hand.get("rot"), 4)
    center, rotation = _vector(quad.get("pos"), 3), _vector(quad.get("rot"), 4)
    size = _vector(quad.get("tam"), 2)
    if origin is None or aim is None or center is None or rotation is None or size is None or min(size) <= 0:
        return None
    for q in (aim, rotation):
        norm = math.hypot(*q)
        if norm < 1e-6 or not math.isfinite(norm):
            return None
        q[:] = [v / norm for v in q]
    inverse = [-rotation[0], -rotation[1], -rotation[2], rotation[3]]
    local = _rotate(inverse, [a - b for a, b in zip(origin, center)])
    direction = _rotate(inverse, _rotate(aim, [0, 0, -1]))
    # OpenXR quad 本地正面朝 +Z，瞄准射线沿手柄 -Z；只接受正面向内的交点。
    if not all(math.isfinite(v) for v in local + direction) or local[2] <= 0 or direction[2] >= -1e-6:
        return None
    distance = -local[2] / direction[2]
    u = 0.5 + (local[0] + distance * direction[0]) / size[0]
    v = 0.5 - (local[1] + distance * direction[1]) / size[1]
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


class QuestController:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._pressed = False
        self._trigger_armed = False
        self._stick_armed = False
        self._last_valid: float | None = None

    def reset(self) -> None:
        self._trigger_armed = self._stick_armed = False
        self._last_valid = None
        try:
            if self._pressed:
                self.backend.mouse_up()
                self._pressed = False
        finally:
            self.backend.release_keys()

    def expire(self) -> float:
        if self._last_valid is not None:
            remaining = 0.5 - (time.monotonic() - self._last_valid)
            if remaining > 0:
                return remaining
            self.reset()
        return 0.5

    def update(self, data: Any, quad: dict[str, Any]) -> None:
        self.expire()
        hands = data.get("maos") if isinstance(data, dict) and data.get("estado") == "focado" else None
        hand = hands.get("dir") if isinstance(hands, dict) else None
        uv = ray_to_uv(hand, quad)
        stick = _vector(hand.get("analogico"), 2) if isinstance(hand, dict) else None
        trigger = hand.get("gatilho") if isinstance(hand, dict) else None
        if uv is None or stick is None or any(abs(v) > 1 for v in stick) or type(trigger) is not bool:
            self.reset()
            return
        if not self.backend.move(*uv):
            self.reset()
            return
        self._last_valid = time.monotonic()
        if trigger:
            self._stick_armed = False
            if self._trigger_armed:
                self._trigger_armed = False
                # 先记录所有权：后端部分发送后抛异常，服务器 finally 仍能释放左键。
                self._pressed = True
                if not self.backend.mouse_down():
                    self.reset()
            return
        self._trigger_armed = True
        if self._pressed:
            self.backend.mouse_up()
            self._pressed = False
        y = stick[1]
        # 回中 0.3、触发 0.7 形成迟滞，抖动及持续拨住都不会连切视频。
        if abs(y) <= 0.3:
            self._stick_armed = True
        elif abs(y) >= 0.7 and self._stick_armed:
            self._stick_armed = False
            if not self.backend.tap_key("up" if y > 0 else "down"):
                self.reset()
