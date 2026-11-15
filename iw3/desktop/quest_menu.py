"""A single PC-rendered settings texture for the existing Pop3D client."""

import copy
import math
import struct
import threading
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .quest_input import _rotate, _vector, ray_to_uv
from .quest_settings import FIELD_MAP, PAGES, get_fields, validate_patch


class QuestMenu:
    WIDTH, HEIGHT = 1536, 1280
    TEXTURE_ID = "iw3_settings"
    ROWS = 6

    def __init__(self, snapshot: Callable[[], dict], submit: Callable[[str, dict], None]) -> None:
        self._snapshot, self._submit = snapshot, submit
        self._lock = threading.RLock()
        self.visible = False
        self._state: dict[str, Any] = {"values": {}, "choices": {}, "status": "", "message": ""}
        self._draft: dict[str, Any] = {}
        self._waiting: dict[str, Any] = {}
        self._page, self._offset = 0, 0
        self._picker: str | None = None
        self._choice_offset = 0
        self._selected = "divergence"
        self._fine = False
        self._small_models = False
        self._message = ""
        self._confirm_until = 0.0
        self._last_texture = -math.inf
        self._dirty = True
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._buttons: list[tuple[tuple[int, int, int, int], str, Any, bool]] = []
        self._quad: dict[str, Any] = {
            "pos": [0, 0, -2.4],
            "rot": [0, 0, 0, 1],
            "tam": [2.2, 2.2 * self.HEIGHT / self.WIDTH],
        }
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.visible = False
            self._b_armed = {"esq": False, "dir": False}
            self._trigger_armed = {"esq": False, "dir": False}
            self._release_gate = True
            self._last_input: float | None = None
            self._pressed: tuple[str, tuple[int, int, int, int], str, Any, bool] | None = None
            self._repeat_at = 0.0
            self._scroll_at = 0.0
            self._laser: dict[str, Any] | None = None
            self._pointer_side = "dir"
            self._confirm_until = 0.0
            self._dirty = True

    def _sync(self) -> None:
        state = self._snapshot()
        if state != self._state:
            self._state = copy.deepcopy(state)
            self._dirty = True
        values = self._state.get("values", {})
        for pending in (self._draft, self._waiting):
            for key in list(pending):
                if key in values and values[key] == pending[key]:
                    del pending[key]
        if self._state.get("status") == "error":
            self._waiting.clear()

    def _place(self, data: dict[str, Any], video_quad: Any) -> None:
        head = data.get("cabeca")
        pos = _vector(head.get("pos"), 3) if isinstance(head, dict) else None
        rot = _vector(head.get("rot"), 4) if isinstance(head, dict) else None
        if pos is not None and rot is not None and 1e-6 < math.hypot(*rot) < math.inf:
            rot = [v / math.hypot(*rot) for v in rot]
            offset = _rotate(rot, [0, 0, -2])
            self._quad["pos"] = [a + b for a, b in zip(pos, offset)]
            self._quad["rot"] = rot
        elif isinstance(video_quad, dict):
            pos, rot = _vector(video_quad.get("pos"), 3), _vector(video_quad.get("rot"), 4)
            if pos is not None and rot is not None and 1e-6 < math.hypot(*rot) < math.inf:
                rot = [v / math.hypot(*rot) for v in rot]
                offset = _rotate(rot, [0, 0, 0.6])
                self._quad["pos"] = [a + b for a, b in zip(pos, offset)]
                self._quad["rot"] = rot

    def update(self, data: Any, video_quad: Any) -> bool:
        # 0x83 dict → 两手 pos[3]/rot[4] → UV[2] → 单纹理像素；按钮只在松开后检测按下。
        with self._lock:
            now = time.monotonic()
            hands = data.get("maos") if isinstance(data, dict) and data.get("estado") == "focado" else None
            if not isinstance(hands, dict) or any(
                not isinstance(hands.get(side), dict)
                or any(type(hands[side].get(key)) is not bool for key in ("b", "gatilho", "aperto"))
                or (stick := _vector(hands[side].get("analogico"), 2)) is None
                or any(abs(v) > 1 for v in stick)
                for side in ("esq", "dir")
            ):
                self.reset()
                return True
            hands = {side: hands[side] for side in ("esq", "dir")}
            if self._last_input is not None and now - self._last_input >= 0.5:
                self.reset()
            self._last_input = now
            toggle_side = next((side for side in ("dir", "esq") if hands[side]["b"] and self._b_armed[side]), None)
            for side in ("esq", "dir"):
                self._b_armed[side] = not hands[side]["b"]
            if toggle_side is not None:
                self._pointer_side = toggle_side
                self.visible = not self.visible
                self._release_gate = True
                self._pressed = self._laser = None
                self._trigger_armed = {"esq": False, "dir": False}
                self._confirm_until = 0
                self._dirty = True
                if self.visible:
                    self._place(data, video_quad)
                    self._sync()
                return True
            if not self.visible:
                if self._release_gate:
                    if all(
                        not hand["gatilho"] and not hand["aperto"] and max(map(abs, hand["analogico"])) <= 0.3
                        for hand in hands.values()
                    ):
                        self._release_gate = False
                    return True
                return False
            self._sync()
            self._laser = None
            if self._release_gate:
                if any(
                    hand["gatilho"] or hand["aperto"] or max(map(abs, hand["analogico"])) > 0.3
                    for hand in hands.values()
                ):
                    return True
                self._release_gate = False
            hits = {side: ray_to_uv(hands[side], self._quad) for side in ("dir", "esq")}
            pointer: str | None = (
                self._pressed[0]
                if self._pressed is not None
                else next(
                    (side for side in ("dir", "esq") if hands[side]["gatilho"] and hits[side] is not None),
                    self._pointer_side,
                )
            )
            if pointer is None or hits[pointer] is None:
                pointer = next((side for side, uv in hits.items() if uv is not None), None)
            if pointer is not None:
                self._pointer_side = pointer
            if pointer is not None and (uv := hits[pointer]) is not None:
                offset = _rotate(
                    self._quad["rot"], [(uv[0] - 0.5) * self._quad["tam"][0], (0.5 - uv[1]) * self._quad["tam"][1], 0]
                )
                self._laser = {
                    "tipo": "laser",
                    "mao": pointer,
                    "ponto": [a + b for a, b in zip(self._quad["pos"], offset)],
                }
            for side in ("dir", "esq"):
                hand, uv = hands[side], hits[side]
                if not hand["gatilho"]:
                    self._trigger_armed[side] = uv is not None
                    if self._pressed is not None and self._pressed[0] == side:
                        self._pressed = None
                    continue
                armed, self._trigger_armed[side] = self._trigger_armed[side], False
                button = self._hit(uv)
                if armed and self._pressed is None and button is not None:
                    rect, action, payload, repeat = button
                    self._pressed = (side, rect, action, payload, repeat)
                    self._repeat_at = now + 0.45
                    self._act(action, payload)
                elif self._pressed is not None and self._pressed[0] == side:
                    _, rect, action, payload, repeat = self._pressed
                    if button is None or button[:3] != (rect, action, payload):
                        self._pressed = None
                    elif repeat and now >= self._repeat_at:
                        self._repeat_at = now + 0.12
                        self._act(action, payload)
            if self._pressed is None:
                y = max((h["analogico"][1] for h in hands.values()), key=abs)
                if abs(y) < 0.3:
                    self._scroll_at = 0
                elif abs(y) >= 0.7 and now >= self._scroll_at:
                    self._scroll(-1 if y > 0 else 1)
                    self._scroll_at = now + 0.22
            return True

    def _hit(self, uv: tuple[float, float] | None) -> tuple[tuple[int, int, int, int], str, Any, bool] | None:
        if uv is not None:
            x, y = uv[0] * self.WIDTH, uv[1] * self.HEIGHT
            return next(
                (
                    button
                    for button in self._buttons
                    if button[0][0] <= x < button[0][2] and button[0][1] <= y < button[0][3]
                ),
                None,
            )
        return None

    def _available(self, key: str) -> list[Any]:
        values = list(self._state.get("choices", {}).get(key, FIELD_MAP[key].choices))
        if key == "depth_model" and self._small_models:
            small = self._state.get("choices", {}).get("depth_model_small")
            if isinstance(small, list):
                values = [v for v in values if v in small]
        if key == "gpu":
            values = [[v] if type(v) is int else v for v in values]
        if key == "window_name" and None not in values:
            values.insert(0, None)
        return values

    def _scroll(self, delta: int) -> None:
        if self._picker is not None:
            self._choice_offset = max(
                0, min(max(0, len(self._available(self._picker)) - self.ROWS), self._choice_offset + delta)
            )
        else:
            self._offset = max(0, min(max(0, len(get_fields(PAGES[self._page])) - self.ROWS), self._offset + delta))
        self._dirty = True

    def _edit(self, key: str, value: Any) -> None:
        field = FIELD_MAP[key]
        if field.immediate:
            patch = validate_patch({key: value}, self._state.get("values", {}), self._state.get("choices", {}))
            self._submit("set", patch)
            self._waiting.update(patch)
        elif self._state.get("values", {}).get(key) == value:
            self._draft.pop(key, None)
        else:
            # 跨字段约束在应用时整体校验，允许先改后端再选窗口等关联编辑。
            validate_patch(
                {key: value},
                choices={key: self._available(key)} if FIELD_MAP[key].kind in {"enum", "gpu"} else {},
                check_constraints=False,
            )
            self._draft[key] = value

    def _act(self, action: str, payload: Any) -> None:
        self._dirty = True
        self._message = ""
        if not (action == "command" and payload == "restart_app"):
            self._confirm_until = 0
        values = {**self._state.get("values", {}), **self._draft, **self._waiting}
        try:
            if action == "page":
                self._page, self._offset, self._picker = payload, 0, None
                fields = get_fields(PAGES[self._page])
                if fields:
                    self._selected = fields[0].key
            elif action == "scroll":
                self._scroll(payload)
            elif action == "help":
                self._selected = payload
            elif action == "fine":
                self._fine = not self._fine
                if self._fine:
                    self._message = "微调：强度 / 会聚 / 前景每步0.005；整数每步1；其余数值使用普通步长。"
            elif action == "small":
                self._small_models = not self._small_models
                if "depth_model_small" not in self._state.get("choices", {}):
                    self._message = "未提供小模型候选，继续显示全部模型"
            elif action == "choices":
                self._selected, self._picker, self._choice_offset = payload, payload, 0
            elif action == "back":
                self._picker = None
            elif action == "choose":
                key, value = payload
                self._edit(key, value)
                self._picker = None
            elif action == "toggle" and payload in values:
                self._selected = payload
                self._edit(payload, not values[payload])
            elif action == "default":
                self._selected = payload
                self._edit(payload, None)
            elif action == "follow" and "edge_dilation" in values:
                old = values["edge_dilation"]
                self._edit("edge_dilation", old if type(old) is int else old[0])
            elif action == "adjust":
                key, direction, axis = payload
                self._selected = key
                if key not in values:
                    return
                field, old = FIELD_MAP[key], values[key]
                if key == "edge_dilation":
                    xy = [old, old] if type(old) is int else list(old)
                    index = 0 if axis == "x" else 1
                    xy[index] = max(0, min(20, xy[index] + direction))
                    self._edit(key, xy[0] if type(old) is int and axis == "x" else xy)
                else:
                    step = (
                        0.005 if self._fine and key in {"divergence", "convergence", "foreground_scale"} else field.step
                    )
                    if self._fine and field.kind in {"int", "resolution"}:
                        step = 1
                    old = field.minimum if old is None and key == "resolution" else old
                    value = round(max(field.minimum, min(field.maximum, old + direction * step)), 6)
                    self._edit(key, int(value) if field.kind in {"int", "resolution"} else value)
            elif action == "command":
                if payload == "apply":
                    patch = validate_patch(self._draft, self._state.get("values", {}), self._state.get("choices", {}))
                    self._submit("apply", patch)
                elif payload == "restart_app":
                    if time.monotonic() < self._confirm_until:
                        self._confirm_until = 0
                        self._submit(payload, {})
                    else:
                        self._confirm_until = time.monotonic() + 5
                        self._message = "5秒内再次点击：保存已应用设置并重开 iw3，头显将短暂断线"
                else:
                    if payload == "restore":
                        self._draft.clear()
                        self._waiting.clear()
                    self._submit(payload, {})
        except (ValueError, RuntimeError) as error:
            self._message = str(error)

    def layers(self) -> list[dict]:
        with self._lock:
            if not self.visible:
                return []
            quad = {
                "tipo": "quad",
                "tex": self.TEXTURE_ID,
                "olho": "ambos",
                "alpha": False,
                "rect": [0, 0, self.WIDTH, self.HEIGHT],
                **copy.deepcopy(self._quad),
            }
            return [quad] + ([copy.deepcopy(self._laser)] if self._laser is not None else [])

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            paths = (
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/PingFang.ttc",
            )
            path = next((p for p in paths if Path(p).is_file()), None)
            if path is None:
                raise RuntimeError("Quest中文菜单需要系统中文字体（微软雅黑或 Noto Sans CJK）")
            self._fonts[size] = ImageFont.truetype(path, size)
        return self._fonts[size]

    def _text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: Any,
        width: int,
        size: int = 27,
        fill: str = "#e5eaf3",
    ) -> None:
        text, font = str(text), self._font(size)
        while text and draw.textlength(text, font=font) > width:
            text = text[:-2] + "…" if not text.endswith("…") else text[:-2] + "…"
        draw.text(xy, text, font=font, fill=fill)

    def _wrap(
        self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, size: int = 27, lines: int = 6
    ) -> None:
        line, count = "", 0
        for char in text:
            if char == "\n" or draw.textlength(line + char, font=self._font(size)) > width:
                self._text(draw, (x, y), line, width, size)
                y, line, count = y + size + 10, "", count + 1
                if count >= lines:
                    return
            if char != "\n":
                line += char
        self._text(draw, (x, y), line, width, size)

    def _button(
        self,
        draw: ImageDraw.ImageDraw,
        rect: tuple[int, int, int, int],
        label: str,
        action: str,
        payload: Any = None,
        repeat: bool = False,
        active: bool = False,
    ) -> None:
        self._buttons.append((rect, action, payload, repeat))
        draw.rounded_rectangle(rect, radius=10, fill="#275b82" if active else "#26364b", outline="#516b86", width=2)
        self._text(draw, (rect[0] + 14, rect[1] + 9), label, rect[2] - rect[0] - 25)

    @staticmethod
    def _value(key: str, value: Any) -> str:
        if key == "window_name" and not value:
            return "整个桌面（所选显示器）"
        if key == "quest_screen_curvature":
            return "平面（0°）" if value == 0 else f"曲面 {value:g}°"
        if key == "edge_dilation":
            return f"横 {value} / 纵跟随" if type(value) is int else f"横 {value[0]} / 纵 {value[1]}"
        if value is None:
            return "默认 / 关闭" if key in {"autocrop", "pad_mode"} else "默认 / 未指定"
        if key == "quest_no_audio":
            return "关" if value else "开"
        if type(value) is bool:
            return "开" if value else "关"
        if key in {"quest_audio_gain", "quest_ambient_brightness"}:
            return f"{value * 100:g}%"
        if type(value) is float:
            return f"{value:g}"
        if key == "pad_mode" and value == "tblr":
            return "关闭"
        return str(value)

    def _render(self) -> Image.Image:
        image = Image.new("RGBA", (self.WIDTH, self.HEIGHT), "#101a29")
        draw = ImageDraw.Draw(image)
        self._buttons = []
        self._text(draw, (32, 25), "iw3 · Quest 完整设置", 1200, 42)
        self._text(draw, (1190, 40), "B / Y 关闭", 315)
        for index, page in enumerate(PAGES):
            tab_width = (self.WIDTH - 64) // len(PAGES)
            x = 32 + index * tab_width
            self._button(draw, (x, 100, x + tab_width - 12, 158), page, "page", index, active=index == self._page)
        status = {"loading": "正在加载", "running": "正在串流", "error": "失败 / 可重试", "ready": "已准备"}.get(
            self._state.get("status", ""), self._state.get("status", "")
        )
        self._text(draw, (34, 174), f"电脑状态：{status}　待应用 {len(self._draft)} 项", 1460)
        message = self._message or self._state.get("message", "")
        self._text(draw, (34, 212), message, 1460, 25, "#ffd28a")
        self._button(draw, (32, 258, 140, 310), "上移", "scroll", -1)
        self._button(draw, (152, 258, 260, 310), "下移", "scroll", 1)
        if self._picker is not None:
            self._button(draw, (280, 258, 430, 310), "返回", "back")
            self._text(draw, (455, 267), FIELD_MAP[self._picker].label + " · 选择候选", 1030)
            candidates = self._available(self._picker)
            for index, value in enumerate(candidates[self._choice_offset : self._choice_offset + self.ROWS]):
                y = 326 + index * 88
                self._button(
                    draw,
                    (40, y, 1496, y + 74),
                    self._value(self._picker, value),
                    "choose",
                    (self._picker, value),
                    active=value == self._draft.get(self._picker, self._state.get("values", {}).get(self._picker)),
                )
            if not candidates:
                self._wrap(
                    draw, "当前没有可用候选。捕获窗口可先刷新；设备或模型列表由电脑实际环境提供。", 60, 360, 1400
                )
        else:
            self._button(
                draw, (280, 258, 560, 310), "微调 " + ("开" if self._fine else "关"), "fine", active=self._fine
            )
            if self._page == 1:
                self._button(
                    draw,
                    (580, 258, 950, 310),
                    "只列小模型 " + ("开" if self._small_models else "关"),
                    "small",
                    active=self._small_models,
                )
            if self._page == 3:
                self._button(draw, (1060, 258, 1496, 310), "刷新可用窗口", "command", "refresh_windows")
            fields = get_fields(PAGES[self._page])
            for index, field in enumerate(fields[self._offset : self._offset + self.ROWS]):
                y, key = 324 + index * 88, field.key
                draw.rounded_rectangle((32, y, 1504, y + 80), radius=10, fill="#17263a")
                self._text(draw, (46, y + 4), field.label, 455)
                self._text(draw, (46, y + 44), "即时" if field.immediate else "待应用 · 需重新加载", 455, 22, "#91c3dc")
                current = self._state.get("values", {})
                self._text(
                    draw,
                    (510, y + 4),
                    "现：" + (self._value(key, current[key]) if key in current else "未提供"),
                    490,
                    25,
                )
                pending = self._draft if key in self._draft else self._waiting
                if key in pending:
                    self._text(
                        draw,
                        (510, y + 43),
                        ("待：" if key in self._draft else "更新中：") + self._value(key, pending[key]),
                        490,
                        23,
                        "#ffd28a",
                    )

                def control_rect(a: int, b: int) -> tuple[int, int, int, int]:
                    return (a, y + 10, b, y + 67)

                if field.kind == "edge":
                    for x, label, axis, direction in (
                        (1012, "横−", "x", -1),
                        (1090, "横+", "x", 1),
                        (1168, "纵−", "y", -1),
                        (1246, "纵+", "y", 1),
                    ):
                        self._button(
                            draw, control_rect(x, x + 72), label, "adjust", (key, direction, axis), repeat=True
                        )
                    self._button(draw, control_rect(1324, 1402), "随", "follow")
                elif field.kind in {"number", "int", "resolution"} and key not in self._state.get("choices", {}):
                    self._button(draw, control_rect(1080, 1170), "−", "adjust", (key, -1, "x"), repeat=True)
                    self._button(draw, control_rect(1182, 1272), "+", "adjust", (key, 1, "x"), repeat=True)
                    if field.kind == "resolution":
                        self._button(draw, control_rect(1284, 1402), "默认", "default", key)
                elif field.kind == "bool":
                    self._button(draw, control_rect(1110, 1402), "切换开关", "toggle", key)
                else:
                    self._button(draw, control_rect(1110, 1402), "选择…", "choices", key)
                self._button(draw, control_rect(1410, 1494), "?", "help", key)
            if self._page == 4:
                self._render_status(draw)
        selected = FIELD_MAP[self._selected]
        title, help_text = selected.label + " · 调节说明", selected.help
        if self._page == 4 and self._picker is None and selected.page != PAGES[4]:
            title = "屏幕与状态 · 操作说明"
            help_text = (
                "菜单打开时暂停抓屏和网页输入。B / Y关闭后，先松开两手扳机与握柄、摇杆回中，再恢复控制。"
                "屏幕居中会恢复默认摆放；三项FPS都来自电脑，不是Quest刷新率，也不是头显接收确认。"
            )
        self._text(draw, (36, 882), title, 1450, 29, "#91d7ff")
        self._wrap(draw, help_text, 36, 932, 1460, lines=4)
        for rect, label, command in (
            ((32, 1120, 456, 1190), "应用并重新加载", "apply"),
            ((470, 1120, 792, 1190), "恢复上次成功设置", "restore"),
            ((806, 1120, 1100, 1190), "重启串流", "restart_stream"),
            (
                (1114, 1120, 1504, 1190),
                "再次点击确认重开" if time.monotonic() < self._confirm_until else "重启 iw3 程序",
                "restart_app",
            ),
        ):
            self._button(draw, rect, label, "command", command)
        self._text(
            draw, (34, 1210), "双手指向 + 扳机点击；长按加减连续调节；摇杆上下滚动。重启使用已应用设置。", 1465, 25
        )
        return image

    def _render_status(self, draw: ImageDraw.ImageDraw) -> None:
        values = self._state.get("values", {})
        lines = [
            "握柄抓住画面并移动：平移屏幕；同手摇杆：上下调远近、左右调大小。",
            "调整屏幕位置只改变观看大小，不增加源像素，也不是AI会聚。",
        ]
        for key, label in (
            ("depth_model", "当前模型"),
            ("method", "合成方法"),
            ("output_size", "输出尺寸"),
            ("estimated_fps", "电脑处理 FPS"),
            ("screenshot_fps", "捕获 FPS"),
            ("streaming_fps", "电脑发送 FPS"),
            ("bind_addr", "当前连接地址"),
            ("screen_surface", "当前屏幕形状"),
        ):
            if key in values:
                lines.append(f"{label}：{values[key]}")
        self._wrap(draw, "\n".join(lines), 40, 418, 1450, 26, lines=10)
        self._button(draw, (40, 796, 450, 858), "屏幕居中（重新定位）", "command", "recenter")

    def texture(self) -> bytes | None:
        with self._lock:
            if not self.visible:
                return None
            self._sync()
            now = time.monotonic()
            if self._confirm_until and now >= self._confirm_until:
                self._confirm_until = 0
                self._message = "重启确认已过期，请重新点击"
                self._dirty = True
            if not self._dirty or now - self._last_texture < 0.1:
                return None
            image = self._render()
            identifier = self.TEXTURE_ID.encode("ascii")
            payload = bytes([len(identifier)]) + identifier + struct.pack("!HHB", self.WIDTH, self.HEIGHT, 1)
            payload += zlib.compress(image.tobytes(), level=3)
            self._dirty, self._last_texture = False, now
            return payload
