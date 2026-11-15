"""Small, edge-colored ambient texture for the existing Pop3D client."""

import math
import struct
import time
import zlib

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

_Rect = tuple[float, float, float, float]
_FULL: _Rect = (0.0, 0.0, 1.0, 1.0)
_SAMPLE_H, _SAMPLE_W = 72, 128


class _ContentWindow:
    def __init__(self) -> None:
        self.active = _FULL
        self.pending: _Rect | None = None
        self.since = 0.0

    def update(self, rect: _Rect, now: float) -> _Rect:
        left, top, right, bottom = self.active
        # 新内容立即放开；只有持续稳定的黑条才缩小透明保护区。
        self.active = (min(left, rect[0]), min(top, rect[1]), max(right, rect[2]), max(bottom, rect[3]))
        if rect == self.active:
            self.pending = None
        elif self.pending is None or any(
            abs(a - b) > tolerance for a, b, tolerance in zip(rect, self.pending, (1 / _SAMPLE_W, 1 / _SAMPLE_H) * 2)
        ):
            self.pending, self.since = rect, now
        elif now >= math.nextafter(self.since + 0.2, -math.inf):
            self.active, self.pending = rect, None
        return self.active


class AmbientBackground:
    WIDTH, HEIGHT = 320, 180
    TEXTURE_ID = "iw3_ambient"

    def __init__(self) -> None:
        x = torch.linspace(0, 1, 32).view(1, 1, 32)
        y = torch.linspace(0, 1, 18).view(1, 18, 1)
        self._weights = (torch.exp(-6 * x), torch.exp(-6 * (1 - x)), torch.exp(-6 * y), torch.exp(-6 * (1 - y)))
        self._weight_sum = sum(self._weights[1:], self._weights[0])
        self._x = torch.linspace(0, 1, self.WIDTH)[None, :]
        self._y = torch.linspace(0, 1, self.HEIGHT)[:, None]
        self._colors: torch.Tensor | None = None
        self._last_tick: float | None = None
        self._signature: tuple[object, ...] | None = None
        self._source_window = _ContentWindow()
        self._display_window = _ContentWindow()

    def reset(self) -> None:
        self._colors = None
        self._last_tick = None
        self._signature = None
        self._source_window = _ContentWindow()
        self._display_window = _ContentWindow()

    @staticmethod
    def _validate_frame(frame: torch.Tensor) -> None:
        if (
            not isinstance(frame, torch.Tensor)
            or frame.ndim != 3
            or frame.shape[0] != 3
            or min(frame.shape[1:]) <= 0
            or not frame.is_floating_point()
        ):
            raise ValueError("ambient frame must be floating RGB [3,H,W]")

    @staticmethod
    def _check_values(reduced: torch.Tensor) -> None:
        low, high = torch.aminmax(reduced)
        if not 0 <= low <= high <= 1:
            raise ValueError("ambient RGB values must be finite and between 0 and 1")

    @staticmethod
    def _content_rect(value: torch.Tensor, strict: bool, high_quantile: float = 0.9) -> _Rect:
        if float((value > 32 / 255).float().mean()) < 0.1:
            return _FULL
        axes = []
        for dim in (0, 1):
            if strict:
                dark = value.amax(dim=dim) <= 16 / 255
            else:
                low, high = value.quantile(0.1, dim=dim), value.quantile(high_quantile, dim=dim)
                dark = (high <= 16 / 255) & (high - low <= 8 / 255)
            positions = torch.nonzero(~dark).flatten()
            total = value.shape[1 - dim]
            if not positions.numel():
                axes.append((0, total))
                continue
            start, stop = int(positions[0]), int(positions[-1]) + 1
            far = total - stop
            confirmed = min(start, far) >= 2 and abs(start - far) <= 2 and stop - start >= total * 0.15
            axes.append((start, stop) if confirmed else (0, total))
        (left, right), (top, bottom) = axes
        content = value[top:bottom, left:right]
        coverage = 0.5 if strict else 0.35
        if float((content > 32 / 255).float().mean()) < coverage or float(content.quantile(0.9)) < 48 / 255:
            return _FULL
        # 显示额外保护一格；取色直接从内容边缘开始，避免再混入黑条。
        guard = 1 if strict else 0
        return (
            max(0, left - guard) / _SAMPLE_W,
            max(0, top - guard) / _SAMPLE_H,
            min(_SAMPLE_W, right + guard) / _SAMPLE_W,
            min(_SAMPLE_H, bottom + guard) / _SAMPLE_H,
        )

    def _display_rect(self, frame: torch.Tensor, feather: float, scale: tuple[float, float]) -> _Rect:
        # 最终每眼RGB[3,H,W] → max-pool[1,72,128]，一像素亮点也保留。
        value = F.adaptive_max_pool2d(frame.detach().amax(dim=0, keepdim=True), (_SAMPLE_H, _SAMPLE_W))
        value = value.squeeze(0).to(device="cpu", dtype=torch.float32)
        self._check_values(value)
        rect = self._content_rect(value, strict=True)
        # 主体用多数行/列判断，避免较高字幕被并入主体后漏掉渐隐保护。
        primary = self._content_rect(value, strict=False, high_quantile=0.5)
        if primary == _FULL:
            return rect
        left, top, right, bottom = primary
        outside = torch.ones_like(value, dtype=torch.bool)
        outside[
            max(0, round(top * _SAMPLE_H) - 2) : min(_SAMPLE_H, round(bottom * _SAMPLE_H) + 2),
            max(0, round(left * _SAMPLE_W) - 2) : min(_SAMPLE_W, round(right * _SAMPLE_W) + 2),
        ] = False
        marks = torch.nonzero(outside & (value > 16 / 255))
        if not marks.numel():
            return rect
        protected = list(rect)
        for axis, coordinate, samples, pixels in ((0, 1, _SAMPLE_W, self.WIDTH), (1, 0, _SAMPLE_H, self.HEIGHT)):
            start = int(marks[:, coordinate].min()) / samples
            stop = (int(marks[:, coordinate].max()) + 1) / samples
            width = rect[axis + 2] - rect[axis]
            # 只扩会碰到黑条亮字的轴；保留一整纹素余量覆盖双线性采样。
            guard = feather * width + scale[axis] / (pixels - 1)
            if start - rect[axis] < guard or rect[axis + 2] - stop < guard:
                padding = guard / (1 - 2 * feather)
                protected[axis] -= padding
                protected[axis + 2] += padding
        return protected[0], protected[1], protected[2], protected[3]

    def _alpha(self, rect: _Rect, scale: tuple[float, float], feather: float) -> Image.Image:
        sx, sy = scale
        u, v = (self._x - 0.5) * sx + 0.5, (self._y - 0.5) * sy + 0.5
        left, top, right, bottom = rect
        distance_x = torch.minimum(u - left, right - u) / (right - left)
        distance_y = torch.minimum(v - top, bottom - v) / (bottom - top)
        distance = torch.minimum(distance_x, distance_y)
        if feather:
            inner = (distance / feather).clamp(0, 1)
            alpha = 1 - inner.square() * (3 - 2 * inner)
        else:
            alpha = (distance < 0).float()
        # 主面内保持完整填光；只在主面外到扩散层外缘之间淡出。
        fade_x = (torch.minimum(self._x, 1 - self._x) / max((1 - 1 / sx) / 2, 1e-6)).clamp(0, 1)
        fade_y = (torch.minimum(self._y, 1 - self._y) / max((1 - 1 / sy) / 2, 1e-6)).clamp(0, 1)
        outer = fade_x.square() * (3 - 2 * fade_x) * fade_y.square() * (3 - 2 * fade_y)
        return Image.fromarray((alpha * outer * 255).round().byte().numpy())

    @torch.inference_mode()
    def update(
        self,
        frame: torch.Tensor,
        brightness: float,
        content_rect: tuple[int, int, int, int] | None = None,
        tick: float | None = None,
        *,
        sbs: torch.Tensor | None = None,
        scale: tuple[float, float] = (2.0, 2.0),
        fill_borders: bool = True,
        feather: float = 0.03,
    ) -> bytes | None:
        self._validate_frame(frame)
        if sbs is not None:
            self._validate_frame(sbs)
            if sbs.shape[-1] % 2:
                raise ValueError("ambient SBS width must be even")
        if type(brightness) not in (int, float) or not math.isfinite(brightness) or not 0 <= brightness <= 1:
            raise ValueError("ambient brightness must be between 0 and 1")
        if type(fill_borders) is not bool:
            raise ValueError("ambient fill_borders must be a boolean")
        if type(feather) not in (int, float) or not math.isfinite(feather) or not 0 <= feather <= 0.1:
            raise ValueError("ambient feather must be between 0 and 0.1")
        if (
            not isinstance(scale, tuple)
            or len(scale) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 1 for value in scale)
        ):
            raise ValueError("ambient scale must contain two finite values >= 1")
        now = time.monotonic() if tick is None else tick
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("ambient tick must be finite")
        if content_rect is not None:
            if (
                not isinstance(content_rect, tuple)
                or len(content_rect) != 4
                or any(type(value) is not int for value in content_rect)
            ):
                raise ValueError("ambient content_rect must contain four integer coordinates")
            left, top, right, bottom = content_rect
            if not 0 <= left < right <= frame.shape[2] or not 0 <= top < bottom <= frame.shape[1]:
                raise ValueError("ambient content_rect must be inside the frame")
        if self._last_tick is not None and now < math.nextafter(self._last_tick + 0.1, -math.inf):
            return None
        signature = (tuple(frame.shape), content_rect, None if sbs is None else tuple(sbs.shape), fill_borders)
        if signature != self._signature:
            self.reset()
            self._signature = signature
        source = frame if content_rect is None else frame[:, top:bottom, left:right]
        # 单目RGB[3,H,W] → 同设备area[3,72,128] → CPU；检测与取色共享小图。
        reduced = F.interpolate(source.detach().unsqueeze(0), size=(_SAMPLE_H, _SAMPLE_W), mode="area").squeeze(0)
        reduced = reduced.to(device="cpu", dtype=torch.float32)
        self._check_values(reduced)
        source_rect = self._content_rect(reduced.amax(0), strict=False) if fill_borders else _FULL
        previous_source = self._source_window.active
        source_rect = self._source_window.update(source_rect, now)
        if source_rect != previous_source:
            self._colors = None
        left_f, top_f, right_f, bottom_f = source_rect
        reduced = reduced[
            :,
            round(top_f * _SAMPLE_H) : round(bottom_f * _SAMPLE_H),
            round(left_f * _SAMPLE_W) : round(right_f * _SAMPLE_W),
        ]
        reduced = F.interpolate(reduced.unsqueeze(0), size=(18, 32), mode="area").squeeze(0)
        display_rect = _FULL
        if fill_borders:
            if sbs is None:
                display_rect = self._display_rect(frame, feather, scale)
            else:
                a, b = (self._display_rect(eye, feather, scale) for eye in sbs.chunk(2, dim=2))
                display_rect = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
        elif sbs is not None:
            self._check_values(F.adaptive_max_pool2d(sbs.detach().amax(dim=0, keepdim=True), (1, 1)).cpu())
        display_rect = self._display_window.update(display_rect, now)
        edges = (
            reduced[:, :, :2].mean(dim=2, keepdim=True),
            reduced[:, :, -2:].mean(dim=2, keepdim=True),
            reduced[:, :2, :].mean(dim=1, keepdim=True),
            reduced[:, -2:, :].mean(dim=1, keepdim=True),
        )
        colors = sum((edge * weight for edge, weight in zip(edges, self._weights)), torch.zeros_like(reduced))
        colors = colors / self._weight_sum
        if self._colors is not None and self._last_tick is not None:
            blend = -math.expm1(-(now - self._last_tick) / 0.25)
            colors = self._colors.lerp(colors, blend)
        # CPU float[3,18,32] → uint8 ndarray[18,32,3] → PIL RGBA(320,180) → zlib bytes。
        rgb = (colors.permute(1, 2, 0) * (255 * brightness)).round().clamp(0, 255).byte()
        image = Image.fromarray(rgb.numpy()).resize((self.WIDTH, self.HEIGHT), Image.Resampling.BILINEAR)
        image = image.filter(ImageFilter.BoxBlur(24)).convert("RGBA")
        image.putalpha(self._alpha(display_rect, scale, feather))
        identifier = self.TEXTURE_ID.encode("ascii")
        payload = bytes([len(identifier)]) + identifier + struct.pack("!HHB", self.WIDTH, self.HEIGHT, 1)
        payload += zlib.compress(image.tobytes(), level=3)
        self._colors, self._last_tick = colors, now
        return payload
