"""现有Pop3D协议的边缘填光、全天域背景与缓存动画。"""

import math
import struct
import time
import zlib

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .quest_ambient_panorama import ATLAS_HEIGHT, FACE_SIZE, GUTTER, MOTION_STYLES, AmbientPanorama
from .quest_settings import AMBIENT_STYLES

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
        self._color_tick: float | None = None
        self._signature: tuple[object, ...] | None = None
        self._source_window = _ContentWindow()
        self._display_window = _ContentWindow()
        self._panorama: AmbientPanorama | None = None
        self._phase_start: float | None = None
        self._render_cache: tuple[torch.Tensor, Image.Image, float, str, bool, bool] | None = None

    def reset(self, preserve_content: bool = False) -> None:
        self._colors = None
        self._last_tick = None
        self._color_tick = None
        self._phase_start = None
        self._render_cache = None
        if not preserve_content:
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

    @staticmethod
    def _refine_black_edges(frame: torch.Tensor, reduced: torch.Tensor, rect: _Rect, strict: bool) -> _Rect:
        left, top, right, bottom = rect
        content = reduced[
            round(top * _SAMPLE_H) : round(bottom * _SAMPLE_H),
            round(left * _SAMPLE_W) : round(right * _SAMPLE_W),
        ]
        if float((content > 32 / 255).float().mean()) < (0.5 if strict else 0.35):
            return rect
        # 原尺寸RGB[3,H,W] → 同设备max[H,W] → 行/列max[H]/[W]；只读回两条投影，定位低清漏掉的窄边。
        value = frame.detach().amax(0)
        refined = list(rect)
        for axis, projection in enumerate((value.amax(0), value.amax(1))):
            positions = torch.nonzero(projection.to(device="cpu", dtype=torch.float32) > 16 / 255).flatten()
            if not positions.numel():
                continue
            start, stop = int(positions[0]) / projection.numel(), (int(positions[-1]) + 1) / projection.numel()
            if strict and rect[axis] > 0 and rect[axis + 2] < 1:
                refined[axis], refined[axis + 2] = start, stop
            # shortcut: 仅对≤2%的连续窄暗带允许单侧处理；更宽不对称暗带仍保守保留。
            if 0 < start <= 0.02:
                refined[axis] = max(refined[axis], start)
            if 0 < 1 - stop <= 0.02:
                refined[axis + 2] = min(refined[axis + 2], stop)
        return refined[0], refined[1], refined[2], refined[3]

    def _display_rect(self, frame: torch.Tensor, feather: float, scale: tuple[float, float]) -> _Rect:
        # 最终每眼RGB[3,H,W] → max-pool[1,72,128]，一像素亮点也保留。
        value = F.adaptive_max_pool2d(frame.detach().amax(dim=0, keepdim=True), (_SAMPLE_H, _SAMPLE_W))
        value = value.squeeze(0).to(device="cpu", dtype=torch.float32)
        self._check_values(value)
        rect = self._content_rect(value, strict=True)
        rect = self._refine_black_edges(frame, value, rect, strict=True)
        inset = list(rect)
        for axis, pixels in enumerate((self.WIDTH, self.HEIGHT)):
            # 已确认黑条的渐隐向内容内走，留一纹素使GL双线性采样的填光端完全不透黑。
            margin = scale[axis] / (pixels - 1)
            if rect[axis] > 0:
                inset[axis] += margin
            if rect[axis + 2] < 1:
                inset[axis + 2] -= margin
        # 主体用多数行/列判断，避免较高字幕被并入主体后漏掉渐隐保护。
        primary = self._content_rect(value, strict=False, high_quantile=0.5)
        if primary == _FULL:
            return inset[0], inset[1], inset[2], inset[3]
        left, top, right, bottom = primary
        outside = torch.ones_like(value, dtype=torch.bool)
        outside[
            max(0, round(top * _SAMPLE_H) - 2) : min(_SAMPLE_H, round(bottom * _SAMPLE_H) + 2),
            max(0, round(left * _SAMPLE_W) - 2) : min(_SAMPLE_W, round(right * _SAMPLE_W) + 2),
        ] = False
        marks = torch.nonzero(outside & (value > 16 / 255))
        if not marks.numel():
            return inset[0], inset[1], inset[2], inset[3]
        protected = inset
        for axis, coordinate, samples, pixels in ((0, 1, _SAMPLE_W, self.WIDTH), (1, 0, _SAMPLE_H, self.HEIGHT)):
            start = int(marks[:, coordinate].min()) / samples
            stop = (int(marks[:, coordinate].max()) + 1) / samples
            width = rect[axis + 2] - rect[axis]
            # 只扩会碰到黑条亮字的轴；保留一整纹素余量覆盖双线性采样。
            guard = feather * width + scale[axis] / (pixels - 1)
            if start - rect[axis] < guard or rect[axis + 2] - stop < guard:
                padding = guard / (1 - 2 * feather)
                protected[axis] = rect[axis] - padding
                protected[axis + 2] = rect[axis + 2] + padding
        return protected[0], protected[1], protected[2], protected[3]

    def _alpha(self, rect: _Rect, scale: tuple[float, float], feather: float, halo: bool = False) -> Image.Image:
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
        if halo:
            # 只加快参考幕面外衰减；竖屏两侧空区仍保持完整填光。
            outer = outer.pow(3)
        return Image.fromarray((alpha * outer * 255).round().byte().numpy())

    @torch.inference_mode()
    def animate(self, *, tick: float | None = None) -> bytes | None:
        cached = self._render_cache
        if cached is None or not cached[-1] or cached[3] not in MOTION_STYLES:
            return None
        now = time.monotonic() if tick is None else tick
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("ambient tick must be finite")
        if self._last_tick is not None and now < math.nextafter(self._last_tick + 0.1, -math.inf):
            return None
        # 暂停时只读CPU float[3,18,32]颜色和L[320,180]遮罩，不访问原视频/CUDA张量。
        payload = self._render(*cached, now)
        self._last_tick = now
        return payload

    def _render(
        self,
        colors: torch.Tensor,
        alpha: Image.Image,
        brightness: float,
        style: str,
        panoramic: bool,
        motion: bool,
        now: float,
    ) -> bytes:
        if self._phase_start is None:
            self._phase_start = now
        atlas = None
        if panoramic or style in MOTION_STYLES:
            if self._panorama is None:
                self._panorama = AmbientPanorama()
            palette = torch.stack(
                (
                    colors[:, :, :2].mean((1, 2)),
                    colors[:, :, -2:].mean((1, 2)),
                    colors[:, :2, :].mean((1, 2)),
                    colors[:, -2:, :].mean((1, 2)),
                )
            )
            phase = (now - self._phase_start) * 0.12 if motion else 0.0
            atlas = self._panorama.atlas(self._panorama.colors(palette, style, phase), brightness)
        if style in MOTION_STYLES:
            assert atlas is not None
            image = atlas.crop((GUTTER, GUTTER, GUTTER + FACE_SIZE, GUTTER + FACE_SIZE))
            image = image.resize((self.WIDTH, self.HEIGHT), Image.Resampling.BILINEAR)
        else:
            # 旧四种非360模式保留已验收的完整像素路径和哈希。
            rgb = (colors.permute(1, 2, 0) * (255 * brightness)).round().clamp(0, 255).byte()
            image = Image.fromarray(rgb.numpy()).resize((self.WIDTH, self.HEIGHT), Image.Resampling.BILINEAR)
            image = image.filter(ImageFilter.BoxBlur(40 if style == "blur" else 24)).convert("RGBA")
        image.putalpha(alpha)
        if panoramic:
            assert atlas is not None
            combined = Image.new("RGBA", (self.WIDTH, self.HEIGHT + ATLAS_HEIGHT))
            combined.paste(image, (0, 0))
            combined.paste(atlas, (0, self.HEIGHT))
            image = combined
        identifier = self.TEXTURE_ID.encode("ascii")
        payload = bytes([len(identifier)]) + identifier + struct.pack("!HHB", *image.size, 1)
        return payload + zlib.compress(image.tobytes(), level=3)

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
        video_rect: _Rect = _FULL,
        style: str = "edge",
        panoramic: bool = False,
        motion: bool = True,
    ) -> bytes | None:
        self._validate_frame(frame)
        if type(style) is not str or style not in AMBIENT_STYLES:
            raise ValueError("ambient style is unsupported")
        if type(panoramic) is not bool or type(motion) is not bool:
            raise ValueError("ambient panoramic and motion must be booleans")
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
        if (
            not isinstance(video_rect, tuple)
            or len(video_rect) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in video_rect)
            or not 0 <= video_rect[0] < video_rect[2] <= 1
            or not 0 <= video_rect[1] < video_rect[3] <= 1
        ):
            raise ValueError("ambient video_rect must be normalized bounds inside the reference screen")
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
        if self._signature is not None and self._signature[-3:] != (style, panoramic, motion):
            # 暂停只重建一次，不能丢弃已稳定的黑条范围再等待后续帧。
            self.reset(preserve_content=True)
            self._signature = (*self._signature[:-3], style, panoramic, motion)
        if self._last_tick is not None and now < math.nextafter(self._last_tick + 0.1, -math.inf):
            return None
        signature = (
            tuple(frame.shape),
            content_rect,
            None if sbs is None else tuple(sbs.shape),
            fill_borders,
            video_rect,
            style,
            panoramic,
            motion,
        )
        if signature != self._signature:
            self.reset()
            self._signature = signature
        source = frame if content_rect is None else frame[:, top:bottom, left:right]
        # 单目RGB[3,H,W] → 同设备area[3,72,128] → CPU粗检；精定位后从原像素区单独取色。
        reduced = F.interpolate(source.detach().unsqueeze(0), size=(_SAMPLE_H, _SAMPLE_W), mode="area").squeeze(0)
        reduced = reduced.to(device="cpu", dtype=torch.float32)
        self._check_values(reduced)
        source_rect = self._content_rect(reduced.amax(0), strict=False) if fill_borders else _FULL
        if fill_borders:
            source_rect = self._refine_black_edges(source, reduced.amax(0), source_rect, strict=False)
        previous_source = self._source_window.active
        source_rect = self._source_window.update(source_rect, now)
        if source_rect != previous_source:
            self._colors = None
        left_f, top_f, right_f, bottom_f = source_rect
        source_content = source[
            :,
            round(top_f * source.shape[1]) : round(bottom_f * source.shape[1]),
            round(left_f * source.shape[2]) : round(right_f * source.shape[2]),
        ]
        # 按原像素裁后再取色，避免1–3px黑条已混入72×128采样、随后无法剔净。
        reduced = F.interpolate(source_content.unsqueeze(0), size=(18, 32), mode="area").squeeze(0)
        reduced = reduced.to(device="cpu", dtype=torch.float32)
        display_rect = _FULL
        video_left, video_top, video_right, video_bottom = video_rect
        video_width, video_height = video_right - video_left, video_bottom - video_top
        local_scale = scale[0] / video_width, scale[1] / video_height
        if fill_borders:
            if sbs is None:
                display_rect = self._display_rect(frame, feather, local_scale)
            else:
                a, b = (self._display_rect(eye, feather, local_scale) for eye in sbs.chunk(2, dim=2))
                display_rect = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
        elif sbs is not None:
            self._check_values(F.adaptive_max_pool2d(sbs.detach().amax(dim=0, keepdim=True), (1, 1)).cpu())
        display_rect = self._display_window.update(display_rect, now)
        # SBS局部检测/迟滞后映射到可信几何框；首帧即填满参考幕面的竖屏空区。
        display_rect = (
            video_left + display_rect[0] * video_width,
            video_top + display_rect[1] * video_height,
            video_left + display_rect[2] * video_width,
            video_top + display_rect[3] * video_height,
        )
        if style == "blur":
            colors = reduced
        elif style == "cinema":
            colors = reduced.mean(dim=(1, 2), keepdim=True).expand_as(reduced) * 0.35
        else:
            edges = (
                reduced[:, :, :2].mean(dim=2, keepdim=True),
                reduced[:, :, -2:].mean(dim=2, keepdim=True),
                reduced[:, :2, :].mean(dim=1, keepdim=True),
                reduced[:, -2:, :].mean(dim=1, keepdim=True),
            )
            colors = sum((edge * weight for edge, weight in zip(edges, self._weights)), torch.zeros_like(reduced))
            colors = colors / self._weight_sum
        if self._colors is not None and self._color_tick is not None:
            blend = -math.expm1(-(now - self._color_tick) / 0.25)
            colors = self._colors.lerp(colors, blend)
        # CPU float[3,18,32] → RGBA[180,320,4]；360时追加atlas，合成[316,320,4]后zlib。
        alpha = self._alpha(display_rect, scale, feather, halo=style == "halo")
        cached = (colors, alpha, brightness, style, panoramic, motion)
        payload = self._render(*cached, now)
        self._render_cache = cached
        self._colors, self._last_tick, self._color_tick = colors, now, now
        return payload
