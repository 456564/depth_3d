"""Small, edge-colored ambient texture for the existing Pop3D client."""

import math
import struct
import time
import zlib

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


class AmbientBackground:
    WIDTH, HEIGHT = 160, 90
    TEXTURE_ID = "iw3_ambient"

    def __init__(self) -> None:
        x = torch.linspace(0, 1, 32).view(1, 1, 32)
        y = torch.linspace(0, 1, 18).view(1, 18, 1)
        self._weights = (torch.exp(-6 * x), torch.exp(-6 * (1 - x)), torch.exp(-6 * y), torch.exp(-6 * (1 - y)))
        self._weight_sum = sum(self._weights[1:], self._weights[0])
        x = torch.linspace(0, 1, self.WIDTH)
        y = torch.linspace(0, 1, self.HEIGHT)
        fade_x = (torch.minimum(x, 1 - x) * 5).clamp(0, 1)
        fade_y = (torch.minimum(y, 1 - y) * 5).clamp(0, 1)
        fade_x = fade_x.square() * (3 - 2 * fade_x)
        fade_y = fade_y.square() * (3 - 2 * fade_y)
        self._alpha = Image.fromarray((fade_y[:, None] * fade_x[None, :] * 255).round().byte().numpy())
        self._colors: torch.Tensor | None = None
        self._last_tick: float | None = None

    def reset(self) -> None:
        self._colors = None
        self._last_tick = None

    @torch.inference_mode()
    def update(
        self,
        frame: torch.Tensor,
        brightness: float,
        content_rect: tuple[int, int, int, int] | None = None,
        tick: float | None = None,
    ) -> bytes | None:
        if (
            not isinstance(frame, torch.Tensor)
            or frame.ndim != 3
            or frame.shape[0] != 3
            or min(frame.shape[1:]) <= 0
            or not frame.is_floating_point()
        ):
            raise ValueError("ambient frame must be floating RGB [3,H,W]")
        if type(brightness) not in (int, float) or not math.isfinite(brightness) or not 0 <= brightness <= 1:
            raise ValueError("ambient brightness must be between 0 and 1")
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
            frame = frame[:, top:bottom, left:right]
        if self._last_tick is not None and now < math.nextafter(self._last_tick + 0.1, -math.inf):
            return None

        # 实测：float RGB[3,H,W] → 同设备[1,3,18,32] → CPU float[3,18,32]，范围[0,1]。
        reduced = F.interpolate(frame.detach().unsqueeze(0), size=(18, 32), mode="area").squeeze(0)
        reduced = reduced.to(device="cpu", dtype=torch.float32)
        low, high = torch.aminmax(reduced)
        if not 0 <= low <= high <= 1:
            raise ValueError("ambient RGB values must be finite and between 0 and 1")
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
        # CPU float[3,18,32] → uint8 ndarray[18,32,3] → PIL RGBA(160,90) → zlib bytes。
        rgb = (colors.permute(1, 2, 0) * (255 * brightness)).round().clamp(0, 255).byte()
        image = Image.fromarray(rgb.numpy()).resize((self.WIDTH, self.HEIGHT), Image.Resampling.BILINEAR)
        image = image.filter(ImageFilter.BoxBlur(12)).convert("RGBA")
        image.putalpha(self._alpha)
        identifier = self.TEXTURE_ID.encode("ascii")
        payload = bytes([len(identifier)]) + identifier + struct.pack("!HHB", self.WIDTH, self.HEIGHT, 1)
        payload += zlib.compress(image.tobytes(), level=3)
        self._colors, self._last_tick = colors, now
        return payload
