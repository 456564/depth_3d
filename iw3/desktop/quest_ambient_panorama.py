"""通过现有quad协议生成低分辨率、方向连续的六面环境光。"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .quest_input import _rotate

MOTION_STYLES = frozenset(("aurora", "mist", "stars"))
_S = math.sqrt(0.5)
_FACES: tuple[tuple[list[float], list[float]], ...] = (
    ([0, 0, -1], [0, 0, 0, 1]),
    ([0, 0, 1], [0, 1, 0, 0]),
    ([1, 0, 0], [0, -_S, 0, _S]),
    ([-1, 0, 0], [0, _S, 0, _S]),
    ([0, 1, 0], [_S, 0, 0, _S]),
    ([0, -1, 0], [-_S, 0, 0, _S]),
)
FACE_SIZE, GUTTER, RADIUS = 64, 2, 50.0
TILE = FACE_SIZE + 2 * GUTTER
ATLAS_WIDTH, ATLAS_HEIGHT = 3 * TILE, 2 * TILE


@lru_cache(maxsize=1)
def _fields() -> tuple[torch.Tensor, ...]:
    coordinate = torch.linspace(-1, 1, FACE_SIZE)
    y, x = torch.meshgrid(-coordinate, coordinate, indexing="ij")
    directions = []
    for position, rotation in _FACES:
        horizontal = torch.tensor(_rotate(rotation, [1, 0, 0]))
        vertical = torch.tensor(_rotate(rotation, [0, 1, 0]))
        points = torch.tensor(position) + x[..., None] * horizontal + y[..., None] * vertical
        directions.append(F.normalize(points, dim=-1))
    direction = torch.stack(directions)
    anchors = torch.tensor([[-1, 0, -1], [1, 0, -1], [0, 1, -1], [0, -1, -1]]) / math.sqrt(2)
    weights = (direction @ anchors.T * 2).softmax(-1)
    longitude = torch.atan2(direction[..., 0], -direction[..., 2])
    # 固定球面星点只计算一次；同一方向函数使公共边取色连续，动画不移动整个背景。
    index = torch.arange(72, dtype=torch.float32)
    height = 1 - 2 * (index + 0.5) / len(index)
    angle = index * (math.pi * (3 - math.sqrt(5)))
    radius = (1 - height.square()).sqrt()
    stars = torch.stack((radius * angle.cos(), height, radius * angle.sin()), dim=1)
    stars = ((direction @ stars.T - 1) * 1500).exp().sum(-1).clamp(0, 1)
    return direction, weights, longitude, stars


class AmbientPanorama:
    def __init__(self) -> None:
        self._direction, self._weights, self._longitude, self._stars = _fields()

    def colors(self, palette: torch.Tensor, style: str, phase: float = 0.0) -> torch.Tensor:
        if palette.shape != (4, 3) or palette.is_cuda or not palette.is_floating_point():
            raise ValueError("ambient palette must be CPU floating RGB [4,3]")
        colors = self._weights @ palette
        x, y, z = self._direction.unbind(-1)
        if style == "cinema":
            colors = palette.mean(0).expand_as(colors)
            gain = torch.ones_like(y)
        elif style == "halo":
            gain = 0.08 + 0.75 * (-z).clamp(0, 1).pow(6)
        elif style == "blur":
            colors = colors * 0.65 + palette.mean(0) * 0.35
            gain = 0.45 + 0.25 * (-z).clamp(0, 1)
        elif style == "aurora":
            band = ((y - 0.18 * (3 * self._longitude + phase).sin()) / 0.14).square().neg().exp()
            band += 0.55 * ((y + 0.4 + 0.12 * (2 * self._longitude - phase).cos()) / 0.1).square().neg().exp()
            gain = 0.10 + 0.75 * band
        elif style == "mist":
            cloud = (x * 3 + y * 2 + z * 4 + phase).sin()
            cloud += 0.5 * (x * 6 - y * 4 + z * 2 - phase * 0.7).sin()
            gain = 0.38 + 0.15 * cloud
        elif style == "stars":
            twinkle = 0.65 + 0.35 * (x * 9 + y * 13 + z * 7 + phase).sin()
            gain = 0.035 + 1.5 * self._stars * twinkle
        else:
            gain = 0.3 + 0.25 * (-z).clamp(0, 1)
        return (colors * gain[..., None]).clamp(0, 1)

    @staticmethod
    def atlas(colors: torch.Tensor, brightness: float) -> Image.Image:
        # CPU float[6,64,64,3] → RGBA[6,68,68,4]（2px护边）→ atlas[136,204,4]。
        rgb = (colors * (255 * brightness)).round().clamp(0, 255).byte()
        rgba = torch.cat((rgb, torch.full_like(rgb[..., :1], 255)), dim=-1)
        tiles = F.pad(rgba.permute(0, 3, 1, 2), (GUTTER,) * 4, mode="replicate").permute(0, 2, 3, 1)
        atlas = tiles.reshape(2, 3, TILE, TILE, 4).permute(0, 2, 1, 3, 4).reshape(ATLAS_HEIGHT, ATLAS_WIDTH, 4)
        return Image.fromarray(atlas.numpy())


def panorama_layers(
    texture_id: str, position: list[float], rotation: list[float], overlay_height: int
) -> list[dict[str, Any]]:
    layers = []
    # OpenXR quad正面是local +Z；14万射线研究自检无漏口。几何沿窗口世界姿态调整，不随头转。
    for index, (center, orientation) in enumerate(_FACES):
        offset = _rotate(rotation, [RADIUS * value for value in center])
        ax, ay, az, aw = rotation
        bx, by, bz, bw = orientation
        composed = [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
        layers.append(
            {
                "tipo": "quad",
                "tex": texture_id,
                "olho": "ambos",
                "alpha": False,
                "pos": [a + b for a, b in zip(position, offset)],
                "rot": composed,
                "tam": [2 * RADIUS, 2 * RADIUS],
                "rect": [index % 3 * TILE + GUTTER, overlay_height + index // 3 * TILE + GUTTER, FACE_SIZE, FACE_SIZE],
            }
        )
    return layers
