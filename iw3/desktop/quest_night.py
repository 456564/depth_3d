"""离线生成的CC0自然夜景；运行时仅加载和压缩固定PNG。"""

import struct
import zlib
from functools import lru_cache
from pathlib import Path

from PIL import Image

TEXTURE_ID = "iw3_night"
FACE_SIZE, GUTTER = 512, 2


@lru_cache(maxsize=1)
def texture() -> bytes:
    with Image.open(Path(__file__).with_name("assets") / "lakeside_night.png") as image:
        if image.size != (1548, 1032) or image.mode != "RGBA":
            raise ValueError("invalid bundled night texture")
        # 实测RGBA[1032,1548,4]；0x10消息沿用既有ID、尺寸、zlib格式，外层由server封装。
        identifier = TEXTURE_ID.encode("ascii")
        header = bytes([len(identifier)]) + identifier + struct.pack("!HHB", *image.size, 1)
        return header + zlib.compress(image.tobytes(), level=3)
