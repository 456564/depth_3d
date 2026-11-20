"""播放时钟；网页媒体必须在写入和读取时使用同一高精度时间域。"""

import time


def now() -> float:
    return time.perf_counter()
