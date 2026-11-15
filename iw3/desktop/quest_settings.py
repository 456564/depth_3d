"""Quest 设置使用真实 args 名称和值；不在网络线程加载模型或设备。"""

import copy
import math
from dataclasses import dataclass
from typing import Any

PAGES = ("画面与立体", "模型与性能", "串流与声音", "捕获与控制", "屏幕与状态")
COMMANDS = frozenset({"set", "apply", "restart_stream", "restart_app", "restore", "refresh_windows", "recenter"})


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    page: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    step: float = 1
    immediate: bool = False
    help: str = ""
    choices: tuple[Any, ...] = ()


FIELDS = (
    Field(
        "divergence",
        "3D 强度",
        PAGES[0],
        "number",
        0,
        100,
        0.25,
        True,
        "作用：控制左右眼差异。增大：立体感通常更强，也更容易拉边、重影和不适。减小：更平，0可作平面对照。代价：不增加源视频清晰度；超训练范围时效果无保证。",
    ),
    Field(
        "convergence_mode",
        "会聚方式",
        PAGES[0],
        "enum",
        help=(
            "固定 constant：用会聚值选择落在屏幕表面的深度。自动 sod_v1：识别主体并随内容选择会聚位置，"
            "需要额外计算，切镜头可能变化。两种方式没有统一优劣。"
        ),
        choices=("constant", "sod_v1"),
    ),
    Field(
        "convergence",
        "会聚平面",
        PAGES[0],
        "number",
        -100,
        100,
        0.1,
        True,
        "作用：选择哪层内容落在虚拟屏幕表面。正常左右眼下，调小通常更多出屏，调大更多入屏。0.5是归一化深度，非0.5米。代价：不移动整块屏幕；自动模式还会随主体变化。",
    ),
    Field(
        "foreground_scale",
        "前景厚度分配",
        PAGES[0],
        "number",
        -3,
        3,
        0.2,
        True,
        "作用：改变前景与背景内部厚度分配。正值通常让前景更厚、背景层次压缩；负值反向；0保持默认。代价：效果依深度模型变化，不是二维放大或把整个人物向前推，视频先用0。",
    ),
    Field(
        "edge_dilation",
        "边缘修复（横 / 纵）",
        PAGES[0],
        "edge",
        0,
        20,
        1,
        True,
        "作用：处理深度前景边界。横增大可能减少交界拉扯，也可能吞头发细节、增加计算；0关闭。纵选跟随时横纵共用一个整数；纵填0只做横向。数字是迭代次数，不是最终像素扩张量。",
    ),
    Field(
        "synthetic_view",
        "合成哪只眼",
        PAGES[0],
        "enum",
        help=(
            "both：双眼都合成，形变分摊。left / right：只合成指定眼，另一眼保留原图，合成眼位移加倍。"
            "代价：一眼可能更干净，另一眼变形更强，不保证总帧率翻倍。通常先选both。"
        ),
        choices=("both", "right", "left"),
    ),
    Field(
        "preserve_screen_border",
        "保持画面边界",
        PAGES[0],
        "bool",
        help="作用：尽量减少画面边缘的立体位移。开：通常减少拉边，同时压缩边缘附近立体感。关：保留边缘视差，可能更易露边。不能修复源视频黑边或保证消除竖条；forward_fill不支持。",
    ),
    Field(
        "ema_normalize",
        "闪烁抑制",
        PAGES[0],
        "bool",
        help="作用：平滑深度归一化范围，减少深浅忽强忽弱。开：使用历史权重；关：范围仅取当前帧，模型自身时序机制仍在。代价：开启后切镜头可能恢复更慢。不是补帧，也不直接磨平彩色视频。",
    ),
    Field(
        "ema_decay",
        "历史权重",
        PAGES[0],
        "number",
        0.1,
        0.999,
        0.01,
        False,
        "作用：调闪烁抑制对过去帧的依赖。增大：通常更稳，切镜头或远近骤变时恢复更慢。减小：响应更快，也可能更跳。代价：只平滑深度范围，不会提高彩色画面帧率。",
    ),
    Field(
        "depth_model",
        "深度模型",
        PAGES[1],
        "enum",
        help="作用：AI估计远近。模型大通常更吃显存与计算，但不保证当前片段更稳、更清楚或更舒服。先用流畅的Any_V2_S。列表表示可用实现，不表示权重已缓存；首次加载可能下载并耗时。",
    ),
    Field(
        "resolution",
        "深度分辨率",
        PAGES[1],
        "resolution",
        192,
        8190,
        64,
        False,
        "作用：控制AI深度输入尺寸。增大：可能改善细轮廓，通常更吃计算和显存；减小：更快但细物体可能变粗。默认由模型决定，实际会对齐。不是彩色视频超分；DepthPro不支持自定尺寸。",
    ),
    Field(
        "method",
        "双眼合成方法",
        PAGES[1],
        "enum",
        help=(
            "sym预测一次位移供双眼采样；v3双眼分别预测；v2为较早网络。"
            "MLBW l2/l4混合2/4层位移，不保证4层更好；l2s超过相应强度会换常规版。"
            "monobw单调映射；forward_fill前向投射并填洞。没有固定画质或速度排名。"
        ),
        choices=(
            "row_flow_v3_sym",
            "row_flow_v3",
            "row_flow_v2",
            "mlbw_l2",
            "mlbw_l4",
            "mlbw_l2s",
            "monobw",
            "forward_fill",
        ),
    ),
    Field(
        "gpu",
        "NVIDIA 计算设备",
        PAGES[1],
        "gpu",
        help=(
            "作用：选择执行AI的设备。Quest链路还需要NVIDIA NVENC，因此不提供CPU降档。"
            "编号只是当前可用设备索引，换设备不保证更快或更清楚；切换需要重载模型并重建串流。"
        ),
    ),
    Field(
        "compile",
        "模型编译",
        PAGES[1],
        "bool",
        help="作用：尝试编译模型以加快重复运行。开：首次准备可能很慢，依环境可能失败或反复编译；关：直接运行。代价：不提高画质，收益取决于模型与尺寸；本机不保证编译支持齐全。",
    ),
    Field(
        "stream_height",
        "视频高度上限",
        PAGES[2],
        "int",
        320,
        4320,
        120,
        False,
        "作用：限制每眼彩色画面最高像素高度。增大：通常更清楚，也更吃捕获、合成、编码与网络；减小：负担少但细节减少。源窗口较小时不放大，不能凭空超分。应用后编码尺寸会变化。",
    ),
    Field(
        "stream_fps",
        "目标发送帧率",
        PAGES[2],
        "int",
        1,
        240,
        1,
        False,
        "作用：限制电脑每秒最多处理和发送的帧数。增大：算力网络够时更顺；减小：减负但运动更跳。代价：跟不上不会自动补帧。不是Quest屏幕刷新率，也不保证实际出画帧率。",
    ),
    Field(
        "quest_bitrate",
        "HEVC 码率（Mbps）",
        PAGES[2],
        "int",
        5,
        100,
        5,
        False,
        "作用：控制压缩数据量。增大：通常保留更多细节，占更多网络，超过吞吐可能更卡。减小：省网络，更容易糊或出现块状。不会提高源视频本身清晰度；需重新建立编码器。",
    ),
    Field(
        "quest_no_audio",
        "发送电脑声音",
        PAGES[2],
        "bool",
        help="作用：把电脑声音送到头显。开：发送声音；关：只保留视频。不是麦克风开关；采音来源由下面选项决定，切换需要重新加载媒体。",
    ),
    Field(
        "quest_audio_source",
        "声音采集方式",
        PAGES[2],
        "enum",
        help="process独立采音：不随Windows主音量或系统静音改变，网页自身静音仍生效，也会采到其他应用。speaker兼容回环：取默认扬声器，受系统音量与静音影响。两者不是音质高低档。",
        choices=("process", "speaker"),
    ),
    Field(
        "quest_audio_gain",
        "头显音量增益",
        PAGES[2],
        "number",
        0,
        4,
        0.1,
        True,
        "作用：缩放发给头显的PCM幅度。1=100%原幅度；0静音。增大更响但过高会削波失真，减小更轻。2=幅度乘2，不代表主观响度翻倍。Quest自身音量键仍独立控制播放响度。",
    ),
    Field(
        "screenshot",
        "捕获后端",
        PAGES[3],
        "enum",
        help=(
            "wc_mp：Windows窗口捕获，独立进程处理，推荐网页控制。mss：截取屏幕区域，遮挡也会入画，可作兼容备用。"
            "wc_cuda：GPU直取，减少CPU中转，不保证总帧率更高；窗口缩放有局限。pil：桌面基础备用，不支持指定窗口/显示器。"
            "电脑控制仅支持wc_mp/mss。它们不是AI画质档，切换后确认窗口与裁剪。"
        ),
        choices=("pil", "mss", "wc_mp", "wc_cuda", "kwcapture", "obs"),
    ),
    Field(
        "monitor_index",
        "显示器编号",
        PAGES[3],
        "int",
        0,
        255,
        1,
        False,
        "作用：未选窗口时选择捕获显示器。0是列表第一台，并非清晰度或优先级。指定窗口后此值忽略。插拔显示器会影响编号；可用候选来自当前设备，应用前需重新确认。",
    ),
    Field(
        "window_name",
        "捕获窗口",
        PAGES[3],
        "window",
        help="作用：指定播放网页的窗口，减少把其他桌面内容传出。默认空通常捕获显示器。电脑控制要求唯一且前台的窗口；同名窗口须先处理。刷新只更新候选，不自动更换当前捕获源。",
    ),
    *(
        Field(
            f"crop_{side}",
            f"{label}裁剪（像素）",
            PAGES[3],
            "int",
            0,
            500,
            5,
            False,
            f"作用：从捕获区域{label}侧去掉内容。增大裁掉更多，0不裁。像素基于电脑捕获区域，不是头显角度；不会恢复未捕获内容，也不会平移画面。四边合计不可把内容裁没，勿用裁剪掩盖分眼错误。",
        )
        for side, label in (("top", "上"), ("bottom", "下"), ("left", "左"), ("right", "右"))
    ),
    Field(
        "autocrop",
        "自动裁剪",
        PAGES[3],
        "enum",
        help="BLACK_TB识别上下黑边；BLACK识别四周黑边；FLAT系列也识别纯色边，可能误裁内容。关闭不作自动判断。桌面流程裁出内容做AI再补回边框以保持尺寸，不会永久放大去掉黑边。",
        choices=(None, "BLACK_TB", "BLACK", "FLAT_TB", "FLAT"),
    ),
    Field(
        "pad_mode",
        "补边到 16:9",
        PAGES[3],
        "enum",
        help="作用：给非16:9内容补黑边，保持原内容比例。开：保持16:9输出框；关：保留原比例。不会拉伸或增加细节。开启手柄控制电脑时不支持额外补边，须关闭后再应用。",
        choices=(None, "tblr", "16:9"),
    ),
    Field(
        "quest_control",
        "手柄控制电脑",
        PAGES[3],
        "bool",
        help="开：右手指向移动光标，扳机左键或拖动，摇杆上下切视频。需Windows、wc_mp或mss、唯一且前台的窗口、无额外补边。关：仍可握柄移动虚拟屏幕。菜单打开时电脑控制和抓屏暂停。",
    ),
)
FIELD_MAP = {field.key: field for field in FIELDS}


def get_fields(page: str | None = None) -> tuple[Field, ...]:
    return tuple(field for field in FIELDS if page is None or field.page == page)


def validate_patch(
    patch: dict[str, Any],
    current: dict[str, Any] | None = None,
    choices: dict[str, Any] | None = None,
    *,
    check_constraints: bool = True,
) -> dict[str, Any]:
    """Validate native args values without coercing strings, booleans or list shapes."""
    if not isinstance(patch, dict) or any(key not in FIELD_MAP for key in patch):
        raise ValueError("未知 Quest 设置字段")
    for key, value in patch.items():
        field = FIELD_MAP[key]
        valid = False
        if field.kind == "bool":
            valid = type(value) is bool
        elif field.kind in {"int", "number", "resolution"}:
            if field.kind == "resolution" and value is None:
                valid = True
            elif type(value) in ((int, float) if field.kind == "number" else (int,)):
                try:
                    valid = math.isfinite(value) and field.minimum <= value <= field.maximum
                except OverflowError:
                    valid = False
        elif field.kind == "edge":
            values = [value] if type(value) is int else value
            valid = (
                isinstance(values, list)
                and len(values) in (1, 2)
                and (type(value) is int or len(values) == 2)
                and all(type(v) is int and 0 <= v <= 20 for v in values)
            )
        elif field.kind == "gpu":
            valid = isinstance(value, list) and 1 <= len(value) <= 16 and all(type(v) is int and v >= 0 for v in value)
        elif field.kind == "window":
            valid = value is None or isinstance(value, str) and len(value) <= 2048
        elif field.kind == "enum":
            valid = value in field.choices if field.choices else isinstance(value, str) and 0 < len(value) <= 128
        if not valid:
            raise ValueError(f"{field.label}：类型或值域无效")
        available = (choices or {}).get(key)
        if isinstance(available, (list, tuple)) and field.kind != "window":
            candidates = [[v] if type(v) is int else v for v in available] if key == "gpu" else available
            if value not in candidates:
                raise ValueError(f"{field.label}：当前设备不支持该选项")
    if not check_constraints:
        return copy.deepcopy(patch)
    merged = {**(current or {}), **patch}
    if merged.get("quest_control"):
        if (
            merged.get("screenshot") not in {"wc_mp", "mss"}
            or not isinstance(merged.get("window_name"), str)
            or not merged["window_name"].strip()
            or merged.get("pad_mode") not in {None, "tblr"}
        ):
            raise ValueError("电脑控制要求 wc_mp/mss、指定窗口且关闭补边")
    if str(merged.get("depth_model", "")).startswith("DepthPro") and merged.get("resolution") is not None:
        raise ValueError("DepthPro 不支持自定义深度分辨率，请选择默认")
    if merged.get("method") == "forward_fill" and merged.get("preserve_screen_border"):
        raise ValueError("forward_fill 不支持保持画面边界，请先关闭该项")
    return copy.deepcopy(patch)
