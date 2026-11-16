# iw3 Quest 原生串流预览版

已发布预览版 `v0.1.0-quest-preview.3` 提供动态环境光背景。当前本地源码另有智能黑边填光、边缘渐隐和头显退出网页全屏，用户已确认效果可用、退出成功；底边、图集／直播全屏与独立射线的修正已由用户确认正常；配套扩展继续处理竖屏网站虚化，新衔接待实测。本地新增功能尚未发布，下文设置说明以当前源码为准，安装命令仍指向已发布的 preview.3。`v0.1.0-quest-preview.2`（`5701fee6`）保留为已验收的固定基线。[更新日志](CHANGELOG.md)记录功能、修复和验证范围。本 fork 基于 [nagadomi/nunif](https://github.com/nagadomi/nunif) 的 `iw3-desktop`，用于把电脑上的网页视频实时转换为立体画面，在 Quest 的现有 Pop3D 接收应用中观看。以下是本 fork 的说明，后面保留完整上游 README。

### 为什么增加这些功能

| 功能 | 目的与实现 |
|---|---|
| HEVC 原生局域网串流 | 在原网页串流之外增加 Full SBS → NVIDIA HEVC NVENC → TCP → Quest 原生硬件解码链路，减少高分辨率逐帧 JPEG 的传输开销。实际流畅度与画质仍需在自己的设备上验证。 |
| 独立音频与头显音量 | Windows 进程回环采集除 iw3 自身进程树外的系统声音，电脑主音量或系统静音不影响发送；0–400% 增益只改变发给 Quest 的声音。网页自身静音仍会消除音源，其他应用和通知也可能被发送。另保留扬声器回环兼容模式。 |
| 网页全屏控制 | 右手柄原生射线映射到所选浏览器窗口，A键单击（长按不连点），扳机点击/拖动，摇杆上下单次切换视频；同一窗口切换全屏时更新捕获与点击坐标。Quest电脑控制启用时，wc_mp/mss不捕获电脑鼠标，避免鼠标干扰黑边填光。网页须支持对应上下键操作。 |
| 左右手调屏 | 任一手柄指向画面后握住握柄并移动，即可平移屏幕；同手摇杆上推远、下拉近，右放大、左缩小。调屏时暂停电脑输入；居中或重连恢复默认屏幕。 |
| 头显内完整设置 | B/Y 打开六页中文面板，任一手指向并用扳机操作；覆盖画质、模型、串流、声音、捕获、控制和环境光设置，显示实际值、待应用值和调节代价。提供恢复设置、重启串流与重启 iw3 程序。 |
| CUDA 与整个桌面控制 | 修复 CUDA 窗口缩小时只显示左上部分的问题；wc_mp/mss/wc_cuda 均支持所选窗口或整台显示器的同帧点击映射。显示器布局变化时阻断旧坐标，重启串流后重新确认。 |
| 可调曲面屏幕 | 0–120° 水平圆柱屏，0° 为平面，电脑滑块或头显“屏幕与状态”页即时调节；保持画面中心、弧长与高度，点击和抓屏跟随实际发送的曲面。客户端未声明支持时保持平面，无需更换 APK。 |
| 动态环境光背景 | 用画面边缘颜色生成主屏周围的柔光，左右上下对应，跟随调屏与内容变化。当前源码可保守识别连续黑条并填光，让内容窄边渐隐到柔光、中央保持清晰。默认关闭；“环境光”页提供五项即时设置，复用现有 APK 的纹理与图层协议。 |
| 对称合成保边设置 | 修正 `row_flow_v3_sym` 忽略“保留屏幕边界”的参数传递，使该选项实际参与合成。它用于减小合成边缘位移，不能保证所有竖边现象消失。 |

### Windows 安装与启动

当前原生串流面向 Windows x64 与支持 HEVC NVENC 的 NVIDIA GPU；独立采音要求 Windows build 20348 或更新版本。Quest 使用已有的 [Pop3D 接收应用](https://github.com/Sindelitcis/pop3d-quest)，无需安装本 fork 提供的 APK。

1. 先按[上游 Windows 分发包说明](windows_package/docs/README.md)完成环境安装。关闭 iw3，保存已有源码改动。以下首次接入命令在分发包目录的 PowerShell 中执行：

   ```powershell
   .\nunif-cmd.bat git remote add quest https://github.com/456564/depth_3d.git
   .\nunif-cmd.bat git fetch quest tag v0.1.0-quest-preview.3
   .\nunif-cmd.bat git switch -c quest-preview v0.1.0-quest-preview.3
   .\nunif-cmd.bat python -m pip install -r requirements.txt -r requirements-gui.txt -r iw3/desktop/requirements-quest.txt
   ```

   `nunif-cmd.bat` 会自动进入 `nunif` 源码目录。PyAV 已由 `requirements.txt` 声明；`requirements-quest.txt` 固定兼容采音所用的 PyAudioWPatch 版本。模型权重按上游安装流程取得，本 fork 不附带本机模型或用户视频。

   首次安装使用新分支名；已有同名分支时先保存本地修改，再按 Git 提示处理，避免覆盖自己的代码。

2. 使用独立采音前，安装 MSVC x64 C++ Build Tools 与 Windows SDK，打开 **x64 Native Tools Command Prompt**，切到分发包内的 `nunif` 源码目录，执行：

   ```bat
   powershell -NoProfile -ExecutionPolicy Bypass -File .\iw3\desktop\native\build.ps1
   ```

   构建生成 `iw3/desktop/native/quest_process_audio.dll`；运行只需该 DLL，编译产物不提交到源码仓库。接口与构建说明见 [native README](iw3/desktop/native/README.md)。不构建时可选择“扬声器回环（兼容）”，其声音仍受 Windows 主音量和系统静音影响。

3. 在分发包目录运行 `iw3-desktop-gui.bat`。退出电脑端 Pop3D，避免端口冲突；保留 Quest 中的 Pop3D 接收应用。电脑和 Quest 连接同一个可信局域网。
4. 显示模式选 **Quest / Pop3D**，首次可用 **Any_V2_S、1080、30 FPS、35 Mbps**，电脑声音按需开启。需要控制网页时，选 **wc_mp、wc_cuda或mss**，选择唯一浏览器窗口或“整个桌面（所选显示器）”，勾选“手柄控制电脑”；整屏控制四边裁剪须为0。指定窗口须保持前台；整屏模式可点击激活屏幕内窗口，方向键须指向前台窗口。点击“开始”后打开 Quest 接收应用。

原生模式自动输出 Full SBS，通常为 3840×1080、每眼 1920×1080。调屏不依赖“手柄控制电脑”开关；松开握柄后，先松扳机和A键、摇杆回中，再操作网页。preview.2可在头显菜单刷新并选择另一个窗口，再应用；旧preview.1需停止后重新选择。停止后改回 `Web Streaming` 即可使用原网页模式。

本预览分支不应随意运行完整 `update.bat`：上游更新脚本在拉取失败时可能执行 `git reset --hard`，丢弃未提交源码改动。升级本 fork 前先保存改动，再显式获取对应版本。

更新源码后须关闭并重新打开 iw3，或使用头显菜单的“重启 iw3 程序”（5 秒内再次点击确认），等 Quest 重连。单点“重启串流”不会重新导入修改后的源码。本版发布源代码，不附 APK、模型、媒体、本机配置或编译二进制。

本地工作空间另配有抖音净屏扩展1.0.4：只隐藏已核实的全屏控件（含左上关闭/搜索、消息、右侧工具列与网页弹幕），兼容消息重挂到嵌套播放器；保留图集、直播及未知媒体。普通播放器填回底部48px布局槽，保持源画面比例，全屏隐藏电脑鼠标；活动竖屏作品只移除已核实的模糊背景图片，保留完整视频和黑底供项目环境光填充。更新后在Edge扩展管理中点“重新加载”，再刷新抖音、重新进入全屏。扩展保存在项目工作空间的`.Codex/tools/douyin-pure-video/`，不包含在当前源码仓库或旧发布包中；无需新APK。禁用扩展并刷新网页即可回退。选择器来自本机实际页面检查，网站改版后可能需要更新。

### 在头显中设置

右 B 或左 Y 打开/关闭菜单。任一手指向、扳机或 A/X 点击；摇杆上下滚动，扳机长按加减连续调整，A/X 长按不连点。“微调”使用更小步长，“?”解释作用和代价。菜单打开时暂停网页控制和握柄调屏；关闭后先松开两手扳机、A/X、握柄并让摇杆回中，再恢复控制。无需新 APK：中文由电脑绘成单张 RGBA 纹理，经现有 `0x10`/`0x11` 纹理和图层协议显示。

| 分页 | 可调设置及用途 |
|---|---|
| 画面与立体 | 3D 强度、会聚方式/平面、前景厚度分配、横纵边缘修复、合成哪只眼、保持画面边界、闪烁抑制及历史权重。更强视差不等于更清楚；过强可能拉边或不适。 |
| 模型与性能 | 深度模型、深度分辨率、双眼合成方法、NVIDIA GPU、模型编译；另有小模型筛选。模型或尺寸更大通常更费资源，不保证本片段效果更好；首次加载可能下载权重。 |
| 串流与声音 | 视频高度上限、目标发送 FPS、HEVC 码率、电脑声音、采音方式、头显增益。更高分辨率/码率增加负担；目标 FPS 不是 Quest 刷新率，过高增益可能失真。 |
| 捕获与控制 | 捕获后端、显示器、刷新/选择窗口、四边裁剪、自动裁剪、16:9 补边、手柄控制电脑。控制支持 wc_mp/mss/wc_cuda，无额外补边。选“整个桌面（所选显示器）”时四边裁剪须为0；窗口模式须唯一且前台。 |
| 屏幕与状态 | 屏幕曲度0–120°、调屏帮助、屏幕居中、退出网页全屏（Esc）、当前模型/方法/输出尺寸、三项电脑端 FPS、连接地址与配置形状。曲度越大越包围，建议30–60°；0回平面。调屏大小与曲度均不增加源像素。 |
| 环境光 | 背景默认关，亮度25%，扩散2倍；智能黑边填光默认开，边缘渐隐默认3%、范围0–10%，均需开启背景才有效。亮度越大柔光越亮，扩散越大覆盖越广；渐隐越大，更多边缘内容融入柔光，越小则保留更宽的清晰区域。不提高分辨率或AI立体强度。 |

强度、会聚平面、前景厚度、边缘修复、头显增益、屏幕曲度和五个环境光设置即时提交；其他设置先成为草稿，按“应用并重新加载”重建媒体。环境光设置随已应用设置保存并在重启后恢复。面板区分“现”与“待”，只有电脑完成首帧及编码后才报告已应用；这不代表头显已确认出画。新配置失败时最多自动尝试一次回退；仍可修改重试，或点“恢复上次成功设置”。

当前源码先排除捕获器补边，再从原单目画面保守识别成对、近似居中的连续黑条，取内容四边颜色。覆盖范围由同帧最终 Full SBS 的左右眼内容框取包围并集，优先保留较大区域，字幕可能导致部分黑边仍保留；全黑、暗场或低置信时保留整幅画面。内容区域扩大立即放开，缩小需连续稳定0.2秒。检测无法完全区分真实暗色内容与像素相同的黑条；网页工具栏、白边仍可能参与，建议全屏。

对低清检测容易漏掉的四周细黑条，另用原尺寸的行列亮度投影定位；只有不超过该轴2%的连续窄暗带允许单侧处理。普通已确认黑条从真实像素边界填光，保留纹理采样余量，减少渐隐部分透黑造成的细缝；字幕保护仍单独保留。视频原尺寸、比例和输入像素不变，但填光边界可能覆盖最外侧约一纹素的内容；双眼范围不同时仍优先保留两眼内容。

柔光为双眼共用的320×180 RGBA、最高10Hz，常规颜色约0.25秒平滑；同一纹理承载颜色与遮罩，中央透明、黑条填光、内容窄边渐隐，外围渐暗。主视频本身仍不透明，边缘效果由柔光覆盖完成，不改编码尺寸或点击映射。跟随窗口移动、缩放和曲面；圆柱柔光与主屏同半径、同轴心，前方上限170°，能力未知或不支持时用平面。不生成360房间；纹理和视频独立更新，无逐帧同步保证。

“重启串流”按已应用设置重建媒体并保留管理连接；“重启 iw3 程序”须在 5 秒内再次点击确认，保存已应用设置，退出并重开电脑程序、自动开始串流，头显短暂断线后重连。两种重启均不包含尚未应用的草稿，也不重启 Windows。

退出网页全屏：B/Y →“屏幕与状态”→“退出网页全屏（Esc）”，任一手柄点击，效果与电脑按一次Esc相同；随后B/Y关闭设置菜单。需已开启“手柄控制电脑”，捕获目标在电脑前台且同帧映射有效；整桌面模式会检查画面中心所指的前台窗口。发送失败会留在菜单提示，不重启串流、不自动切换前台。显示“已发送Esc”只表示电脑接受按键，仍需看网页实际响应。

网页指向使用独立原生射线，不把光标绘入视频或柔光采样。射线在当前已发送的平面／曲面上定位；打开菜单时改用菜单射线，抓屏调节、指向离开画面、失焦、输入超时、暂停或断连时清除网页射线。wc_cuda原生捕获没有可用的鼠标开关，本轮仅由抖音净屏扩展在抖音全屏时隐藏鼠标；CUDA在其他网站或非全屏时仍可能捕获鼠标，不能承诺避免填光干扰。

黑条误判时，先关闭“智能黑边填光”；此时仍保留外围环境光和已设置的窄边渐隐。将“边缘渐隐宽度”设0可关闭窄边渐隐，智能填光可独立保留；关闭“环境光背景”则立即移除柔光层、恢复黑色背景。需要回到已验收代码基线时，先退出 iw3、保存未提交源码和新增文件，确认工作区干净，再执行 `git switch --detach v0.1.0-quest-preview.2`；原开发分支仍保留，切回该分支即可继续。不要使用 `reset --hard` 或强制覆盖未保存修改。

### 协议、来源与许可证

- 基础工程来自 **nagadomi/nunif**，保留 [MIT LICENSE](LICENSE)、[第三方 NOTICE](NOTICE) 和上游文档。
- 本 fork 的 PC 发送端依据公开的 **Pop3D Quest** 协议独立实现互操作，参考 [Sindelitcis/pop3d-quest](https://github.com/Sindelitcis/pop3d-quest/tree/81449d37f8727b385d61f6ccb81c15fdb92ac868) 及 v1.4.4 发行所附 Quest 源码；未复制未确认许可的 Pop3D Windows 自有实现。Quest 客户端继承 Moonlight XR/Android，采用 **GPLv3**，不因本 fork 的 MIT 基础而改为 MIT。
- Quest 经 UDP 52300 发现电脑，再连接 TCP 52301；电脑发送 HEVC、双眼图层和 PCM 音频，Quest 返回头部与手柄状态。解码和 OpenXR 显示由现有 Quest 应用完成。
- Windows 独立采音参考 Microsoft [ApplicationLoopback 示例](https://github.com/microsoft/windows-classic-samples/tree/main/Samples/ApplicationLoopback)，保留 [Microsoft MIT 许可证](iw3/desktop/native/LICENSE-Microsoft.txt)。
- **模型权重与依赖各自遵循其许可证**，不能一并视为 MIT。部分深度模型有非商业限制，见 [iw3 模型说明](iw3/README.md)。PyAV wheel 所带 FFmpeg 还可能影响二进制分发许可，见下方上游 `License Notes`。

### 当前限制

- preview.2 已获用户验收，环境光、填光、渐隐及头显退出网页全屏已有用户可用反馈。preview.3 发布时656项自动回归通过（123项未选中）；该历史统计不代表本轮验证结果。本轮媒体净屏、底部布局槽与独立网页射线修正仍待Quest复测。无线负载、长期舒适度及多屏热插拔仍需实测。
- 协议没有认证或加密，只在自己的可信局域网使用；开启电脑控制后，同网内不可信设备可能影响电脑输入。不要将端口映射到公网。
- 接收端不使用发送 PTS 做音画同步，实际音画延迟仍需实机标定。增益过高可能削波失真。
- 部分场景可见的竖边问题尚未确定根因，本版不宣称已修复。
- 网页控制支持已选的唯一前台窗口或整台显示器。整屏模式可点击激活窗口，但摇杆方向键只发送给指向处的前台窗口。显示器布局变化后需重启串流；失焦、断线或几何切换期间会释放输入。`Streaming FPS` 是电脑发送速率，不代表头显显示帧率。

---

My playground.

For the time being, I will make incompatible changes.

## waifu2x

[waifu2x/README.md](./waifu2x/README.md)

waifu2x: Image Super-Resolution for Anime-Style Art. Also it supports photo models (GAN based models)

The repository contains waifu2x pytorch implementation and pretrained models, started with porting the original [waifu2x](https://github.com/nagadomi/waifu2x).

The demo application can be found at
- https://waifu2x.udp.jp/ (Cloud version)
- https://unlimited.waifu2x.net/ (In-Browser version).

## iw3

[iw3/README.md](./iw3/README.md)

I want to watch any 2D video as 3D video on my VR device, so I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video into side-by-side 3D image/video.

### iw3-desktop

[iw3/docs/desktop.md](./iw3/docs/desktop.md)

iw3.desktop is a tool that converts your PC desktop screen into 3D and streaming over WiFi.

You can watch any image and video/live displayed on your PC as 3D in realtime.

### iw3-player

[iw3/player/README.md](./iw3/player/README.md)

iw3-player is a self-hosted, specialized viewing environment for stereoscopic media.  
It allows you to stream media that has been pre-converted to 3D with iw3 from your PC and enjoy it on VR devices through a WebXR application.

## stilizer

[stlizer/README.md](./stlizer/README.md)

stlizer is a fast conservative video stabilizer.

## cliqa

[cliqa/README.md](./cliqa/README.md)

`cliqa` provides low-vision image quality scores that are more consistent across different images.

It is useful for filtering low-quality images with a threshold value when creating image datasets.

Currently, the following two models are supported.

- JPEGQuality: Predicts JPEG Quality from image content
- GrainNoiseLeve: Predicts Noise Level related to photograph and PSNR degraded by that noise

CLI tools are also available to filter out low quality images using these results.

## Install

### Installer for Windows users

- [nunif windows package](windows_package/docs/README.md)
- [nunif windows package (日本語)](windows_package/docs/README_ja.md)

### For developers

#### Dependencies

- Python 3 (developed with 3.12)
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirements.txt

We usually support the latest version. If there are bugs or compatibility issues, we will specify the version.

Optional dependencies (not installed from the requirements files; install them only if you need the feature):

- [INSTALL-ubuntu](INSTALL-ubuntu.md)
- [INSTALL-windows](INSTALL-windows.md)
- [INSTALL-macos](INSTALL-macos.md)

For Intel GPUs, additionally see section [INSTALL-xpu](INSTALL-xpu.md).  
For older NVIDIA GPUs, additionally see section [INSTALL-cu126](INSTALL-cu126.md).


For container, packages, or special hardware builds, see [extra_build](extra_build).

#### About NUNIF_HOME

If the environment variable `NUNIF_HOME` is defined, downloaded pretrained models, configuration files, cache, temporary files, and lock files will be saved under `NUNIF_HOME`. This may be useful when packaging or in situations where the source directory does not have write permissions.
The `~` character at the beginning of a path string is expanded to the home directory.

### License Notes

Note that if you distribute binary builds, it is possible that it will be GPL.

This is due to PyAV(av) wheel package containing the GPL version of ffmpeg library.
[You can build PyAV with the LGPL version of ffmpeg library](INSTALL-ubuntu.md#building-pyav-from-source).

If you load this repository with torch.hub.load for waifu2x Python API etc, this problem does not exist because PyAV is not a dependent package.
