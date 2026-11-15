# iw3 Quest 原生串流预览版

版本：`v0.1.0-quest-preview.1`。本 fork 基于 [nagadomi/nunif](https://github.com/nagadomi/nunif) 的 `iw3-desktop`，用于把电脑上的网页视频实时转换为立体画面，在 Quest 的现有 Pop3D 接收应用中观看。以下是本 fork 的说明，后面保留完整上游 README。

### 为什么增加这些功能

| 功能 | 目的与实现 |
|---|---|
| HEVC 原生局域网串流 | 在原网页串流之外增加 Full SBS → NVIDIA HEVC NVENC → TCP → Quest 原生硬件解码链路，减少高分辨率逐帧 JPEG 的传输开销。实际流畅度与画质仍需在自己的设备上验证。 |
| 独立音频与头显音量 | Windows 进程回环采集除 iw3 自身进程树外的系统声音，电脑主音量或系统静音不影响发送；0–400% 增益只改变发给 Quest 的声音。网页自身静音仍会消除音源，其他应用和通知也可能被发送。另保留扬声器回环兼容模式。 |
| 网页全屏控制 | 右手柄射线映射到所选浏览器窗口，扳机点击/拖动，摇杆上下单次切换视频；同一窗口切换全屏时更新捕获与点击坐标。网页须支持对应上下键操作。 |
| 左右手调屏 | 任一手柄指向画面后握住握柄并移动，即可平移屏幕；同手摇杆上推远、下拉近，右放大、左缩小。调屏时暂停电脑输入；居中或重连恢复默认屏幕。 |
| 对称合成保边设置 | 修正 `row_flow_v3_sym` 忽略“保留屏幕边界”的参数传递，使该选项实际参与合成。它用于减小合成边缘位移，不能保证所有竖边现象消失。 |

### Windows 安装与启动

当前原生串流面向 Windows x64 与支持 HEVC NVENC 的 NVIDIA GPU；独立采音要求 Windows build 20348 或更新版本。Quest 使用已有的 [Pop3D 接收应用](https://github.com/Sindelitcis/pop3d-quest)，无需安装本 fork 提供的 APK。

1. 先按[上游 Windows 分发包说明](windows_package/docs/README.md)完成环境安装。关闭 iw3，保存已有源码改动。以下首次接入命令在分发包目录的 PowerShell 中执行：

   ```powershell
   .\nunif-cmd.bat git remote add quest https://github.com/456564/depth_3d.git
   .\nunif-cmd.bat git fetch quest tag v0.1.0-quest-preview.1
   .\nunif-cmd.bat git switch -c quest-preview v0.1.0-quest-preview.1
   .\nunif-cmd.bat python -m pip install -r requirements.txt -r requirements-gui.txt -r iw3/desktop/requirements-quest.txt
   ```

   `nunif-cmd.bat` 会自动进入 `nunif` 源码目录。PyAV 已由 `requirements.txt` 声明；`requirements-quest.txt` 固定兼容采音所用的 PyAudioWPatch 版本。模型权重按上游安装流程取得，本 fork 不附带本机模型或用户视频。

2. 使用独立采音前，安装 MSVC x64 C++ Build Tools 与 Windows SDK，打开 **x64 Native Tools Command Prompt**，切到分发包内的 `nunif` 源码目录，执行：

   ```bat
   powershell -NoProfile -ExecutionPolicy Bypass -File .\iw3\desktop\native\build.ps1
   ```

   构建生成 `iw3/desktop/native/quest_process_audio.dll`；运行只需该 DLL，编译产物不提交到源码仓库。接口与构建说明见 [native README](iw3/desktop/native/README.md)。不构建时可选择“扬声器回环（兼容）”，其声音仍受 Windows 主音量和系统静音影响。

3. 在分发包目录运行 `iw3-desktop-gui.bat`。退出电脑端 Pop3D，避免端口冲突；保留 Quest 中的 Pop3D 接收应用。电脑和 Quest 连接同一个可信局域网。
4. 显示模式选 **Quest / Pop3D**，首次可用 **Any_V2_S、1080、30 FPS、35 Mbps**，电脑声音按需开启。需要控制网页时，选 **wc_mp 或 mss**、明确指定唯一浏览器窗口，并勾选“手柄控制电脑”；让浏览器保持前台。点击“开始”后打开 Quest 接收应用。

原生模式自动输出 Full SBS，通常为 3840×1080、每眼 1920×1080。调屏不依赖“手柄控制电脑”开关；松开握柄后，先松扳机、摇杆回中，再操作网页。切换到另一个窗口时，需要停止后重新选择。停止后改回 `Web Streaming` 即可使用原网页模式。

本预览分支不应随意运行完整 `update.bat`：上游更新脚本在拉取失败时可能执行 `git reset --hard`，丢弃未提交源码改动。升级本 fork 前先保存改动，再显式获取对应版本。

### 协议、来源与许可证

- 基础工程来自 **nagadomi/nunif**，保留 [MIT LICENSE](LICENSE)、[第三方 NOTICE](NOTICE) 和上游文档。
- 本 fork 的 PC 发送端依据公开的 **Pop3D Quest** 协议独立实现互操作，参考 [Sindelitcis/pop3d-quest](https://github.com/Sindelitcis/pop3d-quest/tree/81449d37f8727b385d61f6ccb81c15fdb92ac868) 及 v1.4.4 发行所附 Quest 源码；未复制未确认许可的 Pop3D Windows 自有实现。Quest 客户端继承 Moonlight XR/Android，采用 **GPLv3**，不因本 fork 的 MIT 基础而改为 MIT。
- Quest 经 UDP 52300 发现电脑，再连接 TCP 52301；电脑发送 HEVC、双眼图层和 PCM 音频，Quest 返回头部与手柄状态。解码和 OpenXR 显示由现有 Quest 应用完成。
- Windows 独立采音参考 Microsoft [ApplicationLoopback 示例](https://github.com/microsoft/windows-classic-samples/tree/main/Samples/ApplicationLoopback)，保留 [Microsoft MIT 许可证](iw3/desktop/native/LICENSE-Microsoft.txt)。
- **模型权重与依赖各自遵循其许可证**，不能一并视为 MIT。部分深度模型有非商业限制，见 [iw3 模型说明](iw3/README.md)。PyAV wheel 所带 FFmpeg 还可能影响二进制分发许可，见下方上游 `License Notes`。

### 当前限制

- 这是预览版。单元测试和本机链路验证不能替代 Quest 的持续性能、音画观感和调屏手感验收。
- 协议没有认证或加密，只在自己的可信局域网使用；开启电脑控制后，同网内不可信设备可能影响电脑输入。不要将端口映射到公网。
- 接收端不使用发送 PTS 做音画同步，实际音画延迟仍需实机标定。增益过高可能削波失真。
- 部分场景可见的竖边问题尚未确定根因，本版不宣称已修复。
- 网页控制仅支持已选的唯一前台窗口及已校准捕获配置；失焦、断线或几何切换期间会释放输入。`Streaming FPS` 是电脑发送速率，不代表头显显示帧率。

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
