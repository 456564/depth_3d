# Windows 进程音频回环

`quest_process_audio.dll` 通过 Windows `ActivateAudioInterfaceAsync` 采集进程音频。最低 Windows build 20348；x64 DLL。使用 Microsoft ApplicationLoopback 示例公开接口，许可见 `LICENSE-Microsoft.txt`。

## 构建

安装 MSVC x64 C++ Build Tools 与 Windows SDK，确保 `cl.exe` 可发现，在本目录执行 `./build.ps1`。脚本显式传入 SDK include/lib 路径，使用 `/MT` 静态运行库。生成 `quest_process_audio.dll`、obj、lib、exp；运行只需 DLL 和系统自带 ole32/MMDevAPI/KERNEL32。

## C ABI

调用约定 cdecl；Python 使用 `ctypes.CDLL`，HRESULT 按有符号 int32 读取。负值失败；`S_FALSE=1` 表示本次无 PCM。

```c
HRESULT pa_create(uint32_t pid, int include_tree, void** output);
HRESULT pa_start(void* handle);
HRESULT pa_read(void* handle, void* buffer, uint32_t capacity,
                uint32_t timeout_ms, uint32_t* bytes);
HRESULT pa_stop(void* handle);
void pa_free(void* handle);
HRESULT pa_endpoint_state(int* muted, float* volume);
```

- 固定 PCM：48000 Hz、双声道、signed16 little-endian，4 bytes/frame。
- `include_tree=1` 捕获指定 PID 及子进程；`0` 排除指定 PID 及子进程，捕获其余系统音频。生产用法排除 iw3 自身树，并非单浏览器或单标签捕获。
- `pa_read` 容量至少 4 字节且是 4 的倍数，超时 0–200 ms；允许按帧部分消费包。静音包显式补零，最多保留十个最新包，超出丢旧包。
- 每个句柄只能 start 一次，stop 可重复。仅 read 与 stop 允许并发；create/start/stop/free 生命周期操作须串行。先 stop，再 join Python 读取线程，最后 free；不得在 read 进行时 free。
- `pa_endpoint_state` 只读取默认 render/eConsole 端点静音与音量，不改变系统设置。所有音频仅留内存，不写音频文件。

## 本机验证

Windows 11 build 22631，MSVC 14.44/SDK 10.0.22621：构建、导出/依赖检查、参数自检、include/exclude、部分读取和并发 read/stop 均通过。

2026-11-15 用户手动静音 Windows 后，3 秒探针读取 `muted=true, volume=0.18`，298 块全部非零，572160 bytes，peak 28297。说明本机主端点静音后仍可捕获有效 PCM；Quest 实机听感及音画同步仍需验收。网页播放器自身静音仍会消除音源。

官方来源：

- <https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/>
- <https://github.com/microsoft/windows-classic-samples/tree/main/Samples/ApplicationLoopback>
