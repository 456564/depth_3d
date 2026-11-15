#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <wrl.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;
using Microsoft::WRL::FtmBase;
using Microsoft::WRL::Make;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;
using Microsoft::WRL::ClassicCom;

// Microsoft ApplicationLoopback 的公开接口；不复制其 WAV/MF 文件写入链路。
class Activation final : public RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase,
                                              IActivateAudioInterfaceCompletionHandler> {
public:
    HANDLE completed = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HRESULT result = E_PENDING;
    ComPtr<IAudioClient> client;
    AUDIOCLIENT_ACTIVATION_PARAMS parameters{};
    PROPVARIANT variant{};

    ~Activation() { if (completed) CloseHandle(completed); }

    STDMETHOD(ActivateCompleted)(IActivateAudioInterfaceAsyncOperation* operation) override {
        HRESULT activated = E_UNEXPECTED;
        ComPtr<IUnknown> unknown;
        result = operation->GetActivateResult(&activated, &unknown);
        if (SUCCEEDED(result)) result = activated;
        if (SUCCEEDED(result)) result = unknown.As(&client);
        SetEvent(completed);
        return S_OK;
    }
};

class Capture final {
public:
    const DWORD pid;
    const bool include_tree;
    HANDLE stopping_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    std::deque<std::vector<BYTE>> chunks;
    size_t front_offset = 0;
    bool started = false;
    bool ready = false;
    bool stopping = false;
    bool finished = false;
    HRESULT result = E_PENDING;
    HRESULT stop_result = S_OK;

    Capture(DWORD target, bool include) : pid(target), include_tree(include) {}
    ~Capture() { stop(); if (stopping_event) CloseHandle(stopping_event); }

    void announce(HRESULT hr) {
        std::lock_guard<std::mutex> lock(mutex);
        result = hr;
        ready = true;
        changed.notify_all();
    }

    HRESULT capture() {
        struct CloseEvent { HANDLE value = nullptr; ~CloseEvent() { if (value) CloseHandle(value); } } close;
        auto activation = Make<Activation>();
        if (!activation) return E_OUTOFMEMORY;
        if (!activation->completed) return HRESULT_FROM_WIN32(GetLastError());
        activation->parameters.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
        activation->parameters.ProcessLoopbackParams.TargetProcessId = pid;
        activation->parameters.ProcessLoopbackParams.ProcessLoopbackMode = include_tree
            ? PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE : PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;
        activation->variant.vt = VT_BLOB;
        activation->variant.blob.cbSize = sizeof(activation->parameters);
        activation->variant.blob.pBlobData = reinterpret_cast<BYTE*>(&activation->parameters);
        ComPtr<IActivateAudioInterfaceAsyncOperation> operation;
        HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient),
                                                 &activation->variant, activation.Get(), &operation);
        if (FAILED(hr)) return hr;
        // 回调对象自行持有参数和事件，激活超时返回也不会引用已经退栈的存储。
        HANDLE activation_events[] = {stopping_event, activation->completed};
        DWORD waited = WaitForMultipleObjects(2, activation_events, FALSE, 5000);
        if (waited == WAIT_OBJECT_0) return HRESULT_FROM_WIN32(ERROR_CANCELLED);
        if (waited == WAIT_TIMEOUT) return HRESULT_FROM_WIN32(WAIT_TIMEOUT);
        if (waited != WAIT_OBJECT_0 + 1) return HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(activation->result)) return activation->result;
        ComPtr<IAudioClient> client = activation->client;
        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = 2;
        format.nSamplesPerSec = 48000;
        format.wBitsPerSample = 16;
        format.nBlockAlign = 4;
        format.nAvgBytesPerSec = 192000;
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            0, 0, &format, nullptr);
        if (FAILED(hr)) return hr;
        HANDLE samples_ready = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!samples_ready) return HRESULT_FROM_WIN32(GetLastError());
        close.value = samples_ready;
        hr = client->SetEventHandle(samples_ready);
        if (FAILED(hr)) return hr;
        ComPtr<IAudioCaptureClient> reader;
        hr = client->GetService(IID_PPV_ARGS(&reader));
        if (FAILED(hr)) return hr;
        hr = client->Start();
        if (FAILED(hr)) return hr;
        announce(S_OK);
        HANDLE events[] = {stopping_event, samples_ready};
        HRESULT captured = S_OK;
        while (true) {
            waited = WaitForMultipleObjects(2, events, FALSE, 200);
            if (waited == WAIT_OBJECT_0) break;
            if (waited == WAIT_TIMEOUT) continue;
            if (waited != WAIT_OBJECT_0 + 1) { captured = HRESULT_FROM_WIN32(GetLastError()); break; }
            UINT32 frames = 0;
            while (SUCCEEDED(captured = reader->GetNextPacketSize(&frames)) && frames) {
                BYTE* data = nullptr;
                DWORD flags = 0;
                captured = reader->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
                if (FAILED(captured)) break;
                try {
                    std::vector<BYTE> block(static_cast<size_t>(frames) * 4, 0);
                    if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT) && data) std::memcpy(block.data(), data, block.size());
                    std::lock_guard<std::mutex> lock(mutex);
                    // 十个最新包限制积压；只保留内存，不保存用户音频。
                    if (chunks.size() >= 10) { chunks.pop_front(); front_offset = 0; }
                    chunks.push_back(std::move(block));
                    changed.notify_all();
                } catch (...) {
                    reader->ReleaseBuffer(frames);
                    captured = E_OUTOFMEMORY;
                    break;
                }
                captured = reader->ReleaseBuffer(frames);
                if (FAILED(captured)) break;
                if (WaitForSingleObject(stopping_event, 0) == WAIT_OBJECT_0) break;
            }
            if (FAILED(captured)) break;
        }
        stop_result = client->Stop();
        return FAILED(captured) ? captured : stop_result;
    }

    void run() noexcept {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        bool initialized = SUCCEEDED(hr);
        if (initialized) {
            try { hr = capture(); }
            catch (...) { hr = E_OUTOFMEMORY; }
            // capture() 的全部音频 COM 引用先释放，再撤销工作线程 apartment。
            CoUninitialize();
        }
        std::lock_guard<std::mutex> lock(mutex);
        result = hr;
        ready = finished = true;
        changed.notify_all();
    }

    HRESULT stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!started) return S_OK;
            stopping = true;
            SetEvent(stopping_event);
            changed.notify_all();
        }
        if (worker.joinable()) worker.join();
        return stop_result;
    }
};

extern "C" __declspec(dllexport) HRESULT pa_create(uint32_t pid, int include_tree, void** output) noexcept {
    if (!output) return E_POINTER;
    *output = nullptr;
    if (!pid || (include_tree != 0 && include_tree != 1)) return E_INVALIDARG;
    Capture* capture = nullptr;
    try { capture = new (std::nothrow) Capture(pid, include_tree != 0); }
    catch (...) { return E_OUTOFMEMORY; }
    if (!capture) return E_OUTOFMEMORY;
    if (!capture->stopping_event) {
        DWORD error = GetLastError();
        delete capture;
        return HRESULT_FROM_WIN32(error);
    }
    *output = capture;
    return S_OK;
}

extern "C" __declspec(dllexport) HRESULT pa_start(void* handle) noexcept {
    if (!handle) return E_POINTER;
    auto capture = static_cast<Capture*>(handle);
    std::unique_lock<std::mutex> lock(capture->mutex);
    if (capture->started) return HRESULT_FROM_WIN32(ERROR_ALREADY_INITIALIZED);
    capture->started = true;
    try { capture->worker = std::thread(&Capture::run, capture); }
    catch (...) { capture->started = false; return E_OUTOFMEMORY; }
    if (!capture->changed.wait_for(lock, std::chrono::seconds(6), [&] { return capture->ready; })) {
        lock.unlock();
        capture->stop();
        return HRESULT_FROM_WIN32(WAIT_TIMEOUT);
    }
    return capture->result;
}

extern "C" __declspec(dllexport) HRESULT pa_read(void* handle, void* buffer, uint32_t capacity,
                                                 uint32_t timeout_ms, uint32_t* bytes) noexcept {
    if (!bytes) return E_POINTER;
    *bytes = 0;
    if (!handle || !buffer) return E_POINTER;
    if (capacity < 4 || capacity % 4 || timeout_ms > 200) return E_INVALIDARG;
    auto capture = static_cast<Capture*>(handle);
    std::unique_lock<std::mutex> lock(capture->mutex);
    if (!capture->started || !capture->ready) return HRESULT_FROM_WIN32(ERROR_INVALID_STATE);
    capture->changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return !capture->chunks.empty() || capture->finished || capture->stopping;
    });
    if (FAILED(capture->result)) return capture->result;
    if (capture->stopping) return S_FALSE;
    while (!capture->chunks.empty() && *bytes < capacity) {
        auto& chunk = capture->chunks.front();
        auto count = std::min(static_cast<size_t>(capacity - *bytes), chunk.size() - capture->front_offset);
        std::memcpy(static_cast<BYTE*>(buffer) + *bytes, chunk.data() + capture->front_offset, count);
        *bytes += static_cast<uint32_t>(count);
        capture->front_offset += count;
        if (capture->front_offset == chunk.size()) { capture->chunks.pop_front(); capture->front_offset = 0; }
    }
    return *bytes ? S_OK : S_FALSE;
}

extern "C" __declspec(dllexport) HRESULT pa_stop(void* handle) noexcept {
    if (!handle) return E_POINTER;
    return static_cast<Capture*>(handle)->stop();
}

extern "C" __declspec(dllexport) void pa_free(void* handle) noexcept {
    delete static_cast<Capture*>(handle);
}

extern "C" __declspec(dllexport) HRESULT pa_endpoint_state(int* muted, float* volume) noexcept {
    if (!muted || !volume) return E_POINTER;
    HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(initialized) && initialized != RPC_E_CHANGED_MODE) return initialized;
    HRESULT result = [&]() -> HRESULT {
        ComPtr<IMMDeviceEnumerator> enumerator;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator));
        if (FAILED(hr)) return hr;
        ComPtr<IMMDevice> endpoint;
        hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &endpoint);
        if (FAILED(hr)) return hr;
        ComPtr<IAudioEndpointVolume> control;
        hr = endpoint->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr,
                                reinterpret_cast<void**>(control.GetAddressOf()));
        if (FAILED(hr)) return hr;
        BOOL mute = FALSE;
        hr = control->GetMute(&mute);
        if (FAILED(hr)) return hr;
        *muted = mute ? 1 : 0;
        return control->GetMasterVolumeLevelScalar(volume);
    }();
    if (SUCCEEDED(initialized)) CoUninitialize();
    return result;
}
