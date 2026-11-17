/* 观察播放器实际压缩输入；不绘制画面，不改变原媒体调用。 */
(() => {
  'use strict';
  const channel = 'iw3-browser-video-v1';
  const diagnostics = {version: '1.3.0', stage: 'starting', ticks: 0, mse_urls: 0, blob_urls: 0, other_urls: 0,
    video_tracks: 0, audio_tracks: 0, other_tracks: 0, appends: 0, bytes: 0, chunks: 0, states: 0, current: 'none',
    src_object_type: 'none', width: 0, height: 0, current_rtc: false, rtc_video_tracks: 0, rtc_transform_supported: false, fullscreen:false};
  function publishDiagnostics() {
    if (document.documentElement) document.documentElement.dataset.iw3VideoMain = JSON.stringify(diagnostics);
  }
  setInterval(publishDiagnostics, 500);
  publishDiagnostics();
  const session = crypto.randomUUID();
  const sources = new WeakMap(), buffers = new WeakMap(), blobs = new Map();
  const rtcPeers = new WeakSet(), rtcReceivers = new WeakMap();
  try {
    diagnostics.rtc_transform_supported = typeof RTCRtpScriptTransform === 'function' && typeof RTCRtpReceiver === 'function' && 'transform' in RTCRtpReceiver.prototype;
    if (typeof RTCPeerConnection === 'function') {
      const setRemoteDescription = RTCPeerConnection.prototype.setRemoteDescription;
      RTCPeerConnection.prototype.setRemoteDescription = function () {
        try {
          if (!rtcPeers.has(this)) {
            this.addEventListener('track', event => {
              try {
                if (event.track?.kind === 'video' && event.receiver && !rtcReceivers.has(event.track)) {
                  rtcReceivers.set(event.track, event.receiver); diagnostics.rtc_video_tracks++;
                }
              } catch {}
            });
            rtcPeers.add(this);
          }
        } catch {}
        return Reflect.apply(setRemoteDescription, this, arguments);
      };
    }
  } catch {}
  let serial = 0, target, fullscreenRoot;
  if (globalThis.IW3MediaSource) IW3MediaSource.install();
  function sourceRecord(source) {
    if (!sources.has(source)) sources.set(source, { id: crypto.randomUUID(), tracks: [] });
    return sources.get(source);
  }
  function emit(message) { window.postMessage({ channel, session, ...message }, '*'); }
  const createURL = URL.createObjectURL;
  URL.createObjectURL = function (object) {
    const url = Reflect.apply(createURL, this, arguments);
    if (object instanceof MediaSource) {
      diagnostics.mse_urls++;
      blobs.set(url, sourceRecord(object));
      if (blobs.size > 16) blobs.delete(blobs.keys().next().value);
    } else if (typeof Blob !== 'undefined' && object instanceof Blob) diagnostics.blob_urls++;
    else diagnostics.other_urls++;
    return url;
  };
  const revokeURL = URL.revokeObjectURL;
  URL.revokeObjectURL = function (url) {
    // 播放器会在绑定video后立即revoke，currentSrc仍需关联原MediaSource。
    return Reflect.apply(revokeURL, this, arguments);
  };
  const addBuffer = MediaSource.prototype.addSourceBuffer;
  MediaSource.prototype.addSourceBuffer = function (mime) {
    const buffer = Reflect.apply(addBuffer, this, arguments);
    if (/^video\//i.test(mime)) diagnostics.video_tracks++;
    else if (/^audio\//i.test(mime)) diagnostics.audio_tracks++;
    else diagnostics.other_tracks++;
    const record = sourceRecord(this);
    const track = { buffer, mime, generation: `${session}:${record.id}:${crypto.randomUUID()}`, sequence: 0, bytes: 0, offset: null, failed: false };
    record.tracks.push(track); buffers.set(buffer, track);
    return buffer;
  };
  const removeBuffer = MediaSource.prototype.removeSourceBuffer;
  MediaSource.prototype.removeSourceBuffer = function (buffer) {
    const result = Reflect.apply(removeBuffer, this, arguments);
    const record = sources.get(this);
    if (record) record.tracks = record.tracks.filter(t => t.buffer !== buffer);
    buffers.delete(buffer);
    return result;
  };
  const append = SourceBuffer.prototype.appendBuffer;
  SourceBuffer.prototype.appendBuffer = function (data) {
    const track = buffers.get(this);
    const video = track ? activeVideo().video : undefined;
    // 只跳过活动直取视频本身；下一条预加载仍须保留 ftyp 等初始化。
    const directTrack = !video?.srcObject && blobs.get(video?.currentSrc)?.tracks.includes(track) &&
      globalThis.IW3MediaSource?.sourceFor(video);
    let copy;
    if (track && !directTrack && /^video\/(mp4|webm)/i.test(track.mime)) {
      const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
      if (bytes.byteLength <= 256 * 1024 * 1024) copy = bytes.slice();
    }
    const result = Reflect.apply(append, this, arguments);
    diagnostics.appends++;
    diagnostics.bytes += data.byteLength;
    if (copy) {
      const isInit = copy.length >= 8 && String.fromCharCode(...copy.subarray(4,8)) === 'ftyp';
      if (isInit && track.sequence) {
        track.generation = `${session}:${crypto.randomUUID()}`; track.sequence = 0; track.bytes = 0; track.offset = null; track.failed = false;
      }
      if (track.offset !== null && track.offset !== this.timestampOffset) track.failed = true;
      track.offset = this.timestampOffset;
      track.bytes += copy.byteLength;
      if (track.bytes > 256 * 1024 * 1024) track.failed = true;
      if (!track.failed) {
        for (let offset = 0; offset < copy.byteLength; offset += 8 * 1024 * 1024) {
          const piece = copy.slice(offset, offset + 8 * 1024 * 1024);
          emit({ kind: 'chunk', generation: track.generation, sequence: track.sequence++, data: piece.buffer });
          diagnostics.chunks++;
        }
      } else emit({ kind: 'error', message: '视频缓存过大或时间轴已改变；等待下一条视频。' });
    } else if (track && !directTrack && /^video\//i.test(track.mime)) {
      track.failed = true;
      emit({ kind: 'error', message: '视频片段超过缓存上限；等待下一条视频。' });
    }
    return result;
  };
  function validFullscreenRoot(root) {
    const native = document.fullscreenElement;
    return !!root?.isConnected && (native ? native === root : root.classList?.contains('isCssFullScreen') === true);
  }
  function videoFullscreenRoot(video) {
    const native = document.fullscreenElement;
    if (native && (native === video || native.contains(video))) return native;
    const css = video.closest('.isCssFullScreen');
    return validFullscreenRoot(css) ? css : undefined;
  }
  function activeVideo() {
    let selected, selectedRoot, area = 0;
    const bound = validFullscreenRoot(fullscreenRoot) ? fullscreenRoot : undefined;
    for (const video of document.querySelectorAll('video')) {
      const native = document.fullscreenElement;
      if (native && native !== video && !native.contains(video)) continue;
      const root = videoFullscreenRoot(video);
      // 切条移除video时仍绑定原全屏容器，不让页面预览抢走目标。
      if (bound && bound !== video && !bound.contains(video) && !root) continue;
      const r = video.getBoundingClientRect();
      const visible = Math.max(0, Math.min(r.right, innerWidth) - Math.max(0, r.left)) * Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(0, r.top));
      if (visible > 0 && (root && !selectedRoot || !!root === !!selectedRoot && visible > area)) {
        selected = video; selectedRoot = root; area = visible;
      }
    }
    return {video:selected,root:selectedRoot};
  }
  function current() {
    const {video,root} = activeVideo();
    if (root) fullscreenRoot = root;
    else if (!validFullscreenRoot(fullscreenRoot)) fullscreenRoot = undefined;
    const fullscreen = !!fullscreenRoot && document.visibilityState !== 'hidden' && document.hasFocus();
    if (!video && !target && !fullscreen) return {};
    const src = video?.currentSrc, object = video?.srcObject;
    const descriptor = globalThis.IW3MediaSource?.sourceFor(video) || undefined;
    const sourceKey = descriptor ? JSON.stringify(descriptor) : '';
    const source = !object && video?.videoWidth && video.videoHeight ? blobs.get(src) : undefined;
    const track = source?.tracks.find(t => /^video\//i.test(t.mime) && t.sequence > 0 && !t.failed);
    // 操作代次隔离全屏/前台切换；压缩缓存键不随路由变化，回切可重放初始化。
    if (!target || target.video !== video || target.src !== src || target.object !== object || target.media_generation !== track?.generation || target.source_key !== sourceKey ||
      target.fullscreen !== fullscreen || target.root !== fullscreenRoot)
      target = {video,src,object,media_generation:track?.generation,source_key:sourceKey,fullscreen,root:fullscreenRoot,generation:`${session}:${crypto.randomUUID()}`};
    return { video, track, descriptor, fullscreen, generation:target.generation };
  }
  function publishState() {
    const { video, track, descriptor, fullscreen, generation } = current();
    const object = video?.srcObject;
    diagnostics.ticks++;
    diagnostics.fullscreen = fullscreen === true;
    diagnostics.width = video?.videoWidth || 0; diagnostics.height = video?.videoHeight || 0;
    diagnostics.src_object_type = !object ? 'none' : typeof MediaStream === 'function' && object instanceof MediaStream ? 'media_stream' :
      typeof MediaSourceHandle === 'function' && object instanceof MediaSourceHandle ? 'media_source_handle' : 'other';
    diagnostics.current_rtc = false;
    diagnostics.source_kind = descriptor?.kind || 'none';
    diagnostics.renditions = descriptor?.renditions.length || 0;
    diagnostics.known_works = globalThis.IW3MediaSource?.knownWorks() || 0;
    try {
      if (diagnostics.src_object_type === 'media_stream') diagnostics.current_rtc = object.getVideoTracks().some(track => rtcReceivers.has(track));
    } catch {}
    diagnostics.current = !video ? 'none' : object ? 'srcObject' : !video.currentSrc ? 'empty' :
      video.currentSrc.startsWith('blob:') ? (blobs.has(video.currentSrc) ? 'blob_mapped' : 'blob_unmapped') : 'url';
    if (!generation) { publishDiagnostics(); return; }
    const offset = descriptor ? 0 : track?.offset || 0;
    emit({ kind: 'state', state: {
      session, generation, serial: ++serial, media_available:!!(descriptor || track), fullscreen:fullscreen === true,
      ...(descriptor ? {source:descriptor} : track ? {media_generation:track.generation} : {}),
      media_time: Number.isFinite(video?.currentTime) ? Math.max(0, video.currentTime - offset) : 0,
      paused: video?.paused ?? true, playback_rate: video?.playbackRate ?? 1,
      duration: Number.isFinite(video?.duration) ? Math.max(0, video.duration - offset) : null,
      width: video?.videoWidth || 0, height: video?.videoHeight || 0,
    } });
    diagnostics.states++;
    publishDiagnostics();
  }
  setInterval(publishState, 150);
  document.addEventListener('fullscreenchange', publishState);
  document.addEventListener('visibilitychange', publishState);
  window.addEventListener('focus', publishState);
  window.addEventListener('blur', publishState);
  window.addEventListener('message', event => {
    if (event.source !== window || event.data?.channel !== channel || event.data.kind !== 'commands') return;
    for (const command of event.data.commands || []) {
      const { video, track, descriptor, generation } = current();
      if ((!video && !validFullscreenRoot(fullscreenRoot)) || generation !== event.data.generation) break;
      if (command.generation !== generation) continue;
      if (command.name === 'pause_toggle') {
        if (!track && !descriptor) continue;
        if (video.paused) video.play().catch(() => {}); else video.pause();
      } else if (command.name === 'next' || command.name === 'previous') {
        const direction = command.name === 'next' ? 'next' : 'prev';
        document.querySelector(`[data-e2e="video-switch-${direction}-arrow"]`)?.click();
        break;
      } else if (command.name === 'exit_fullscreen') {
        if (fullscreenRoot && document.fullscreenElement === fullscreenRoot) document.exitFullscreen().catch(() => {});
        else if (validFullscreenRoot(fullscreenRoot)) {
          const player = video?.closest('.xgplayer') || fullscreenRoot.querySelector('.xgplayer') || fullscreenRoot;
          const native = player?.querySelector('.xgplayer-fullscreen');
          const control = /退出全屏/.test(native?.textContent || '') ? native : player?.querySelector('[data-e2e="xgplayer-page-full-screen"]');
          control?.click();
        }
      }
    }
  });
  diagnostics.stage = 'ready';
  publishDiagnostics();
})();
