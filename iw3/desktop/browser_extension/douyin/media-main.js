/* 观察播放器实际压缩输入；不绘制画面，不改变原媒体调用。 */
(() => {
  'use strict';
  const channel = 'iw3-browser-video-v1';
  const session = crypto.randomUUID();
  const sources = new WeakMap(), buffers = new WeakMap(), blobs = new Map();
  let serial = 0;
  function sourceRecord(source) {
    if (!sources.has(source)) sources.set(source, { id: crypto.randomUUID(), tracks: [] });
    return sources.get(source);
  }
  function emit(message) { window.postMessage({ channel, session, ...message }, '*'); }
  const createURL = URL.createObjectURL;
  URL.createObjectURL = function (object) {
    const url = Reflect.apply(createURL, this, arguments);
    if (object instanceof MediaSource) {
      blobs.set(url, sourceRecord(object));
      if (blobs.size > 16) blobs.delete(blobs.keys().next().value);
    }
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
    let copy;
    if (track && /^video\/(mp4|webm)/i.test(track.mime)) {
      const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
      if (bytes.byteLength <= 256 * 1024 * 1024) copy = bytes.slice();
    }
    const result = Reflect.apply(append, this, arguments);
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
        }
      } else emit({ kind: 'error', message: '视频缓存过大或时间轴已改变；等待下一条视频。' });
    } else if (track && /^video\//i.test(track.mime)) {
      track.failed = true;
      emit({ kind: 'error', message: '视频片段超过缓存上限；等待下一条视频。' });
    }
    return result;
  };
  function activeVideo() {
    let selected, area = 0;
    for (const video of document.querySelectorAll('video')) {
      const r = video.getBoundingClientRect();
      const visible = Math.max(0, Math.min(r.right, innerWidth) - Math.max(0, r.left)) * Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(0, r.top));
      if (visible > area && video.videoWidth && video.videoHeight) { selected = video; area = visible; }
    }
    return selected;
  }
  function current() {
    const video = activeVideo();
    const source = video && blobs.get(video.currentSrc);
    const track = source?.tracks.find(t => /^video\//i.test(t.mime) && t.sequence > 0 && !t.failed);
    return { video, track };
  }
  setInterval(() => {
    const { video, track } = current();
    if (!track) return;
    emit({ kind: 'state', state: {
      session, generation: track.generation, serial: ++serial,
      media_time: Math.max(0, video.currentTime - track.offset),
      paused: video.paused, playback_rate: video.playbackRate,
      duration: Number.isFinite(video.duration) ? Math.max(0, video.duration - track.offset) : null,
      width: video.videoWidth, height: video.videoHeight,
    } });
  }, 150);
  window.addEventListener('message', event => {
    if (event.source !== window || event.data?.channel !== channel || event.data.kind !== 'commands') return;
    const { video, track } = current();
    if (!track || track.generation !== event.data.generation) return;
    for (const command of event.data.commands || []) {
      if (command.generation !== track.generation) continue;
      if (command.name === 'pause_toggle') {
        if (video.paused) video.play().catch(() => {}); else video.pause();
      } else if (command.name === 'next' || command.name === 'previous') {
        const direction = command.name === 'next' ? 'next' : 'prev';
        document.querySelector(`[data-e2e="video-switch-${direction}-arrow"]`)?.click();
      } else if (command.name === 'exit_fullscreen') {
        if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
        else if (document.querySelector('.isCssFullScreen')) {
          const player = video.closest('.xgplayer');
          const native = player?.querySelector('.xgplayer-fullscreen');
          const control = /退出全屏/.test(native?.textContent || '') ? native : player?.querySelector('[data-e2e="xgplayer-page-full-screen"]');
          control?.click();
        }
      }
    }
  });
})();
