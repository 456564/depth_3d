/* 压缩字节只经扩展后台送入本机桥；页面不能指定网络地址。 */
(() => {
  'use strict';
  const channel = 'iw3-browser-video-v1';
  const maxChunk = 8 * 1024 * 1024, maxCache = 256 * 1024 * 1024;
  const caches = new Map();
  const names = new Set(['pause_toggle', 'next', 'previous', 'exit_fullscreen']);
  let cachedBytes = 0, latestState, activeState, pumping = false, ready = false, lastStateAt = 0;
  const diagnostics = {version: '1.1.2', events: 0, states: 0, chunks: 0, bytes: 0, ticks: 0, window_seen: 0,
    rejected_source: 0, rejected_channel: 0, rejected_session: 0, rejected_kind: 0, rejected_state: 0, rejected_chunk: 0,
    report_sent: 0, report_ok: 0, report_rejected: 0, report_pending: 0, last_error: 'none'};
  const bridgeDiagnostics = {state: 'waiting', bytes: 0, next_sequence: 0};
  let lastDiagnosticAt = Date.now(), reporting = false;
  function publishDiagnostics() {
    if (document.documentElement) document.documentElement.dataset.iw3VideoRelay = JSON.stringify(diagnostics);
  }
  function readMarker(name) {
    const text = document.documentElement?.dataset[name];
    if (typeof text !== 'string' || text.length > 4096) return null;
    try { return JSON.parse(text); } catch { return null; }
  }
  function reportDiagnostics() {
    if (reporting || Date.now() - lastDiagnosticAt < 1000) return;
    lastDiagnosticAt = Date.now();
    let sending = false;
    try {
      const main = readMarker('iw3VideoMain');
      let selected, area = 0;
      for (const video of document.querySelectorAll('video')) {
        const r = video.getBoundingClientRect();
        const visible = Math.max(0, Math.min(r.right, innerWidth) - Math.max(0, r.left)) * Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(0, r.top));
        if (visible > area && video.videoWidth && video.videoHeight) { selected = video; area = visible; }
      }
      const video = selected ? {kind: selected.srcObject ? 'srcObject' : !selected.currentSrc ? 'empty' : selected.currentSrc.startsWith('blob:') ? 'blob' : 'url',
        width: selected.videoWidth, height: selected.videoHeight, paused: selected.paused, ready_state: selected.readyState} : {kind: 'none', width: 0, height: 0, paused: true, ready_state: 0};
      reporting = true; sending = true; diagnostics.report_pending = 1; diagnostics.report_sent++;
      publishDiagnostics();
      void Promise.resolve(chrome.runtime.sendMessage({kind: 'diagnostics', diagnostics: {main, relay: {...diagnostics}, video,
        bridge: {...bridgeDiagnostics}}}))
        .then(reply => {
          if (reply?.ok === true) { diagnostics.report_ok++; diagnostics.last_error = 'none'; }
          else { diagnostics.report_rejected++; diagnostics.last_error = 'report_rejected'; }
        })
        .catch(() => { diagnostics.report_rejected++; diagnostics.last_error = 'runtime_error'; })
        .finally(() => { reporting = false; diagnostics.report_pending = 0; publishDiagnostics(); });
    } catch {
      diagnostics.report_rejected++; diagnostics.last_error = sending ? 'runtime_sync_error' : 'diagnostic_build_error';
      reporting = false; diagnostics.report_pending = 0; publishDiagnostics();
    }
  }
  publishDiagnostics();
  const identifier = value => typeof value === 'string' && value.length > 0 && value.length <= 256 && /^[\x21-\x7e]+$/.test(value);
  function validState(value, session) {
    return value && value.session === session && identifier(session) && identifier(value.generation) &&
      Number.isSafeInteger(value.serial) && value.serial >= 0 &&
      Number.isFinite(value.media_time) && value.media_time >= 0 && value.media_time <= 604800 &&
      typeof value.paused === 'boolean' && Number.isFinite(value.playback_rate) && value.playback_rate >= 0.05 && value.playback_rate <= 16 &&
      [value.width, value.height].every(size => Number.isInteger(size) && size > 0 && size <= 8192) &&
      (value.duration === null || Number.isFinite(value.duration) && value.duration >= 0 && value.duration <= 604800);
  }
  function summary(state, message = '') {
    const record = activeState && caches.get(activeState.generation);
    Object.assign(bridgeDiagnostics, {state, bytes: record?.bytes || 0, next_sequence: record?.next || 0});
    if (!document.documentElement) return;
    document.documentElement.dataset.iw3VideoBridge = JSON.stringify({
      state, message, generation: activeState?.generation || null,
      bytes: record?.bytes || 0, next_sequence: record?.next || 0,
    });
  }
  function discard(generation) {
    const record = caches.get(generation);
    if (record) cachedBytes -= record.bytes;
    caches.delete(generation);
  }
  function acceptChunk(message) {
    if (!identifier(message.generation) || !Number.isSafeInteger(message.sequence) || message.sequence < 0 ||
      Object.prototype.toString.call(message.data) !== '[object ArrayBuffer]' || !message.data.byteLength || message.data.byteLength > maxChunk) return;
    let record = caches.get(message.generation);
    if (!record) {
      if (message.sequence !== 0) return;
      while (caches.size >= 3) {
        const candidate = [...caches.keys()].find(key => key !== activeState?.generation);
        if (!candidate) return;
        discard(candidate);
      }
      record = {chunks: [], bytes: 0, next: 0, session: message.session};
      caches.set(message.generation, record);
    }
    if (record.session !== message.session || message.sequence !== record.chunks.length) return;
    while (cachedBytes + message.data.byteLength > maxCache) {
      const candidate = [...caches.keys()].find(key => key !== activeState?.generation && key !== message.generation);
      if (!candidate) { summary('error', '压缩视频缓存已满；请切换视频。'); return; }
      discard(candidate);
    }
    const bytes = new Uint8Array(message.data).slice();
    record.chunks.push(bytes);
    record.bytes += bytes.length; cachedBytes += bytes.length;
    diagnostics.chunks++; diagnostics.bytes += bytes.length;
  }
  function base64(bytes) {
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
    return btoa(binary);
  }
  async function pump() {
    if (pumping) return;
    pumping = true;
    try {
      while (true) {
        if (latestState) {
          activeState = latestState; latestState = undefined; ready = false;
          const reply = await chrome.runtime.sendMessage({kind: 'state', state: activeState, focused: document.hasFocus()});
          if (!reply?.ok) { summary('waiting', '本机视频桥暂不可用；等待重连。'); break; }
          ready = true;
          const record = caches.get(activeState.generation);
          if (record && Number.isSafeInteger(reply.next_sequence) && reply.next_sequence >= 0) record.next = reply.next_sequence;
          if (!latestState || latestState.generation === activeState.generation) {
            const commands = Array.isArray(reply.commands) ? reply.commands.filter(command =>
              names.has(command?.name) && command.generation === activeState.generation) : [];
            if (commands.length) window.postMessage({channel, session: activeState.session, kind: 'commands', generation: activeState.generation, commands}, '*');
          }
          summary(record?.next > 0 ? 'source' : 'waiting', record ? '' : '等待当前视频的初始化字节。');
          continue;
        }
        const record = activeState && caches.get(activeState.generation);
        if (!ready || !record || record.next >= record.chunks.length) break;
        const sequence = record.next;
        const reply = await chrome.runtime.sendMessage({
          kind: 'chunk', session: activeState.session, generation: activeState.generation,
          sequence, data: base64(record.chunks[sequence]),
        });
        if (!reply?.ok || !Number.isSafeInteger(reply.next_sequence) || reply.next_sequence <= sequence) {
          ready = false; summary('waiting', '压缩视频发送中断；等待重连。'); break;
        }
        record.next = reply.next_sequence;
        summary('source');
      }
    } catch {
      ready = false; summary('waiting', '扩展连接已改变；请刷新抖音。');
    } finally { pumping = false; }
  }
  window.addEventListener('message', event => {
    const message = event.data;
    diagnostics.window_seen++;
    if (event.source !== window) { diagnostics.rejected_source++; return; }
    if (message?.channel !== channel) { diagnostics.rejected_channel++; return; }
    if (!identifier(message.session)) { diagnostics.rejected_session++; return; }
    diagnostics.events++;
    if (message.kind === 'chunk') {
      const accepted = diagnostics.chunks;
      acceptChunk(message);
      if (diagnostics.chunks === accepted) diagnostics.rejected_chunk++;
    } else if (message.kind === 'state' && validState(message.state, message.session)) {
      diagnostics.states++;
      latestState = message.state; lastStateAt = Date.now();
    } else if (message.kind === 'error') summary('error', '网页未提供可用压缩视频；不会回退录屏。');
    else { diagnostics[message.kind === 'state' ? 'rejected_state' : 'rejected_kind']++; return; }
    void pump();
  });
  setInterval(() => {
    diagnostics.ticks++;
    publishDiagnostics();
    reportDiagnostics();
    if (activeState && Date.now() - lastStateAt > 1000) {
      ready = false; summary('waiting', '等待当前网页视频；不会回退录屏。');
    }
  }, 250);
})();
