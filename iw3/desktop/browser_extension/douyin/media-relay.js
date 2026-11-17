/* 压缩字节只经扩展后台送入本机桥；页面不能指定网络地址。 */
(() => {
  'use strict';
  const channel = 'iw3-browser-video-v1';
  const maxChunk = 8 * 1024 * 1024, maxCache = 256 * 1024 * 1024;
  const caches = new Map();
  const names = new Set(['pause_toggle', 'next', 'previous', 'exit_fullscreen']);
  let cachedBytes = 0, latestState, activeState, pumping = false, ready = false, lastStateAt = 0;
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
    if (!document.documentElement) return;
    const record = activeState && caches.get(activeState.generation);
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
    if (event.source !== window || message?.channel !== channel || !identifier(message.session)) return;
    if (message.kind === 'chunk') acceptChunk(message);
    else if (message.kind === 'state' && validState(message.state, message.session)) {
      latestState = message.state; lastStateAt = Date.now();
    } else if (message.kind === 'error') summary('error', '网页未提供可用压缩视频；不会回退录屏。');
    else return;
    void pump();
  });
  setInterval(() => {
    if (activeState && Date.now() - lastStateAt > 1000) {
      ready = false; summary('waiting', '等待当前网页视频；不会回退录屏。');
    }
  }, 250);
})();
