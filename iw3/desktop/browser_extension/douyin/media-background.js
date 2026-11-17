/* 仅本扩展的抖音顶层页面可访问固定本机视频桥。 */
(() => {
  'use strict';
  const endpoint = 'http://127.0.0.1:52302';
  const maxChunk = 8 * 1024 * 1024;
  let token, pairing, lease;
  const identifier = value => typeof value === 'string' && value.length > 0 && value.length <= 256 && /^[\x21-\x7e]+$/.test(value);
  function validState(value) {
    return value && identifier(value.session) && identifier(value.generation) &&
      Number.isSafeInteger(value.serial) && value.serial >= 0 &&
      Number.isFinite(value.media_time) && value.media_time >= 0 && value.media_time <= 604800 &&
      typeof value.paused === 'boolean' && Number.isFinite(value.playback_rate) && value.playback_rate >= 0.05 && value.playback_rate <= 16 &&
      [value.width, value.height].every(size => Number.isInteger(size) && size > 0 && size <= 8192) &&
      (value.duration === null || Number.isFinite(value.duration) && value.duration >= 0 && value.duration <= 604800);
  }
  async function pair() {
    if (!pairing) pairing = (async () => {
      const response = await fetch(`${endpoint}/hello`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(1500),
      });
      if (!response.ok) throw new Error('Bridge pairing failed');
      const result = await response.json();
      if (typeof result.token !== 'string' || result.token.length > 128) throw new Error('Invalid bridge token');
      token = result.token;
    })().finally(() => { pairing = undefined; });
    await pairing;
  }
  async function post(path, body, headers = {}) {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (!token) await pair();
      const response = await fetch(`${endpoint}${path}`, {
        method: 'POST', headers: {'X-IW3-Token': token, ...headers}, body,
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(1500),
      });
      if (response.status === 403 && attempt === 0) { token = undefined; continue; }
      if (!response.ok) return {ok: false, message: '本机视频桥暂不可用'};
      const result = await response.json();
      return {ok: true, commands: result.commands || [], next_sequence: result.next_sequence, need_media: result.need_media};
    }
    return {ok: false, message: '本机视频桥认证失败'};
  }
  async function handle(message, sender) {
    try {
      if (sender.frameId !== 0 || !Number.isInteger(sender.tab?.id) || new URL(sender.url).origin !== 'https://www.douyin.com')
        return {ok: false, message: '仅允许抖音顶层页面'};
      const now = Date.now();
      if (message?.kind === 'state' && validState(message.state)) {
        if (lease && lease.tab !== sender.tab.id && now < lease.until && !(sender.tab.active && message.focused === true))
          return {ok: false, message: '另一个抖音标签正在提供视频'};
        const state = {};
        for (const key of ['session','generation','serial','media_time','paused','playback_rate','duration','width','height']) state[key] = message.state[key];
        lease = {tab: sender.tab.id, session: state.session, generation: state.generation, until: now + 1000};
        return await post('/state', JSON.stringify(state), {'Content-Type': 'application/json'});
      }
      if (message?.kind !== 'chunk' || !lease || lease.tab !== sender.tab.id || now > lease.until ||
        message.session !== lease.session || message.generation !== lease.generation ||
        !Number.isSafeInteger(message.sequence) || message.sequence < 0 ||
        typeof message.data !== 'string' || !message.data.length || message.data.length > 4 * Math.ceil(maxChunk / 3) ||
        message.data.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(message.data))
        return {ok: false, message: '无效或过期的视频字节'};
      const binary = atob(message.data);
      if (!binary.length || binary.length > maxChunk) return {ok: false, message: '视频字节超出上限'};
      const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
      return await post('/chunk', bytes, {
        'Content-Type': 'application/octet-stream', 'X-IW3-Session': message.session,
        'X-IW3-Generation': message.generation, 'X-IW3-Sequence': String(message.sequence),
      });
    } catch {
      return {ok: false, message: '本机视频桥暂不可用'};
    }
  }
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    void handle(message, sender).then(sendResponse);
    return true;
  });
})();
