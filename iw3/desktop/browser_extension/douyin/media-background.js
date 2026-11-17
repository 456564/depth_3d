/* 仅本扩展的抖音顶层页面可访问固定本机视频桥。 */
(() => {
  'use strict';
  const endpoint = 'http://127.0.0.1:52302';
  const maxChunk = 8 * 1024 * 1024;
  let token, pairing, lease, delivery;
  const pages = new Map();
  const transport = {stage:'idle',http_status:null,error:null,ok:false,hello:0,state:0,chunk:0,bytes:0,updated_at:null};
  let playback = null;
  const identifier = value => typeof value === 'string' && value.length > 0 && value.length <= 256 && /^[\x21-\x7e]+$/.test(value);
  function counts(value, keys) {
    if (!value || typeof value !== 'object') return null;
    const safe = {};
    for (const key of keys) {
      if (!Number.isSafeInteger(value[key]) || value[key] < 0) return null;
      safe[key] = value[key];
    }
    return safe;
  }
  function cleanDiagnostics(value) {
    if (!value || typeof value !== 'object') return null;
    const relay = counts(value.relay, ['events','states','chunks','bytes']);
    const video = counts(value.video, ['width','height','ready_state']);
    const bridge = counts(value.bridge, ['bytes','next_sequence']);
    if (!relay || !['1.1.1','1.1.2'].includes(value.relay.version) || !video || video.width > 8192 || video.height > 8192 || video.ready_state > 4 ||
      !['none','empty','url','blob','srcObject'].includes(value.video.kind) || typeof value.video.paused !== 'boolean' ||
      !bridge || bridge.bytes > 256 * 1024 * 1024 || !['waiting','source','error'].includes(value.bridge.state)) return null;
    for (const key of ['ticks','window_seen','rejected_source','rejected_channel','rejected_session','rejected_kind','rejected_state','rejected_chunk','report_sent','report_ok','report_rejected','report_pending']) {
      if (value.relay[key] === undefined) continue;
      if (!Number.isSafeInteger(value.relay[key]) || value.relay[key] < 0 || key === 'report_pending' && value.relay[key] > 1) return null;
      relay[key] = value.relay[key];
    }
    if (value.relay.last_error !== undefined) {
      if (!['none','report_rejected','runtime_error','runtime_sync_error','diagnostic_build_error'].includes(value.relay.last_error)) return null;
      relay.last_error = value.relay.last_error;
    }
    let main = null;
    if (value.main !== null && value.main !== undefined) {
      main = counts(value.main, ['ticks','mse_urls','blob_urls','other_urls','video_tracks','audio_tracks','other_tracks','appends','bytes','chunks','states']);
      if (!main || !['1.1.1','1.1.2'].includes(value.main.version) || !['starting','ready'].includes(value.main.stage) ||
        !['none','empty','url','blob_mapped','blob_unmapped','srcObject'].includes(value.main.current)) return null;
      if (value.main.src_object_type !== undefined) {
        if (!['none','media_stream','media_source_handle','other'].includes(value.main.src_object_type)) return null;
        main.src_object_type = value.main.src_object_type;
      }
      if (value.main.width !== undefined || value.main.height !== undefined) {
        const dimensions = counts(value.main,['width','height']);
        if (!dimensions || dimensions.width > 8192 || dimensions.height > 8192) return null;
        Object.assign(main,dimensions);
      }
      Object.assign(main, {version:value.main.version,stage:value.main.stage,current:value.main.current});
    }
    return {main,relay:{version:value.relay.version,...relay},video:{...video,kind:value.video.kind,paused:value.video.paused},bridge:{...bridge,state:value.bridge.state}};
  }
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
      Object.assign(transport, {stage:'hello',http_status:null}); transport.hello++;
      const response = await fetch(`${endpoint}/hello`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(1500),
      });
      transport.http_status = response.status;
      if (!response.ok) { transport.ok = false; transport.error = 'http'; throw new Error('Bridge pairing failed'); }
      const result = await response.json();
      if (typeof result.token !== 'string' || result.token.length > 128) throw new Error('Invalid bridge token');
      token = result.token;
    })().finally(() => { pairing = undefined; });
    await pairing;
  }
  async function post(path, body, headers = {}) {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (!token) await pair();
      Object.assign(transport, {stage:path.slice(1),http_status:null}); transport[path.slice(1)]++;
      const response = await fetch(`${endpoint}${path}`, {
        method: 'POST', headers: {'X-IW3-Token': token, ...headers}, body,
        cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(1500),
      });
      transport.http_status = response.status;
      if (response.status === 403 && attempt === 0) { transport.ok = false; transport.error = 'http'; token = undefined; continue; }
      if (!response.ok) { transport.ok = false; transport.error = 'http'; return {ok: false, message: '本机视频桥暂不可用'}; }
      const result = await response.json();
      transport.ok = true; transport.error = null; transport.updated_at = Date.now();
      if (path === '/chunk') transport.bytes += body.byteLength;
      return {ok: true, commands: result.commands || [], next_sequence: result.next_sequence, need_media: result.need_media};
    }
    return {ok: false, message: '本机视频桥认证失败'};
  }
  async function handle(message, sender) {
    try {
      if (message?.kind === 'status') {
        if (sender.id !== chrome.runtime.id || sender.tab || sender.url !== chrome.runtime.getURL('popup.html'))
          return {ok:false,message:'仅允许扩展状态面板'};
        const now = Date.now();
        const records = [...pages.values()];
        const record = (lease && now < lease.until ? pages.get(lease.tab) : null) || records.findLast(page => page.active) || records.at(-1);
        const source = !!lease && record === pages.get(lease.tab) && now < lease.until;
        const confirmed = source && delivery?.tab === lease.tab && delivery.session === lease.session && delivery.generation === lease.generation ? delivery : null;
        return {
          ok:true,
          page:record ? {diagnostics:record.diagnostics,age_ms:Math.max(0,now-record.received_at),fresh:now-record.received_at<=3000,
            source,delivered:!!confirmed && confirmed.bytes > 0 && confirmed.sequence > 0 && transport.ok && now-transport.updated_at<=3000,
            confirmed_bytes:confirmed?.bytes || 0,confirmed_sequence:confirmed?.sequence || 0} : null,
          pages:pages.size,
          transport:{...transport,age_ms:transport.updated_at === null ? null : Math.max(0,now-transport.updated_at)},
          playback,
        };
      }
      if (sender.frameId !== 0 || !Number.isInteger(sender.tab?.id) || new URL(sender.url).origin !== 'https://www.douyin.com')
        return {ok: false, message: '仅允许抖音顶层页面'};
      const now = Date.now();
      if (message?.kind === 'diagnostics') {
        const diagnostics = cleanDiagnostics(message.diagnostics);
        if (!diagnostics) return {ok:false,message:'无效诊断统计'};
        pages.delete(sender.tab.id);
        pages.set(sender.tab.id, {diagnostics,received_at:now,active:sender.tab.active === true});
        while (pages.size > 8) pages.delete(pages.keys().next().value);
        return {ok:true};
      }
      if (message?.kind === 'state' && validState(message.state)) {
        if (lease && lease.tab !== sender.tab.id && now < lease.until && !(sender.tab.active && message.focused === true))
          return {ok: false, message: '另一个抖音标签正在提供视频'};
        const state = {};
        for (const key of ['session','generation','serial','media_time','paused','playback_rate','duration','width','height']) state[key] = message.state[key];
        lease = {tab: sender.tab.id, session: state.session, generation: state.generation, until: now + 1000};
        if (!delivery || delivery.tab !== lease.tab || delivery.session !== lease.session || delivery.generation !== lease.generation)
          delivery = {tab:lease.tab,session:lease.session,generation:lease.generation,bytes:0,sequence:0};
        const currentDelivery = delivery;
        playback = {width:state.width,height:state.height,media_time:state.media_time,paused:state.paused,playback_rate:state.playback_rate};
        const reply = await post('/state', JSON.stringify(state), {'Content-Type': 'application/json'});
        // 桥重启会清空同一视频的序号；换对象可阻断旧请求的迟到确认。
        if (currentDelivery === delivery && reply.ok && reply.next_sequence === 0)
          delivery = {...currentDelivery,bytes:0,sequence:0};
        return reply;
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
      const currentDelivery = delivery;
      const reply = await post('/chunk', bytes, {
        'Content-Type': 'application/octet-stream', 'X-IW3-Session': message.session,
        'X-IW3-Generation': message.generation, 'X-IW3-Sequence': String(message.sequence),
      });
      if (currentDelivery === delivery && reply.ok && Number.isSafeInteger(reply.next_sequence) && reply.next_sequence > message.sequence &&
        message.sequence >= currentDelivery.sequence) {
        currentDelivery.bytes += bytes.byteLength; currentDelivery.sequence = reply.next_sequence;
      }
      return reply;
    } catch (error) {
      transport.ok = false;
      transport.error = transport.http_status >= 400 ? 'http' : error?.name === 'TimeoutError' ? 'timeout' : 'network_or_response';
      return {ok: false, message: '本机视频桥暂不可用'};
    }
  }
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    void handle(message, sender).then(sendResponse);
    return true;
  });
})();
