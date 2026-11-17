/* 来源描述仅在内存流转；页面 JSON 不作为“当前作品”的身份证明。 */
(() => {
  'use strict';
  const works = new Map(), maxJson = 4 * 1024 * 1024;
  const identifier = value => typeof value === 'string' && /^[\w:-]{1,128}$/.test(value);
  const workId = value => typeof value === 'string' && /^\d{1,32}$/.test(value);
  function mediaURL(value, upgrade = false) {
    if (typeof value !== 'string' || value.length > 4096) return null;
    try {
      const url = new URL(value);
      if (upgrade && url.protocol === 'http:') url.protocol = 'https:';
      if (url.protocol !== 'https:' || url.username || url.password || url.hash || url.port && url.port !== '443' ||
          !['douyinvod.com','douyincdn.com'].some(domain => url.hostname.endsWith('.' + domain))) return null;
      return url.href;
    } catch { return null; }
  }
  function clean(value) {
    if (!value || !['vod','live'].includes(value.kind) || !workId(value.id) ||
        !Array.isArray(value.renditions) || value.renditions.length < 1 || value.renditions.length > 8) return null;
    const renditions = [], ids = new Set();
    for (const item of value.renditions) {
      if (!item || !identifier(item.id) || ids.has(item.id) || !['h264','hevc','av1','unknown'].includes(item.codec) ||
          ![item.width,item.height].every(size => Number.isInteger(size) && size >= (value.kind === 'live' ? 0 : 1) && size <= 8192) ||
          !(item.width > 0 && item.height > 0 || value.kind === 'live' && item.width === 0 && item.height === 0) ||
          !Number.isSafeInteger(item.bitrate) || item.bitrate < 0 || item.bitrate > 1000000000 ||
          !Array.isArray(item.urls) || item.urls.length < 1 || item.urls.length > 3) return null;
      const urls = item.urls.map(url => mediaURL(url));
      if (urls.some(url => !url)) return null;
      ids.add(item.id);
      renditions.push({id:item.id,width:item.width,height:item.height,bitrate:item.bitrate,codec:item.codec,urls});
    }
    return {kind:value.kind,id:value.id,renditions};
  }
  function observeVod(value) {
    if (!value || typeof value !== 'object') return;
    const entries = [value.aweme_detail,...(Array.isArray(value.aweme_list) ? value.aweme_list.slice(0,64) : [])];
    for (const entry of entries) {
      if (!entry || !workId(entry.aweme_id) || !Array.isArray(entry.video?.bit_rate)) continue;
      const renditions = [];
      for (const item of entry.video.bit_rate.slice(0,64)) {
        const address = item?.play_addr;
        if (!address || !Array.isArray(address.url_list)) continue;
        const urls = [...new Set(address.url_list.map(url => mediaURL(url)).filter(Boolean))].slice(0,3);
        const codec = item.is_h265 === 1 ? 'hevc' : item.is_h265 === 0 ? 'h264' : 'unknown';
        const gear = identifier(item.gear_name) ? item.gear_name.slice(0,64) : 'native';
        const id = `${gear}:${address.width}x${address.height}:${codec}:${item.bit_rate}`;
        const rendition = {id,width:address.width,height:address.height,bitrate:item.bit_rate,codec,urls};
        if (clean({kind:'vod',id:entry.aweme_id,renditions:[rendition]}) && !renditions.some(other => other.id === id)) renditions.push(rendition);
      }
      renditions.sort((a,b) => Math.min(b.width,b.height)-Math.min(a.width,a.height) || b.width*b.height-a.width*a.height ||
        (a.codec === b.codec ? b.bitrate-a.bitrate : a.codec.localeCompare(b.codec)) || a.id.localeCompare(b.id));
      if (renditions.length) {
        works.delete(entry.aweme_id);
        works.set(entry.aweme_id,{kind:'vod',id:entry.aweme_id,renditions:renditions.slice(0,8)});
        while (works.size > 64) works.delete(works.keys().next().value);
      }
    }
  }
  function parseText(text) {
    if (typeof text !== 'string' || text.length > maxJson) return;
    for (const input of [text,(() => {try{return decodeURIComponent(text);}catch{return '';}})()]) {
      try { observeVod(JSON.parse(input)); return; } catch {}
    }
  }
  function apiURL(value) {
    try {
      const url = new URL(typeof value === 'string' ? value : value?.url, location.href);
      return url.origin === 'https://www.douyin.com' && /^\/aweme\/v1\/web\/(aweme\/(detail|related|post)|tab\/feed)\//.test(url.pathname);
    } catch { return false; }
  }
  async function observeResponse(response) {
    let reader;
    try {
      if (!response.ok || !/json/i.test(response.headers.get('content-type') || '') || Number(response.headers.get('content-length') || 0) > maxJson) return;
      const copy = response.clone(); reader = copy.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder(); let text = '', bytes = 0;
      while (true) {
        const result = await reader.read();
        if (result.done) break;
        bytes += result.value.byteLength;
        if (bytes > maxJson) return;
        text += decoder.decode(result.value,{stream:true});
      }
      parseText(text + decoder.decode());
    } catch {} finally { if (reader) void reader.cancel().catch(() => {}); }
  }
  function install() {
    if (typeof fetch === 'function') {
      const original = fetch;
      globalThis.fetch = function () {
        const result = Reflect.apply(original,this,arguments);
        if (apiURL(arguments[0])) void Promise.resolve(result).then(response => observeResponse(response)).catch(() => {});
        return result;
      };
    }
    if (typeof XMLHttpRequest === 'function') {
      const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send, requests = new WeakMap();
      XMLHttpRequest.prototype.open = function () {
        const result = Reflect.apply(open,this,arguments); requests.set(this,apiURL(arguments[1])); return result;
      };
      XMLHttpRequest.prototype.send = function () {
        if (requests.get(this)) this.addEventListener('load',() => {
          try {
            if (this.status < 200 || this.status >= 300) return;
            if (this.responseType === 'json') observeVod(this.response);
            else if (!this.responseType || this.responseType === 'text') parseText(this.responseText);
          } catch {}
        },{once:true});
        return Reflect.apply(send,this,arguments);
      };
    }
    const embedded = () => {
      for (const script of document.querySelectorAll('script[type="application/json"],script#RENDER_DATA,script#__NEXT_DATA__')) parseText(script.textContent);
    };
    embedded(); document.addEventListener('DOMContentLoaded',embedded,{once:true});
  }
  function liveRoom(value) {
    const found = new Map(), seen = new WeakSet(); let nodes = 0;
    function visit(item,depth) {
      if (++nodes > 10000 || depth > 24) return;
      if (typeof item === 'string') {
        if (item.length > 2 * 1024 * 1024 || !item.includes('stream_url')) return;
        try { visit(JSON.parse(item),depth+1); } catch {}
        for (const line of item.split('\n').slice(0,1000)) {
          const colon = line.indexOf(':');
          if (colon > 0 && /^[\da-f]+$/i.test(line.slice(0,colon))) try { visit(JSON.parse(line.slice(colon+1)),depth+1); } catch {}
        }
        return;
      }
      if (!item || typeof item !== 'object' || seen.has(item)) return;
      seen.add(item);
      const room = item.state?.roomStore?.roomInfo;
      if (workId(room?.room?.id_str) && room.web_stream_url?.flv_pull_url) {
        const renditions = [];
        for (const id of ['FULL_HD1','SD1','SD2']) {
          const url = mediaURL(room.web_stream_url.flv_pull_url[id],true);
          if (url && !renditions.some(rendition => rendition.urls.includes(url))) renditions.push({id,width:0,height:0,bitrate:0,codec:id === 'FULL_HD1' ? 'hevc' : 'unknown',urls:[url]});
        }
        if (renditions.length) found.set(room.room.id_str,{kind:'live',id:room.room.id_str,renditions});
      }
      for (const child of Object.values(item).slice(0,1000)) visit(child,depth+1);
    }
    visit(value,0);
    return found.size === 1 ? found.values().next().value : null;
  }
  // 内部 room id 尚无可靠 public path 映射，只信任本次文档加载的初始直播路径。
  const livePath = typeof location !== 'undefined' && location.hostname === 'live.douyin.com' &&
    /^\/\d{1,32}\/?$/.test(location.pathname) ? location.pathname : null;
  let liveDeparted = false, flight, flightCount = 0, liveSource;
  const own = (value,key) => value && Object.getOwnPropertyDescriptor(value,key)?.value;
  function cardWorkId(video) {
    for (let node=video,depth=0;node && depth<8;node=node.parentElement,depth++) {
      if (typeof node.querySelectorAll === 'function' && node.querySelectorAll('video').length > 1) break;
      for (const name of Object.getOwnPropertyNames(node).filter(key => /^__react(Props|Fiber)\$/.test(key))) {
        const value = own(node,name);
        let fiber = name.startsWith('__reactFiber') ? value : null;
        if (fiber) {
          let root=fiber,count=0;
          while (own(root,'return') && count++<80) root=own(root,'return');
          const current=own(own(root,'stateNode'),'current');
          if (!current) continue;
          if (current !== root) fiber=own(fiber,'alternate');
        }
        for (let count=0;count<8;count++) {
          const props=fiber ? own(fiber,'memoizedProps') : count === 0 && !name.startsWith('__reactFiber') ? value : null;
          // 真实活动video的祖先Fiber暴露awemeInfo；nextVideoInfo是预加载的另一作品。
          const direct=own(props,'awemeId'),nested=own(own(props,'awemeInfo'),'awemeId');
          if (workId(direct) && workId(nested) && direct !== nested) return null;
          const id=workId(nested) ? nested : direct;
          if (workId(id)) return id;
          if (!fiber) break;
          fiber=own(fiber,'return');
        }
      }
    }
    return null;
  }
  function sourceFor(video) {
    try {
      if (location.hostname === 'live.douyin.com') {
        if (!livePath || liveDeparted || livePath !== location.pathname) {
          liveDeparted = true; liveSource = null;
          return null;
        }
        if (!video) return null;
        const value = Object.getOwnPropertyDescriptor(globalThis,'__pace_f')?.value;
        if (Array.isArray(value) && (value !== flight || value.length !== flightCount)) {
          const source = liveRoom(value.slice(value === flight ? flightCount : 0));
          flight = value; flightCount = value.length;
          if (source && liveSource && source.id !== liveSource.id) {
            liveDeparted = true; liveSource = null;
            return null;
          }
          if (source) liveSource = source;
        }
        return liveSource || null;
      }
      if (!video) return null;
      let id;
      try { id = new URL(video.currentSrc).searchParams.get('__vid'); } catch {}
      // URL携带的作品ID优先；不使用feed顺序、尺寸或时长匹配。
      if (!id && location.hostname === 'www.douyin.com') id=cardWorkId(video);
      if (workId(id)) return works.get(id) || null;
      for (const source of works.values()) if (source.renditions.some(item => item.urls.includes(video.currentSrc))) return source;
    } catch {}
    return null;
  }
  globalThis.IW3MediaSource = Object.freeze({clean,observeVod,install,sourceFor,knownWorks:() => works.size});
})();
