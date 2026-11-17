const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function browser() {
  const events = [], timers = [], listeners = {};
  let uuid = 0;
  class SourceBuffer {
    timestampOffset = 0;
    appendBuffer(data) { if (this.fail) throw new Error('append failed'); this.last = data; }
  }
  class MediaSource { addSourceBuffer() { return new SourceBuffer(); } }
  class MediaStream {
    getTracks() { throw new Error('must not inspect tracks'); }
    getVideoTracks() { return this.videoTracks || []; }
  }
  class MediaSourceHandle {}
  class RTCPeerConnection {
    listeners=[];
    addEventListener(name,callback) { this.listeners.push({name,callback}); }
    setRemoteDescription() {
      if (this.failure) throw this.failure;
      this.arguments=Array.from(arguments); return this.result ||= Promise.resolve();
    }
  }
  class RTCRtpReceiver {}
  Object.defineProperty(RTCRtpReceiver.prototype,'transform',{set(){throw new Error('must not install transform');},get(){throw new Error('must not read transform');}});
  class RTCRtpScriptTransform { constructor() { throw new Error('must not install transform'); } }
  const video = {
    currentSrc: '', videoWidth: 1080, videoHeight: 1920, currentTime: 2,
    duration: 10, paused: false, playbackRate: 1,
    getBoundingClientRect: () => ({left:0,top:0,right:1080,bottom:1920,width:1080,height:1920}),
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
    closest() { return null; },
  };
  const context = {
    console, MediaSource, SourceBuffer, MediaStream, MediaSourceHandle, RTCPeerConnection, RTCRtpReceiver, RTCRtpScriptTransform, ArrayBuffer, Uint8Array, Blob,
    crypto: { randomUUID: () => `id${++uuid}` },
    URL: { createObjectURL: () => `blob:${++uuid}`, revokeObjectURL() {} },
    innerWidth: 1080, innerHeight: 1920,
    document: { querySelectorAll: s => s === 'video' ? [video] : [], querySelector: () => null, documentElement: {dataset:{}}, fullscreenElement:null,
      visibilityState:'visible',hasFocus:()=>true,addEventListener:(name,f)=>(listeners[name] ||= []).push(f) },
    setInterval: f => timers.push(f),
    addEventListener: (name,f) => (listeners[name] ||= []).push(f),
    postMessage: data => events.push(data),
    performance: {now:()=>0},
  };
  context.window = context;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../desktop/browser_extension/douyin/media-main.js'), 'utf8'),context);
  return { context, events, video, timers, listeners, dispatch:name=>(listeners[name] || []).forEach(f=>f()) };
}
const init = () => new Uint8Array([0,0,0,16,102,116,121,112,105,115,111,109,0,0,0,0]);
function cssRoot(video) {
  const root={isConnected:true,css:true,contains:node=>node===video,classList:{contains:name=>name==='isCssFullScreen' && root.css}};
  video.closest=selector=>selector==='.isCssFullScreen' && root.css ? root : null;
  return root;
}

test('actual MediaSource bytes bind to active blob and video track only', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const audio = source.addSourceBuffer('audio/mp4; codecs="mp4a.40.2"');
  const video = source.addSourceBuffer('video/mp4; codecs="avc1.640028"');
  audio.appendBuffer(init()); video.appendBuffer(init());
  b.timers.forEach(f => f());
  const state = b.events.find(e => e.kind === 'state');
  const chunks = b.events.filter(e => e.kind === 'chunk');
  assert.ok(state?.state.generation);
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].generation, state.state.media_generation);
  assert.equal(state.state.media_available, true);
  assert.equal(state.state.width, 1080); assert.equal(state.state.height, 1920);
  assert.equal(state.state.media_time, 2);
});
test('append copies typed-array view and preserves original API', () => {
  const b=browser(), s=new b.context.MediaSource();
  b.video.currentSrc=b.context.URL.createObjectURL(s);
  const sb=s.addSourceBuffer('video/mp4');
  const bytes=new Uint8Array([99,...init(),88]); const view=bytes.subarray(1,bytes.length-1);
  sb.appendBuffer(view); bytes.fill(0);
  const chunk=b.events.find(e=>e.kind==='chunk');
  assert.deepEqual(Array.from(new Uint8Array(chunk.data)), Array.from(init()));
  assert.equal(sb.last, view);
  sb.fail=true;
  assert.throws(()=>sb.appendBuffer(init()), /append failed/);
  assert.equal(b.events.filter(e=>e.kind==='chunk').length,1);
});
test('reinitializing one SourceBuffer changes generation and resets sequence', () => {
  const b=browser(),s=new b.context.MediaSource(); b.video.currentSrc=b.context.URL.createObjectURL(s);
  const sb=s.addSourceBuffer('video/mp4'); sb.appendBuffer(init()); sb.appendBuffer(new Uint8Array([0,0,0,8,109,100,97,116])); sb.appendBuffer(init());
  const chunks=b.events.filter(e=>e.kind==='chunk');
  assert.notEqual(chunks[0].generation,chunks[2].generation);
  assert.deepEqual(chunks.map(e=>e.sequence),[0,1,0]);
});
test('unrelated blob stays controllable and disappearing video explicitly clears its target', () => {
  const b=browser(); b.video.currentSrc='blob:unrelated'; b.timers.forEach(f=>f());
  const control=b.events.find(e=>e.kind==='state')?.state;
  assert.equal(control?.media_available,false);
  assert.equal(control.media_generation,undefined);
  assert.equal(b.events.filter(e=>e.kind==='chunk').length,0);
  b.events.length=0; b.video.getBoundingClientRect=()=>({left:2000,top:0,right:3080,bottom:1920});
  b.timers.forEach(f=>f());
  const gone=b.events.find(e=>e.kind==='state')?.state;
  assert.equal(gone?.fullscreen,false); assert.equal(gone?.media_available,false);
  assert.deepEqual([gone.width,gone.height],[0,0]);
  assert.notEqual(gone.generation,control.generation);
});

test('diagnostics report a ready observer even without a mapped video track', () => {
  const b = browser();
  b.video.currentSrc = 'blob:private-source';
  b.timers.forEach(f => f());
  const stats = JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain || '{}');
  assert.equal(stats.stage, 'ready');
  assert.equal(stats.current, 'blob_unmapped');
  assert.equal(stats.ticks, 1);
  assert.equal(stats.appends, 0);
  assert.equal(stats.bytes, 0);
  assert.ok(!JSON.stringify(stats).includes('private-source'));
});

test('diagnostics count successful media calls without including payload or MIME text', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const sb = source.addSourceBuffer('video/mp4; codecs="private-codec"');
  source.addSourceBuffer('audio/mp4');
  b.context.URL.createObjectURL({});
  sb.appendBuffer(init());
  sb.fail = true;
  assert.throws(() => sb.appendBuffer(init()), /append failed/);
  b.timers.forEach(f => f());
  const stats = JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain || '{}');
  assert.equal(stats.current, 'blob_mapped');
  assert.equal(stats.mse_urls, 1);
  assert.equal(stats.other_urls, 1);
  assert.equal(stats.video_tracks, 1);
  assert.equal(stats.audio_tracks, 1);
  assert.equal(stats.appends, 1);
  assert.equal(stats.bytes, init().byteLength);
  assert.equal(stats.chunks, 1);
  assert.equal(stats.states, 1);
  assert.ok(!JSON.stringify(stats).includes('private-codec'));
});

test('ordinary Blob inputs are identified without inventing a MediaSource track', () => {
  const b = browser();
  b.video.currentSrc = b.context.URL.createObjectURL(new Blob([init()], {type:'video/mp4'}));
  b.timers.forEach(f => f());
  const stats = JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain);
  assert.equal(stats.blob_urls, 1);
  assert.equal(stats.mse_urls, 0);
  assert.equal(stats.current, 'blob_unmapped');
  assert.equal(stats.appends, 0);
  assert.equal(b.events.find(e => e.kind === 'state')?.state.media_available, false);
});

test('current srcObject diagnostics classify native types without reading frames or tracks', () => {
  const b = browser();
  for (const [object, type] of [[new b.context.MediaStream(),'media_stream'],[new b.context.MediaSourceHandle(),'media_source_handle'],[{},'other'],[null,'none']]) {
    b.video.srcObject = object;
    b.timers.forEach(f=>f());
    const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain);
    assert.equal(stats.src_object_type,type);
    assert.equal(stats.width,1080);
    assert.equal(stats.height,1920);
  }
  assert.equal(b.events.filter(e=>e.kind==='chunk').length,0);
  assert.ok(b.events.filter(e=>e.kind==='state').every(e=>e.state.media_available===false));
});

test('passive RTC observation preserves native return and binds current video tracks by identity only', () => {
  const b=browser(), pc=new b.context.RTCPeerConnection(), description={private:'SDP not for diagnostics'};
  const result=Promise.resolve(); pc.result=result;
  assert.equal(pc.setRemoteDescription(description,7),result);
  assert.equal(pc.arguments[0],description); assert.equal(pc.arguments[1],7);
  pc.setRemoteDescription(description);
  assert.equal(pc.listeners.length,1);
  assert.equal(pc.arguments[0],description);
  const track={kind:'video',get label(){throw new Error('must not read label');}}, receiver=new b.context.RTCRtpReceiver();
  pc.listeners[0].callback({track,receiver}); pc.listeners[0].callback({track,receiver});
  pc.listeners[0].callback({track:{kind:'audio'},receiver});
  const stream=new b.context.MediaStream(); stream.videoTracks=[track]; b.video.srcObject=stream;
  b.timers.forEach(f=>f());
  const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain);
  assert.equal(stats.current_rtc,true); assert.equal(stats.rtc_video_tracks,1); assert.equal(stats.rtc_transform_supported,true);
  assert.ok(!JSON.stringify(stats).includes('SDP'));
  assert.equal(b.events.at(-1).state.media_available,false);
  stream.videoTracks=[{kind:'video'}]; b.timers.forEach(f=>f());
  assert.equal(JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain).current_rtc,false);
});

test('RTC diagnostics never replace native failures or interrupt control heartbeats', () => {
  const b=browser(), pc=new b.context.RTCPeerConnection(), failure=new Error('native failure');
  pc.failure=failure; assert.throws(()=>pc.setRemoteDescription({}),error=>error===failure);
  const other=new b.context.RTCPeerConnection(); other.addEventListener=()=>{throw new Error('listener failed');};
  const expected=Promise.resolve(); other.result=expected; assert.equal(other.setRemoteDescription({}),expected);
  const stream=new b.context.MediaStream(); stream.getVideoTracks=()=>{throw new Error('private track failure');}; b.video.srcObject=stream;
  b.timers.forEach(f=>f());
  const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain);
  assert.equal(stats.current_rtc,false);
  assert.equal(b.events.at(-1).state.media_available,false);
  assert.ok(!JSON.stringify(stats).includes('private'));
});

test('control generation follows continuous video identity and never reuses a departed live target', () => {
  const b=browser(), stream=new b.context.MediaStream();
  b.video.srcObject=stream;
  b.timers.forEach(f=>f());
  const first=b.events.at(-1).state;
  assert.equal(first?.media_available,false);
  b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.generation,first.generation);
  b.video.srcObject=new b.context.MediaStream(); b.timers.forEach(f=>f());
  assert.notEqual(b.events.at(-1).state.generation,first.generation);
  b.video.srcObject=stream; b.timers.forEach(f=>f());
  assert.notEqual(b.events.at(-1).state.generation,first.generation);
  const returned=b.events.at(-1).state.generation;
  b.video.currentSrc='blob:changed'; b.timers.forEach(f=>f());
  assert.notEqual(b.events.at(-1).state.generation,returned);
});

test('visible live target without metadata sends a finite zero-size control heartbeat', () => {
  const b=browser(); b.video.srcObject=new b.context.MediaStream();
  b.video.videoWidth=0; b.video.videoHeight=0; b.video.currentTime=NaN; b.video.duration=Infinity;
  b.timers.forEach(f=>f());
  const current=b.events.find(e=>e.kind==='state')?.state;
  assert.equal(current?.media_available,false);
  assert.deepEqual([current.width,current.height,current.media_time,current.duration],[0,0,0,null]);
});

test('returning to cached MSE uses fresh operation generation and preserves the compressed cache key', () => {
  const b=browser(), source=new b.context.MediaSource();
  b.video.currentSrc=b.context.URL.createObjectURL(source);
  source.addSourceBuffer('video/mp4').appendBuffer(init()); b.timers.forEach(f=>f());
  const first=b.events.at(-1).state;
  b.video.srcObject=new b.context.MediaStream(); b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.media_available,false);
  assert.notEqual(b.events.at(-1).state.generation,first.generation);
  b.video.srcObject=null; b.timers.forEach(f=>f());
  const returned=b.events.at(-1).state;
  assert.equal(returned.media_available,true);
  assert.equal(returned.media_generation,first.media_generation);
  assert.notEqual(returned.generation,first.generation);
});

test('live controls reject pause and stale targets and stop the command batch after navigation', () => {
  const b=browser(); b.video.srcObject=new b.context.MediaStream(); b.timers.forEach(f=>f());
  const generation=b.events.at(-1).state?.generation;
  assert.ok(generation);
  let clicks=0;
  b.context.document.querySelector=selector=>selector.includes('video-switch-') ? {click:()=>{clicks++;b.video.srcObject=new b.context.MediaStream();}} : null;
  const commands=['pause_toggle','next','previous'].map(name=>({name,generation}));
  const send=()=>b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation,commands}});
  send();
  assert.equal(b.video.paused,false);
  assert.equal(clicks,1);
  send(); assert.equal(clicks,1);
});

test('current media dimensions reset when no video is visible', () => {
  const b = browser();
  b.timers.forEach(f=>f());
  b.video.getBoundingClientRect=()=>({left:2000,top:0,right:3080,bottom:1920});
  b.timers.forEach(f=>f());
  const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoMain);
  assert.equal(stats.current,'none');
  assert.equal(stats.src_object_type,'none');
  assert.equal(stats.width,0);
  assert.equal(stats.height,0);
});

test('exit command uses the webpage fullscreen control for Y mode', () => {
  const b=browser(), s=new b.context.MediaSource(); b.video.currentSrc=b.context.URL.createObjectURL(s);
  let pageClicks=0, nativeClicks=0;
  const root=cssRoot(b.video), player={querySelector:selector=>selector==='[data-e2e="xgplayer-page-full-screen"]' ? {click:()=>pageClicks++} : {textContent:'进入全屏H',click:()=>nativeClicks++}};
  b.video.closest=selector=>selector==='.isCssFullScreen' ? root : selector==='.xgplayer' ? player : null;
  s.addSourceBuffer('video/mp4').appendBuffer(init()); b.timers.forEach(f=>f());
  const generation=b.events.find(e=>e.kind==='state').state.generation;
  b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation,commands:[{name:'exit_fullscreen',generation}]}});
  assert.equal(pageClicks+nativeClicks,1);
  assert.equal(pageClicks,1); assert.equal(nativeClicks,0);
});

test('CSS ancestor fullscreen rotates operation identity while preserving the MSE cache', () => {
  const b=browser(), source=new b.context.MediaSource(); b.video.currentSrc=b.context.URL.createObjectURL(source);
  source.addSourceBuffer('video/mp4').appendBuffer(init()); b.timers.forEach(f=>f());
  const ordinary=b.events.at(-1).state;
  assert.equal(ordinary.fullscreen,false);
  const root=cssRoot(b.video); b.timers.forEach(f=>f());
  const entered=b.events.at(-1).state;
  assert.equal(entered.fullscreen,true); assert.notEqual(entered.generation,ordinary.generation);
  assert.equal(entered.media_generation,ordinary.media_generation);
  root.css=false; b.timers.forEach(f=>f());
  const exited=b.events.at(-1).state;
  assert.equal(exited.fullscreen,false); assert.notEqual(exited.generation,entered.generation);
  assert.equal(exited.media_generation,ordinary.media_generation);
  b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation:entered.generation,commands:[{name:'pause_toggle',generation:entered.generation}]}});
  assert.equal(b.video.paused,false);
});

test('native fullscreen includes the video itself and nested descendants only', () => {
  for(const direct of [false,true]) {
    const b=browser();
    const root=direct ? b.video : {contains:node=>node===b.video}; root.isConnected=true;
    b.context.document.fullscreenElement=root; b.dispatch('fullscreenchange');
    assert.equal(b.events.at(-1)?.state.fullscreen,true);
    b.context.document.fullscreenElement={isConnected:true,contains:()=>false}; b.dispatch('fullscreenchange');
    assert.equal(b.events.at(-1).state.fullscreen,false);
  }
});

test('the fullscreen video wins over a larger unrelated preview', () => {
  const b=browser(); cssRoot(b.video);
  b.video.getBoundingClientRect=()=>({left:0,top:0,right:400,bottom:400});
  const preview={...b.video,closest:()=>null,currentTime:7,getBoundingClientRect:()=>({left:0,top:0,right:1080,bottom:1920})};
  b.context.document.querySelectorAll=()=>[preview,b.video]; b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.media_time,2); assert.equal(b.events.at(-1).state.fullscreen,true);
});

test('native fullscreen excludes an unrelated CSS fullscreen preview underneath it', () => {
  const b=browser();
  b.video.getBoundingClientRect=()=>({left:0,top:0,right:400,bottom:400});
  const preview={...b.video,currentTime:7,getBoundingClientRect:()=>({left:0,top:0,right:1080,bottom:1920})}; cssRoot(preview);
  b.context.document.querySelectorAll=()=>[preview,b.video];
  b.context.document.fullscreenElement={isConnected:true,contains:node=>node===b.video}; b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.media_time,2); assert.equal(b.events.at(-1).state.fullscreen,true);
});

test('a bound fullscreen root keeps a zero-size waiting heartbeat across video replacement', () => {
  const b=browser(), root=cssRoot(b.video); b.timers.forEach(f=>f());
  const first=b.events.at(-1).state;
  const preview={...b.video,closest:()=>null,currentTime:7};
  b.context.document.querySelectorAll=()=>[preview]; b.timers.forEach(f=>f());
  const waiting=b.events.at(-1).state;
  assert.equal(waiting.fullscreen,true); assert.equal(waiting.media_available,false);
  assert.deepEqual([waiting.width,waiting.height,waiting.media_time,waiting.duration],[0,0,0,null]);
  assert.notEqual(waiting.generation,first.generation);
  b.timers.forEach(f=>f()); assert.equal(b.events.at(-1).state.generation,waiting.generation);
  root.css=false; b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.fullscreen,false);
  assert.equal(b.events.at(-1).state.media_time,7);
});

test('disconnected fullscreen roots and unrelated global markers never retain fullscreen', () => {
  const b=browser(), root=cssRoot(b.video); b.timers.forEach(f=>f());
  root.isConnected=false; b.context.document.querySelectorAll=()=>[];
  b.context.document.querySelector=()=>({isConnected:true}); b.timers.forEach(f=>f());
  assert.equal(b.events.at(-1).state.fullscreen,false);
  assert.equal(b.events.at(-1).state.media_available,false);
});

test('blur and hidden immediately invalidate fullscreen and focus cannot revive an old operation', () => {
  const b=browser(); cssRoot(b.video); b.timers.forEach(f=>f());
  const first=b.events.at(-1).state;
  b.context.document.hasFocus=()=>false; b.dispatch('blur');
  assert.equal(b.events.at(-1).state.fullscreen,false);
  b.context.document.hasFocus=()=>true; b.dispatch('focus');
  assert.equal(b.events.at(-1).state.fullscreen,true);
  assert.notEqual(b.events.at(-1).state.generation,first.generation);
  b.context.document.visibilityState='hidden'; b.dispatch('visibilitychange');
  assert.equal(b.events.at(-1).state.fullscreen,false);
  b.context.document.visibilityState='visible'; b.dispatch('visibilitychange');
  assert.equal(b.events.at(-1).state.fullscreen,true);
});

test('a bound native fullscreen root can still exit while its video is being replaced', () => {
  const b=browser(), root={isConnected:true,contains:node=>node===b.video};
  let exits=0; b.context.document.fullscreenElement=root;
  b.context.document.exitFullscreen=()=>{exits++; b.context.document.fullscreenElement=null; return Promise.resolve();};
  b.timers.forEach(f=>f()); b.context.document.querySelectorAll=()=>[]; b.timers.forEach(f=>f());
  const waiting=b.events.at(-1).state;
  assert.equal(waiting.fullscreen,true); assert.equal(waiting.media_available,false);
  b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation:waiting.generation,commands:[{name:'exit_fullscreen',generation:waiting.generation}]}});
  assert.equal(exits,1); b.dispatch('fullscreenchange');
  assert.equal(b.events.at(-1).state.fullscreen,false);
});

test('a bound CSS fullscreen root can exit without borrowing another player video', () => {
  const b=browser(), root=cssRoot(b.video); let clicks=0;
  root.querySelector=selector=>selector==='.xgplayer-fullscreen' ? {textContent:'退出全屏',click:()=>{clicks++;root.css=false;}} : null;
  b.timers.forEach(f=>f()); b.context.document.querySelectorAll=()=>[]; b.timers.forEach(f=>f());
  const waiting=b.events.at(-1).state;
  b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation:waiting.generation,commands:[{name:'exit_fullscreen',generation:waiting.generation}]}});
  assert.equal(clicks,1); b.timers.forEach(f=>f()); assert.equal(b.events.at(-1).state.fullscreen,false);
});
