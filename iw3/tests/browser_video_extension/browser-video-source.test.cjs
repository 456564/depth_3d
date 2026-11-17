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
  const video = {
    currentSrc: '', videoWidth: 1080, videoHeight: 1920, currentTime: 2,
    duration: 10, paused: false, playbackRate: 1,
    getBoundingClientRect: () => ({left:0,top:0,right:1080,bottom:1920,width:1080,height:1920}),
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
  };
  const context = {
    console, MediaSource, SourceBuffer, ArrayBuffer, Uint8Array,
    crypto: { randomUUID: () => `id${++uuid}` },
    URL: { createObjectURL: () => `blob:${++uuid}`, revokeObjectURL() {} },
    innerWidth: 1080, innerHeight: 1920,
    document: { querySelectorAll: s => s === 'video' ? [video] : [], documentElement: {dataset:{}}, fullscreenElement:null },
    setInterval: f => timers.push(f),
    addEventListener: (name,f) => (listeners[name] ||= []).push(f),
    postMessage: data => events.push(data),
    performance: {now:()=>0},
  };
  context.window = context;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../desktop/browser_extension/douyin/media-main.js'), 'utf8'),context);
  return { context, events, video, timers, listeners };
}
const init = () => new Uint8Array([0,0,0,16,102,116,121,112,105,115,111,109,0,0,0,0]);

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
  assert.equal(chunks[0].generation, state.state.generation);
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
test('unrelated blob or invisible video does not fabricate media state', () => {
  const b=browser(); b.video.currentSrc='blob:unrelated'; b.timers.forEach(f=>f());
  assert.equal(b.events.filter(e=>e.kind==='state').length,0);
});

test('exit command uses the webpage fullscreen control for Y mode', () => {
  const b=browser(), s=new b.context.MediaSource(); b.video.currentSrc=b.context.URL.createObjectURL(s);
  s.addSourceBuffer('video/mp4').appendBuffer(init()); b.timers.forEach(f=>f());
  const generation=b.events.find(e=>e.kind==='state').state.generation;
  let pageClicks=0, nativeClicks=0;
  b.context.document.querySelector=()=>({});
  b.video.closest=()=>({querySelector:selector=>selector==='[data-e2e="xgplayer-page-full-screen"]' ? {click:()=>pageClicks++} : {textContent:'进入全屏H',click:()=>nativeClicks++}});
  b.listeners.message[0]({source:vm.runInNewContext('window',b.context),data:{channel:'iw3-browser-video-v1',kind:'commands',generation,commands:[{name:'exit_fullscreen',generation}]}});
  assert.equal(pageClicks+nativeClicks,1);
  assert.equal(pageClicks,1); assert.equal(nativeClicks,0);
});
