const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function browser() {
  const events = [], timers = [];
  let uuid = 0;
  class SourceBuffer {
    timestampOffset = 0;
    appendBuffer() {}
    changeType() {}
  }
  class MediaSource {
    sourceBuffers = [];
    addSourceBuffer() {
      const source = new SourceBuffer();
      this.sourceBuffers.push(source);
      return source;
    }
    removeSourceBuffer(source) {
      this.sourceBuffers.splice(this.sourceBuffers.indexOf(source), 1);
    }
  }
  const video = {
    currentSrc: '', videoWidth: 1080, videoHeight: 1920, currentTime: 2,
    duration: 10, paused: false, playbackRate: 1,
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 1080, bottom: 1920 }),
  };
  const context = {
    MediaSource, SourceBuffer, ArrayBuffer, Uint8Array,
    crypto: { randomUUID: () => `id${++uuid}` },
    URL: { createObjectURL: () => `blob:${++uuid}`, revokeObjectURL() {} },
    innerWidth: 1080, innerHeight: 1920,
    document: { querySelectorAll: () => [video] },
    setInterval: callback => timers.push(callback), addEventListener() {},
    postMessage: data => events.push(data),
  };
  context.window = context;
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, '../../desktop/browser_extension/douyin/media-main.js'), 'utf8'), context,
  );
  return { context, events, video, timers };
}

const init = () => new Uint8Array([0, 0, 0, 16, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 0, 0]);

test('removed SourceBuffer never remains the current video track', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const first = source.addSourceBuffer('video/mp4');
  first.appendBuffer(init());
  source.removeSourceBuffer(first);
  const second = source.addSourceBuffer('video/mp4');
  second.appendBuffer(init());
  b.timers.forEach(callback => callback());
  const chunks = b.events.filter(event => event.kind === 'chunk');
  const current = b.events.find(event => event.kind === 'state');
  assert.equal(current?.state.generation, chunks.at(-1).generation);
});

test('accepted append larger than transport chunk limit loses no compressed bytes', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const buffer = source.addSourceBuffer('video/mp4');
  const large = new Uint8Array(9 * 1024 * 1024);
  large.set(init()); large[large.length - 1] = 77;
  buffer.appendBuffer(large);
  const chunks = b.events.filter(event => event.kind === 'chunk');
  assert.equal(chunks.reduce((total, event) => total + event.data.byteLength, 0), large.byteLength);
  assert.ok(chunks.every(event => event.data.byteLength <= 8 * 1024 * 1024));
  assert.equal(new Uint8Array(chunks.at(-1).data).at(-1), 77);
});

test('revoking an already attached media blob preserves video source identity', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const buffer = source.addSourceBuffer('video/mp4');
  buffer.appendBuffer(init());
  b.context.URL.revokeObjectURL(b.video.currentSrc);
  b.timers.forEach(callback => callback());
  const chunk = b.events.find(event => event.kind === 'chunk');
  const state = b.events.find(event => event.kind === 'state');
  assert.equal(state?.state.generation, chunk.generation);
});

test('negative timestamp offset adjusts media time and duration onto the same raw timeline', () => {
  const b = browser(), source = new b.context.MediaSource();
  b.video.currentSrc = b.context.URL.createObjectURL(source);
  const buffer = source.addSourceBuffer('video/mp4');
  buffer.timestampOffset = -5;
  buffer.appendBuffer(init());
  b.video.currentTime = 8;
  b.video.duration = 10;
  b.timers.forEach(callback => callback());
  const state = b.events.find(event => event.kind === 'state').state;
  assert.equal(state.media_time, 13);
  assert.equal(state.duration, 15);
});
