const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const directory = path.join(__dirname, '../../desktop/browser_extension/douyin');
const flush = async () => { for (let i=0;i<4;i++) await new Promise(resolve=>setImmediate(resolve)); };
const source = (id='123') => ({kind:'vod',id,renditions:[{id:'1440',width:1440,height:2560,bitrate:1200000,codec:'hevc',urls:['https://v.example.douyinvod.com/video?token=private']} ]});
const metadata = (id='123') => ({aweme_detail:{aweme_id:id,video:{width:1440,height:2560,bit_rate:[
  {gear_name:'adapt_1080',bit_rate:900000,is_h265:0,play_addr:{width:1080,height:1920,url_list:['https://v.example.douyinvod.com/low']}},
  {gear_name:'adapt_1440',bit_rate:1200000,is_h265:1,play_addr:{width:1440,height:2560,url_list:['https://v.example.douyinvod.com/high?token=private']}}
]}}});

function browser(initialURL='https://www.douyin.com/video/123') {
  const events=[],timers=[],listeners={}; let uuid=0;
  class SourceBuffer {timestampOffset=0;appendBuffer(data){this.last=data;}}
  class MediaSource {addSourceBuffer(){return new SourceBuffer();}}
  class BrowserURL extends URL {static createObjectURL(){return `blob:${++uuid}`;}static revokeObjectURL(){}}
  const video={currentSrc:'https://v.example.douyinvod.com/current?__vid=123',videoWidth:1080,videoHeight:1920,currentTime:2,duration:10,paused:false,playbackRate:1,
    getBoundingClientRect:()=>({left:0,top:0,right:1080,bottom:1920}),closest:()=>null,pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}};
  const context={URL:BrowserURL,MediaSource,SourceBuffer,ArrayBuffer,Uint8Array,Blob,Response,TextDecoder,WeakMap,Map,
    location:new URL(initialURL),fetch:async()=>Response.json(metadata()),
    crypto:{randomUUID:()=>`id${++uuid}`},innerWidth:1080,innerHeight:1920,
    document:{querySelectorAll:selector=>selector==='video'?[video]:[],querySelector:()=>null,documentElement:{dataset:{}},hasFocus:()=>true,visibilityState:'visible',addEventListener:(name,fn)=>(listeners[name]||=[]).push(fn)},
    setInterval:fn=>timers.push(fn),addEventListener:(name,fn)=>(listeners[name]||=[]).push(fn),postMessage:value=>events.push(value)};
  context.window=context;
  vm.createContext(context);
  const helper=path.join(directory,'media-source.js');
  if(fs.existsSync(helper))vm.runInContext(fs.readFileSync(helper,'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(directory,'media-main.js'),'utf8'),context);
  const tick=()=>{timers.forEach(fn=>fn());return events.filter(event=>event.kind==='state').at(-1)?.state;};
  const observe=async()=>{await context.fetch('https://www.douyin.com/aweme/v1/web/aweme/detail/');await flush();return tick();};
  return {context,video,events,timers,listeners,tick,observe};
}

test('current video gets all real rendition sizes without changing the website player',async()=>{
  const b=browser(),old=b.video.currentSrc;
  const state=await b.observe();
  assert.equal(state.source?.id,'123');
  assert.deepEqual(Array.from(state.source.renditions,r=>[r.width,r.height]),[[1440,2560],[1080,1920]]);
  assert.equal(state.media_available,true);
  assert.equal(state.media_generation,undefined);
  assert.equal(b.video.currentSrc,old);
  assert.equal(b.video.currentTime,2);
  assert.ok(!b.context.document.documentElement.dataset.iw3VideoMain.includes('private'));
  assert.equal(b.tick().generation,state.generation,'heartbeat must retain source generation');
});

test('metadata for a different work never binds by feed ordering or dimensions',async()=>{
  const b=browser();b.video.currentSrc='blob:unknown';b.context.location=new URL('https://www.douyin.com/');
  await b.observe();assert.equal(b.tick().source,undefined);
  b.video.currentSrc='https://v.example.douyinvod.com/current?__vid=456';
  assert.equal(b.tick().source,undefined);
});

test('blob identity follows only the current video card committed awemeInfo and ignores nextVideoInfo',async()=>{
  const b=browser();await b.observe();b.video.currentSrc='blob:active';b.context.location=new URL('https://www.douyin.com/?recommend=1');
  const root={stateNode:{}};root.stateNode.current=root;
  const props={awemeInfo:{awemeId:'123'},nextVideoInfo:{awemeId:'456'}};
  const owner={memoizedProps:props,return:root};
  const fiber={memoizedProps:{},return:owner};
  const card={parentElement:null,querySelectorAll:()=>[b.video],__reactFiber$test:fiber};b.video.parentElement=card;
  assert.equal(b.tick().source?.id,'123');
  const second=metadata('456');b.context.IW3MediaSource.observeVod(second);
  owner.memoizedProps={awemeInfo:{awemeId:'456'},nextVideoInfo:{awemeId:'123'}};
  assert.equal(b.tick().source?.id,'456');
  card.querySelectorAll=()=>[b.video,{}];
  assert.equal(b.tick().source,undefined,'feed container with multiple videos is not a card identity');
});

test('uncommitted React props cannot revive a departed blob work',async()=>{
  const b=browser();await b.observe();b.context.IW3MediaSource.observeVod(metadata('456'));
  b.video.currentSrc='blob:active';b.context.location=new URL('https://www.douyin.com/');
  const oldRoot={stateNode:{}},currentRoot={stateNode:{}};oldRoot.stateNode.current=currentRoot;currentRoot.stateNode.current=currentRoot;
  const current={memoizedProps:{awemeInfo:{awemeId:'456'}},return:currentRoot};
  const stale={memoizedProps:{awemeInfo:{awemeId:'123'}},return:oldRoot,alternate:current};
  b.video.parentElement={parentElement:null,querySelectorAll:()=>[b.video],__reactFiber$test:stale};
  assert.equal(b.tick().source?.id,'456');
});

test('detail route alone cannot identify a replacement or unassociated blob',async()=>{
  const b=browser();await b.observe();b.video.currentSrc='blob:unmatched';
  assert.equal(b.tick().source,undefined);
});

test('changed rendition URLs invalidate pending frames while equal metadata keeps the operation',async()=>{
  const b=browser();const first=await b.observe();
  await b.observe();assert.equal(b.tick().generation,first.generation);
  b.context.IW3MediaSource.observeVod(metadata());
  const changed=metadata();changed.aweme_detail.video.bit_rate[1].play_addr.url_list[0]+='&new=1';
  b.context.IW3MediaSource.observeVod(changed);
  assert.notEqual(b.tick().generation,first.generation);
});

test('direct source stops duplicate MSE uploads and retains semantic pause',async()=>{
  const b=browser();await b.observe();
  const s=new b.context.MediaSource();b.video.currentSrc=b.context.URL.createObjectURL(s);
  b.video.parentElement={parentElement:null,querySelectorAll:()=>[b.video],__reactProps$test:{awemeInfo:{awemeId:'123'}}};
  const sb=s.addSourceBuffer('video/mp4');
  b.tick();b.events.length=0;
  const init=new Uint8Array([0,0,0,16,102,116,121,112,105,115,111,109,0,0,0,0]);sb.appendBuffer(init);
  const state=b.tick();assert.equal(state.source?.id,'123');
  assert.equal(b.events.filter(event=>event.kind==='chunk').length,0);
  b.context.commandForTest={channel:'iw3-browser-video-v1',kind:'commands',generation:state.generation,commands:[{name:'pause_toggle',generation:state.generation}]};
  b.context.listenersForTest=b.listeners.message;
  vm.runInContext('listenersForTest.forEach(fn=>fn({source:window,data:commandForTest}))',b.context);
  assert.equal(b.video.paused,true);
});

test('active descriptor A does not discard preloaded B MSE initialization',async()=>{
  const b=browser();await b.observe();
  assert.equal(b.tick().source?.id,'123');
  const next=new b.context.MediaSource(),url=b.context.URL.createObjectURL(next);
  const buffer=next.addSourceBuffer('video/mp4');
  const init=new Uint8Array([0,0,0,16,102,116,121,112,105,115,111,109,0,0,0,0]);
  buffer.appendBuffer(init);
  const first=b.events.filter(event=>event.kind==='chunk');
  assert.equal(first.length,1,'preloading another track must preserve its ftyp');
  assert.deepEqual(Array.from(new Uint8Array(first[0].data)),Array.from(init));
  b.video.currentSrc=url;
  const active=b.tick();
  assert.equal(active.source,undefined);
  assert.equal(active.media_available,true);
  assert.equal(active.media_generation,first[0].generation);
  buffer.appendBuffer(new Uint8Array([0,0,0,8,109,111,111,102]));
  assert.deepEqual(b.events.filter(event=>event.kind==='chunk').map(event=>event.sequence),[0,1]);
});

const liveFlight=(id='987')=>[1,'a:'+JSON.stringify({state:{roomStore:{roomInfo:{room:{id_str:id},
  web_stream_url:{flv_pull_url:{FULL_HD1:'https://pull.example.douyincdn.com/full.flv'}}}}}})+'\n'];

test('late old-room Flight append cannot bind a different SPA room path',()=>{
  const b=browser('https://live.douyin.com/456');b.video.currentSrc='';b.video.srcObject={};
  b.context.__pace_f=[liveFlight()];
  assert.equal(b.tick().source?.id,'987');
  b.context.location=new URL('https://live.douyin.com/999');
  assert.equal(b.tick().source,undefined);
  b.context.__pace_f.push(liveFlight());
  assert.equal(b.tick().source,undefined,'old room packet is not evidence for the new path');
  b.context.__pace_f=[liveFlight('654')];
  assert.equal(b.tick().source,undefined,'new array still needs proven route identity or a fresh document');
});

test('initial document path is captured before the first video observation',()=>{
  const b=browser('https://live.douyin.com/456');b.video.currentSrc='';b.video.srcObject={};
  b.context.__pace_f=[liveFlight()];
  b.context.location=new URL('https://live.douyin.com/999');
  assert.equal(b.tick().source,undefined,'initial old document state cannot bind a new route');
});

test('navigation without a visible video still invalidates the initial live document',()=>{
  const b=browser('https://live.douyin.com/456');b.video.currentSrc='';b.video.srcObject={};
  b.context.__pace_f=[liveFlight()];
  assert.equal(b.tick().source?.id,'987');
  const original=b.video.getBoundingClientRect;
  b.context.location=new URL('https://live.douyin.com/999');
  b.video.getBoundingClientRect=()=>({left:0,top:0,right:0,bottom:0});
  assert.equal(b.tick().source,undefined);
  b.context.location=new URL('https://live.douyin.com/456');
  b.video.getBoundingClientRect=original;
  assert.equal(b.tick().source,undefined,'an absent video must not hide the route departure');
});

test('another room in a later Flight packet cannot replace the initial room on the same path',()=>{
  const b=browser('https://live.douyin.com/456');b.video.currentSrc='';b.video.srcObject={};
  b.context.__pace_f=[liveFlight()];
  assert.equal(b.tick().source?.id,'987');
  b.context.__pace_f.push(liveFlight('654'));
  assert.equal(b.tick().source,undefined,'a different internal room identity must invalidate rather than replace');
  b.context.__pace_f.push(liveFlight());
  assert.equal(b.tick().source,undefined,'an ambiguous document needs a fresh identity boundary');
});

test('independent live room uses HEVC table and drops stale Flight state after navigation',()=>{
  const b=browser('https://live.douyin.com/456');
  b.video.currentSrc='';b.video.srcObject={};
  b.context.__pace_f=[[1,'a:'+JSON.stringify({state:{roomStore:{roomInfo:{room:{id_str:'987'},web_stream_url:{flv_pull_url:{FULL_HD1:'http://pull.example.douyincdn.com/full.flv?secret=1',SD1:'https://pull.example.douyincdn.com/sd.flv'}}}}}})+'\n']];
  const current=b.tick();assert.equal(current.source?.kind,'live');assert.equal(current.source.id,'987');
  assert.deepEqual(Array.from(current.source.renditions,r=>[r.id,r.width,r.height,r.codec]),[['FULL_HD1',0,0,'hevc'],['SD1',0,0,'unknown']]);
  assert.match(current.source.renditions[0].urls[0],/^https:/);
  b.context.location=new URL('https://live.douyin.com/999');
  assert.equal(b.tick().source,undefined);
  b.context.location=new URL('https://live.douyin.com/');b.tick();
  b.context.location=new URL('https://live.douyin.com/456');
  assert.equal(b.tick().source,undefined,'returning after leaving room cannot revive old Flight state');
});

test('descriptor validation rejects network escape, excessive input and hidden fields',()=>{
  const b=browser(),helper=b.context.IW3MediaSource;
  assert.ok(helper,'source descriptor helper must exist');
  const clean=helper.clean({...source(),cookie:'never forwarded'});
  assert.equal(clean.cookie,undefined);
  for(const url of ['http://v.example.douyinvod.com/file','https://127.0.0.1/file','https://v.example.douyinvod.com.evil.test/file','https://a:b@v.example.douyinvod.com/file','https://v.example.douyinvod.com/file#extra','https://douyinvod.com/file','https://v.example.douyinvod.com/'+ 'x'.repeat(4096)]){
    const invalid=source();invalid.renditions[0].urls=[url];assert.equal(helper.clean(invalid),null);
  }
  const invalid=source();invalid.renditions=Array(9).fill(invalid.renditions[0]);assert.equal(helper.clean(invalid),null);
  assert.equal(helper.clean({...source(),kind:'vod',renditions:[{...source().renditions[0],width:0}]}),null);
  assert.equal(helper.clean({...source(),kind:'live',renditions:[{...source().renditions[0],width:0}]}),null);
});

module.exports={source};
