const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const directory = path.join(__dirname, '../../desktop/browser_extension/douyin');
const channel = 'iw3-browser-video-v1';
const tick = () => new Promise(resolve => setImmediate(resolve));
const state = (serial=1, generation='g') => ({session:'s',generation,serial,media_time:2,paused:false,playback_rate:1,duration:10,width:1080,height:1920});

function relay(sendMessage = async message => ({ok:true,commands:[],next_sequence:message.kind==='chunk'?message.sequence+1:0,need_media:true})) {
  const calls=[], listeners={}, timers=[], posted=[];
  let now=0;
  const context={
    console, ArrayBuffer, Uint8Array, Map, Date:{now:()=>now},
    btoa:s=>Buffer.from(s,'binary').toString('base64'),
    document:{documentElement:{dataset:{}},hasFocus:()=>true,querySelectorAll:()=>[]},
    chrome:{runtime:{sendMessage:async message=>{calls.push(message);return sendMessage(message);}}},
    setInterval:f=>timers.push(f),
    addEventListener:(name,fn)=>(listeners[name] ||= []).push(fn),
    postMessage:message=>posted.push(message),
  };
  context.window=context;
  const file=path.join(directory,'media-relay.js');
  assert.ok(fs.existsSync(file),'media relay missing');
  vm.runInNewContext(fs.readFileSync(file,'utf8'),context);
  context.listenersForTest=listeners.message||[];
  const emit=data=>{
    context.messageForTest={channel,session:'s',...data};
    vm.runInContext('listenersForTest.forEach(fn=>fn({source:window,data:messageForTest}))',context);
  };
  return {context,calls,posted,timers,emit,advance:ms=>{now+=ms;timers.forEach(f=>f());}};
}

function background(handler = async (url, options) => ({status:200,ok:true,json:async()=>url.endsWith('/hello')?{token:'local-token'}:{commands:[],next_sequence:options.body==='bytes'?1:0,need_media:true}})) {
  let listener, now=0;
  const calls=[];
  const context={
    console, URL, Uint8Array, ArrayBuffer, AbortSignal,
    atob:s=>Buffer.from(s,'base64').toString('binary'), Date:{now:()=>now},
    chrome:{runtime:{id:'a'.repeat(32),getURL:p=>`chrome-extension://${'a'.repeat(32)}/${p}`,onMessage:{addListener:fn=>listener=fn}}},
    fetch:async(url,options)=>{calls.push({url,options});return handler(url,options);},
  };
  const file=path.join(directory,'media-background.js');
  assert.ok(fs.existsSync(file),'media background missing');
  vm.runInNewContext(fs.readFileSync(file,'utf8'),context);
  const send=(message,sender={url:'https://www.douyin.com/?recommend=1',frameId:0,tab:{id:5,active:true}})=>new Promise(resolve=>listener(message,sender,resolve));
  return {calls,send,advance:ms=>{now+=ms;}};
}

test('relay forwards actual compressed bytes through JSON runtime boundary after state',async()=>{
  const b=relay();
  b.emit({kind:'chunk',generation:'g',sequence:0,data:new Uint8Array([0,1,2,255]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  assert.equal(b.calls[0].kind,'state');
  assert.equal(b.calls[1].kind,'chunk');
  assert.deepEqual(Array.from(Buffer.from(b.calls[1].data,'base64')),[0,1,2,255]);
  assert.equal(b.calls[1].generation,'g');
  assert.equal(b.calls[1].sequence,0);
});

test('relay has a loading heartbeat before the first media event', () => {
  const b = relay();
  b.advance(250);
  const stats = JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay || '{}');
  assert.equal(stats.version, '1.1.2');
  assert.equal(stats.events, 0);
  assert.equal(stats.chunks, 0);
  assert.equal(b.calls.length, 0);
});

test('relay diagnostics count only accepted chunk bytes', async () => {
  const b = relay();
  b.emit({kind:'chunk',generation:'g',sequence:0,data:new Uint8Array([1,2,3]).buffer});
  b.emit({kind:'chunk',generation:'g',sequence:0,data:new Uint8Array([1,2,3]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  b.advance(250);
  const stats = JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay || '{}');
  assert.equal(stats.events, 3);
  assert.equal(stats.states, 1);
  assert.equal(stats.chunks, 1);
  assert.equal(stats.bytes, 3);
});

test('relay separates timer heartbeat and each window-message rejection layer', async () => {
  const b=relay();
  b.context.messageForTest={channel,session:'s',kind:'state',state:state()};
  vm.runInContext('listenersForTest.forEach(fn=>fn({source:{},data:messageForTest}))',b.context);
  b.emit({channel:'foreign'});
  b.emit({session:''});
  b.emit({kind:'unknown'});
  b.emit({kind:'state',state:{...state(),paused:'false'}});
  b.emit({kind:'chunk',generation:'g',sequence:1,data:new Uint8Array([1]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  b.advance(250);
  const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.ticks,1);
  assert.equal(stats.window_seen,7);
  assert.equal(stats.rejected_source,1);
  assert.equal(stats.rejected_channel,1);
  assert.equal(stats.rejected_session,1);
  assert.equal(stats.rejected_kind,1);
  assert.equal(stats.rejected_state,1);
  assert.equal(stats.rejected_chunk,1);
  assert.equal(stats.states,1);
});

test('relay exposes pending, accepted, rejected and failed diagnostic reports without raw errors', async () => {
  let release, mode='pending';
  const b=relay(message=>{
    if (message.kind!=='diagnostics') return Promise.resolve({ok:true,next_sequence:0});
    if(mode==='pending') return new Promise(resolve=>{release=resolve;});
    if(mode==='rejected') return Promise.resolve({ok:false,message:'private rejection text'});
    return Promise.reject(new Error('private URL and token'));
  });
  b.advance(1000); await tick();
  b.advance(250);
  let stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.report_sent,1);
  assert.equal(stats.report_pending,1);
  assert.equal(stats.report_ok,0);
  release({ok:true}); await tick();
  stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.report_pending,0);
  assert.equal(stats.report_ok,1);
  mode='rejected'; b.advance(1000); await tick();
  stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.report_rejected,1);
  assert.equal(stats.last_error,'report_rejected');
  mode='failed'; b.advance(1000); await tick();
  stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.report_rejected,2);
  assert.equal(stats.last_error,'runtime_error');
  assert.ok(!JSON.stringify(stats).includes('private'));
});

test('a diagnostic report that never returns remains pending while timer heartbeat advances', async () => {
  const b=relay(()=>new Promise(()=>{}));
  b.advance(1000); await tick();
  b.advance(5000); await tick();
  const stats=JSON.parse(b.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.ticks,2);
  assert.equal(stats.report_pending,1);
  assert.equal(stats.report_sent,1);
  assert.equal(stats.report_ok,0);
  assert.equal(stats.report_rejected,0);
  assert.equal(b.calls.length,1);
});

test('diagnostics distinguish construction failure and synchronous runtime exceptions', async () => {
  const broken=relay();
  broken.context.document.querySelectorAll=()=>{throw new Error('private page detail');};
  broken.advance(1000); await tick();
  let stats=JSON.parse(broken.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.last_error,'diagnostic_build_error');
  assert.equal(stats.report_sent,0);
  assert.equal(stats.report_pending,0);
  const sync=relay();
  sync.context.chrome.runtime.sendMessage=()=>{throw new Error('private context detail');};
  sync.advance(1000); await tick();
  stats=JSON.parse(sync.context.document.documentElement.dataset.iw3VideoRelay);
  assert.equal(stats.last_error,'runtime_sync_error');
  assert.equal(stats.report_sent,1);
  assert.equal(stats.report_pending,0);
  assert.ok(!JSON.stringify(stats).includes('private'));
});

test('relay reports script diagnostics before media without exposing marker URLs', async () => {
  const b = relay();
  b.context.document.documentElement.dataset.iw3VideoMain = '{"stage":"ready"}';
  b.context.document.documentElement.dataset.iw3VideoBridge = '{"state":"waiting","bytes":0,"next_sequence":0,"generation":"private"}';
  b.advance(1000);
  await tick();
  const report = b.calls.find(message => message.kind === 'diagnostics');
  assert.equal(report.diagnostics.main.stage, 'ready');
  assert.equal(report.diagnostics.video.kind, 'none');
  assert.equal(report.diagnostics.bridge.state, 'waiting');
  assert.ok(!JSON.stringify(report).includes('private'));
  assert.equal(b.calls.filter(message => message.kind !== 'diagnostics').length, 0);
});

test('relay diagnostics tolerate document_start without an element and malformed markers', async () => {
  const b = relay();
  b.context.document.documentElement = null;
  b.advance(1000);
  await tick();
  assert.equal(b.calls.at(-1).diagnostics.main, null);
  b.context.document.documentElement = {dataset: {iw3VideoMain: '{not json'}};
  b.advance(1000);
  await tick();
  assert.equal(b.calls.at(-1).diagnostics.main, null);
});

test('actual relay startup diagnostics reach background before any media is available', async () => {
  const worker = background();
  const b = relay(message => worker.send(message));
  b.advance(1000);
  await tick();
  const status = await worker.send({kind:'status'}, {id:'a'.repeat(32),url:`chrome-extension://${'a'.repeat(32)}/popup.html`});
  assert.equal(status.page?.fresh, true);
  assert.equal(status.page.diagnostics.main, null);
  assert.equal(status.page.diagnostics.video.kind, 'none');
  assert.equal(status.page.diagnostics.bridge.state, 'waiting');
  assert.equal(worker.calls.length, 0);
});

test('page-writable bridge marker cannot replace relay acknowledgement statistics', async () => {
  const b = relay();
  b.context.document.documentElement.dataset.iw3VideoBridge = '{"state":"source","bytes":999,"next_sequence":5}';
  b.advance(1000);
  await tick();
  const report = b.calls.find(message => message.kind === 'diagnostics');
  assert.equal(report.diagnostics.bridge.state, 'waiting');
  assert.equal(report.diagnostics.bridge.bytes, 0);
  assert.equal(report.diagnostics.bridge.next_sequence, 0);
});

test('relay ignores forged source, unknown channel and invalid metadata',async()=>{
  const b=relay();
  b.emit({kind:'state',state:state(1,'')});
  b.emit({channel:'unknown',kind:'state',state:state()});
  b.emit({session:'other',kind:'state',state:state()});
  b.emit({kind:'state',state:{...state(),paused:'false'}});
  await tick();
  assert.equal(b.calls.length,0);
});

test('relay keeps state heartbeat ahead of pending chunks and gates commands by generation',async()=>{
  let release;
  const b=relay(message=>message.kind==='chunk'?new Promise(resolve=>release=resolve):Promise.resolve({ok:true,next_sequence:0,commands:[{name:'pause_toggle',generation:message.state.generation},{name:'next',generation:'old'}]}));
  b.emit({kind:'chunk',generation:'g',sequence:0,data:new Uint8Array([1]).buffer});
  b.emit({kind:'chunk',generation:'g',sequence:1,data:new Uint8Array([2]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  b.emit({kind:'state',state:state(2)});
  release({ok:true,next_sequence:1});
  await tick();
  assert.equal(b.calls[2].kind,'state');
  assert.ok(b.posted.every(message=>message.commands.every(command=>command.generation==='g')));
  release?.({ok:true,next_sequence:2});
});

test('relay retransmits cached init when PC bridge resets its sequence',async()=>{
  let next=0;
  const b=relay(async message=>{
    if(message.kind==='chunk'){next=message.sequence+1;return {ok:true,next_sequence:next};}
    return {ok:true,next_sequence:next,commands:[]};
  });
  b.emit({kind:'chunk',generation:'g',sequence:0,data:new Uint8Array([7,8]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  next=0;
  b.emit({kind:'state',state:state(2)});
  await tick();
  assert.equal(b.calls.filter(message=>message.kind==='chunk').length,2);
});

test('relay does not fabricate missing init or retry endlessly on a stopped bridge',async()=>{
  const b=relay(async()=>({ok:false,message:'本机视频桥暂不可用'}));
  b.emit({kind:'chunk',generation:'g',sequence:1,data:new Uint8Array([7]).buffer});
  b.emit({kind:'state',state:state()});
  await tick();
  assert.equal(b.calls.length,1);
  assert.match(b.context.document.documentElement.dataset.iw3VideoBridge,/waiting/);
  b.advance(2000);
  assert.equal(b.calls.filter(message => message.kind !== 'diagnostics').length,1);
});

test('background rejects non-Douyin and subframe senders before any network',async()=>{
  const b=background();
  for(const sender of [{url:'https://example.com',frameId:0,tab:{id:5}}, {url:'https://www.douyin.com/',frameId:1,tab:{id:5}}, {url:'https://www.douyin.com.evil/',frameId:0,tab:{id:5}}]){
    assert.equal((await b.send({kind:'state',state:state()},sender)).ok,false);
  }
  assert.equal(b.calls.length,0);
});

test('background uses only fixed loopback paths and never returns the pairing token',async()=>{
  const b=background();
  const result=await b.send({kind:'state',state:state(),focused:true,url:'https://evil/'});
  assert.equal(result.ok,true);
  assert.deepEqual(b.calls.map(call=>call.url),['http://127.0.0.1:52302/hello','http://127.0.0.1:52302/state']);
  assert.equal(b.calls[1].options.headers['X-IW3-Token'],'local-token');
  assert.equal(JSON.stringify(result).includes('local-token'),false);
  assert.equal((await b.send({kind:'fetch',url:'https://evil/'})).ok,false);
  assert.equal(b.calls.length,2);
});

test('background converts only bounded valid base64 and sends binary body',async()=>{
  const b=background();
  await b.send({kind:'state',state:state(),focused:true});
  const result=await b.send({kind:'chunk',session:'s',generation:'g',sequence:0,data:'AAEC/w=='});
  assert.equal(result.ok,true);
  const call=b.calls.at(-1);
  assert.ok(call.url.endsWith('/chunk'));
  assert.deepEqual(Array.from(call.options.body),[0,1,2,255]);
  assert.equal(call.options.headers['X-IW3-Generation'],'g');
  const before=b.calls.length;
  for(const data of ['!!!','AA','=AAA']) assert.equal((await b.send({kind:'chunk',session:'s',generation:'g',sequence:0,data})).ok,false);
  assert.equal(b.calls.length,before);
});

test('background isolates active tab lease and requires current generation for chunks',async()=>{
  const b=background();
  await b.send({kind:'state',state:state(),focused:true});
  const sender={url:'https://www.douyin.com/',frameId:0,tab:{id:6,active:false}};
  assert.equal((await b.send({kind:'state',state:state(1,'other'),focused:false},sender)).ok,false);
  assert.equal((await b.send({kind:'chunk',session:'s',generation:'old',sequence:0,data:'AA=='})).ok,false);
  b.advance(1100);
  assert.equal((await b.send({kind:'state',state:state(1,'other'),focused:false},sender)).ok,true);
});

test('background re-pairs once after 403 but does not expose raw fetch errors',async()=>{
  let rejected=false;
  const b=background(async url=>{
    if(url.endsWith('/hello')) return {ok:true,status:200,json:async()=>({token:'secret'})};
    if(!rejected){rejected=true;return {ok:false,status:403,json:async()=>({})};}
    return {ok:true,status:200,json:async()=>({commands:[],next_sequence:0})};
  });
  assert.equal((await b.send({kind:'state',state:state(),focused:true})).ok,true);
  assert.equal(b.calls.filter(call=>call.url.endsWith('/hello')).length,2);
  const bad=background(async()=>{throw new Error('sensitive signed url and token');});
  const result=await bad.send({kind:'state',state:state(),focused:true});
  assert.equal(result.ok,false);
  assert.equal(JSON.stringify(result).includes('sensitive'),false);
});
