const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const directory = path.join(__dirname, '../../desktop/browser_extension/douyin');
const extensionId = 'a'.repeat(32);
const popupSender = {id:extensionId, url:`chrome-extension://${extensionId}/popup.html`};
const pageSender = (id = 5, active = true) => ({id:extensionId,url:'https://www.douyin.com/',frameId:0,tab:{id,active}});
const state = () => ({session:'private-session',generation:'private-generation',serial:1,media_time:2,paused:false,playback_rate:1,duration:10,width:1080,height:1920});
const diagnostics = () => ({
  main:{version:'1.1.1',stage:'ready',ticks:1,mse_urls:1,blob_urls:0,other_urls:0,video_tracks:1,audio_tracks:1,other_tracks:0,appends:1,bytes:4,chunks:1,states:1,current:'blob_mapped'},
  relay:{version:'1.1.1',events:2,states:1,chunks:1,bytes:4},
  video:{kind:'blob',width:1080,height:1920,paused:false,ready_state:4},
  bridge:{state:'source',bytes:4,next_sequence:1},
});

function background(handler) {
  let listener, now = 100;
  const calls = [];
  const context = {
    URL, Uint8Array, ArrayBuffer, AbortSignal, Date:{now:()=>now},
    atob:s=>Buffer.from(s,'base64').toString('binary'),
    chrome:{runtime:{id:extensionId,getURL:p=>`chrome-extension://${extensionId}/${p}`,onMessage:{addListener:fn=>listener=fn}}},
    fetch:async (url, options) => {
      calls.push({url,options});
      if (handler) return handler(url,options);
      return {ok:true,status:200,json:async()=>url.endsWith('/hello')?{token:'private-token'}:{commands:[],next_sequence:1}};
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(directory,'media-background.js'),'utf8'),context);
  return {
    calls,
    send:(message,sender=pageSender())=>new Promise(resolve=>listener(message,sender,resolve)),
    advance:ms=>{now+=ms;},
  };
}

test('popup reports missing and stale page heartbeats without network requests', async () => {
  const b = background();
  const missing = await b.send({kind:'status'},popupSender);
  assert.equal(missing.ok,true);
  assert.equal(missing.page,null);
  assert.equal((await b.send({kind:'diagnostics',diagnostics:diagnostics()})).ok,true);
  const fresh = await b.send({kind:'status'},popupSender);
  assert.equal(fresh.page.fresh,true);
  assert.equal(fresh.page.source,false);
  assert.equal(fresh.page.diagnostics.main.current,'blob_mapped');
  b.advance(4000);
  const stale = await b.send({kind:'status'},popupSender);
  assert.equal(stale.page.fresh,false);
  assert.equal(stale.page.age_ms,4000);
  assert.equal(b.calls.length,0);
});

test('only this extension popup can query transport status', async () => {
  const b = background();
  for (const sender of [pageSender(),{...popupSender,id:'b'.repeat(32)},{...popupSender,url:'https://evil.example/'},{...popupSender,tab:{id:5}},{...popupSender,url:`chrome-extension://${extensionId}/other.html`}]) {
    const reply = await b.send({kind:'status'},sender);
    assert.equal(reply.ok,false);
    assert.equal(reply.transport,undefined);
  }
  assert.equal(b.calls.length,0);
});

test('live room loads both media scripts and can report only from its HTTPS top level', async () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(directory,'manifest.json'),'utf8'));
  for (const script of manifest.content_scripts) {
    assert.ok(script.matches.includes('https://live.douyin.com/*'));
  }
  const b = background();
  const live = {...pageSender(),url:'https://live.douyin.com/123'};
  assert.equal((await b.send({kind:'diagnostics',diagnostics:diagnostics()},live)).ok,true);
  assert.equal((await b.send({kind:'state',state:state(),focused:true},live)).ok,true);
  for (const url of ['http://live.douyin.com/123','https://evil.douyin.com/123','https://live.douyin.com.evil.example/123']) {
    assert.equal((await b.send({kind:'diagnostics',diagnostics:diagnostics()},{...live,url})).ok,false);
  }
  assert.equal((await b.send({kind:'diagnostics',diagnostics:diagnostics()},{...live,frameId:1})).ok,false);
});

test('diagnostics accept only top-level Douyin and bounded safe fields', async () => {
  const b = background();
  for (const sender of [{...pageSender(),frameId:1},{...pageSender(),url:'https://evil.example/'}]) {
    assert.equal((await b.send({kind:'diagnostics',diagnostics:diagnostics()},sender)).ok,false);
  }
  const bad = diagnostics(); bad.main.current = 'https://private.example/?token=secret';
  assert.equal((await b.send({kind:'diagnostics',diagnostics:bad})).ok,false);
  const huge = diagnostics(); huge.relay.bytes = Infinity;
  assert.equal((await b.send({kind:'diagnostics',diagnostics:huge})).ok,false);
  const extra = diagnostics(); extra.url = 'https://private.example/'; extra.main.token = 'private-token';
  assert.equal((await b.send({kind:'diagnostics',diagnostics:extra})).ok,true);
  const safe = JSON.stringify(await b.send({kind:'status'},popupSender));
  assert.ok(!safe.includes('private'));
});

test('new diagnostic fields are validated while old 1.1.1 reports stay compatible', async () => {
  const b=background();
  const old=diagnostics();
  assert.equal((await b.send({kind:'diagnostics',diagnostics:old})).ok,true);
  const current=diagnostics();
  Object.assign(current.main,{version:'1.1.2',src_object_type:'media_stream',width:540,height:960});
  Object.assign(current.relay,{version:'1.1.2',ticks:5,window_seen:9,rejected_source:1,rejected_channel:1,rejected_session:1,rejected_kind:1,rejected_state:1,rejected_chunk:1,
    report_sent:2,report_ok:1,report_rejected:0,report_pending:1,last_error:'none'});
  assert.equal((await b.send({kind:'diagnostics',diagnostics:current})).ok,true);
  const status=await b.send({kind:'status'},popupSender);
  assert.equal(status.page.diagnostics.main.src_object_type,'media_stream');
  assert.equal(status.page.diagnostics.main.width,540);
  assert.equal(status.page.diagnostics.relay.ticks,5);
  assert.equal(status.page.diagnostics.relay.report_pending,1);
  const bad=structuredClone(current); bad.main.src_object_type='private-type';
  assert.equal((await b.send({kind:'diagnostics',diagnostics:bad})).ok,false);
  const malformed=structuredClone(current); malformed.relay.report_pending=2;
  assert.equal((await b.send({kind:'diagnostics',diagnostics:malformed})).ok,false);
  const unsafe=structuredClone(current); unsafe.relay.last_error='private URL';
  assert.equal((await b.send({kind:'diagnostics',diagnostics:unsafe})).ok,false);
});

test('RTC diagnostics allow only safe counts and booleans alongside legacy reports', async () => {
  const b=background(), report=diagnostics();
  Object.assign(report.main,{current_rtc:true,rtc_video_tracks:2,rtc_transform_supported:true});
  assert.equal((await b.send({kind:'diagnostics',diagnostics:report})).ok,true);
  const clean=(await b.send({kind:'status'},popupSender)).page.diagnostics.main;
  assert.equal(clean.current_rtc,true); assert.equal(clean.rtc_video_tracks,2); assert.equal(clean.rtc_transform_supported,true);
  for(const extra of [{current_rtc:'private label'},{rtc_video_tracks:-1},{rtc_transform_supported:1}]) {
    const bad=diagnostics(); Object.assign(bad.main,extra);
    assert.equal((await b.send({kind:'diagnostics',diagnostics:bad})).ok,false);
  }
});

test('status counts successful compressed sends without exposing credentials or source identifiers', async () => {
  const b = background();
  await b.send({kind:'state',state:state(),focused:true});
  await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.transport.hello,1);
  assert.equal(status.transport.state,1);
  assert.equal(status.transport.chunk,1);
  assert.equal(status.transport.bytes,4);
  assert.equal(status.transport.http_status,200);
  assert.equal(status.transport.ok,true);
  assert.equal(status.playback.width,1080);
  assert.ok(!JSON.stringify(status).includes('private'));
});

test('control-only heartbeat is independently online and cannot retain old delivery', async () => {
  const b=background();
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  await b.send({kind:'state',state:{...state(),serial:2,media_available:false,width:0,height:0},focused:true});
  const status=await b.send({kind:'status'},popupSender);
  assert.equal(status.page.controllable,true);
  assert.equal(status.page.media_available,false);
  assert.equal(status.page.delivered,false);
  assert.equal(status.page.confirmed_bytes,0);
  const before=b.calls.length;
  assert.equal((await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AA=='})).ok,false);
  assert.equal(b.calls.length,before);
});

test('a new target must confirm its own control heartbeat before being reported online', async () => {
  let release, hold=false;
  const b=background(async url=>url.endsWith('/hello') ? {ok:true,status:200,json:async()=>({token:'local'})} : hold ?
    new Promise(resolve=>{release=resolve;}) : {ok:true,status:200,json:async()=>({next_sequence:0,commands:[]})});
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  assert.equal((await b.send({kind:'status'},popupSender)).page.controllable,true);
  hold=true;
  const pending=b.send({kind:'state',state:{...state(),serial:2,generation:'new-live',media_available:false},focused:true});
  await settle();
  assert.equal((await b.send({kind:'status'},popupSender)).page.controllable,false);
  release({ok:true,status:200,json:async()=>({next_sequence:0,commands:[]})}); await pending;
  assert.equal((await b.send({kind:'status'},popupSender)).page.controllable,true);
});

test('page markers and successful playback state never prove compressed media delivery', async () => {
  const b = background();
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.page.delivered,false);
  assert.equal(status.page.confirmed_bytes,0);
  assert.equal(status.page.confirmed_sequence,0);
});

test('changing generation clears the confirmed media delivery for the same tab', async () => {
  const b = background();
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  assert.equal((await b.send({kind:'status'},popupSender)).page.delivered,true);
  await b.send({kind:'state',state:{...state(),serial:2,generation:'next-generation'},focused:true});
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.page.delivered,false);
  assert.equal(status.page.confirmed_bytes,0);
});

test('an old asynchronous chunk response cannot confirm a newer generation', async () => {
  let release;
  const b = background(async url => {
    if (url.endsWith('/hello')) return {ok:true,status:200,json:async()=>({token:'private-token'})};
    if (url.endsWith('/chunk')) return new Promise(resolve=>{release=resolve;});
    return {ok:true,status:200,json:async()=>({commands:[],next_sequence:0})};
  });
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  const old = b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  await settle();
  await b.send({kind:'state',state:{...state(),serial:2,generation:'next-generation'},focused:true});
  release({ok:true,status:200,json:async()=>({commands:[],next_sequence:1})}); await old;
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.page.delivered,false);
  assert.equal(status.page.confirmed_bytes,0);
});

test('same-source bridge sequence reset clears old delivery until chunks are acknowledged again', async () => {
  let sequence=0;
  const b = background(async url => {
    if (url.endsWith('/hello')) return {ok:true,status:200,json:async()=>({token:'private-token'})};
    if (url.endsWith('/chunk')) sequence++;
    return {ok:true,status:200,json:async()=>({commands:[],next_sequence:sequence})};
  });
  await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:state(),focused:true});
  await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  assert.equal((await b.send({kind:'status'},popupSender)).page.delivered,true);
  sequence=0;
  await b.send({kind:'state',state:{...state(),serial:2},focused:true});
  assert.equal((await b.send({kind:'status'},popupSender)).page.delivered,false);
  await b.send({kind:'chunk',session:'private-session',generation:'private-generation',sequence:0,data:'AAEC/w=='});
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.page.delivered,true);
  assert.equal(status.page.confirmed_bytes,4);
});

test('transport errors expose a safe category and never raw fetch text', async () => {
  const b = background(async()=>{throw new Error('private signed URL and token');});
  await b.send({kind:'state',state:state(),focused:true});
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.transport.ok,false);
  assert.equal(status.transport.error,'network_or_response');
  assert.equal(status.transport.stage,'hello');
  assert.ok(!JSON.stringify(status).includes('private'));
});

test('source lease keeps popup on the actual source tab and bounded reports are evicted', async () => {
  const b = background();
  await b.send({kind:'diagnostics',diagnostics:diagnostics()},pageSender(5));
  await b.send({kind:'state',state:state(),focused:true},pageSender(5));
  const other = diagnostics(); other.video.width = 640;
  await b.send({kind:'diagnostics',diagnostics:other},pageSender(6,false));
  const source = await b.send({kind:'status'},popupSender);
  assert.equal(source.page.diagnostics.video.width,1080);
  assert.equal(source.page.source,true);
  b.advance(4000);
  for (let id=10;id<20;id++) await b.send({kind:'diagnostics',diagnostics:other},pageSender(id,false));
  const evicted = await b.send({kind:'status'},popupSender);
  assert.equal(evicted.page.diagnostics.video.width,640);
  assert.equal(evicted.page.source,false);
  assert.equal(evicted.pages,8);
});

test('expired source lease does not hide a new active page without a video track', async () => {
  const b = background();
  await b.send({kind:'diagnostics',diagnostics:diagnostics()},pageSender(5));
  await b.send({kind:'state',state:state(),focused:true},pageSender(5));
  const waiting = diagnostics(); waiting.video = {kind:'none',width:0,height:0,paused:true,ready_state:0}; waiting.main=null;
  await b.send({kind:'diagnostics',diagnostics:waiting},pageSender(6));
  b.advance(1100);
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.page.diagnostics.video.kind,'none');
  assert.equal(status.page.source,false);
});

test('a pending request preserves the last completed success for its normal TTL', async () => {
  let hold=false, release;
  const b = background(async url => {
    if (url.endsWith('/hello')) return {ok:true,status:200,json:async()=>({token:'private-token'})};
    if (hold) return new Promise(resolve=>{release=resolve;});
    return {ok:true,status:200,json:async()=>({commands:[],next_sequence:0})};
  });
  await b.send({kind:'state',state:state(),focused:true});
  hold=true;
  const pending = b.send({kind:'state',state:{...state(),serial:2},focused:true});
  await settle();
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.transport.ok,true);
  release({ok:true,status:200,json:async()=>({commands:[],next_sequence:0})});
  await pending;
});

test('a pending retry retains the last completed error until a response completes', async () => {
  let retry=false, release;
  const b = background(async url => {
    if (retry) return new Promise(resolve=>{release=resolve;});
    throw new Error('private network failure');
  });
  await b.send({kind:'state',state:state(),focused:true});
  retry=true;
  const pending = b.send({kind:'state',state:{...state(),serial:2},focused:true});
  await settle();
  const status = await b.send({kind:'status'},popupSender);
  assert.equal(status.transport.ok,false);
  assert.equal(status.transport.error,'network_or_response');
  release({ok:false,status:503,json:async()=>({})}); await pending;
});

test('an authentication rejection stays visible during automatic pairing retry', async () => {
  let hellos=0, release;
  const b = background(async url => {
    if (url.endsWith('/hello')) {
      hellos++;
      if (hellos>1) return new Promise(resolve=>{release=resolve;});
      return {ok:true,status:200,json:async()=>({token:'private-token'})};
    }
    return {ok:false,status:403,json:async()=>({})};
  });
  const pending=b.send({kind:'state',state:state(),focused:true});
  await settle();
  const status=await b.send({kind:'status'},popupSender);
  assert.equal(status.transport.ok,false);
  assert.equal(status.transport.error,'http');
  release({ok:false,status:503,json:async()=>({})}); await pending;
});

test('manifest opens a local status popup without adding permissions', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(directory,'manifest.json'),'utf8'));
  assert.equal(manifest.version,'1.2.0');
  assert.equal(manifest.action?.default_popup,'popup.html');
  assert.equal(manifest.permissions,undefined);
  assert.deepEqual(manifest.host_permissions,['http://127.0.0.1/*']);
});

function popup(reply) {
  const elements = new Map(), timers = [], sent = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id,{textContent:'',dataset:{}});
    return elements.get(id);
  };
  const context = {
    document:{getElementById:element},
    chrome:{runtime:{getManifest:()=>({version:'1.1.1'}),sendMessage:async message=>{sent.push(message);return reply;}}},
    setInterval:fn=>timers.push(fn),
  };
  vm.runInNewContext(fs.readFileSync(path.join(directory,'popup.js'),'utf8'),context);
  return {elements,timers,sent,element};
}

const settle = () => new Promise(resolve => setImmediate(resolve));
const connectedStatus = () => {
  const data=diagnostics();
  Object.assign(data.main,{version:'1.1.2',width:1080,height:1920,src_object_type:'none'});
  return {ok:true,page:{fresh:true,source:true,delivered:true,confirmed_bytes:4,confirmed_sequence:1,age_ms:150,diagnostics:data},transport:{stage:'chunk',http_status:200,error:null,ok:true,hello:1,state:3,chunk:1,bytes:4,age_ms:100},playback:null};
};

test('popup renders live source statistics with text nodes and polls only status', async () => {
  const b = popup(connectedStatus());
  await settle();
  assert.equal(b.element('summary').textContent,'真实视频字节已送达电脑');
  assert.equal(b.element('size').textContent,'1080 × 1920');
  assert.equal(b.element('sent').textContent,'4 B');
  b.timers[0](); await settle();
  assert.deepEqual(b.sent.map(message=>message.kind),['status','status']);
  assert.equal(b.element('summary').dataset.state,'ok');
});

test('popup clearly marks missing, stale and failed status and never injects page HTML', async () => {
  const missing = connectedStatus(); missing.page = null;
  const a = popup(missing); await settle();
  assert.equal(a.element('summary').textContent,'未收到抖音页面心跳');
  assert.match(a.element('hint').textContent,/刷新抖音/);
  const stale = connectedStatus(); stale.page.fresh=false; stale.page.age_ms=4000;
  const b = popup(stale); await settle();
  assert.equal(b.element('summary').textContent,'页面心跳已过期');
  assert.match(b.element('hint').textContent,/历史统计/);
  const failed = connectedStatus(); failed.transport.ok=false; failed.transport.error='http'; failed.transport.http_status=403;
  const c = popup(failed); await settle();
  assert.equal(c.element('connection').textContent,'HTTP 拒绝');
  assert.match(c.element('request').textContent,/403/);
  const text = fs.readFileSync(path.join(directory,'popup.js'),'utf8');
  assert.ok(!text.includes('innerHTML'));
});

test('another tab historical sends never prove this page delivered video', async () => {
  const other = connectedStatus(); other.page.source=false; other.page.delivered=false;
  const b = popup(other); await settle();
  assert.notEqual(b.element('summary').textContent,'真实视频字节已送达电脑');
  assert.equal(b.element('summary').dataset.state,'waiting');
});

test('popup delivery summary follows trusted background confirmation rather than page bridge markers', async () => {
  const forged = connectedStatus(); forged.page.delivered=false;
  const a = popup(forged); await settle();
  assert.notEqual(a.element('summary').textContent,'真实视频字节已送达电脑');
  const confirmed = connectedStatus(); confirmed.page.diagnostics.bridge.state='waiting';
  const b = popup(confirmed); await settle();
  assert.equal(b.element('summary').textContent,'真实视频字节已送达电脑');
  assert.equal(b.element('sequence').textContent,'1 · 4 B');
});

test('popup distinguishes online page control from unavailable live media', async () => {
  const status=connectedStatus();
  Object.assign(status.page,{controllable:true,media_available:false,delivered:false,confirmed_bytes:0,confirmed_sequence:0});
  const b=popup(status); await settle();
  assert.equal(b.element('summary').textContent,'页面控制在线，当前视频未提供可解码输入');
  assert.match(b.element('hint').textContent,/网页.*控件/);
  assert.equal(b.element('summary').dataset.state,'waiting');
  const offline=connectedStatus(); Object.assign(offline.page,{controllable:false,media_available:false,delivered:false});
  const c=popup(offline); await settle();
  assert.notEqual(c.element('summary').textContent,'页面控制在线，当前视频未提供可解码输入');
});

test('popup describes RTC association as observed metadata rather than decoded media', async () => {
  const status=connectedStatus();
  Object.assign(status.page,{controllable:true,media_available:false,delivered:false});
  Object.assign(status.page.diagnostics.main,{current:'srcObject',src_object_type:'media_stream',current_rtc:true,rtc_video_tracks:1,rtc_transform_supported:true});
  const b=popup(status); await settle();
  assert.match(b.element('input').textContent,/RTC 已关联/);
  assert.match(b.element('input').textContent,/视频轨道 1/);
  status.page.diagnostics.main.current_rtc=false;
  const c=popup(status); await settle();
  assert.match(c.element('input').textContent,/RTC 尚未观察/);
  assert.doesNotMatch(c.element('summary').textContent,/字节已送达/);
});

test('popup distinguishes current MAIN media from stale relay video snapshots', async () => {
  const status=connectedStatus();
  Object.assign(status.page.diagnostics.main,{version:'1.1.2',current:'none',src_object_type:'none',width:0,height:0});
  const b=popup(status); await settle();
  assert.equal(b.element('size').textContent,'尚无当前视频');
  assert.match(b.element('relay-video').textContent,/历史\/不一致快照.*1080 × 1920/);
  status.page.fresh=false;
  const stale=popup(status); await settle();
  assert.match(stale.element('input').textContent,/历史/);
  Object.assign(status.page.diagnostics.main,{current:'srcObject',src_object_type:'media_stream',width:540,height:960});
  status.page.fresh=true;
  const live=popup(status); await settle();
  assert.match(live.element('input').textContent,/MediaStream/);
  assert.equal(live.element('size').textContent,'540 × 960');
});

test('background exposes only the current fullscreen request and its own acknowledgement', async () => {
  const b=background(); await b.send({kind:'diagnostics',diagnostics:diagnostics()});
  await b.send({kind:'state',state:{...state(),fullscreen:true},focused:true});
  const full=await b.send({kind:'status'},popupSender);
  assert.equal(full.page.fullscreen,true); assert.equal(full.playback.fullscreen,true);
  assert.equal(full.page.controllable,true);
  b.advance(1100);
  const stale=await b.send({kind:'status'},popupSender);
  assert.equal(stale.page.fullscreen,null); assert.equal(stale.page.source,false);
});

test('popup reports fullscreen request delivery without claiming the runtime switched', async () => {
  const status=connectedStatus(); Object.assign(status.page,{fullscreen:true,controllable:true});
  const full=popup(status); await settle();
  assert.match(full.element('fullscreen').textContent,/全屏/);
  assert.match(full.element('route-request').textContent,/本机桥.*视频流/);
  assert.match(full.element('route-request').textContent,/实际来源.*iw3/);
  status.page.fullscreen=false;
  const desktop=popup(status); await settle();
  assert.match(desktop.element('fullscreen').textContent,/非全屏/);
  assert.match(desktop.element('route-request').textContent,/请求桌面/);
  status.page.controllable=false;
  const pending=popup(status); await settle();
  assert.match(pending.element('route-request').textContent,/等待.*确认/);
  status.page.source=false; status.page.fullscreen=null;
  const missing=popup(status); await settle();
  assert.match(missing.element('fullscreen').textContent,/等待/);
});
