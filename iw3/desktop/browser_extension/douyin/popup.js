(() => {
  'use strict';
  const text = (id, value) => { document.getElementById(id).textContent = value; };
  const bytes = value => value >= 1048576 ? `${(value / 1048576).toFixed(2)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} B`;
  const sourceNames = {none:'未找到视频',empty:'等待视频源',url:'网址媒体',blob:'blob 媒体',blob_mapped:'已关联的 MSE 压缩流',blob_unmapped:'blob 尚未关联',srcObject:'srcObject 输入'};
  const objectNames = {none:'无对象',media_stream:'MediaStream',media_source_handle:'MediaSourceHandle',other:'其他对象'};
  const reportErrors = {none:'无',report_rejected:'后台拒绝诊断报告',runtime_error:'扩展消息调用失败',runtime_sync_error:'扩展消息同步异常',diagnostic_build_error:'诊断报告构建失败'};
  const stages = {idle:'尚无请求',hello:'配对',state:'播放状态',chunk:'压缩片段'};
  const errors = {http:'HTTP 拒绝',timeout:'请求超时',network_or_response:'连接或响应失败'};
  let reading = false;
  text('version', chrome.runtime.getManifest().version);
  async function refresh() {
    if (reading) return;
    reading = true;
    try {
      const status = await chrome.runtime.sendMessage({kind:'status'});
      if (!status?.ok) throw new Error('Status unavailable');
      const page = status.page, data = page?.diagnostics;
      const main = data?.main, relay = data?.relay, video = data?.video;
      const transport = status.transport;
      const history = page && !page.fresh ? '历史：' : '';
      const videoMismatch = main && (main.width !== undefined ? main.width !== video?.width || main.height !== video?.height : main.current === 'none' && video?.width > 0);
      const delivered = page?.fresh && page.delivered === true;
      const controlOnly = page?.fresh && page.controllable && page.media_available === false;
      const rtc = main?.src_object_type === 'media_stream' && main.current_rtc !== undefined ? ` · RTC ${main.current_rtc ? '已关联' : '尚未观察'} / 视频轨道 ${main.rtc_video_tracks} / 编码变换接口${main.rtc_transform_supported ? '可用' : '未检出'}` : '';
      text('summary', !page ? '未收到抖音页面心跳' : !page.fresh ? '页面心跳已过期' : delivered ? '真实视频字节已送达电脑' : controlOnly ? '页面控制在线，当前视频未提供可解码输入' : '等待可用视频源或本机连接');
      document.getElementById('summary').dataset.state = delivered ? 'ok' : 'waiting';
      text('hint', !page ? '请重载本扩展，然后刷新抖音并播放普通视频。' : !page.fresh ? '以下为历史统计；诊断心跳过期不等于视频断流。' : controlOnly ? '此输入尚未取源；切条需网页本身提供上一条／下一条控件。' : !main ? '转发脚本已加载，尚未收到媒体观察脚本状态。' : main.current === 'blob_unmapped' ? '播放器的 blob 尚未关联到可观察的压缩输入。' : delivered ? '取源成功；继续在 iw3 检查硬解和头显串流。' : 'iw3 需选择 browser_video 来源并开始串流。');
      text('heartbeat', page ? `${page.fresh ? '在线' : '已过期'} · ${(page.age_ms / 1000).toFixed(1)} 秒前` : '未收到');
      text('main', main ? `${history}${main.stage === 'ready' ? '已加载' : '启动中'} v${main.version} · 心跳 ${main.ticks}` : '未收到');
      text('relay', relay ? `${history}心跳 ${relay.ticks ?? '旧版未报告'} · 状态 ${relay.states} / 片段 ${relay.chunks}` : '未收到');
      text('input', history + (main ? sourceNames[main.current] + (main.current === 'srcObject' ? ` · ${objectNames[main.src_object_type] || '旧版未报告类型'}${rtc}` : '') : 'MAIN 未报告当前输入'));
      text('size', history + (main?.width > 0 ? `${main.width} × ${main.height}` : main?.current === 'none' || main?.width === 0 ? '尚无当前视频' : video?.width ? `转发快照：${video.width} × ${video.height}` : '尚无当前尺寸'));
      text('relay-video', video?.width ? `${history || videoMismatch ? '历史/不一致快照：' : '转发快照：'}${sourceNames[video.kind]} · ${video.width} × ${video.height}` : `${history}转发层未找到视频`);
      text('playback', video?.width ? `${history || videoMismatch ? '快照：' : ''}${video.paused ? '暂停' : '播放'} · 就绪 ${video.ready_state}/4` : '等待视频');
      text('objects', main ? `MSE ${main.mse_urls} / Blob ${main.blob_urls} / 其他 ${main.other_urls}` : '等待');
      text('tracks', main ? `视频 ${main.video_tracks} / 音频 ${main.audio_tracks} / 其他 ${main.other_tracks}；追加 ${main.appends}` : '等待');
      text('observed', main ? `${bytes(main.bytes)} · ${main.chunks} 个片段` : '等待');
      text('cached', relay ? bytes(relay.bytes) : '等待');
      text('window-events', relay?.window_seen === undefined ? '旧版未报告' : `监听 ${relay.window_seen} / 入口通过 ${relay.events}；源过滤 ${relay.rejected_source}`);
      text('window-rejected', relay?.ticks === undefined ? '旧版未报告' : `协议 ${relay.rejected_channel} / 会话 ${relay.rejected_session} / 未知 ${relay.rejected_kind} / 状态 ${relay.rejected_state} / 片段 ${relay.rejected_chunk}`);
      text('diagnostic-reports', relay?.report_sent === undefined ? '旧版未报告' : `尝试 ${relay.report_sent} / 成功 ${relay.report_ok} / 失败 ${relay.report_rejected} / 发送时待回 ${relay.report_pending}`);
      text('diagnostic-error', relay?.last_error === undefined ? '旧版未报告' : reportErrors[relay.last_error]);
      text('connection', transport.ok ? `${transport.age_ms <= 3000 ? '最近请求成功' : '历史请求成功'} · ${(transport.age_ms / 1000).toFixed(1)} 秒前` : transport.error ? errors[transport.error] : '未测试（尚无媒体状态）');
      text('request', `${stages[transport.stage]}${transport.http_status === null ? '' : ` · HTTP ${transport.http_status}`}`);
      text('requests', `配对 ${transport.hello} / 状态 ${transport.state} / 片段 ${transport.chunk}`);
      text('sent', bytes(transport.bytes));
      text('sequence', page ? `${page.confirmed_sequence} · ${bytes(page.confirmed_bytes)}` : '等待');
    } catch {
      text('summary', '扩展状态暂不可用');
      text('hint', '请关闭并重新打开面板；如刚重载扩展，请刷新抖音。');
      document.getElementById('summary').dataset.state = 'waiting';
    } finally { reading = false; }
  }
  void refresh();
  setInterval(() => { void refresh(); }, 1000);
})();
