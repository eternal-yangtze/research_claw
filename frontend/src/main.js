import './style.css';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { api, post, uploadFile } from './api';

const $ = id => document.getElementById(id);
const icons = {
  plus: '<path d="M12 5v14M5 12h14"/>', chat: '<path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-3 3V11.5A8.5 8.5 0 0 1 9.5 3h3a8.5 8.5 0 0 1 8.5 8.5Z"/>',
  book: '<path d="M4 4h6a3 3 0 0 1 3 3v14a4 4 0 0 0-4-3H3V4Zm9 3a3 3 0 0 1 3-3h5v14h-4a4 4 0 0 0-4 3"/>',
  arrow: '<path d="m6 12 6-6 6 6M12 6v14"/>', clip: '<path d="m8 13 7-7a3 3 0 0 1 4 4L9 20a5 5 0 0 1-7-7L13 2"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>', close: '<path d="m6 6 12 12M18 6 6 18"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>', check: '<path d="m5 12 4 4L19 6"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${icons[name] || icons.chat}</svg>`;
const brand = '<span class="brand-mark">R<span>c</span></span>';
const state = { user: null, project: 'default', session: null, sessions: [], messages: [], library: null, actions: [], run: null, submitting: false, uploading: false, refreshBusy: false, epoch: 0, sources: [], mode: 'login' };
const scope = () => ({ project_id: state.project });
const query = extra => new URLSearchParams({ ...scope(), ...extra }).toString();
function el(tag, className, text) { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
function button(label, handler, className = '') { const b = el('button', className, label); b.type = 'button'; b.onclick = handler; return b; }
function notice(error) { const n = $('notice'); n.textContent = error.message || String(error); n.hidden = false; clearTimeout(notice.timer); notice.timer = setTimeout(() => { n.hidden = true; }, 9000); }
function markdown(text) {
  const node = el('div', 'markdown'); node.innerHTML = DOMPurify.sanitize(marked.parse(text || '', { breaks: true }), { FORBID_TAGS: ['img', 'style', 'input', 'form'], FORBID_ATTR: ['style'] });
  for (const link of node.querySelectorAll('a')) { link.target = '_blank'; link.rel = 'noopener noreferrer'; }
  return node;
}
function storageKey() { return `researchclaw-chat:${state.user}:${state.project}`; }
function saveSelection() { localStorage.setItem(storageKey(), state.session || ''); }

$('app').innerHTML = `
<section id="auth-screen" class="auth-screen" hidden>
  <div class="auth-story"><div class="brand">${brand}<span>ResearchClaw</span></div><div><span class="eyebrow">A SPACE FOR YOUR RESEARCH</span><h1>让一个想法，<br>慢慢成为一篇论文。</h1><p>读懂文献，追问证据，校准方向。<br>你的研究始终由你掌握。</p><div class="story-lines"><i></i><i></i><i></i><span>从提问开始 · 在证据中前进</span></div></div><small>为硕士科研而设计的个人研究助手</small></div>
  <div class="auth-form-wrap"><form id="login-form"><span class="eyebrow">YOUR RESEARCH DESK</span><h2 id="auth-title">欢迎回来</h2><p id="auth-description">登录后，继续你的研究对话。</p><label>用户 ID<input id="user" autocomplete="username" pattern="[a-zA-Z0-9_-]{1,64}" placeholder="例如 researcher" required maxlength="64"></label><label>个人密码<input id="password" type="password" autocomplete="current-password" placeholder="输入你的个人密码" required maxlength="256"></label><label id="invite-label" hidden>本机邀请码<input id="invite" type="password" autocomplete="off" placeholder=".env.web 中的访问密码"></label><p id="auth-error" class="error" role="alert"></p><button id="login-submit" class="primary" type="submit">登录工作区 ${icon('chevron')}</button><div class="auth-switch"><button type="button" id="show-register">创建账号</button><button type="button" id="show-claim">认领旧会话账号</button><button type="button" id="show-login" hidden>返回登录</button></div><small class="muted">账号在本机保存。邀请码仅用于注册或认领旧 ID；登录使用个人密码。</small></form></div>
</section>
<div id="shell" class="shell" hidden>
  <aside id="sidebar" class="sidebar"><div class="brand">${brand}<span>ResearchClaw<small>个人科研空间</small></span><button id="close-sidebar" class="icon-button mobile-only" aria-label="收起侧边栏">${icon('close')}</button></div><button id="new-session" class="new-chat">${icon('plus')}<span>开启新对话</span><kbd>新</kbd></button><div class="history-label"><span>研究对话</span><span id="session-count"></span></div><input id="session-search" class="session-search" placeholder="搜索对话标题" aria-label="搜索对话"><div id="session-list" class="session-list"></div><div class="sidebar-bottom"><button id="open-library" class="library-button">${icon('book')}<span>论文知识库<small id="sidebar-count">正在读取…</small></span><span>↗</span></button><div class="profile"><span id="avatar" class="avatar"></span><span><b id="user-label"></b><small>单会话记忆 · 共享论文知识</small></span><button id="logout" title="退出登录">退出</button></div></div></aside>
  <main class="chat-main"><header class="chat-header"><div class="header-title"><button id="toggle-sidebar" class="icon-button" aria-label="切换侧边栏">${icon('menu')}</button><div><h1 id="session-title">新对话</h1><span class="header-subtitle"><i></i> 科研 Agent · 基于论文证据</span></div></div><div class="header-actions"><button id="session-settings">会话设置</button><button id="header-library">${icon('book')}<span>论文与进度</span><span id="job-dot" hidden></span></button></div></header>
  <div id="conversation-scroll" class="conversation-scroll"><section id="welcome" class="welcome"><div class="welcome-symbol">${brand}<span class="spark">✦</span></div><span class="eyebrow">THINK DEEPER, WITH EVIDENCE</span><h2>今天，想研究什么？</h2><p>从一个疑问、一个尚不成熟的 idea，<br class="mobile-only">或一篇值得读的论文开始。</p><div class="suggestions"><button data-prompt="我的方向是时间序列数据压缩与误差控制。先判断当前论文库能支持哪些研究问题，列出相关文献与缺口；不要开始下载或实验。"><span>↗</span><b>梳理研究方向</b><small>看看已有文献，找到值得追问的问题</small></button><button data-prompt="我想用自适应分段的分段线性近似压缩时间序列，并保证逐点最大绝对误差。请基于知识库分析已有方法重合之处，区分有依据的结论与待验证创新点。"><span>✧</span><b>推敲一个 idea</b><small>比较已有工作，识别假设与创新空间</small></button><button data-prompt="请解释时间序列压缩中逐点最大绝对误差和均方误差的区别，检索论文证据，用初学者能理解的例子回答，并给出原文位置。"><span>⌕</span><b>问一个文献问题</b><small>自然语言提问，沿着引用回到原文</small></button><button data-prompt="针对具有逐点绝对误差保证的时间序列压缩，帮我拟定硕士可完成的实验方案：基线、数据、误差设置、指标和消融。先给方案供我修改，不执行实验。"><span>⌘</span><b>设计一次实验</b><small>讨论基线、指标与可执行的小规模验证</small></button></div><div class="welcome-foot"><span class="tiny-line"></span>无需固定步骤，随时追问或调整方向<span class="tiny-line"></span></div></section><div id="messages" class="messages" aria-live="polite"></div><div id="run-status" class="run-status" hidden></div><div id="action-cards" class="action-cards"></div></div>
  <div class="composer-zone"><div id="upload-strip" class="upload-strip" hidden></div><div id="active-job" class="active-job" hidden></div><form id="composer" class="composer"><textarea id="message-input" rows="1" maxlength="100000" placeholder="描述你的研究问题，或继续追问…" aria-label="消息输入"></textarea><div class="composer-bottom"><div><button id="attach" type="button" class="icon-button" title="上传论文 PDF">${icon('clip')}</button><span class="composer-hint">论文问答 · Idea 分析 · 实验方案</span></div><button id="stop-run" type="button" hidden>停止</button><button id="send" class="send-button" aria-label="发送消息" type="submit">${icon('arrow')}</button></div></form><p class="composer-disclaimer">结论请结合原文核验；创新点是基于当前文献的判断。下载、入库与实验执行均需确认。</p></div>
  </main>
  <aside id="paper-panel" class="paper-panel" hidden><header><div><span class="eyebrow">YOUR PAPER LIBRARY</span><h2>论文与知识库</h2></div><button id="close-library" class="icon-button" aria-label="关闭论文面板">${icon('close')}</button></header><div class="paper-panel-body"><div id="library-stats" class="library-stats"></div><div id="paper-drop" class="paper-drop" tabindex="0" role="button" aria-label="选择或拖入论文 PDF">${icon('clip')}<b>拖入论文，或点击上传</b><small>PDF · 每篇最多 20 MiB · 支持批量与哈希去重</small></div><p class="panel-note">上传后先暂存。核对文件，再确认增量入库；完成后，同项目新会话也能检索。</p><div class="section-heading"><h3>已上传文件</h3><button id="ingest" class="primary small">确认入库</button></div><div id="documents"></div><div class="section-heading"><h3>构建任务</h3><button id="refresh-library" class="text-button">刷新</button></div><div id="jobs"></div><details><summary>当前可检索论文</summary><input id="paper-filter" placeholder="按标题筛选论文" aria-label="筛选论文"><div id="papers"></div></details></div></aside>
</div><input id="file-input" type="file" multiple accept=".pdf,application/pdf" hidden><div id="notice" class="notice" role="status" hidden></div><dialog id="modal"><div class="modal-header"><h2 id="modal-title"></h2><button id="close-modal" class="icon-button" aria-label="关闭弹窗">${icon('close')}</button></div><div id="modal-body"></div><div id="modal-footer" class="modal-footer"></div></dialog>`;

function authMode(mode) {
  state.mode = mode; $('invite-label').hidden = mode === 'login'; $('invite').required = mode !== 'login';
  $('show-login').hidden = mode === 'login'; $('show-register').hidden = mode !== 'login'; $('show-claim').hidden = mode !== 'login';
  $('auth-title').textContent = { login: '欢迎回来', register: '创建你的科研空间', claim: '找回原有研究对话' }[mode];
  $('auth-description').textContent = mode === 'login' ? '登录后，继续你的研究对话。' : mode === 'claim' ? '输入原有用户 ID，使用本机邀请码设置个人密码。' : '使用本机邀请码注册，设置独立的个人密码（至少 10 位）。';
  $('login-submit').textContent = mode === 'login' ? '登录工作区 →' : mode === 'claim' ? '认领并登录 →' : '注册并登录 →';
  $('password').minLength = mode === 'login' ? 1 : 10; $('auth-error').textContent = '';
}
$('user').pattern = '[a-zA-Z0-9_@.\\-]{1,64}';
$('user').placeholder = '例如 demo2@2026';
$('user').title = '1–64 位英文字母、数字、_、-、@ 或点号；不能是单独的 . 或 ..';
$('show-register').onclick = () => authMode('register'); $('show-claim').onclick = () => authMode('claim'); $('show-login').onclick = () => authMode('login');
$('login-form').onsubmit = async event => {
  event.preventDefault(); $('login-submit').disabled = true; $('auth-error').textContent = '';
  try {
    const body = { user_id: $('user').value.trim(), password: $('password').value };
    if (state.mode !== 'login') await post(state.mode === 'claim' ? '/auth/claim-history' : '/auth/register', { ...body, invite_code: $('invite').value });
    await post('/auth/login', body); $('password').value = $('invite').value = ''; await boot(body.user_id);
  } catch (e) { if (state.user) notice(e); else $('auth-error').textContent = e.status === 401 ? '用户 ID 或个人密码不正确。首次使用请创建账号；已有历史 ID 请先认领。' : e.message; }
  finally { $('login-submit').disabled = false; }
};

async function boot(user) {
  state.user = user; $('auth-screen').hidden = true; $('shell').hidden = false;
  $('user-label').textContent = user; $('avatar').textContent = user.slice(0, 1).toUpperCase();
  await loadSessions(); const saved = localStorage.getItem(storageKey());
  const first = state.sessions.find(s => s.id === saved && !s.archived) || state.sessions.find(s => !s.archived);
  if (first) await selectSession(first.id); else await newSession();
}
async function loadSessions() { state.sessions = await api('/sessions?' + query()); renderSessions(); }
function renderSessions() {
  const list = $('session-list'); list.replaceChildren(); const term = $('session-search').value.toLowerCase();
  const sessions = state.sessions.filter(s => !s.archived && s.title.toLowerCase().includes(term));
  $('session-count').textContent = state.sessions.filter(s => !s.archived).length;
  for (const s of sessions) {
    const row = button('', () => selectSession(s.id).catch(notice), 'session-item' + (state.session === s.id ? ' selected' : ''));
    row.innerHTML = icon('chat'); const content = el('span'); content.append(el('b', '', s.title || '新对话'), el('small', '', new Date(s.created * 1000).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }))); row.append(content); list.append(row);
  }
  if (!sessions.length) list.append(el('p', 'empty-small', term ? '没有匹配的对话' : '你的研究对话会保存在这里'));
}
async function selectSession(id) {
  const epoch = ++state.epoch; state.session = id; state.sources = []; state.run = null; state.actions = [];
  $('action-cards').replaceChildren(); $('run-status').hidden = true; $('ingest').disabled = true;
  busy(); renderSessions(); saveSelection();
  const detail = await api(`/sessions/${id}?` + query()); if (epoch !== state.epoch) return;
  state.messages = detail.messages; state.detail = detail; $('session-title').textContent = detail.title;
  renderMessages(); $('sidebar').classList.remove('mobile-open'); await refresh();
  const debug = await api('/debug?' + query({ session_id: id })); if (epoch !== state.epoch) return;
  const latest = debug.model_runs[0]; if (latest?.status === 'running') { state.run = latest.id; watchRun(latest.id, id); }
  else if (latest && ['interrupted', 'failed'].includes(latest.status)) showRecovery(latest.id, id);
  else if (latest?.status === 'completed') await loadSources(latest.id, id);
}
async function newSession() {
  const result = await post('/sessions', { ...scope(), title: '新对话' }); await loadSessions(); await selectSession(result.id); $('message-input').focus();
}
$('new-session').onclick = () => newSession().catch(notice); $('session-search').oninput = renderSessions;

function renderMessages() {
  $('welcome').hidden = state.messages.some(m => ['user', 'assistant'].includes(m.role) && m.content);
  const list = $('messages'); list.replaceChildren();
  for (const m of state.messages) {
    if (!['user', 'assistant'].includes(m.role) || !m.content) continue;
    const article = el('article', 'message ' + m.role); const avatar = el('div', 'message-avatar');
    if (m.role === 'assistant') avatar.innerHTML = brand; else avatar.textContent = state.user.slice(0, 1).toUpperCase();
    const body = el('div', 'message-body'); body.append(el('span', 'message-author', m.role === 'assistant' ? 'ResearchClaw' : '你'), markdown(String(m.content)));
    if (m.role === 'assistant') body.append(button('复制', () => navigator.clipboard.writeText(m.content).then(() => notice('已复制')).catch(notice), 'copy-button'));
    article.append(avatar, body); list.append(article);
  }
  scrollBottom();
}
function scrollBottom() { requestAnimationFrame(() => { $('conversation-scroll').scrollTop = $('conversation-scroll').scrollHeight; }); }
function busy() { $('send').disabled = !!state.run || state.submitting || !state.session; $('stop-run').hidden = !state.run; }
$('message-input').oninput = () => { $('message-input').style.height = 'auto'; $('message-input').style.height = Math.min(180, $('message-input').scrollHeight) + 'px'; };
$('message-input').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } };
for (const b of document.querySelectorAll('[data-prompt]')) b.onclick = () => { $('message-input').value = b.dataset.prompt; $('message-input').dispatchEvent(new Event('input')); $('message-input').focus(); };
$('composer').onsubmit = async event => {
  event.preventDefault(); const text = $('message-input').value.trim(); if (!text || state.run || state.submitting) return;
  const sid = state.session; state.submitting = true; busy();
  try {
    const result = await post('/chat', { ...scope(), session_id: sid, message: text });
    if (state.session === sid) { $('message-input').value = ''; $('message-input').dispatchEvent(new Event('input')); state.messages.push({ role: 'user', content: text }); renderMessages(); state.run = result.run_id; watchRun(result.run_id, sid); }
  } catch (e) { notice(e); } finally { state.submitting = false; busy(); }
};
const watchers = new Map();
async function watchRun(rid, sid) {
  if (watchers.has(rid)) return; watchers.set(rid, true); busy();
  let cycles = 0;
  try {
    while (state.user && state.session === sid && state.run === rid) {
      const result = await api(`/runs/${rid}`);
      if (result.status !== 'running') {
        state.run = null; const detail = await api(`/sessions/${sid}?` + query()); if (state.session !== sid) break;
        state.messages = detail.messages; state.detail = detail; renderMessages(); $('run-status').hidden = true;
        await refresh(); await loadSources(rid, sid);
        if (result.status !== 'completed') {
          if (result.error) notice(result.error);
          if (result.error?.startsWith('论文下载失败')) { $('run-status').hidden = false; $('run-status').replaceChildren(el('span', '', '下载没有完成。可更换公开链接重新提出下载，或手动上传 PDF；不会自动重放。')); }
          else showRecovery(rid, sid);
        }
        else if (detail.title === '新对话') suggestTitle(sid, true).catch(() => {});
        break;
      }
      $('run-status').hidden = false; $('run-status').replaceChildren(el('span', 'spinner'), el('span', '', '正在理解问题与检查证据…'));
      if (cycles++ % 3 === 0) {
        try { const graph = await api(`/runs/${rid}/graph?` + query()); if (state.session !== sid) break;
          const nodes = graph.graph.nodes || []; $('run-status').append(el('small', '', `${graph.model_calls} 次推理 · ${graph.tool_calls} 次工具调用${nodes.length ? ' · ' + nodes.filter(n => n.status === 'accepted').length + '/' + nodes.length + ' 项已核验' : ''}`));
        } catch { /* Journal not yet created; the run status remains authoritative. */ }
      }
      await new Promise(resolve => setTimeout(resolve, 1400));
    }
  } catch (e) {
    if (state.session === sid) { notice(e); $('run-status').hidden = false; $('run-status').replaceChildren(el('span', '', '连接中断。已发送的问题仍保存在会话中。'), button('重新连接', () => { watchers.delete(rid); watchRun(rid, sid); }, 'text-button')); }
  } finally { watchers.delete(rid); busy(); }
}
function showRecovery(rid, sid) {
  $('run-status').hidden = false; $('run-status').replaceChildren(el('span', '', '上轮未完成。可以继续追问，或尝试从检查点恢复。'), button('尝试恢复', async () => {
    try { const r = await post(`/runs/${rid}/recover?` + query()); if (state.session === sid) { state.run = r.run_id; watchRun(r.run_id, sid); } } catch (e) { notice(e); }
  }, 'text-button'));
}
$('stop-run').onclick = async () => { try { await post(`/runs/${state.run}/cancel`); notice('已请求停止；正在进行的模型调用结束后生效。'); } catch (e) { notice(e); } };

async function loadSources(rid, sid) {
  try { const sources = await api(`/runs/${rid}/sources?` + query()); if (state.session !== sid || !sources.length) return;
    const last = [...$('messages').querySelectorAll('.assistant .message-body')].at(-1); if (!last) return;
    last.querySelector('.sources')?.remove(); const details = el('details', 'sources'); details.append(el('summary', '', `查看本轮读取的 ${sources.length} 条原文证据`));
    details.append(el('p', 'muted', '这些是实际读取的来源；不代表每条都被最终答案引用或支持全部结论。'));
    for (const source of sources) { const box = el('div', 'source-card'); const link = el('a', '', source.title || source.paper_title || source.paper_id); link.href = '/api/v1/library/paper?' + query({ paper_id: source.paper_id }); link.target = '_blank'; link.rel = 'noopener'; box.append(link, el('small', '', `PDF 第 ${source.page || '?'} 页 · ${source.section || '章节未识别'} · ${source.chunk_id}`)); if (source.source_type === 'bibliography') box.append(el('small', 'muted', '参考文献条目：仅证明名称与引用位置，未验证被引论文的内容。')); box.append(el('p', '', source.text || '')); details.append(box); }
    last.append(details);
  } catch { /* Answers are still visible if original evidence cannot be loaded. */ }
}

async function refresh() {
  if (!state.user || !state.session || state.refreshBusy) return; state.refreshBusy = true; const sid = state.session;
  try {
    const [library, actions] = await Promise.all([api('/library?' + query({ session_id: sid })), api(`/sessions/${sid}/actions?` + query())]);
    if (state.session !== sid) return; state.library = library; state.actions = actions; renderLibrary(); renderActions();
  } catch (e) { if (e.status === 401) { state.user = null; state.run = null; $('shell').hidden = true; $('auth-screen').hidden = false; notice('登录已过期，请重新登录'); } else notice(e); }
  finally { state.refreshBusy = false; }
}
function openLibrary() { $('paper-panel').hidden = false; refresh(); }
$('open-library').onclick = $('header-library').onclick = openLibrary; $('close-library').onclick = () => { $('paper-panel').hidden = true; };
$('refresh-library').onclick = () => refresh(); $('attach').onclick = () => $('file-input').click();
$('paper-drop').onclick = () => $('file-input').click(); $('paper-drop').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); } };
$('paper-drop').ondragover = event => { event.preventDefault(); $('paper-drop').classList.add('dragover'); }; $('paper-drop').ondragleave = () => $('paper-drop').classList.remove('dragover');
$('paper-drop').ondrop = event => { event.preventDefault(); $('paper-drop').classList.remove('dragover'); upload(event.dataTransfer.files); };
$('file-input').onchange = event => upload(event.target.files);
async function upload(files) {
  if (state.versionSwitching) { notice('正在切换测试版本，请完成后再上传。'); return; }
  if (state.uploading || !state.session) return; state.uploading = true; openLibrary(); const sid = state.session;
  try {
    for (const file of [...files]) {
      if (!file.name.toLowerCase().endsWith('.pdf') || !file.size || file.size > 20 * 1024 * 1024) { notice(`${file.name}：仅支持非空 PDF，每篇 ≤20 MiB`); continue; }
      const strip = $('upload-strip'); strip.hidden = false; strip.replaceChildren(el('span', '', `正在上传 ${file.name}`)); const progress = document.createElement('progress'); progress.max = 100; progress.value = 0; strip.append(progress);
      await uploadFile(file, { ...scope(), session_id: sid }, value => { progress.value = value; });
      await refresh();
    }
    notice('文件已暂存。请在论文面板核对后点击“确认入库”。');
  } catch (e) { notice(e); } finally { state.uploading = false; $('upload-strip').hidden = true; $('file-input').value = ''; }
}
function bar(label, value) { const wrap = el('div', 'progress-row'); const line = el('div', 'progress-label'); line.append(el('span', '', label), el('span', '', `${Math.round(value || 0)}%`)); const p = document.createElement('progress'); p.max = 100; p.value = value || 0; wrap.append(line, p); return wrap; }
function duration(seconds) { if (seconds == null) return '估算中'; return seconds < 60 ? `${Math.round(seconds)} 秒` : `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`; }
function renderLibrary() {
  const data = state.library; if (!data) return;
  $('sidebar-count').textContent = `${data.paper_count} 篇可检索论文`;
  $('library-stats').replaceChildren();
  let versionBox = $('test-version-controls');
  if (!versionBox) { versionBox = el('div', 'panel-note'); versionBox.id = 'test-version-controls'; $('library-stats').after(versionBox); }
  const testing = data.test_version?.status === 'active';
  versionBox.replaceChildren(el('p', '', testing ? '测试版本已开启：新论文进入独立增量索引，可撤销。' : '试用扩库前可开启测试版本，原库保留。'));
  const toggleVersion = button(testing ? '一键撤销测试新增…' : '开启可撤销测试版本', async () => {
    state.versionSwitching = true; toggleVersion.disabled = true;
    try {
      if (!testing) { await post('/library/test-version/start', { ...scope(), session_id: state.session }); await refresh(); notice('已开启测试版本；后续确认入库进入独立测试索引。'); return; }
      const sid = state.session;
      const preview = await post('/library/test-version/prepare-rollback', { ...scope(), session_id: sid });
      const body = el('div'); body.append(el('p', '', preview.note), el('p', '', `本轮新增 ${preview.files_to_archive.length} 个上传文件将归档。会话和原有论文不会删除。`));
      const yes = button('确认撤销测试版本', async () => { yes.disabled = true; try { await post('/library/test-version/rollback', { ...scope(), session_id: sid, confirmation_token: preview.confirmation_token }); $('modal').close(); await refresh(); notice('测试版本已撤销，新论文退出检索；原库保留。'); } catch (e) { notice(e); yes.disabled = false; } }, 'primary');
      modal('撤销测试新增论文', body, [button('取消', () => $('modal').close()), yes]);
    } catch (e) { notice(e); }
    finally { state.versionSwitching = false; toggleVersion.disabled = false; }
  }, 'text-button');
  toggleVersion.id = 'toggle-test-version'; toggleVersion.disabled = state.uploading || !!state.run || data.jobs.some(j => ['queued', 'running'].includes(j.status)); versionBox.append(toggleVersion);
  for (const [value, label] of [[data.paper_count, '可检索论文'], [data.documents.filter(d => !d.indexed).length, '待核对文件'], [data.capacity, '论文容量上限']]) { const n = el('div'); n.append(el('strong', '', value), el('small', '', label)); $('library-stats').append(n); }
  $('documents').replaceChildren();
  for (const d of data.documents) {
    const n = el('div', 'document-row'); const text = el('div'); text.append(el('b', '', d.name), el('small', '', `${(d.bytes / 1024 / 1024).toFixed(2)} MiB · ${d.indexed ? '同哈希论文已在库中' : d.supported ? '已上传，等待确认' : '此入口仅构建 PDF'}`));
    const a = el('a', 'text-button', '查看'); a.href = '/api/v1/workbench/file?' + query({ document: d.document }); a.target = '_blank'; a.rel = 'noopener'; n.append(text, a); $('documents').append(n);
  }
  if (!data.documents.length) $('documents').append(el('p', 'empty-small', '还没有上传论文。也可以在对话中请求查找开放论文。'));
  const active = data.jobs.find(j => ['queued', 'running'].includes(j.status));
  $('job-dot').hidden = !active; $('ingest').disabled = !!active || state.uploading || !data.documents.some(d => !d.indexed && d.supported);
  $('jobs').replaceChildren();
  for (const job of data.jobs.filter(j => j.kind === 'rag-publication').slice(0, 8)) {
    const box = el('section', 'job-card ' + job.status); box.append(el('b', '', job.publication_active === false ? '测试版本已撤销 · 该批论文已退出当前检索' : job.stage));
    box.append(el('small', '', `${job.completed_files}/${job.total_files} 份文件 · 已用 ${duration(job.elapsed_seconds)}`));
    box.append(bar('论文解析', job.parse_progress), bar('知识库构建与发布', job.build_progress));
    if (['queued', 'running'].includes(job.status)) { const eta = el('small', 'muted', `预计剩余：${duration(job.eta_seconds)}`); eta.title = job.eta_basis; box.append(eta); }
    if (job.document) { const document = data.documents.find(d => d.document.endsWith('/' + job.document)); box.append(el('small', 'muted', '当前文件：' + (document?.name || job.document))); }
    if (job.error) box.append(el('p', 'error', job.error));
    if (job.summary) box.append(el('p', 'job-summary', `本次编码 ${job.summary.papers_encoded} 篇 / ${job.summary.vectors_encoded} 个向量；复用 ${job.summary.papers_reused} 篇 / ${job.summary.vectors_reused} 个向量。`));
    if (job.rejected_papers?.length) box.append(el('p', 'error', `${job.rejected_papers.length} 篇质量不达标，未发布，请修复原文后重试。`));
    $('jobs').append(box);
  }
  if (!$('jobs').children.length) $('jobs').append(el('p', 'empty-small', '确认论文入库后，显示真实解析、向量化与发布进度。'));
  const activeBox = $('active-job'); activeBox.hidden = !active;
  if (active) { activeBox.replaceChildren(el('span', 'spinner'), el('span', '', active.stage + ' · ' + (active.build_progress ?? 0) + '%'), button('查看进度', openLibrary, 'text-button')); }
  renderPapers();
}
function renderPapers() {
  const list = $('papers'); list.replaceChildren(); const term = $('paper-filter').value.toLowerCase();
  const unique = new Map(); for (const p of state.library?.papers || []) if (!unique.has(p.content_hash)) unique.set(p.content_hash, p);
  for (const p of [...unique.values()].filter(p => p.title.toLowerCase().includes(term))) { const n = el('div', 'paper-row'); const a = el('a', '', p.title); a.href = '/api/v1/library/paper?' + query({ paper_id: p.paper_id }); a.target = '_blank'; a.rel = 'noopener'; n.append(a, el('small', '', `${p.scope === 'project' ? '本项目新增' : '已有文献库'} · 文本质量 ${p.quality_status}`)); list.append(n); }
}
$('paper-filter').oninput = renderPapers;
$('ingest').onclick = async () => { try { const result = await post('/library/propose-ingest', { ...scope(), session_id: state.session }); await refresh(); await inspectAction(result.action_id); } catch (e) { notice(e); } };
function renderActions() {
  const target = $('action-cards'); target.replaceChildren();
  const labels = { download_pdf: '下载论文', ingest_documents: '增量构建知识库', implement_experiment: '提交编码 / 实验实现' };
  for (const action of state.actions.filter(a => a.status === 'pending')) { const n = el('div', 'action-card'); n.append(el('span', 'action-badge', '需要你确认'), el('b', '', labels[action.action] || action.action), el('p', '', action.reason), button('查看并决定', () => inspectAction(action.id).catch(notice), 'primary small')); target.append(n); }
}
function modal(title, content, footer = []) { $('modal-title').textContent = title; $('modal-body').replaceChildren(content); $('modal-footer').replaceChildren(...footer); if (!$('modal').open) $('modal').showModal(); }
$('close-modal').onclick = () => $('modal').close();
async function inspectAction(id) {
  const preview = await post(`/actions/${id}/prepare?` + query()); const sid = state.session;
  const body = el('div'); body.append(el('p', '', preview.reason));
  const args = el('pre', 'confirmation-payload', JSON.stringify(preview.arguments, null, 2)); body.append(args);
  if (preview.action === 'ingest_documents') { body.append(el('p', '', '本次将检查上传目录中的 PDF，仅处理未编码的论文；相同哈希去重、已有向量复用。入库成功后，同项目其他会话可检索这些论文。')); const ul = el('ul'); for (const d of state.library?.documents.filter(d => !d.indexed && d.supported) || []) ul.append(el('li', '', d.name)); body.append(ul); }
  if (preview.action === 'download_pdf') body.append(el('p', '', '仅下载暂存，不自动入库。下载成功后还需核对并确认构建知识库。'));
  if (preview.action === 'implement_experiment') body.append(el('p', 'error', '这会调用已有编码实现适配器并消耗资源。工程验证不等于论文实验成功，请核对冻结的实验合同。'));
  body.append(el('small', 'muted', '确认仅对当前参数与当前会话状态有效，5 分钟后过期。取消不会执行。'));
  const confirm = button('确认执行', async () => { confirm.disabled = true; try { const r = await post(`/actions/${id}/confirm`, { ...scope(), confirmation_token: preview.confirmation_token }); $('modal').close(); if (state.session === sid) { state.run = r.run_id; watchRun(r.run_id, sid); } await refresh(); } catch (e) { notice(e); confirm.disabled = false; } }, 'primary');
  modal('核对操作 · 由你决定', body, [button('拒绝此操作', async () => { try { await post(`/actions/${id}/reject?` + query()); $('modal').close(); await refresh(); } catch (e) { notice(e); } }), button('稍后决定', () => $('modal').close()), confirm]);
}

async function suggestTitle(sid, automatic = false) {
  const suggestion = await post(`/sessions/${sid}/suggest-title`, scope());
  const current = state.sessions.find(s => s.id === sid);
  if (automatic && current?.title !== '新对话') return;
  await post(`/sessions/${sid}`, { ...scope(), title: suggestion.title }); await loadSessions();
  if (state.session === sid) $('session-title').textContent = suggestion.title;
  return suggestion.title;
}
$('session-settings').onclick = async () => {
  const sid = state.session; if (!sid) return;
  try {
    const [detail, memories] = await Promise.all([api(`/sessions/${sid}?` + query()), api('/memories?' + query({ session_id: sid }))]);
    const body = el('div', 'settings');
    const label = el('label', '', '会话名称'); const title = el('input'); title.value = detail.title; title.maxLength = 120; title.id = 'rename-input'; label.append(title); body.append(label);
    const controls = el('div', 'settings-actions'); controls.append(button('保存名称', async () => { try { await post(`/sessions/${sid}`, { ...scope(), title: title.value.trim() || '新对话' }); await loadSessions(); if (state.session === sid) $('session-title').textContent = title.value; notice('名称已保存'); } catch (e) { notice(e); } }), button('让 AI 取名', async () => { try { const r = await post(`/sessions/${sid}/suggest-title`, scope()); title.value = r.title; notice('已生成建议名称，点击保存后生效'); } catch (e) { notice(e); } })); body.append(controls);
    const memoLabel = el('label', '', '当前会话约束与笔记'); const memo = el('textarea'); memo.id = 'memory-input'; memo.maxLength = 8000; memo.value = detail.memory || ''; memo.placeholder = '例如：只讨论逐点最大绝对误差；基线先不执行。'; memoLabel.append(memo); body.append(memoLabel, button('保存会话笔记', async () => { try { await post(`/sessions/${sid}`, { ...scope(), memory: memo.value }); notice('笔记已保存，仅本会话使用'); } catch (e) { notice(e); } }));
    const details = el('details'); details.append(el('summary', '', '查看当前研究状态'), el('pre', 'state-preview', JSON.stringify(detail.state, null, 2))); body.append(details);
    for (const m of memories.filter(m => m.status === 'proposed')) { const row = el('div', 'memory-proposal'); row.append(el('p', '', m.content), button('采纳这条记忆', async () => { try { await post(`/memories/${m.id}`, { ...scope(), action: 'confirm' }); row.remove(); notice('已采纳，仅本会话可用'); } catch (e) { notice(e); } })); body.append(row); }
    body.append(el('p', 'muted', '新会话不读取本会话的私有记忆。论文知识库可复用；删除会话会删除其关联记忆，保留论文。'));
    body.append(button('删除此会话…', () => deleteSession(sid).catch(notice), 'danger'));
    modal('管理当前会话', body, [button('完成', () => $('modal').close())]);
  } catch (e) { notice(e); }
};
async function deleteSession(sid) {
  const preview = await post(`/sessions/${sid}/prepare-delete`, scope());
  const body = el('div'); body.append(el('p', '', '确认删除此会话及其关联记忆？删除后其他会话不能再召回这些记忆。论文知识库保留。'), el('pre', 'confirmation-payload', JSON.stringify(preview.impact, null, 2)));
  const confirm = button('确认删除会话与记忆', async () => { confirm.disabled = true; try { await post(`/sessions/${sid}`, { ...scope(), delete: true, confirmation_token: preview.token }); $('modal').close(); await loadSessions(); const next = state.sessions.find(s => !s.archived); if (next) await selectSession(next.id); else await newSession(); } catch (e) { notice(e); confirm.disabled = false; } }, 'danger');
  modal('删除会话 · 二次确认', body, [button('取消', () => $('modal').close()), confirm]);
}
$('toggle-sidebar').onclick = () => { if (innerWidth < 900) $('sidebar').classList.toggle('mobile-open'); else $('shell').classList.toggle('sidebar-collapsed'); }; $('close-sidebar').onclick = () => $('sidebar').classList.remove('mobile-open');
$('logout').onclick = async () => { try { await post('/auth/logout'); state.user = null; state.run = null; state.epoch++; location.reload(); } catch (e) { notice(e); } };
setInterval(() => { if (state.user) refresh(); }, 2500);
api('/auth/me').then(r => boot(r.user_id)).catch(e => { state.user = null; $('shell').hidden = true; $('auth-screen').hidden = false; if (e.status !== 401) notice(e); });
