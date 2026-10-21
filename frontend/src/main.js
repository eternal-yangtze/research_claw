import './style.css';

const el = id => document.getElementById(id);
let session = null, events = null;
async function api(path, options = {}) {
  const response = await fetch('/api/v1' + path, { credentials: 'include', ...options });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '请求失败，请检查输入或重新登录。');
  return value;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const report = error => { el('status').textContent = error.message; };
function addMessage(role, content) {
  const article = document.createElement('article');
  article.className = role;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? '你' : 'ResearchOps';
  const text = document.createElement('p'); text.textContent = content;
  article.append(label, text); el('messages').append(article); article.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function reset() { session = null; el('messages').replaceChildren(); }
function busy(value) { for (const id of ['send', 'new-session', 'project', 'logout']) el(id).disabled = value; }
el('login').onsubmit = async event => {
  event.preventDefault();
  try {
    await post('/auth/login', { user_id: el('user').value, password: el('password').value });
    el('password').value = ''; reset(); el('login-panel').hidden = true; el('workspace').hidden = false; el('status').textContent = '';
  } catch (error) { report(error); }
};
el('new-session').onclick = reset;
el('project').onchange = reset;
el('logout').onclick = async () => {
  try { await post('/auth/logout', {}); if (events) events.close(); reset(); el('workspace').hidden = true; el('login-panel').hidden = false; }
  catch (error) { report(error); }
};
el('chat').onsubmit = async event => {
  event.preventDefault(); const message = el('message').value.trim(); if (!message) return;
  busy(true); el('status').textContent = '正在整理回答…';
  try {
    const run = await post('/chat', { message, project_id: el('project').value, session_id: session });
    session = run.session_id; addMessage('user', message); el('message').value = '';
    events = new EventSource('/api/v1/runs/' + run.run_id + '/events');
    events.onmessage = event => {
      const result = JSON.parse(event.data); events.close(); busy(false);
      if (result.status === 'completed') { addMessage('assistant', result.answer); el('status').textContent = ''; }
      else report(new Error('本次任务未完成，请查看后台运行记录。'));
    };
    events.onerror = async () => {
      events.close(); busy(false);
      try {
        const result = await api('/runs/' + run.run_id);
        if (result.status === 'completed') addMessage('assistant', result.answer);
        else report(new Error('连接已断开，任务状态：' + result.status));
      } catch (error) { report(error); }
    };
  } catch (error) { busy(false); report(error); }
};
el('upload').onsubmit = async event => {
  event.preventDefault(); const file = el('document').files[0]; if (!file) return;
  if (file.size > 20 * 1024 * 1024) return report(new Error('请选择小于 20 MiB 的文件。'));
  el('upload-button').disabled = true; el('upload-status').textContent = '正在解析资料…';
  try {
    const suffix = file.name.split('.').pop().toLowerCase();
    await api('/knowledge/upload?project_id=' + encodeURIComponent(el('project').value) + '&suffix=' + encodeURIComponent(suffix), { method: 'POST', body: file });
    el('upload-status').textContent = file.name + ' 已加入当前项目。';
  } catch (error) { el('upload-status').textContent = ''; report(error); }
  finally { el('upload-button').disabled = false; }
};
api('/health').then(() => { el('connection').textContent = '服务已连接'; }).catch(() => { el('connection').textContent = '服务暂不可用'; });
