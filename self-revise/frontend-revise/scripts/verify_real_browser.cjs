/* Actual browser registration and conversation with the configured REAL LLM. */
const { chromium } = require('../../../frontend/node_modules/playwright');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../../..');
const work = 'self-revise/frontend-revise/local-runtime/browser-real-' + Date.now();
const out = path.join(root, process.env.RESEARCHCLAW_BROWSER_EVIDENCE || 'self-revise/frontend-revise/evidence/registration-rollback-v20261110');
fs.mkdirSync(out, { recursive: true });
const log = fs.openSync(path.join(out, 'real-browser.log'), 'w');
const env = { ...process.env, RESEARCHCLAW_FRONTEND_ORIGINS: 'http://127.0.0.1:5182', RESEARCHCLAW_RETRIEVAL_BACKEND: 'neural' };
const backend = spawn(path.join(root, '.venv/Scripts/python.exe'), ['-X', 'utf8', '-B', 'self-revise/frontend-revise/scripts/verify_research_chat_live.py', '--serve-browser', '--workspace', work], { cwd: root, env, windowsHide: true, stdio: ['ignore', log, log] });
const frontend = spawn(process.execPath, [path.join(root, 'frontend/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5182', '--strictPort'], { cwd: path.join(root, 'frontend'), env: { ...env, RESEARCHCLAW_API_TARGET: 'http://127.0.0.1:8022' }, windowsHide: true, stdio: ['ignore', log, log] });
const manifest = path.join(root, 'knowledge_base/rag_corpora/time_series_domain_release_v3_1/rag-search-index.json');
const digest = () => crypto.createHash('sha256').update(fs.readFileSync(manifest)).digest('hex');
async function ready(url) { for (let i = 0; i < 180; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise(r => setTimeout(r, 500)); } throw Error('Service not ready: ' + url); }
(async () => {
  let browser, page;
  const result = { workspace: work, kind: 'real browser + real configured LLM, not a deterministic response fixture', shared_before: digest() };
  try {
    await ready('http://127.0.0.1:8022/api/v1/health'); await ready('http://127.0.0.1:5182');
    browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(60000);
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:5182');
    await page.click('#show-register'); await page.fill('#user', 'demo2@2026');
    await page.fill('#password', 'Demo2@2026!Browser'); await page.fill('#invite', 'browser-real-invite');
    const registration = page.waitForResponse(r => r.url().endsWith('/auth/register'));
    await page.click('#login-submit'); result.registration_status = (await registration).status();
    await page.waitForSelector('#shell:visible');
    assert.equal(result.registration_status, 201);
    const question = process.env.RESEARCHCLAW_BROWSER_QUESTION || '请检索知识库，用一个异常点例子简要解释逐点最大绝对误差和均方误差的区别，引用一处实际读取的原文。最多250字，不下载、不实验。';
    const started = Date.now(); result.question = question;
    await page.fill('#message-input', question);
    const dispatched = page.waitForResponse(r => r.url().endsWith('/api/v1/chat') && r.request().method() === 'POST');
    await page.click('#send'); const request = await (await dispatched).json(); result.run_id = request.run_id;
    const api = async url => page.evaluate(async url => { const r = await fetch('/api/v1' + url, { credentials: 'include' }); if (!r.ok) throw Error('HTTP ' + r.status); return r.json(); }, url);
    let run;
    for (let i = 0; i < 660; i++) { run = await api('/runs/' + request.run_id); if (run.status !== 'running') break; await page.waitForTimeout(1000); }
    result.run = run; result.graph = await api('/runs/' + request.run_id + '/graph');
    result.sources = await api('/runs/' + request.run_id + '/sources'); result.seconds = (Date.now() - started) / 1000;
    assert.equal(run.status, 'completed'); assert.equal(result.graph.status, 'complete'); assert.ok(result.sources.length);
    await page.waitForSelector('.message.assistant'); await page.waitForSelector('.sources');
    await page.waitForFunction(() => document.querySelector('#session-title').textContent !== '新对话');
    result.ai_title = await page.locator('#session-title').innerText();
    await page.screenshot({ path: path.join(out, 'real-browser-answer.png'), fullPage: true });
    result.errors = errors; assert.deepEqual(errors, []);
    result.shared_after = digest(); assert.equal(result.shared_after, result.shared_before);
    result.status = 'passed';
    console.log(JSON.stringify({ status: result.status, registration_status: result.registration_status, seconds: result.seconds, sources: result.sources.length, ai_title: result.ai_title }));
  } catch (error) { result.status = 'failed'; result.error = String(error); if (page) { result.visible_text = await page.locator('body').innerText(); await page.screenshot({path:path.join(out,'real-browser-failure.png'),fullPage:true}); } throw error; }
  finally { fs.writeFileSync(path.join(out, 'real-browser.json'), JSON.stringify(result, null, 2)); if (browser) await browser.close(); backend.kill(); frontend.kill(); fs.closeSync(log); }
})().catch(error => { console.error(error); process.exitCode = 1; });
