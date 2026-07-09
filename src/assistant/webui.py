"""A minimal, self-contained web chat UI served at ``GET /ui``.

One inline HTML page — no build step, no CDN, no external assets — that streams
replies from ``POST /chat/stream``. The page itself carries no data, so it is not
token-gated; the *API* calls it makes are. When ``API_TOKEN`` is set, the page
asks for the token once and keeps it in ``localStorage``, sending it as a bearer
header on every request.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assistant</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111; --mut:#666; --line:#e3e3e3; --me:#eef2ff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c2f36; --me:#232a3d; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,-apple-system,sans-serif;
         display:flex; flex-direction:column; height:100dvh; }
  header { padding:.75rem 1rem; border-bottom:1px solid var(--line); font-weight:600; }
  #log { flex:1; overflow-y:auto; padding:1rem; display:flex; flex-direction:column; gap:.75rem; }
  .msg { max-width:min(70ch,90%); padding:.6rem .8rem; border-radius:.75rem; white-space:pre-wrap; word-wrap:break-word; }
  .me { align-self:flex-end; background:var(--me); }
  .bot { align-self:flex-start; border:1px solid var(--line); }
  .err { align-self:center; color:#c00; font-size:.9rem; }
  form { display:flex; gap:.5rem; padding:.75rem 1rem; border-top:1px solid var(--line); }
  input,button { font:inherit; }
  #q { flex:1; padding:.6rem .8rem; border:1px solid var(--line); border-radius:.5rem;
       background:var(--bg); color:var(--fg); }
  button { padding:.6rem 1rem; border:0; border-radius:.5rem; background:#3b5bdb; color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  footer { padding:.4rem 1rem; color:var(--mut); font-size:.78rem; border-top:1px solid var(--line); }
</style>
</head>
<body>
<header>Assistant</header>
<div id="log"></div>
<form id="f">
  <input id="q" autocomplete="off" placeholder="Message…" autofocus>
  <button id="send">Send</button>
</form>
<footer>Streaming over <code>/chat/stream</code>. Reply "undo" to revert the last calendar or task change.</footer>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const box = document.getElementById('q');
const send = document.getElementById('send');
let threadId = localStorage.getItem('thread_id') || null;

function bubble(cls, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function headers() {
  const h = {'Content-Type': 'application/json'};
  const t = localStorage.getItem('api_token');
  if (t) h['Authorization'] = 'Bearer ' + t;
  return h;
}

async function ask(text) {
  bubble('me', text);
  const out = bubble('bot', '');
  let res;
  try {
    res = await fetch('/chat/stream', {
      method: 'POST', headers: headers(),
      body: JSON.stringify({message: text, thread_id: threadId}),
    });
  } catch (e) { out.remove(); bubble('err', 'Network error.'); return; }

  if (res.status === 401) {
    out.remove();
    const t = prompt('API token required:');
    if (t) { localStorage.setItem('api_token', t); return ask(text); }
    bubble('err', 'Unauthorized.');
    return;
  }
  if (!res.ok) { out.remove(); bubble('err', 'Server error ' + res.status); return; }

  // Parse the SSE stream: `data:` frames carry reply text, `event: done`
  // carries the thread id, `event: error` reports a model failure.
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const frames = buf.split('\\n\\n');
    buf = frames.pop();
    for (const frame of frames) {
      if (frame.startsWith('event: done')) {
        const id = frame.slice(frame.indexOf('data: ') + 6);
        if (id) { threadId = id; localStorage.setItem('thread_id', id); }
      } else if (frame.startsWith('event: error')) {
        bubble('err', frame.slice(frame.indexOf('data: ') + 6));
      } else if (frame.startsWith('data: ')) {
        out.textContent += frame.slice(6);
        log.scrollTop = log.scrollHeight;
      }
    }
  }
  if (!out.textContent) out.textContent = '(empty reply)';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = box.value.trim();
  if (!text) return;
  box.value = '';
  send.disabled = true;
  try { await ask(text); } finally { send.disabled = false; box.focus(); }
});
</script>
</body>
</html>
"""
