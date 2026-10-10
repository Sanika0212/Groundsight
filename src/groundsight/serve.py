"""A live local view of stored audits (standard library only).

``groundsight serve`` lists the audits in a store, newest first, refreshing as new ones
arrive, and opens each as the full report. It binds to 127.0.0.1, answers only GET
requests, and refuses requests whose Host header is not local (a guard against DNS
rebinding), so other sites in your browser cannot read your audits.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from groundsight.capture import DEFAULT_STORE, Store
from groundsight.report import _CSS, render

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:"
)

_INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Groundsight</title>
<style>{css}
.list {{ max-width:1240px; margin:0 auto; padding:10px 24px 40px; }}
.list a {{ color:var(--accent); text-decoration:none; }}
.empty {{ color:var(--ink2); padding:30px 0; }}
</style></head><body>
<header><h1>Groundsight</h1><div class="sub">Audits in <code>{store}</code>, newest first. This page
updates as new audits arrive.</div></header>
<div class="list card"><table><thead><tr><th>When</th><th>Question</th><th>Supported</th>
<th>Flagged</th><th>Checker</th></tr></thead><tbody id="rows"></tbody></table>
<div class="empty" id="empty" hidden>No audits yet. Run your pipeline inside
<code>groundsight.trace()</code> or call <code>groundsight.audit()</code> with a store.</div></div>
<script>
async function refresh() {{
  try {{
    const audits = await (await fetch('/api/audits')).json();
    const rows = document.getElementById('rows'); rows.textContent = '';
    document.getElementById('empty').hidden = audits.length > 0;
    for (const a of audits) {{
      const tr = document.createElement('tr');
      const cells = [new Date(a.created * 1000).toLocaleString(), null,
                     Math.round(a.groundedness * 100) + '%', String(a.flagged), a.checker];
      cells.forEach((value, i) => {{
        const td = document.createElement('td');
        if (i === 1) {{ const link = document.createElement('a'); link.href = '/audit/' + a.id;
          link.textContent = a.question || a.answer.slice(0, 80); td.append(link); }}
        else td.textContent = value;
        tr.append(td);
      }});
      rows.append(tr);
    }}
  }} catch (e) {{ /* server stopped; keep the last view */ }}
}}
refresh(); setInterval(refresh, 2000);
</script></body></html>
"""


def summaries(store: Store) -> list[dict[str, Any]]:
    return [
        {
            "id": a.id,
            "created": a.created,
            "question": a.question,
            "answer": a.answer,
            "groundedness": a.groundedness,
            "flagged": len(a.flagged),
            "checker": a.checker,
        }
        for a in store.all()
    ]


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "groundsight"

        def log_message(self, format: str, *args: Any) -> None:
            return  # keep the terminal quiet

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Security-Policy", _CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in _LOCAL_HOSTS:
                self._send(403, "forbidden: not a local host", "text/plain; charset=utf-8")
                return
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                page = _INDEX.format(css=_CSS, store=html.escape(str(store.path)))
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/api/audits":
                self._send(200, json.dumps(summaries(store)), "application/json")
            elif path.startswith("/audit/"):
                try:
                    result = store.get(path.removeprefix("/audit/"))
                except KeyError:
                    self._send(404, "no such audit", "text/plain; charset=utf-8")
                    return
                self._send(200, render(result), "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")

    return Handler


def make_server(
    store: Store | str | Path = DEFAULT_STORE, *, host: str = "127.0.0.1", port: int = 7860
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("groundsight serve only binds to the local machine")
    target = store if isinstance(store, Store) else Store(store)
    return ThreadingHTTPServer((host, port), make_handler(target))


def serve_in_background(
    store: Store | str | Path = DEFAULT_STORE, *, port: int = 7860
) -> tuple[ThreadingHTTPServer, str]:
    """Start the viewer on a daemon thread; returns the server and its URL."""
    server = make_server(store, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"
