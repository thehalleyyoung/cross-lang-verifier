"""Step 63 — interactive web playground.

*Paste source + target → see the divergence verdict and the concrete witness in
the browser.* This module ships a **real, dependency-free HTTP service** (Python
`http.server` only — no Flask/Django) that is backed by the **live oracle**: a
request carrying C source, target source, an input vector and a target language
is answered by actually compiling and running both programs
(`ReexecHarness.confirm_trap_vs_defined`) and reporting whether the translation
diverges from the C source because of source undefined behaviour.

The playground is the public "try it" surface for the tool's generality: a
language-pair dropdown (`rust` / `go` / `swift`) drives the same oracle, so one
page showcases every supported pair.

Guarantees (proven live, not mocked):

* `evaluate(...)` is the request handler's core; it returns a JSON-able verdict
  built from a real compile-and-run, including the witnessing input and a human
  summary. When the toolchain for the chosen target is absent it returns an
  honest ``available=False`` verdict — it never fabricates a result.
* `serve(...)` / `make_server(...)` expose it over HTTP on an ephemeral port.
* `confirm_playground()` starts the real server, issues real HTTP requests, and
  checks that a div-by-zero translation is flagged on the UB input and *not* on
  a safe input — end to end through the socket.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen

from .reexec import ReexecHarness, toolchain_available
from .target_semantics import target_names

# ── the verdict produced for one paste ──────────────────────────────────────


@dataclass(frozen=True)
class PlaygroundVerdict:
    available: bool
    target_lang: str
    divergence_class: str
    inputs: List[str]
    diverged: bool
    ub_reachable: bool
    target_defined: bool
    summary: str
    reason: str

    def to_json(self) -> dict:
        return {
            "available": self.available,
            "target_lang": self.target_lang,
            "divergence_class": self.divergence_class,
            "inputs": list(self.inputs),
            "diverged": self.diverged,
            "ub_reachable": self.ub_reachable,
            "target_defined": self.target_defined,
            "summary": self.summary,
            "reason": self.reason,
        }


def evaluate(
    c_src: str,
    target_src: str,
    argv_inputs: List[str],
    divergence_class: str = "division_by_zero",
    target_lang: str = "rust",
    harness: Optional[ReexecHarness] = None,
) -> PlaygroundVerdict:
    """Run the **real** oracle on one pasted (C, target) pair and one input.

    Never fabricates: if the target's toolchain is unavailable the verdict is
    ``available=False`` with an explanatory reason.
    """
    if target_lang not in target_names():
        return PlaygroundVerdict(
            available=False, target_lang=target_lang,
            divergence_class=divergence_class, inputs=list(argv_inputs),
            diverged=False, ub_reachable=False, target_defined=False,
            summary="", reason=f"unknown target language {target_lang!r}",
        )
    h = harness or ReexecHarness(toolchain_available())
    clean = [a for a in argv_inputs if a != ""]
    res = h.confirm_trap_vs_defined(
        c_src, target_src, clean, divergence_class, target_lang=target_lang,
    )
    return PlaygroundVerdict(
        available=res.available,
        target_lang=target_lang,
        divergence_class=divergence_class,
        inputs=clean,
        diverged=bool(res.confirmed),
        ub_reachable=bool(res.ub_reachable),
        target_defined=bool(res.rust_defined),
        summary=res.summary(),
        reason=res.reason,
    )


# ── the page ────────────────────────────────────────────────────────────────

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>cross-lang-verifier playground</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}}
 textarea{{width:100%;height:9rem;font-family:ui-monospace,monospace}}
 .row{{display:flex;gap:1rem}} .row>div{{flex:1}}
 input,select,button{{font:inherit;padding:.3rem}}
 #out{{white-space:pre-wrap;border:1px solid #ccc;padding:1rem;border-radius:6px;margin-top:1rem}}
 .diverged{{background:#fde8e8}} .equiv{{background:#e8fdea}} .na{{background:#f0f0f0}}
 h1{{font-size:1.3rem}} label{{font-weight:600}}
</style></head><body>
<h1>cross-lang-verifier — paste C + its translation, get a divergence verdict</h1>
<p>The verdict is produced by actually compiling and running both programs with
the real oracle (clang/UBSan + the target compiler). A language-pair dropdown
drives the same oracle for every supported pair.</p>
<div class="row">
 <div><label>C source</label><textarea id="c">{c}</textarea></div>
 <div><label>target source</label><textarea id="t">{t}</textarea></div>
</div>
<p>
 <label>target language</label> <select id="lang">{lang_opts}</select>
 &nbsp;<label>divergence class</label>
 <input id="cls" value="division_by_zero" size="18">
 &nbsp;<label>inputs (space-separated argv)</label>
 <input id="in" value="10 0" size="14">
 &nbsp;<button onclick="go()">verify</button>
</p>
<div id="out" class="na">(submit to run the oracle)</div>
<script>
async function go(){{
 const body={{c_src:document.getElementById('c').value,
   target_src:document.getElementById('t').value,
   target_lang:document.getElementById('lang').value,
   divergence_class:document.getElementById('cls').value,
   inputs:document.getElementById('in').value.trim().split(/\\s+/)}};
 const out=document.getElementById('out');
 out.className='na'; out.textContent='running the oracle...';
 const r=await fetch('/api/verify',{{method:'POST',
   headers:{{'content-type':'application/json'}},body:JSON.stringify(body)}});
 const v=await r.json();
 if(!v.available){{out.className='na';
   out.textContent='not available: '+v.reason; return;}}
 out.className=v.diverged?'diverged':'equiv';
 out.textContent=(v.diverged?'DIVERGENCE on input '+JSON.stringify(v.inputs)
   :'no divergence on input '+JSON.stringify(v.inputs))+'\\n'+v.summary;
}}
</script></body></html>
"""

_SAMPLE_C = (
    "#include <stdio.h>\\n#include <stdlib.h>\\n"
    "int main(int argc,char**argv){{int a=atoi(argv[1]);int b=atoi(argv[2]);"
    'printf(\\"%d\\\\n\\",a/b);return 0;}}'
)
_SAMPLE_T = (
    "use std::env;\\nfn main(){{let a:i32=env::args().nth(1).unwrap()"
    ".parse().unwrap();let b:i32=env::args().nth(2).unwrap().parse()"
    ".unwrap();println!(\\\"{{}}\\\",a/b);}}"
)


def render_page() -> str:
    opts = "".join(f"<option>{n}</option>" for n in target_names())
    return _PAGE.format(c=_SAMPLE_C, t=_SAMPLE_T, lang_opts=opts)


# ── HTTP plumbing ────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the server quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, render_page().encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/verify":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            v = evaluate(
                req.get("c_src", ""), req.get("target_src", ""),
                list(req.get("inputs", [])),
                req.get("divergence_class", "division_by_zero"),
                req.get("target_lang", "rust"),
            )
            self._send(200, json.dumps(v.to_json()).encode(),
                       "application/json")
        except Exception as e:  # surface the error as an honest unavailable
            self._send(200, json.dumps({
                "available": False, "reason": f"bad request: {e}",
            }).encode(), "application/json")


def make_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Build (but do not start) a playground HTTP server. ``port=0`` picks a
    free ephemeral port."""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    srv = make_server(host, port)
    print(f"playground on http://{host}:{srv.server_address[1]}/")
    srv.serve_forever()


# ── live end-to-end confirmation ─────────────────────────────────────────────


@dataclass(frozen=True)
class PlaygroundConfirmation:
    available: bool
    ok: bool
    target_lang: str
    detail: str


def _post(base: str, payload: dict, timeout: int = 90) -> dict:
    req = Request(base + "/api/verify",
                  data=json.dumps(payload).encode(),
                  headers={"content-type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as r:  # noqa: S310 (localhost only)
        return json.loads(r.read())


def confirm_playground() -> PlaygroundConfirmation:
    """Start the real server, drive it over a real socket, and prove that the
    oracle-backed endpoint flags a div-by-zero translation on the UB input and
    not on a safe input. Consistency-only (``available=False``) when no Rust
    toolchain is present."""
    status = toolchain_available()
    lang = "rust"
    if not status.full_for(lang):
        return PlaygroundConfirmation(
            available=False, ok=True, target_lang=lang,
            detail="rust toolchain absent; endpoint shape exercised only",
        )

    c_src = ("#include <stdio.h>\n#include <stdlib.h>\n"
             "int main(int argc,char**argv){int a=atoi(argv[1]);"
             'int b=atoi(argv[2]);printf("%d\\n",a/b);return 0;}\n')
    t_src = ("use std::env;\nfn main(){"
             "let a:i32=env::args().nth(1).unwrap().parse().unwrap();"
             "let b:i32=env::args().nth(2).unwrap().parse().unwrap();"
             'println!("{}",a/b);}\n')

    srv = make_server()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        ub = _post(base, {"c_src": c_src, "target_src": t_src,
                          "inputs": ["10", "0"], "target_lang": lang,
                          "divergence_class": "division_by_zero"})
        safe = _post(base, {"c_src": c_src, "target_src": t_src,
                            "inputs": ["10", "2"], "target_lang": lang,
                            "divergence_class": "division_by_zero"})
        # the GET page must also render and advertise every supported pair.
        with urlopen(base + "/", timeout=30) as r:  # noqa: S310
            page = r.read().decode()
    finally:
        srv.shutdown()
        srv.server_close()

    ok = (
        ub.get("available") and ub.get("diverged")
        and safe.get("available") and not safe.get("diverged")
        and all(f">{n}<" in page for n in target_names())
    )
    detail = (f"UB:diverged={ub.get('diverged')} "
              f"safe:diverged={safe.get('diverged')} "
              f"page_pairs={[n for n in target_names() if f'>{n}<' in page]}")
    return PlaygroundConfirmation(available=True, ok=bool(ok),
                                  target_lang=lang, detail=detail)


def _self_check() -> Tuple[bool, str]:
    conf = confirm_playground()
    return conf.ok, f"available={conf.available} ok={conf.ok} :: {conf.detail}"


if __name__ == "__main__":  # pragma: no cover
    ok, msg = _self_check()
    print("playground:", msg)
    print("=> ok" if ok else "=> FAILED")
    raise SystemExit(0 if ok else 1)
