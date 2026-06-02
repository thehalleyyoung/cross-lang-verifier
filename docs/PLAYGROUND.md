# Interactive Web Playground

*Paste C + its translation → get a divergence verdict and the concrete witness
in your browser.* Implemented in `src/ub_oracle/playground.py` as a **real,
dependency-free HTTP service** (Python `http.server` only) backed by the **live
oracle** — every verdict is produced by actually compiling and running both
programs (`ReexecHarness.confirm_trap_vs_defined`), never mocked.

## Run it

```bash
python -m ub_oracle.playground            # or: serve(host, port)
# playground on http://127.0.0.1:8000/
```

Open the page, paste C source and the target source, choose a **target language**
from the dropdown (`rust` / `go` / `swift` — the same oracle drives every pair),
give an `argv` input vector and a divergence class, and hit **verify**. The page
POSTs to `/api/verify` and shows whether the translation diverges from the C
source because of source undefined behaviour, plus the witnessing input and a
one-line summary.

## The endpoint

`POST /api/verify` with JSON:

```json
{ "c_src": "...", "target_src": "...", "target_lang": "rust",
  "divergence_class": "division_by_zero", "inputs": ["10", "0"] }
```

returns

```json
{ "available": true, "diverged": true, "ub_reachable": true,
  "target_defined": true, "inputs": ["10","0"], "summary": "..." }
```

When the chosen target's toolchain is not installed the response is an honest
`{"available": false, "reason": "..."}` — the service **never fabricates** a
verdict.

## Proven live

`confirm_playground()` starts the real server on an ephemeral port, issues real
HTTP requests over a socket, and checks that a div-by-zero translation is flagged
on the UB input `["10","0"]` and **not** on the safe input `["10","2"]`, and that
the rendered page advertises every supported language pair — end to end through
the network stack.
