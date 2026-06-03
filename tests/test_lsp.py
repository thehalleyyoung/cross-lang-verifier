from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict

from src.ub_oracle.lsp import (
    LSP_DIAGNOSTIC_WARNING,
    diagnostics_for_units,
    path_to_uri,
    read_message,
    write_message,
)
from src.ub_oracle.reexec import ToolchainStatus


ROOT = Path(__file__).resolve().parents[1]
NO_TOOLCHAIN = ToolchainStatus(
    cc=None,
    ubsan=False,
    targets=(("rust", None),),
)


def _write_fixture(tmp_path: Path) -> Path:
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "overflow.c").write_text(
        "int f(int x) {\n"
        "  return x + 1;\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "c" / "safe.c").write_text(
        "int g(int x) {\n"
        "  return x + 0;\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "units_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "name": "lsp_signed_overflow",
                        "kind": "binop_const",
                        "op": "add",
                        "const": 1,
                        "width": 32,
                        "var": "x",
                        "signed": True,
                        "probe": "signed_overflow",
                        "source_lang": "c",
                        "target_lang": "rust",
                        "source_file": "c/overflow.c",
                        "line": 2,
                    },
                    {
                        "name": "lsp_safe_control",
                        "kind": "binop_const",
                        "op": "add",
                        "const": 0,
                        "width": 32,
                        "var": "x",
                        "signed": True,
                        "probe": "signed_overflow",
                        "source_lang": "c",
                        "target_lang": "rust",
                        "source_file": "c/safe.c",
                        "line": 2,
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_lsp_diagnostics_mirror_verifier_candidates(tmp_path: Path):
    manifest = _write_fixture(tmp_path)
    units = json.loads(manifest.read_text(encoding="utf-8"))["units"]

    batch = diagnostics_for_units(
        units,
        manifest_path=manifest,
        confirm=False,
        status=NO_TOOLCHAIN,
        document_texts={
            os.path.realpath(tmp_path / "c" / "overflow.c"):
                (tmp_path / "c" / "overflow.c").read_text(encoding="utf-8"),
            os.path.realpath(tmp_path / "c" / "safe.c"):
                (tmp_path / "c" / "safe.c").read_text(encoding="utf-8"),
        },
    )

    overflow_diags = batch.by_path[os.path.realpath(tmp_path / "c" / "overflow.c")]
    safe_diags = batch.by_path[os.path.realpath(tmp_path / "c" / "safe.c")]
    assert len(overflow_diags) == 1
    assert overflow_diags[0]["severity"] == LSP_DIAGNOSTIC_WARNING
    assert overflow_diags[0]["data"]["verdict"] == "candidate"
    assert "NOT re-executed" in overflow_diags[0]["message"]
    assert safe_diags == []


def _send(proc: subprocess.Popen, payload: Dict) -> None:
    assert proc.stdin is not None
    write_message(proc.stdin, payload)


def _read_until(proc: subprocess.Popen, predicate: Callable[[Dict], bool]) -> Dict:
    assert proc.stdout is not None
    deadline = 5.0
    while True:
        readable, _, _ = select.select([proc.stdout], [], [], deadline)
        assert readable, "timed out waiting for LSP message"
        message = read_message(proc.stdout)
        assert message is not None
        if predicate(message):
            return message


def test_lsp_stdio_session_publishes_diagnostics_for_opened_files(tmp_path: Path):
    manifest = _write_fixture(tmp_path)
    overflow = tmp_path / "c" / "overflow.c"
    safe = tmp_path / "c" / "safe.c"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["XLEV_NO_TOOLCHAIN"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "ub_oracle.lsp", "--stdio"],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "rootUri": path_to_uri(tmp_path),
                    "initializationOptions": {
                        "manifest": str(manifest),
                        "confirm": False,
                    },
                },
            },
        )
        init = _read_until(proc, lambda msg: msg.get("id") == 1)
        assert init["result"]["capabilities"]["textDocumentSync"] == 1

        _send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": path_to_uri(overflow),
                        "languageId": "c",
                        "version": 1,
                        "text": overflow.read_text(encoding="utf-8"),
                    }
                },
            },
        )
        overflow_msg = _read_until(
            proc,
            lambda msg: msg.get("method") == "textDocument/publishDiagnostics"
            and msg["params"]["uri"] == path_to_uri(overflow),
        )
        overflow_diags = overflow_msg["params"]["diagnostics"]
        assert len(overflow_diags) == 1
        assert overflow_diags[0]["severity"] == LSP_DIAGNOSTIC_WARNING
        assert overflow_diags[0]["data"]["verdict"] == "candidate"

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": path_to_uri(safe),
                        "languageId": "c",
                        "version": 1,
                        "text": safe.read_text(encoding="utf-8"),
                    }
                },
            },
        )
        safe_msg = _read_until(
            proc,
            lambda msg: msg.get("method") == "textDocument/publishDiagnostics"
            and msg["params"]["uri"] == path_to_uri(safe),
        )
        assert safe_msg["params"]["diagnostics"] == []

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "shutdown"})
        shutdown = _read_until(proc, lambda msg: msg.get("id") == 2)
        assert shutdown["result"] is None
        _send(proc, {"jsonrpc": "2.0", "method": "exit"})
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
