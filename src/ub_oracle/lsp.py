"""Minimal Language Server Protocol diagnostics for verifier manifests.

The server is intentionally a thin editor-facing adapter over the same manifest
and ``verify_unit`` path used by the CLI, pre-commit hook, and GitHub Action. It
does not parse unsaved editor buffers; it publishes diagnostics for units whose
manifest declares a physical ``source_file`` or ``target_file``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from . import cli
from .report import pair_of
from .reexec import ToolchainStatus, toolchain_available
from .verify import VerifyReport, VerifyVerdict, verify_unit

JSON = Dict[str, object]

LSP_DIAGNOSTIC_ERROR = 1
LSP_DIAGNOSTIC_WARNING = 2
TEXT_DOCUMENT_SYNC_FULL = 1

_DEFAULT_MANIFEST_NAMES = (
    "units_manifest.json",
    "cross-lang-verify.json",
    ".cross-lang-verify.json",
)
_ACTIONABLE = {
    VerifyVerdict.DIVERGENT,
    VerifyVerdict.CANDIDATE,
}


@dataclass(frozen=True)
class DiagnosticBatch:
    """Diagnostics grouped by canonical absolute file path."""

    by_path: Dict[str, List[JSON]]
    manifest: Optional[str]


def _canonical(path: Path) -> str:
    return os.path.realpath(os.fspath(path))


def path_to_uri(path: Path) -> str:
    return path.resolve(strict=False).as_uri()


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"only file:// URIs are supported, got {uri!r}")
    if parsed.netloc and parsed.netloc not in ("localhost", "127.0.0.1"):
        raise ValueError(f"remote file URI is not supported: {uri!r}")
    return Path(url2pathname(unquote(parsed.path)))


def _unit_label(unit: Dict, index: int) -> str:
    return str(unit.get("name") or unit.get("id") or f"unit[{index}]")


def _report_class(report: VerifyReport) -> str:
    if report.divergence is not None:
        return report.divergence.divergence_class
    for result in report.oracle_results:
        if result.counterexample is not None:
            return result.divergence_class
    return str(report.unit.get("probe") or "cross-lang-divergence")


def _line_col(unit: Dict) -> Tuple[int, int]:
    try:
        line = int(unit.get("line", 1))
    except (TypeError, ValueError):
        line = 1
    try:
        column = int(unit.get("column", 1))
    except (TypeError, ValueError):
        column = 1
    return max(0, line - 1), max(0, column - 1)


def _clamp_range(unit: Dict, text: Optional[str]) -> JSON:
    line, column = _line_col(unit)
    if text is not None:
        lines = text.splitlines() or [""]
        line = min(line, len(lines) - 1)
        column = min(column, len(lines[line]))
        end_column = min(column + 1, max(column + 1, len(lines[line])))
    else:
        end_column = column + 1
    return {
        "start": {"line": line, "character": column},
        "end": {"line": line, "character": end_column},
    }


def diagnostic_from_report(
    report: VerifyReport,
    *,
    index: int = 0,
    document_text: Optional[str] = None,
) -> Optional[JSON]:
    """Convert an actionable verifier report into one LSP diagnostic.

    Mirrors SARIF's policy: only confirmed divergences and symbolic candidates are
    inline diagnostics by default. Clean, unknown, and not-covered reports do not
    mark a file as problematic.
    """

    if report.verdict not in _ACTIONABLE:
        return None
    severity = (
        LSP_DIAGNOSTIC_ERROR
        if report.verdict is VerifyVerdict.DIVERGENT
        else LSP_DIAGNOSTIC_WARNING
    )
    cls = _report_class(report)
    unit = report.unit
    return {
        "range": _clamp_range(unit, document_text),
        "severity": severity,
        "code": cls,
        "source": "cross-lang-verifier",
        "message": report.banner(),
        "data": {
            "verdict": report.verdict.value,
            "unit": _unit_label(unit, index),
            "pair": pair_of(report),
            "divergenceClass": cls,
        },
    }


def _unit_files(unit: Dict, base_dir: Path) -> List[str]:
    files: List[str] = []
    for key in ("source_file", "target_file"):
        value = unit.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = base_dir / path
        files.append(_canonical(path))
    return files


def diagnostics_for_units(
    units: Iterable[Dict],
    *,
    manifest_path: Path,
    confirm: bool = False,
    status: Optional[ToolchainStatus] = None,
    document_texts: Optional[Dict[str, str]] = None,
) -> DiagnosticBatch:
    """Verify manifest units and group actionable diagnostics by file path."""

    base_dir = manifest_path.resolve(strict=False).parent
    status = status or toolchain_available()
    by_path: Dict[str, List[JSON]] = {}
    document_texts = document_texts or {}

    for index, unit in enumerate(units):
        report = verify_unit(unit, confirm=confirm, status=status)
        files = _unit_files(unit, base_dir)
        if not files:
            continue
        for file_path in files:
            diag = diagnostic_from_report(
                report,
                index=index,
                document_text=document_texts.get(file_path),
            )
            by_path.setdefault(file_path, [])
            if diag is not None:
                by_path[file_path].append(diag)
    return DiagnosticBatch(by_path=by_path, manifest=_canonical(manifest_path))


def _find_manifest(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    current = root.resolve(strict=False)
    while True:
        for name in _DEFAULT_MANIFEST_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _header_value(line: bytes, name: bytes) -> Optional[str]:
    if not line.lower().startswith(name.lower() + b":"):
        return None
    return line.split(b":", 1)[1].strip().decode("ascii")


def read_message(stream) -> Optional[JSON]:
    """Read one LSP-framed JSON message from a binary stream."""

    content_length: Optional[int] = None
    while True:
        line = stream.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        value = _header_value(line.rstrip(b"\r\n"), b"Content-Length")
        if value is not None:
            content_length = int(value)
    if content_length is None:
        raise ValueError("missing Content-Length header")
    body = stream.read(content_length)
    if len(body) != content_length:
        raise ValueError("truncated LSP message body")
    message = json.loads(body.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("JSON-RPC message must be an object")
    return message


def write_message(stream, payload: JSON) -> None:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


class CrossLangVerifierLspServer:
    """Synchronous stdio LSP server for cross-language verifier diagnostics."""

    def __init__(
        self,
        *,
        manifest: Optional[Path] = None,
        confirm: bool = False,
        instream=None,
        outstream=None,
    ) -> None:
        self.manifest = manifest
        self.confirm = confirm
        self.instream = instream or sys.stdin.buffer
        self.outstream = outstream or sys.stdout.buffer
        self._shutdown_requested = False
        self._open_docs: Dict[str, str] = {}
        self._status = toolchain_available()

    def serve(self) -> int:
        while True:
            message = read_message(self.instream)
            if message is None:
                return 0
            should_exit = self._handle_message(message)
            if should_exit:
                return 0

    def _send(self, payload: JSON) -> None:
        write_message(self.outstream, payload)

    def _respond(self, request_id: object, result: object = None) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _error(self, request_id: object, code: int, message: str) -> None:
        self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        })

    def _notify(self, method: str, params: JSON) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _log(self, message: str, *, level: int = 3) -> None:
        self._notify("window/logMessage", {"type": level, "message": message})

    def _handle_message(self, message: JSON) -> bool:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}

        if method == "initialize":
            self._initialize(params)  # type: ignore[arg-type]
            self._respond(request_id, {
                "capabilities": {
                    "textDocumentSync": TEXT_DOCUMENT_SYNC_FULL,
                    "executeCommandProvider": {
                        "commands": ["cross-lang-verifier.reverify"]
                    },
                },
                "serverInfo": {
                    "name": "cross-lang-verifier-lsp",
                    "version": "0.3.0",
                },
            })
            return False
        if method == "initialized":
            if self.manifest is not None:
                self._log(f"using manifest {self.manifest}")
            else:
                self._log("no verifier manifest configured", level=2)
            return False
        if method == "shutdown":
            self._shutdown_requested = True
            self._respond(request_id, None)
            return False
        if method == "exit":
            return True
        if method == "textDocument/didOpen":
            self._did_open(params)  # type: ignore[arg-type]
            return False
        if method == "textDocument/didChange":
            self._did_change(params)  # type: ignore[arg-type]
            return False
        if method == "textDocument/didSave":
            self._did_save(params)  # type: ignore[arg-type]
            return False
        if method == "workspace/executeCommand":
            result = self._execute_command(params)  # type: ignore[arg-type]
            self._respond(request_id, result)
            return False

        if request_id is not None:
            self._error(request_id, -32601, f"method not found: {method}")
        return False

    def _initialize(self, params: JSON) -> None:
        options = params.get("initializationOptions")
        if not isinstance(options, dict):
            options = {}
        manifest_option = (
            options.get("manifest")
            or options.get("manifestPath")
            or options.get("units")
        )
        if manifest_option:
            self.manifest = Path(str(manifest_option)).expanduser()
        else:
            root_uri = params.get("rootUri")
            root = uri_to_path(str(root_uri)) if isinstance(root_uri, str) else None
            self.manifest = self.manifest or _find_manifest(root)
        if "confirm" in options:
            self.confirm = bool(options["confirm"])

    def _did_open(self, params: JSON) -> None:
        doc = params.get("textDocument")
        if not isinstance(doc, dict):
            return
        uri = str(doc.get("uri") or "")
        self._open_docs[uri] = str(doc.get("text") or "")
        self._publish_for_uri(uri)

    def _did_change(self, params: JSON) -> None:
        doc = params.get("textDocument")
        changes = params.get("contentChanges")
        if not isinstance(doc, dict) or not isinstance(changes, list):
            return
        uri = str(doc.get("uri") or "")
        if changes and isinstance(changes[-1], dict):
            self._open_docs[uri] = str(changes[-1].get("text") or "")
        self._publish_for_uri(uri)

    def _did_save(self, params: JSON) -> None:
        doc = params.get("textDocument")
        if not isinstance(doc, dict):
            return
        self._publish_for_uri(str(doc.get("uri") or ""))

    def _execute_command(self, params: JSON) -> JSON:
        command = params.get("command")
        if command != "cross-lang-verifier.reverify":
            return {"reverified": 0, "reason": "unsupported command"}
        count = 0
        for uri in list(self._open_docs):
            self._publish_for_uri(uri)
            count += 1
        return {"reverified": count}

    def _document_texts_by_path(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        for uri, text in self._open_docs.items():
            try:
                docs[_canonical(uri_to_path(uri))] = text
            except ValueError:
                continue
        return docs

    def _publish_for_uri(self, uri: str) -> None:
        if not uri:
            return
        try:
            opened_path = _canonical(uri_to_path(uri))
        except ValueError as exc:
            self._log(str(exc), level=2)
            return
        diagnostics: List[JSON] = []
        if self.manifest is not None:
            try:
                units = cli._load_units(str(self.manifest))
                batch = diagnostics_for_units(
                    units,
                    manifest_path=self.manifest,
                    confirm=self.confirm,
                    status=self._status,
                    document_texts=self._document_texts_by_path(),
                )
                diagnostics = batch.by_path.get(opened_path, [])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                diagnostics = [{
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                    "severity": LSP_DIAGNOSTIC_ERROR,
                    "code": "manifest-error",
                    "source": "cross-lang-verifier",
                    "message": f"cross-lang-verifier manifest error: {exc}",
                }]

        self._notify("textDocument/publishDiagnostics", {
            "uri": uri,
            "diagnostics": diagnostics,
        })


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cross-lang-verify-lsp",
        description=(
            "Language Server Protocol diagnostics for cross-lang-verifier "
            "unit manifests."
        ),
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="serve LSP over stdin/stdout (default)",
    )
    parser.add_argument(
        "--manifest",
        metavar="MANIFEST.json",
        help="units manifest to use before initializationOptions are received",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="run real compiler confirmation; default is fast --no-confirm mode",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    manifest = Path(args.manifest).expanduser() if args.manifest else None
    server = CrossLangVerifierLspServer(manifest=manifest, confirm=args.confirm)
    return server.serve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
