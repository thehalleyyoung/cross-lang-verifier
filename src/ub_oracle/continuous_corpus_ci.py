"""Step 167 -- continuous corpus drift CI.

The committed artifact is a deterministic plan derived from the corpus
datasheet: which corpus surfaces exist, which validation commands protect them,
and which stable content hashes those commands are expected to preserve.  The
nightly runtime report is intentionally separate and untracked; it executes the
existing corpus ``*-check`` gates and fails loudly when any gate reports drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import corpus_datasheet

SCHEMA_VERSION = "continuous-corpus-ci/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "continuous_corpus_ci"
RESULTS_PATH = EXPERIMENT_DIR / "continuous_corpus_ci.json"
DOC_PATH = _ROOT / "docs" / "continuous_corpus_ci.md"
DEFAULT_REPORT_PATH = EXPERIMENT_DIR / "run_reports" / "latest_report.json"

DEFAULT_COMMAND_TIMEOUT_SECONDS = 900
_TAIL_LIMIT = 4000


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _tail(text: str, *, limit: int = _TAIL_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _command_id(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()[:16]


def _normalize_command(argv: Sequence[str]) -> List[str]:
    if not argv:
        return []
    if argv[0] in {"python", "python3"}:
        return [sys.executable, *argv[1:]]
    return list(argv)


def _record_summary(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        "corpus_id": record["corpus_id"],
        "content_hash": record["content_hash"],
        "population_size": record["population_size"],
        "unit": record["unit"],
        "label_balance": record["label_balance"],
        "language_pairs": record["language_pairs"],
        "validation_commands": record["validation_commands"],
    }


def _command_records(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_command: Dict[str, List[str]] = {}
    for record in records:
        for command in record["validation_commands"]:
            by_command.setdefault(str(command), []).append(str(record["corpus_id"]))

    commands: List[Dict[str, object]] = []
    for command in sorted(by_command):
        commands.append(
            {
                "command_id": _command_id(command),
                "command": command,
                "corpus_ids": sorted(set(by_command[command])),
                "timeout_seconds": DEFAULT_COMMAND_TIMEOUT_SECONDS,
            }
        )
    return commands


def plan_document(datasheet_doc: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Build the deterministic nightly corpus-CI plan."""

    doc = corpus_datasheet.results_document() if datasheet_doc is None else datasheet_doc
    records = [_record_summary(record) for record in doc["records"]]
    commands = _command_records(doc["records"])
    stable = {
        "schema": SCHEMA_VERSION,
        "source_datasheet_schema": doc["schema"],
        "source_datasheet_hash": doc["content_hash"],
        "records": records,
        "commands": commands,
    }
    return {
        **stable,
        "verdict_layer_hash": _sha256_obj(stable),
        "n_corpora": len(records),
        "n_commands": len(commands),
        "purpose": (
            "Nightly re-verification of every corpus surface in the generated "
            "datasheet. Each command is an existing corpus check whose nonzero "
            "exit is treated as a drift/regression alert and uploaded with an "
            "ephemeral runtime report."
        ),
    }


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = json.dumps(plan_document(), indent=2, sort_keys=True) + "\n"
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return False, f"{path} does not match regenerated continuous corpus CI plan"
    doc = json.loads(actual)
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    if not doc.get("commands"):
        return False, "continuous corpus CI plan has no commands"
    covered = {cid for command in doc["commands"] for cid in command["corpus_ids"]}
    expected_corpora = {record["corpus_id"] for record in doc["records"]}
    if covered != expected_corpora:
        return False, f"command coverage mismatch: {sorted(expected_corpora - covered)}"
    return True, "OK"


def _fmt_mapping(mapping: Mapping[str, object]) -> str:
    if not mapping:
        return "(none)"
    return ", ".join(f"`{key}`={mapping[key]}" for key in sorted(mapping))


def markdown_document(doc: Optional[Mapping[str, object]] = None) -> str:
    doc = plan_document() if doc is None else doc
    lines = [
        "# Continuous corpus CI",
        "",
        "Generated by `ub_oracle.continuous_corpus_ci`; do not hand-edit. "
        "Run `make continuous-corpus-ci-check` to assert the committed plan is fresh.",
        "",
        "The nightly workflow re-runs the validation commands for every corpus "
        "surface listed in the generated corpus datasheet. Runtime reports are "
        "uploaded as workflow artifacts and intentionally excluded from git; the "
        "committed file below is only the deterministic plan and expected verdict layer.",
        "",
        f"**Schema:** `{doc['schema']}`",
        f"**Source datasheet hash:** `{doc['source_datasheet_hash']}`",
        f"**Verdict-layer hash:** `{doc['verdict_layer_hash']}`",
        f"**Corpora:** {doc['n_corpora']}",
        f"**Validation commands:** {doc['n_commands']}",
        "",
        "## Nightly validation commands",
        "",
        "| command | corpora protected | timeout |",
        "|---|---:|---:|",
    ]
    for command in doc["commands"]:
        corpora = ", ".join(f"`{cid}`" for cid in command["corpus_ids"])
        lines.append(
            f"| `{command['command']}` | {corpora} | "
            f"{command['timeout_seconds']}s |"
        )

    lines.extend(
        [
            "",
            "## Corpus verdict layer",
            "",
            "| corpus | population | labels | hash |",
            "|---|---:|---|---|",
        ]
    )
    for record in doc["records"]:
        lines.append(
            f"| `{record['corpus_id']}` | {record['population_size']} "
            f"{record['unit']} | {_fmt_mapping(record['label_balance'])} | "
            f"`{record['content_hash']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(path: Path = DOC_PATH) -> None:
    path.write_text(markdown_document(), encoding="utf-8")


def check_markdown(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = markdown_document()
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated continuous corpus CI docs"
    return True, "OK"


def write_all() -> None:
    write_results()
    write_markdown()


def check_all() -> Tuple[bool, str]:
    ok, detail = check_results()
    if not ok:
        return ok, detail
    return check_markdown()


@dataclass(frozen=True)
class CommandOutcome:
    command_id: str
    command: str
    corpus_ids: Tuple[str, ...]
    outcome: str
    alert_kind: str
    returncode: int
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.outcome == "pass"

    def to_dict(self) -> Dict[str, object]:
        return {
            "command_id": self.command_id,
            "command": self.command,
            "corpus_ids": list(self.corpus_ids),
            "outcome": self.outcome,
            "alert_kind": self.alert_kind,
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


Runner = Callable[..., subprocess.CompletedProcess]


def classify_failure(command: str, returncode: int, output: str) -> str:
    lowered = output.lower()
    if returncode == 124 or "timed out" in lowered or "timeout" in lowered:
        return "infrastructure-timeout"
    infra_needles = (
        "command not found",
        "no such file or directory",
        "missing toolchain",
        "needs clang",
        "needs c+ubsan",
        "needs clang+ubsan",
        "could not find",
    )
    if any(needle in lowered for needle in infra_needles):
        return "infrastructure-error"
    drift_needles = (
        "does not match",
        "not fresh",
        "mismatch",
        "drift",
        "hash",
        "changed",
        "regression",
        "failure",
    )
    if any(needle in lowered for needle in drift_needles):
        return "verdict-drift"
    if command.startswith("make ") or "pytest" in command:
        return "corpus-check-failure"
    return "command-failure"


def run_command(
    command_record: Mapping[str, object],
    *,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    runner: Runner = subprocess.run,
    env: Optional[Mapping[str, str]] = None,
) -> CommandOutcome:
    command = str(command_record["command"])
    argv = _normalize_command(shlex.split(command))
    run_env = dict(os.environ if env is None else env)
    started = time.monotonic()
    try:
        proc = runner(
            argv,
            cwd=str(_ROOT),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=int(command_timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return CommandOutcome(
            command_id=str(command_record["command_id"]),
            command=command,
            corpus_ids=tuple(str(cid) for cid in command_record["corpus_ids"]),
            outcome="alert",
            alert_kind="infrastructure-timeout",
            returncode=124,
            elapsed_seconds=round(time.monotonic() - started, 3),
            stdout_tail=_tail(str(stdout)),
            stderr_tail=_tail(str(stderr) + "\ncommand timed out"),
        )

    elapsed = round(time.monotonic() - started, 3)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    returncode = int(proc.returncode)
    if returncode == 0:
        outcome = "pass"
        alert_kind = "none"
    else:
        outcome = "alert"
        alert_kind = classify_failure(command, returncode, stdout + "\n" + stderr)
    return CommandOutcome(
        command_id=str(command_record["command_id"]),
        command=command,
        corpus_ids=tuple(str(cid) for cid in command_record["corpus_ids"]),
        outcome=outcome,
        alert_kind=alert_kind,
        returncode=returncode,
        elapsed_seconds=elapsed,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def _selected_commands(
    plan: Mapping[str, object],
    commands: Optional[Iterable[str]] = None,
) -> List[Mapping[str, object]]:
    command_records = list(plan["commands"])
    if commands is None:
        return command_records
    wanted = set(commands)
    selected = [record for record in command_records if record["command"] in wanted]
    missing = sorted(wanted - {record["command"] for record in selected})
    if missing:
        raise ValueError(f"unknown continuous corpus CI command(s): {missing}")
    return selected


def run_nightly(
    *,
    commands: Optional[Iterable[str]] = None,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    runner: Runner = subprocess.run,
    report_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute the selected corpus checks and return an ephemeral runtime report."""

    plan = plan_document()
    selected = _selected_commands(plan, commands)
    started = time.monotonic()
    outcomes = [
        run_command(
            command,
            command_timeout=command_timeout,
            runner=runner,
        )
        for command in selected
    ]
    alert_count = sum(1 for outcome in outcomes if not outcome.ok)
    report: Dict[str, object] = {
        "schema": SCHEMA_VERSION + "/runtime-report",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ok": alert_count == 0,
        "alert_count": alert_count,
        "total_seconds": round(time.monotonic() - started, 3),
        "plan": {
            "schema": plan["schema"],
            "source_datasheet_hash": plan["source_datasheet_hash"],
            "verdict_layer_hash": plan["verdict_layer_hash"],
            "n_corpora": plan["n_corpora"],
            "n_commands": len(selected),
        },
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "SCHEMA_VERSION",
    "RESULTS_PATH",
    "DOC_PATH",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_COMMAND_TIMEOUT_SECONDS",
    "CommandOutcome",
    "plan_document",
    "write_results",
    "load_results",
    "check_results",
    "markdown_document",
    "write_markdown",
    "check_markdown",
    "write_all",
    "check_all",
    "classify_failure",
    "run_command",
    "run_nightly",
]
