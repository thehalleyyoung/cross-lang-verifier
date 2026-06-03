from __future__ import annotations

import subprocess

import pytest

from src.ub_oracle import continuous_corpus_ci as cci
from src.ub_oracle import corpus_datasheet


def test_continuous_corpus_ci_plan_covers_every_datasheet_corpus_and_command():
    datasheet = corpus_datasheet.results_document()
    plan = cci.plan_document(datasheet)

    assert plan["schema"] == cci.SCHEMA_VERSION
    assert plan["source_datasheet_hash"] == datasheet["content_hash"]
    assert plan["n_corpora"] == datasheet["n_records"]
    assert len(plan["verdict_layer_hash"]) == 64

    expected_corpora = {record["corpus_id"] for record in datasheet["records"]}
    planned_corpora = {record["corpus_id"] for record in plan["records"]}
    covered_corpora = {cid for command in plan["commands"] for cid in command["corpus_ids"]}
    assert planned_corpora == expected_corpora
    assert covered_corpora == expected_corpora

    expected_commands = {
        command
        for record in datasheet["records"]
        for command in record["validation_commands"]
    }
    assert {command["command"] for command in plan["commands"]} == expected_commands
    assert "make negative-corpus-check" in expected_commands


def test_continuous_corpus_ci_artifacts_are_byte_fresh():
    ok, detail = cci.check_all()
    assert ok, detail

    on_disk = cci.load_results()
    assert on_disk == cci.plan_document()
    assert cci.DOC_PATH.read_text(encoding="utf-8") == cci.markdown_document()


def test_continuous_corpus_ci_run_report_uses_injected_runner(tmp_path):
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="OK: negative corpus stayed fresh\n",
            stderr="",
        )

    report = cci.run_nightly(
        commands=("make negative-corpus-check",),
        command_timeout=3,
        runner=fake_runner,
        report_path=tmp_path / "report.json",
    )

    assert report["ok"]
    assert report["alert_count"] == 0
    assert report["plan"]["n_commands"] == 1
    assert report["outcomes"][0]["command"] == "make negative-corpus-check"
    assert report["outcomes"][0]["outcome"] == "pass"
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["make", "negative-corpus-check"]
    assert kwargs["cwd"] == str(cci.RESULTS_PATH.parents[2])
    assert isinstance(kwargs["env"], dict)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 3
    assert kwargs["check"] is False
    assert (tmp_path / "report.json").exists()


def test_continuous_corpus_ci_classifies_drift_and_timeouts():
    def drift_runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="experiments/foo.json does not match regenerated verdict hash\n",
            stderr="",
        )

    drift_report = cci.run_nightly(
        commands=("make negative-corpus-check",),
        command_timeout=3,
        runner=drift_runner,
    )
    assert not drift_report["ok"]
    assert drift_report["alert_count"] == 1
    assert drift_report["outcomes"][0]["alert_kind"] == "verdict-drift"

    def timeout_runner(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, timeout=3, output="partial", stderr="still running")

    timeout_report = cci.run_nightly(
        commands=("make negative-corpus-check",),
        command_timeout=3,
        runner=timeout_runner,
    )
    assert not timeout_report["ok"]
    assert timeout_report["outcomes"][0]["alert_kind"] == "infrastructure-timeout"


def test_continuous_corpus_ci_rejects_unknown_command():
    with pytest.raises(ValueError):
        cci.run_nightly(commands=("make does-not-exist",), runner=lambda *_args, **_kwargs: None)
