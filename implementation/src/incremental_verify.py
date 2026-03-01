"""Incremental verification with file watching and CI integration.

Provides watch mode for development, CI-optimised re-verification of
only changed function pairs, git diff integration, and result caching.
"""

import os
import re
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Callable
from pathlib import Path

from .api import verify_equivalence, VerificationResult
from .project_scanner import (
    scan_project, match_functions, extract_c_functions, extract_rust_functions,
    CFunction, RustFunction, FunctionMatch, FunctionVerification,
    ProjectScanResult,
)


# ---------------------------------------------------------------------------
# Verification cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    c_hash: str
    rust_hash: str
    result_equivalent: bool
    result_json: str
    timestamp: float


class VerificationCache:
    """Persistent cache of verification results keyed by content hashes."""

    def __init__(self, cache_dir: str = ".xequiv-cache"):
        self.cache_dir = cache_dir
        self._cache: Dict[str, CacheEntry] = {}
        self._load()

    def _cache_file(self) -> str:
        return os.path.join(self.cache_dir, "verification_cache.json")

    def _load(self) -> None:
        path = self._cache_file()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for key, entry in data.items():
                self._cache[key] = CacheEntry(**entry)

    def _save(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        data = {k: vars(v) for k, v in self._cache.items()}
        with open(self._cache_file(), "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _content_hash(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _make_key(self, c_code: str, rust_code: str) -> str:
        return f"{self._content_hash(c_code)}:{self._content_hash(rust_code)}"

    def lookup(self, c_code: str, rust_code: str) -> Optional[VerificationResult]:
        key = self._make_key(c_code, rust_code)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (entry.c_hash != self._content_hash(c_code) or
                entry.rust_hash != self._content_hash(rust_code)):
            del self._cache[key]
            return None
        data = json.loads(entry.result_json)
        return VerificationResult(
            equivalent=data["equivalent"],
            confidence=data.get("confidence", 1.0),
            duration_ms=data.get("duration_ms", 0.0),
            method=data.get("method", "cached"),
        )

    def store(self, c_code: str, rust_code: str, result: VerificationResult) -> None:
        key = self._make_key(c_code, rust_code)
        self._cache[key] = CacheEntry(
            c_hash=self._content_hash(c_code),
            rust_hash=self._content_hash(rust_code),
            result_equivalent=result.equivalent,
            result_json=json.dumps({
                "equivalent": result.equivalent,
                "confidence": result.confidence,
                "duration_ms": result.duration_ms,
                "method": result.method,
            }),
            timestamp=time.time(),
        )
        self._save()

    def invalidate(self, c_code: str, rust_code: str) -> None:
        key = self._make_key(c_code, rust_code)
        self._cache.pop(key, None)
        self._save()

    def clear(self) -> None:
        self._cache.clear()
        self._save()

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Git diff integration
# ---------------------------------------------------------------------------

def _run_git(args: List[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, cwd=cwd, timeout=30
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def git_changed_files(directory: str, base_ref: str = "HEAD~1",
                      head_ref: str = "HEAD") -> List[str]:
    """Return list of files changed between two git refs."""
    output = _run_git(["diff", "--name-only", base_ref, head_ref], cwd=directory)
    if not output:
        return []
    return [os.path.join(directory, f) for f in output.splitlines() if f.strip()]


def git_uncommitted_changes(directory: str) -> List[str]:
    """Return list of files with uncommitted changes (staged + unstaged)."""
    staged = _run_git(["diff", "--name-only", "--cached"], cwd=directory)
    unstaged = _run_git(["diff", "--name-only"], cwd=directory)
    untracked = _run_git(["ls-files", "--others", "--exclude-standard"], cwd=directory)
    files: Set[str] = set()
    for output in (staged, unstaged, untracked):
        if output:
            files.update(
                os.path.join(directory, f.strip())
                for f in output.splitlines() if f.strip()
            )
    return sorted(files)


def identify_affected_pairs(changed_files: List[str],
                            scan: ProjectScanResult) -> List[FunctionMatch]:
    """Given a list of changed files, find which function pairs need re-verification."""
    changed_set = set(os.path.abspath(f) for f in changed_files)
    affected: List[FunctionMatch] = []
    for match in scan.matches:
        c_abs = os.path.abspath(match.c_function.file_path)
        r_abs = os.path.abspath(match.rust_function.file_path)
        if c_abs in changed_set or r_abs in changed_set:
            affected.append(match)
    return affected


# ---------------------------------------------------------------------------
# CI integration
# ---------------------------------------------------------------------------

@dataclass
class CICheckResult:
    total_checked: int
    passed: int
    failed: int
    cached: int
    errors: int
    verifications: List[FunctionVerification]
    duration_ms: float = 0.0
    exit_code: int = 0

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0


def ci_check(c_dir: str, rust_dir: str,
             changed_files: Optional[List[str]] = None,
             base_ref: str = "HEAD~1",
             timeout_s: float = 120.0,
             method: str = "hybrid",
             cache_dir: str = ".xequiv-cache") -> CICheckResult:
    """Only re-verify changed function pairs. Ideal for CI pipelines.

    If changed_files is None, uses git diff to detect changes automatically.
    """
    start = time.time()
    cache = VerificationCache(cache_dir)
    scan = scan_project(c_dir, rust_dir)

    if changed_files is None:
        c_changed = git_changed_files(c_dir, base_ref=base_ref)
        r_changed = git_changed_files(rust_dir, base_ref=base_ref)
        changed_files = c_changed + r_changed

    if changed_files:
        pairs_to_check = identify_affected_pairs(changed_files, scan)
    else:
        pairs_to_check = scan.matches

    verifications: List[FunctionVerification] = []
    passed = failed = cached_count = errors = 0

    for match in pairs_to_check:
        fv = FunctionVerification(match=match, status="running")
        cached = cache.lookup(match.c_function.raw_source,
                              match.rust_function.raw_source)
        if cached is not None:
            fv.result = cached
            fv.status = "passed" if cached.equivalent else "failed"
            cached_count += 1
        else:
            try:
                result = verify_equivalence(
                    match.c_function.raw_source,
                    match.rust_function.raw_source,
                    timeout_s=timeout_s,
                    method=method,
                )
                fv.result = result
                fv.status = "passed" if result.equivalent else "failed"
                cache.store(match.c_function.raw_source,
                            match.rust_function.raw_source, result)
            except Exception as exc:
                fv.status = "error"
                fv.error_message = str(exc)
                errors += 1

        if fv.status == "passed":
            passed += 1
        elif fv.status == "failed":
            failed += 1
        verifications.append(fv)

    duration = (time.time() - start) * 1000
    return CICheckResult(
        total_checked=len(pairs_to_check),
        passed=passed,
        failed=failed,
        cached=cached_count,
        errors=errors,
        verifications=verifications,
        duration_ms=duration,
        exit_code=1 if failed > 0 or errors > 0 else 0,
    )


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

@dataclass
class FileChangeEvent:
    path: str
    event_type: str  # "modified", "created", "deleted"
    timestamp: float


class _SimplePoller:
    """Poll-based file watcher (no external dependencies)."""

    def __init__(self, directories: List[str], extensions: Set[str],
                 poll_interval: float = 1.0):
        self.directories = directories
        self.extensions = extensions
        self.poll_interval = poll_interval
        self._snapshot: Dict[str, float] = {}
        self._take_snapshot()

    def _collect_files(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for d in self.directories:
            for root, dirs, files in os.walk(d):
                dirs[:] = [x for x in dirs if x not in ("target", ".git", "build")]
                for fname in files:
                    if any(fname.endswith(ext) for ext in self.extensions):
                        fpath = os.path.join(root, fname)
                        try:
                            result[fpath] = os.path.getmtime(fpath)
                        except OSError:
                            pass
        return result

    def _take_snapshot(self) -> None:
        self._snapshot = self._collect_files()

    def poll(self) -> List[FileChangeEvent]:
        current = self._collect_files()
        events: List[FileChangeEvent] = []
        now = time.time()
        for fpath, mtime in current.items():
            if fpath not in self._snapshot:
                events.append(FileChangeEvent(fpath, "created", now))
            elif mtime > self._snapshot[fpath]:
                events.append(FileChangeEvent(fpath, "modified", now))
        for fpath in self._snapshot:
            if fpath not in current:
                events.append(FileChangeEvent(fpath, "deleted", now))
        self._snapshot = current
        return events


def watch(c_dir: str, rust_dir: str,
          on_change: Optional[Callable[[List[FileChangeEvent], CICheckResult], None]] = None,
          poll_interval: float = 2.0,
          timeout_s: float = 60.0,
          method: str = "hybrid",
          cache_dir: str = ".xequiv-cache",
          max_iterations: Optional[int] = None) -> None:
    """Watch C and Rust directories for changes and re-verify affected pairs.

    Args:
        c_dir: Path to C source directory
        rust_dir: Path to Rust source directory
        on_change: Callback invoked with change events and verification result
        poll_interval: Seconds between polls
        timeout_s: Per-pair verification timeout
        method: Verification method
        cache_dir: Cache directory path
        max_iterations: Stop after N iterations (None = run forever)
    """
    poller = _SimplePoller(
        directories=[c_dir, rust_dir],
        extensions={".c", ".h", ".rs"},
        poll_interval=poll_interval,
    )

    iteration = 0
    print(f"[xequiv] Watching {c_dir} and {rust_dir} for changes...")

    while max_iterations is None or iteration < max_iterations:
        time.sleep(poll_interval)
        events = poller.poll()
        if not events:
            continue

        changed_paths = [e.path for e in events]
        print(f"[xequiv] Detected {len(events)} change(s), re-verifying...")

        result = ci_check(
            c_dir, rust_dir,
            changed_files=changed_paths,
            timeout_s=timeout_s,
            method=method,
            cache_dir=cache_dir,
        )

        status_str = "✅ PASS" if result.success else "❌ FAIL"
        print(f"[xequiv] {status_str}: {result.passed} passed, "
              f"{result.failed} failed, {result.cached} cached "
              f"({result.duration_ms:.0f}ms)")

        if on_change:
            on_change(events, result)

        iteration += 1


# ---------------------------------------------------------------------------
# Convenience: run from CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for incremental verification."""
    import argparse
    parser = argparse.ArgumentParser(description="XEquiv incremental verification")
    sub = parser.add_subparsers(dest="command")

    ci_parser = sub.add_parser("ci", help="CI check — verify changed pairs")
    ci_parser.add_argument("c_dir", help="C source directory")
    ci_parser.add_argument("rust_dir", help="Rust source directory")
    ci_parser.add_argument("--base-ref", default="HEAD~1")
    ci_parser.add_argument("--timeout", type=float, default=120.0)
    ci_parser.add_argument("--method", default="hybrid")

    watch_parser = sub.add_parser("watch", help="Watch mode")
    watch_parser.add_argument("c_dir", help="C source directory")
    watch_parser.add_argument("rust_dir", help="Rust source directory")
    watch_parser.add_argument("--poll-interval", type=float, default=2.0)
    watch_parser.add_argument("--timeout", type=float, default=60.0)

    args = parser.parse_args()

    if args.command == "ci":
        result = ci_check(args.c_dir, args.rust_dir,
                          base_ref=args.base_ref,
                          timeout_s=args.timeout,
                          method=args.method)
        for fv in result.verifications:
            icon = "✅" if fv.status == "passed" else "❌" if fv.status == "failed" else "⚠️"
            print(f"  {icon} {fv.match.c_function.name} ↔ {fv.match.rust_function.name}: {fv.status}")
        print(f"\n{'PASS' if result.success else 'FAIL'}: "
              f"{result.passed} passed, {result.failed} failed, "
              f"{result.cached} cached ({result.duration_ms:.0f}ms)")
        raise SystemExit(result.exit_code)

    elif args.command == "watch":
        watch(args.c_dir, args.rust_dir,
              poll_interval=args.poll_interval,
              timeout_s=args.timeout)


if __name__ == "__main__":
    main()
