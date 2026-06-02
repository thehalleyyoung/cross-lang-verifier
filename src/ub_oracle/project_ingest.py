"""Step 31 — whole-project ingestion.

Real cross-language migration never happens one file at a time: the source side
is a whole C build tree and the target side is a Cargo workspace (or another
multi-crate build graph). This module ingests *both* build descriptions that the
real toolchains already emit, so the oracle sees the complete symbol surface of a
project rather than a single hand-picked translation unit.

Source side: a Clang ``compile_commands.json`` compilation database (the de-facto
standard emitted by CMake/Bear/etc.). We read every entry, recover its compile
flags and the translation unit's source, and lower each TU through the
:mod:`ir_ingest` clang-AST path into a shared :class:`ProjectModule`. Include
directories from the recorded ``-I`` flags are threaded back into the AST dump so
multi-file projects with headers parse correctly.

Target side: a Cargo workspace, enumerated by ``cargo metadata`` — the workspace
members, their package names, and the source root of every target. This is the
build graph Cargo itself uses, so we discover exactly the crates that will be
compiled, no guessing.

Everything self-confirms against the *real* clang and cargo on a generated
multi-file project: :func:`confirm_compile_db` builds a two-file C project plus a
faithful ``compile_commands.json`` and checks the union of recovered functions;
:func:`confirm_cargo_workspace` builds a two-member Cargo workspace and checks the
members and target source roots that ``cargo metadata`` reports.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ir_ingest as _iri

CLANG = _iri.CLANG
CARGO = "/opt/homebrew/bin/cargo"


# ---------------------------------------------------------------------------
# Shared whole-project model.
# ---------------------------------------------------------------------------
@dataclass
class SourceUnit:
    """One translation unit recovered from a compilation database entry."""
    path: str
    functions: Dict[str, _iri.IRFunction] = field(default_factory=dict)
    include_dirs: Tuple[str, ...] = ()


@dataclass
class CratePackage:
    """One Cargo package/target discovered via ``cargo metadata``."""
    name: str
    target_name: str
    src_path: str
    kind: Tuple[str, ...] = ()


@dataclass
class ProjectModule:
    """The whole-project symbol surface for one side of a migration."""
    units: Dict[str, SourceUnit] = field(default_factory=dict)
    packages: Dict[str, CratePackage] = field(default_factory=dict)

    def all_functions(self) -> Dict[str, _iri.IRFunction]:
        out: Dict[str, _iri.IRFunction] = {}
        for u in self.units.values():
            out.update(u.functions)
        return out


# ---------------------------------------------------------------------------
# Source side — Clang compilation database (compile_commands.json).
# ---------------------------------------------------------------------------
def _entry_command(entry: dict) -> List[str]:
    """Return the argv of a compile_commands entry (arguments[] or command str)."""
    if "arguments" in entry and entry["arguments"]:
        return list(entry["arguments"])
    if "command" in entry and entry["command"]:
        return shlex.split(entry["command"])
    return []


def _include_dirs_from_argv(argv: List[str], directory: str) -> Tuple[str, ...]:
    """Extract -I include directories (both ``-Ifoo`` and ``-I foo`` forms)."""
    dirs: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-I" and i + 1 < len(argv):
            dirs.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("-I") and len(a) > 2:
            dirs.append(a[2:])
        i += 1
    resolved = []
    for d in dirs:
        resolved.append(d if os.path.isabs(d) else os.path.normpath(os.path.join(directory, d)))
    return tuple(resolved)


def ingest_compile_db(db_path: str) -> Optional[ProjectModule]:
    """Ingest every translation unit named in a compile_commands.json.

    Each entry's source file is lowered through the clang-AST ingester; recorded
    ``-I`` directories are passed back to the AST dump so headers resolve.
    """
    if not os.path.exists(CLANG):
        return None
    try:
        with open(db_path, "r", encoding="utf-8") as fh:
            db = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    proj = ProjectModule()
    for entry in db:
        fpath = entry.get("file")
        directory = entry.get("directory", os.path.dirname(db_path))
        if not fpath:
            continue
        if not os.path.isabs(fpath):
            fpath = os.path.normpath(os.path.join(directory, fpath))
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        argv = _entry_command(entry)
        incs = _include_dirs_from_argv(argv, directory)
        extra = []
        for d in incs:
            extra += ["-I", d]
        mod = _iri.ingest_clang(src, extra_args=extra)
        unit = SourceUnit(path=fpath, include_dirs=incs)
        if mod is not None:
            unit.functions = dict(mod.functions)
        proj.units[fpath] = unit
    return proj


# ---------------------------------------------------------------------------
# Target side — Cargo workspace (cargo metadata).
# ---------------------------------------------------------------------------
def cargo_metadata(workspace_dir: str) -> Optional[dict]:
    """Run ``cargo metadata`` on a workspace and return the parsed JSON."""
    if not os.path.exists(CARGO):
        return None
    env = dict(os.environ)
    env.setdefault("CARGO_HOME", os.path.join(workspace_dir, ".cargo_home"))
    try:
        r = subprocess.run(
            [CARGO, "metadata", "--format-version", "1", "--no-deps"],
            cwd=workspace_dir, env=env, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def ingest_cargo_workspace(workspace_dir: str) -> Optional[ProjectModule]:
    """Enumerate the workspace's packages/targets via ``cargo metadata``."""
    meta = cargo_metadata(workspace_dir)
    if meta is None:
        return None
    proj = ProjectModule()
    member_ids = set(meta.get("workspace_members", []))
    for pkg in meta.get("packages", []):
        # only keep packages that are workspace members (not external deps)
        if member_ids and pkg.get("id") not in member_ids:
            continue
        for tgt in pkg.get("targets", []):
            cp = CratePackage(
                name=pkg.get("name", ""),
                target_name=tgt.get("name", ""),
                src_path=tgt.get("src_path", ""),
                kind=tuple(tgt.get("kind", [])),
            )
            proj.packages[cp.target_name] = cp
    return proj


# ---------------------------------------------------------------------------
# Self-confirmation against the real toolchains.
# ---------------------------------------------------------------------------
@dataclass
class CompileDbConfirmation:
    available: bool
    ok: bool
    project: Optional[ProjectModule] = None


def confirm_compile_db() -> CompileDbConfirmation:
    """Build a real two-file C project + compile_commands.json and ingest it."""
    if not os.path.exists(CLANG):
        return CompileDbConfirmation(available=False, ok=False)
    with tempfile.TemporaryDirectory() as d:
        inc = os.path.join(d, "inc")
        os.makedirs(inc)
        with open(os.path.join(inc, "api.h"), "w") as fh:
            fh.write("int add(int a, int b);\n")
        f1 = os.path.join(d, "math.c")
        f2 = os.path.join(d, "strops.c")
        with open(f1, "w") as fh:
            fh.write('#include "api.h"\n'
                     "int add(int a, int b) { return a + b; }\n"
                     "static int helper(int x) { return x + 1; }\n")
        with open(f2, "w") as fh:
            fh.write("unsigned slen(const char *p) {\n"
                     "    unsigned n = 0; while (p[n]) n++; return n;\n}\n")
        db = [
            {"directory": d, "file": f1,
             "arguments": [CLANG, "-c", "-I", inc, f1]},
            {"directory": d, "file": f2,
             "arguments": [CLANG, "-c", f2]},
        ]
        dbpath = os.path.join(d, "compile_commands.json")
        with open(dbpath, "w") as fh:
            json.dump(db, fh)
        proj = ingest_compile_db(dbpath)
    if proj is None:
        return CompileDbConfirmation(available=True, ok=False)
    fns = proj.all_functions()
    ok = (
        len(proj.units) == 2
        and "add" in fns and fns["add"].ret_type == "int"
        and tuple(p.type for p in fns["add"].params) == ("int", "int")
        and "helper" in fns and fns["helper"].storage == "static"
        and "slen" in fns and fns["slen"].ret_type == "unsigned int"
    )
    return CompileDbConfirmation(available=True, ok=ok, project=proj)


@dataclass
class CargoWorkspaceConfirmation:
    available: bool
    ok: bool
    project: Optional[ProjectModule] = None


def _write_cargo_workspace(root: str) -> None:
    for name, fn in (("alpha", "pub fn alpha_fn() -> i32 { 1 }\n"),
                     ("beta", "pub fn beta_fn() -> i32 { 2 }\n")):
        srcdir = os.path.join(root, name, "src")
        os.makedirs(srcdir)
        with open(os.path.join(root, name, "Cargo.toml"), "w") as fh:
            fh.write(f'[package]\nname = "{name}"\n'
                     'version = "0.1.0"\nedition = "2021"\n')
        with open(os.path.join(srcdir, "lib.rs"), "w") as fh:
            fh.write(fn)
    with open(os.path.join(root, "Cargo.toml"), "w") as fh:
        fh.write('[workspace]\nmembers = ["alpha", "beta"]\nresolver = "2"\n')


def confirm_cargo_workspace() -> CargoWorkspaceConfirmation:
    """Build a real two-member Cargo workspace and enumerate it via metadata."""
    if not os.path.exists(CARGO):
        return CargoWorkspaceConfirmation(available=False, ok=False)
    with tempfile.TemporaryDirectory() as d:
        _write_cargo_workspace(d)
        proj = ingest_cargo_workspace(d)
        if proj is None:
            return CargoWorkspaceConfirmation(available=True, ok=False)
        names = set(proj.packages.keys())
        ok = {"alpha", "beta"} <= names
        for tn in ("alpha", "beta"):
            cp = proj.packages.get(tn)
            if cp is None or not cp.src_path.endswith(os.path.join("src", "lib.rs")):
                ok = False
            if cp is not None and "lib" not in cp.kind:
                ok = False
    return CargoWorkspaceConfirmation(available=True, ok=ok, project=proj)


PROJECT_INGEST_SPI = {
    "ingest_compile_db": ingest_compile_db,
    "ingest_cargo_workspace": ingest_cargo_workspace,
    "confirm_compile_db": confirm_compile_db,
    "confirm_cargo_workspace": confirm_cargo_workspace,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_compile_db()
    w = confirm_cargo_workspace()
    print("compile_db ok:", c.ok, "available:", c.available)
    if c.project:
        print("  project functions:", sorted(c.project.all_functions()))
    print("cargo workspace ok:", w.ok, "available:", w.available)
    if w.project:
        print("  packages:", {k: v.kind for k, v in w.project.packages.items()})
