"""Step 136 — formal SPI contract bridge.

``formal/SPIContract.lean`` states the target-pack and divergence-oracle SPI
obligations as Lean typeclasses.  This module ties that kernel-checked contract
back to the executable Python registry:

* every registered ``TargetPack`` has a corresponding Lean instance;
* every executable pack-conformance obligation key is named in the Lean file;
* the formal oracle method list matches the real ``DivergenceOracle`` surface;
* Lake accepts the Lean development when Lean/Lake are installed.

When Lake is unavailable, the report is explicitly source-contract only: it can
be ``ok`` if the source/interface checks pass, but ``fully_checked`` remains
``False``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from typing import Tuple

from . import pack_conformance
from . import plugin
from .mechanized_soundness import FORMAL_DIR, LAKEFILE, _ROOT, _lake_binary, _tail
from .target_semantics import PACKS

SPI_CONTRACT_SOURCE = os.path.join("formal", "SPIContract.lean")
SPI_CONTRACT_TARGET = "SPIContract"

TARGET_PACK_OBLIGATION_KEYS: Tuple[str, ...] = pack_conformance.OBLIGATION_KEYS
PLUGIN_METHOD_KEYS: Tuple[str, ...] = (
    "applies_to",
    "find_divergence",
    "confirm",
)
PLUGIN_ABSTRACT_METHOD_KEYS: Tuple[str, ...] = (
    "applies_to",
    "find_divergence",
)

REQUIRED_DECLARATIONS: Tuple[str, ...] = (
    "class TargetPackSPI",
    "class OraclePluginSPI",
    "theorem canonicalizeFromCodes_sound",
    "theorem target_pack_obligation_keys_cover_executable_suite",
    "theorem oracle_plugin_method_keys_cover_python_spi",
    "theorem recorded_observable_oracle_sound",
)


def _read_source() -> str | None:
    try:
        with open(os.path.join(_ROOT, SPI_CONTRACT_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


def _instance_name(pack_name: str) -> str:
    return f"{pack_name}TargetPackSPI"


def _decl_present(src: str, declaration: str) -> bool:
    return declaration in src


def _pack_instance_present(src: str, pack_name: str) -> bool:
    return f"instance {_instance_name(pack_name)}" in src


def _quoted_token_present(src: str, token: str) -> bool:
    return f'"{token}"' in src


def python_oracle_spi_methods_ok() -> bool:
    """Return True iff the formal method list matches the real Python SPI."""

    abstract = set(getattr(plugin.DivergenceOracle, "__abstractmethods__", set()))
    if abstract != set(PLUGIN_ABSTRACT_METHOD_KEYS):
        return False
    return all(callable(getattr(plugin.DivergenceOracle, name, None))
               for name in PLUGIN_METHOD_KEYS)


@dataclass(frozen=True)
class SPIContractReport:
    available: bool
    source_present: bool
    lakefile_present: bool
    declarations_present: Tuple[str, ...]
    declarations_missing: Tuple[str, ...]
    pack_instances_present: Tuple[str, ...]
    pack_instances_missing: Tuple[str, ...]
    obligation_keys_present: Tuple[str, ...]
    obligation_keys_missing: Tuple[str, ...]
    plugin_methods_present: Tuple[str, ...]
    plugin_methods_missing: Tuple[str, ...]
    plugin_methods_match_python: bool
    pack_conformance_ok: bool
    kernel_accepted: bool | None
    source_hash: str
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def source_contract_ok(self) -> bool:
        return (
            self.source_present
            and self.lakefile_present
            and not self.declarations_missing
            and not self.pack_instances_missing
            and not self.obligation_keys_missing
            and not self.plugin_methods_missing
            and self.plugin_methods_match_python
            and self.pack_conformance_ok
        )

    @property
    def ok(self) -> bool:
        if not self.source_contract_ok:
            return False
        if self.available:
            return self.kernel_accepted is True
        return True

    @property
    def fully_checked(self) -> bool:
        return self.ok and self.available and self.kernel_accepted is True

    def detail(self) -> str:
        if self.ok:
            mode = "Lake-checked" if self.fully_checked else "source-contract"
            return (
                f"{mode} SPI contract: {len(self.pack_instances_present)} packs, "
                f"{len(self.obligation_keys_present)} obligations, "
                f"{len(self.plugin_methods_present)} plugin methods"
            )
        parts = []
        if self.declarations_missing:
            parts.append(f"missing declarations={list(self.declarations_missing)}")
        if self.pack_instances_missing:
            parts.append(f"missing pack instances={list(self.pack_instances_missing)}")
        if self.obligation_keys_missing:
            parts.append(f"missing obligation keys={list(self.obligation_keys_missing)}")
        if self.plugin_methods_missing:
            parts.append(f"missing plugin methods={list(self.plugin_methods_missing)}")
        if not self.plugin_methods_match_python:
            parts.append("Python DivergenceOracle SPI mismatch")
        if not self.pack_conformance_ok:
            parts.append("registered TargetPack conformance failed")
        if self.available and self.kernel_accepted is not True:
            parts.append(f"Lake rejected SPIContract (exit={self.exit_code})")
        return "; ".join(parts) or "SPI contract source missing"


def confirm_formal_spi_contract(timeout: int = 300) -> SPIContractReport:
    """Confirm the formal SPI contract against Lean and the Python registry."""

    src = _read_source()
    lakefile_present = os.path.exists(os.path.join(_ROOT, LAKEFILE))
    pack_conf = pack_conformance.confirm_pack_conformance()

    if src is None:
        return SPIContractReport(
            available=False,
            source_present=False,
            lakefile_present=lakefile_present,
            declarations_present=(),
            declarations_missing=REQUIRED_DECLARATIONS,
            pack_instances_present=(),
            pack_instances_missing=tuple(_instance_name(name) for name in PACKS),
            obligation_keys_present=(),
            obligation_keys_missing=TARGET_PACK_OBLIGATION_KEYS,
            plugin_methods_present=(),
            plugin_methods_missing=PLUGIN_METHOD_KEYS,
            plugin_methods_match_python=python_oracle_spi_methods_ok(),
            pack_conformance_ok=pack_conf.ok,
            kernel_accepted=None,
            source_hash="",
        )

    src_hash = hashlib.sha256(src.encode()).hexdigest()
    declarations_present = tuple(d for d in REQUIRED_DECLARATIONS
                                 if _decl_present(src, d))
    declarations_missing = tuple(d for d in REQUIRED_DECLARATIONS
                                 if d not in declarations_present)

    pack_instances_present = tuple(_instance_name(name) for name in PACKS
                                   if _pack_instance_present(src, name))
    pack_instances_missing = tuple(_instance_name(name) for name in PACKS
                                   if _instance_name(name)
                                   not in pack_instances_present)

    obligation_keys_present = tuple(k for k in TARGET_PACK_OBLIGATION_KEYS
                                    if _quoted_token_present(src, k))
    obligation_keys_missing = tuple(k for k in TARGET_PACK_OBLIGATION_KEYS
                                    if k not in obligation_keys_present)

    plugin_methods_present = tuple(k for k in PLUGIN_METHOD_KEYS
                                   if _quoted_token_present(src, k))
    plugin_methods_missing = tuple(k for k in PLUGIN_METHOD_KEYS
                                   if k not in plugin_methods_present)

    lake = _lake_binary()
    if lake is None or not lakefile_present:
        return SPIContractReport(
            available=lake is not None,
            source_present=True,
            lakefile_present=lakefile_present,
            declarations_present=declarations_present,
            declarations_missing=declarations_missing,
            pack_instances_present=pack_instances_present,
            pack_instances_missing=pack_instances_missing,
            obligation_keys_present=obligation_keys_present,
            obligation_keys_missing=obligation_keys_missing,
            plugin_methods_present=plugin_methods_present,
            plugin_methods_missing=plugin_methods_missing,
            plugin_methods_match_python=python_oracle_spi_methods_ok(),
            pack_conformance_ok=pack_conf.ok,
            kernel_accepted=None,
            source_hash=src_hash,
        )

    try:
        proc = subprocess.run(
            [lake, "build", SPI_CONTRACT_TARGET],
            cwd=FORMAL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SPIContractReport(
            available=True,
            source_present=True,
            lakefile_present=True,
            declarations_present=declarations_present,
            declarations_missing=declarations_missing,
            pack_instances_present=pack_instances_present,
            pack_instances_missing=pack_instances_missing,
            obligation_keys_present=obligation_keys_present,
            obligation_keys_missing=obligation_keys_missing,
            plugin_methods_present=plugin_methods_present,
            plugin_methods_missing=plugin_methods_missing,
            plugin_methods_match_python=python_oracle_spi_methods_ok(),
            pack_conformance_ok=pack_conf.ok,
            kernel_accepted=proc.returncode == 0,
            source_hash=src_hash,
            exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:  # pragma: no cover
        return SPIContractReport(
            available=True,
            source_present=True,
            lakefile_present=True,
            declarations_present=declarations_present,
            declarations_missing=declarations_missing,
            pack_instances_present=pack_instances_present,
            pack_instances_missing=pack_instances_missing,
            obligation_keys_present=obligation_keys_present,
            obligation_keys_missing=obligation_keys_missing,
            plugin_methods_present=plugin_methods_present,
            plugin_methods_missing=plugin_methods_missing,
            plugin_methods_match_python=python_oracle_spi_methods_ok(),
            pack_conformance_ok=pack_conf.ok,
            kernel_accepted=False,
            source_hash=src_hash,
            stderr_tail=f"lake invocation failed: {exc}",
        )
