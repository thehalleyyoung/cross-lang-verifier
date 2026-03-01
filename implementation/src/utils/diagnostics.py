"""Diagnostic reporting for the Cross-Language Equivalence Verifier.

Provides structured diagnostics with source locations, severity levels,
fix suggestions, and formatted output.
"""

from __future__ import annotations

import sys
import io
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional, List, Sequence, TextIO, Dict

from .source_location import SourceLocation, SourceRange, SourceFile, SourceFileRegistry


class DiagnosticLevel(IntEnum):
    """Severity level for diagnostics."""
    ERROR = 4
    WARNING = 3
    NOTE = 2
    HELP = 1

    def label(self) -> str:
        return self.name.lower()

    def color_code(self) -> str:
        return {
            DiagnosticLevel.ERROR: "\033[1;31m",
            DiagnosticLevel.WARNING: "\033[1;33m",
            DiagnosticLevel.NOTE: "\033[1;36m",
            DiagnosticLevel.HELP: "\033[1;32m",
        }[self]


@dataclass
class FixSuggestion:
    """A suggested fix for a diagnostic."""
    message: str
    location: Optional[SourceLocation] = None
    replacement: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"message": self.message}
        if self.location:
            d["location"] = self.location.to_dict()
        if self.replacement is not None:
            d["replacement"] = self.replacement
        return d


@dataclass
class DiagnosticNote:
    """Additional context note attached to a diagnostic."""
    message: str
    location: Optional[SourceLocation] = None

    def to_dict(self) -> dict:
        d: dict = {"message": self.message}
        if self.location:
            d["location"] = self.location.to_dict()
        return d


@dataclass
class Diagnostic:
    """A single diagnostic message with optional location, notes, and fix suggestions."""
    level: DiagnosticLevel
    message: str
    code: Optional[str] = None
    location: Optional[SourceLocation] = None
    source_range: Optional[SourceRange] = None
    notes: List[DiagnosticNote] = field(default_factory=list)
    fixes: List[FixSuggestion] = field(default_factory=list)
    category: str = ""

    @staticmethod
    def error(message: str, location: Optional[SourceLocation] = None,
              code: Optional[str] = None) -> Diagnostic:
        return Diagnostic(DiagnosticLevel.ERROR, message, code=code, location=location)

    @staticmethod
    def warning(message: str, location: Optional[SourceLocation] = None,
                code: Optional[str] = None) -> Diagnostic:
        return Diagnostic(DiagnosticLevel.WARNING, message, code=code, location=location)

    @staticmethod
    def note(message: str, location: Optional[SourceLocation] = None) -> Diagnostic:
        return Diagnostic(DiagnosticLevel.NOTE, message, location=location)

    @staticmethod
    def help(message: str, location: Optional[SourceLocation] = None) -> Diagnostic:
        return Diagnostic(DiagnosticLevel.HELP, message, location=location)

    def add_note(self, message: str, location: Optional[SourceLocation] = None) -> Diagnostic:
        self.notes.append(DiagnosticNote(message, location))
        return self

    def add_fix(self, message: str, location: Optional[SourceLocation] = None,
                replacement: Optional[str] = None) -> Diagnostic:
        self.fixes.append(FixSuggestion(message, location, replacement))
        return self

    def with_range(self, source_range: SourceRange) -> Diagnostic:
        self.source_range = source_range
        return self

    def with_category(self, category: str) -> Diagnostic:
        self.category = category
        return self

    @property
    def is_error(self) -> bool:
        return self.level == DiagnosticLevel.ERROR

    @property
    def is_warning(self) -> bool:
        return self.level == DiagnosticLevel.WARNING

    def to_dict(self) -> dict:
        d: dict = {
            "level": self.level.label(),
            "message": self.message,
        }
        if self.code:
            d["code"] = self.code
        if self.location:
            d["location"] = self.location.to_dict()
        if self.source_range:
            d["range"] = self.source_range.to_dict()
        if self.notes:
            d["notes"] = [n.to_dict() for n in self.notes]
        if self.fixes:
            d["fixes"] = [f.to_dict() for f in self.fixes]
        if self.category:
            d["category"] = self.category
        return d

    def format_plain(self) -> str:
        """Format diagnostic without colors."""
        parts: List[str] = []
        if self.location:
            parts.append(f"{self.location}: ")
        parts.append(f"{self.level.label()}")
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(f": {self.message}")
        return "".join(parts)

    def format_colored(self) -> str:
        """Format diagnostic with ANSI colors."""
        reset = "\033[0m"
        bold = "\033[1m"
        color = self.level.color_code()
        parts: List[str] = []
        if self.location:
            parts.append(f"{bold}{self.location}{reset}: ")
        parts.append(f"{color}{self.level.label()}{reset}")
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(f": {bold}{self.message}{reset}")
        return "".join(parts)


class DiagnosticCollection:
    """Collects diagnostics and provides filtering, counting, and output."""

    def __init__(self):
        self._diagnostics: List[Diagnostic] = []
        self._source_registry: Optional[SourceFileRegistry] = None
        self._max_errors: int = 100
        self._error_count: int = 0
        self._warning_count: int = 0

    def set_source_registry(self, registry: SourceFileRegistry) -> None:
        self._source_registry = registry

    def set_max_errors(self, max_errors: int) -> None:
        self._max_errors = max_errors

    def add(self, diagnostic: Diagnostic) -> None:
        self._diagnostics.append(diagnostic)
        if diagnostic.is_error:
            self._error_count += 1
        elif diagnostic.is_warning:
            self._warning_count += 1

    def error(self, message: str, location: Optional[SourceLocation] = None,
              code: Optional[str] = None) -> Diagnostic:
        d = Diagnostic.error(message, location, code)
        self.add(d)
        return d

    def warning(self, message: str, location: Optional[SourceLocation] = None,
                code: Optional[str] = None) -> Diagnostic:
        d = Diagnostic.warning(message, location, code)
        self.add(d)
        return d

    def note(self, message: str, location: Optional[SourceLocation] = None) -> Diagnostic:
        d = Diagnostic.note(message, location)
        self.add(d)
        return d

    def help_msg(self, message: str, location: Optional[SourceLocation] = None) -> Diagnostic:
        d = Diagnostic.help(message, location)
        self.add(d)
        return d

    @property
    def diagnostics(self) -> List[Diagnostic]:
        return list(self._diagnostics)

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.level == DiagnosticLevel.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.level == DiagnosticLevel.WARNING]

    @property
    def notes(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.level == DiagnosticLevel.NOTE]

    @property
    def num_errors(self) -> int:
        return self._error_count

    @property
    def num_warnings(self) -> int:
        return self._warning_count

    @property
    def has_errors(self) -> bool:
        return self._error_count > 0

    @property
    def is_empty(self) -> bool:
        return len(self._diagnostics) == 0

    def clear(self) -> None:
        self._diagnostics.clear()
        self._error_count = 0
        self._warning_count = 0

    def merge(self, other: DiagnosticCollection) -> None:
        for d in other._diagnostics:
            self.add(d)

    def filter_by_level(self, level: DiagnosticLevel) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.level == level]

    def filter_by_file(self, filepath: str) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.location and d.location.file == filepath]

    def filter_by_category(self, category: str) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.category == category]

    def group_by_file(self) -> Dict[str, List[Diagnostic]]:
        groups: Dict[str, List[Diagnostic]] = {}
        for d in self._diagnostics:
            key = d.location.file if d.location else "<no location>"
            groups.setdefault(key, []).append(d)
        return groups

    def group_by_category(self) -> Dict[str, List[Diagnostic]]:
        groups: Dict[str, List[Diagnostic]] = {}
        for d in self._diagnostics:
            key = d.category or "<uncategorized>"
            groups.setdefault(key, []).append(d)
        return groups

    def sort_by_location(self) -> None:
        def sort_key(d: Diagnostic):
            if d.location:
                return (d.location.file, d.location.line, d.location.column)
            return ("", 0, 0)
        self._diagnostics.sort(key=sort_key)

    def to_list(self) -> List[dict]:
        return [d.to_dict() for d in self._diagnostics]

    def format_plain(self, stream: Optional[TextIO] = None) -> str:
        """Format all diagnostics without colors."""
        buf = io.StringIO() if stream is None else stream
        for d in self._diagnostics:
            buf.write(d.format_plain() + "\n")
            if self._source_registry and d.location and not d.location.is_unknown:
                snippet = self._source_registry.get_snippet(d.location, context=1)
                if snippet:
                    buf.write(snippet + "\n")
            for note in d.notes:
                loc_str = f"{note.location}: " if note.location else ""
                buf.write(f"  note: {loc_str}{note.message}\n")
            for fix in d.fixes:
                loc_str = f"{fix.location}: " if fix.location else ""
                buf.write(f"  help: {loc_str}{fix.message}\n")
                if fix.replacement is not None:
                    buf.write(f"    suggested: {fix.replacement}\n")
        if self._error_count > 0 or self._warning_count > 0:
            buf.write(f"\n{self._error_count} error(s), {self._warning_count} warning(s)\n")
        if stream is None:
            return buf.getvalue()
        return ""

    def format_colored(self, stream: Optional[TextIO] = None) -> str:
        """Format all diagnostics with ANSI colors."""
        buf = io.StringIO() if stream is None else stream
        reset = "\033[0m"
        cyan = "\033[36m"
        green = "\033[32m"
        for d in self._diagnostics:
            buf.write(d.format_colored() + "\n")
            if self._source_registry and d.location and not d.location.is_unknown:
                snippet = self._source_registry.get_snippet(d.location, context=1)
                if snippet:
                    buf.write(snippet + "\n")
            for note_item in d.notes:
                loc_str = f"{note_item.location}: " if note_item.location else ""
                buf.write(f"  {cyan}note{reset}: {loc_str}{note_item.message}\n")
            for fix in d.fixes:
                loc_str = f"{fix.location}: " if fix.location else ""
                buf.write(f"  {green}help{reset}: {loc_str}{fix.message}\n")
                if fix.replacement is not None:
                    buf.write(f"    suggested: {fix.replacement}\n")
        if self._error_count > 0 or self._warning_count > 0:
            buf.write(f"\n{self._error_count} error(s), {self._warning_count} warning(s)\n")
        if stream is None:
            return buf.getvalue()
        return ""

    def print(self, colored: bool = True, stream: Optional[TextIO] = None) -> None:
        out = stream or sys.stderr
        if colored:
            self.format_colored(out)
        else:
            self.format_plain(out)

    def summary(self) -> str:
        if self.is_empty:
            return "No diagnostics."
        parts: List[str] = []
        if self._error_count:
            parts.append(f"{self._error_count} error(s)")
        if self._warning_count:
            parts.append(f"{self._warning_count} warning(s)")
        note_count = len(self.notes)
        if note_count:
            parts.append(f"{note_count} note(s)")
        return ", ".join(parts)

    def __len__(self) -> int:
        return len(self._diagnostics)

    def __iter__(self):
        return iter(self._diagnostics)

    def __bool__(self) -> bool:
        return not self.is_empty


class DiagnosticFormatter:
    """Configurable formatter for diagnostic output."""

    def __init__(self, use_color: bool = True, show_snippets: bool = True,
                 show_notes: bool = True, show_fixes: bool = True,
                 context_lines: int = 2, max_diagnostics: int = 0):
        self.use_color = use_color
        self.show_snippets = show_snippets
        self.show_notes = show_notes
        self.show_fixes = show_fixes
        self.context_lines = context_lines
        self.max_diagnostics = max_diagnostics
        self._source_registry: Optional[SourceFileRegistry] = None

    def set_source_registry(self, registry: SourceFileRegistry) -> None:
        self._source_registry = registry

    def format(self, collection: DiagnosticCollection) -> str:
        buf = io.StringIO()
        diagnostics = collection.diagnostics
        if self.max_diagnostics > 0:
            diagnostics = diagnostics[:self.max_diagnostics]

        for d in diagnostics:
            if self.use_color:
                buf.write(d.format_colored() + "\n")
            else:
                buf.write(d.format_plain() + "\n")

            if self.show_snippets and self._source_registry and d.location and not d.location.is_unknown:
                snippet = self._source_registry.get_snippet(d.location, self.context_lines)
                if snippet:
                    buf.write(snippet + "\n")

            if self.show_notes:
                for note_item in d.notes:
                    loc_str = f"{note_item.location}: " if note_item.location else ""
                    buf.write(f"  note: {loc_str}{note_item.message}\n")

            if self.show_fixes:
                for fix in d.fixes:
                    loc_str = f"{fix.location}: " if fix.location else ""
                    buf.write(f"  help: {loc_str}{fix.message}\n")
                    if fix.replacement is not None:
                        buf.write(f"    suggested: {fix.replacement}\n")
            buf.write("\n")

        if self.max_diagnostics > 0 and len(collection.diagnostics) > self.max_diagnostics:
            remaining = len(collection.diagnostics) - self.max_diagnostics
            buf.write(f"... and {remaining} more diagnostic(s)\n")

        buf.write(collection.summary() + "\n")
        return buf.getvalue()

    def format_json(self, collection: DiagnosticCollection) -> List[dict]:
        return collection.to_list()
