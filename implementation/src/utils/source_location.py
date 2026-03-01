"""Source location tracking for cross-language verification.

Provides SourceLocation, SourceRange, and SourceFile for mapping IR constructs
back to original C or Rust source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict


class LanguageOrigin(Enum):
    """Language a source location originates from."""
    C = auto()
    RUST = auto()
    GENERATED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class SourceLocation:
    """A precise position in a source file (file:line:col)."""
    file: str
    line: int
    column: int
    language: LanguageOrigin = LanguageOrigin.UNKNOWN

    def __post_init__(self):
        if self.line < 0:
            raise ValueError(f"Line number must be non-negative, got {self.line}")
        if self.column < 0:
            raise ValueError(f"Column number must be non-negative, got {self.column}")

    @staticmethod
    def unknown() -> SourceLocation:
        return SourceLocation(file="<unknown>", line=0, column=0)

    @property
    def is_unknown(self) -> bool:
        return self.file == "<unknown>" and self.line == 0 and self.column == 0

    @property
    def basename(self) -> str:
        return os.path.basename(self.file)

    def format(self, style: str = "gcc") -> str:
        """Format location. style: 'gcc' -> file:line:col, 'msvc' -> file(line,col)."""
        if style == "msvc":
            return f"{self.file}({self.line},{self.column})"
        return f"{self.file}:{self.line}:{self.column}"

    def format_short(self) -> str:
        return f"{self.basename}:{self.line}:{self.column}"

    def with_column(self, col: int) -> SourceLocation:
        return SourceLocation(self.file, self.line, col, self.language)

    def with_line(self, line: int) -> SourceLocation:
        return SourceLocation(self.file, line, self.column, self.language)

    def offset_line(self, delta: int) -> SourceLocation:
        return SourceLocation(self.file, max(1, self.line + delta), self.column, self.language)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"SourceLocation({self.file!r}, {self.line}, {self.column})"

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "language": self.language.name,
        }

    @staticmethod
    def from_dict(d: dict) -> SourceLocation:
        lang = LanguageOrigin[d.get("language", "UNKNOWN")]
        return SourceLocation(d["file"], d["line"], d["column"], lang)


@dataclass(frozen=True)
class SourceRange:
    """A range spanning from start to end in a source file."""
    start: SourceLocation
    end: SourceLocation

    def __post_init__(self):
        if self.start.file != self.end.file:
            raise ValueError("SourceRange start and end must be in the same file")

    @staticmethod
    def from_location(loc: SourceLocation, length: int = 1) -> SourceRange:
        return SourceRange(loc, SourceLocation(loc.file, loc.line, loc.column + length, loc.language))

    @staticmethod
    def single_line(file: str, line: int, col_start: int, col_end: int,
                    language: LanguageOrigin = LanguageOrigin.UNKNOWN) -> SourceRange:
        return SourceRange(
            SourceLocation(file, line, col_start, language),
            SourceLocation(file, line, col_end, language),
        )

    @property
    def file(self) -> str:
        return self.start.file

    @property
    def is_single_line(self) -> bool:
        return self.start.line == self.end.line

    @property
    def num_lines(self) -> int:
        return self.end.line - self.start.line + 1

    @property
    def language(self) -> LanguageOrigin:
        return self.start.language

    def contains(self, loc: SourceLocation) -> bool:
        if loc.file != self.file:
            return False
        if loc.line < self.start.line or loc.line > self.end.line:
            return False
        if loc.line == self.start.line and loc.column < self.start.column:
            return False
        if loc.line == self.end.line and loc.column > self.end.column:
            return False
        return True

    def overlaps(self, other: SourceRange) -> bool:
        if self.file != other.file:
            return False
        return not (self.end.line < other.start.line or
                    other.end.line < self.start.line or
                    (self.end.line == other.start.line and self.end.column < other.start.column) or
                    (other.end.line == self.start.line and other.end.column < self.start.column))

    def merge(self, other: SourceRange) -> SourceRange:
        if self.file != other.file:
            raise ValueError("Cannot merge ranges from different files")
        start = self.start if (self.start.line, self.start.column) <= (other.start.line, other.start.column) else other.start
        end = self.end if (self.end.line, self.end.column) >= (other.end.line, other.end.column) else other.end
        return SourceRange(start, end)

    def format(self) -> str:
        if self.is_single_line:
            return f"{self.file}:{self.start.line}:{self.start.column}-{self.end.column}"
        return f"{self.file}:{self.start.line}:{self.start.column}-{self.end.line}:{self.end.column}"

    def __str__(self) -> str:
        return self.format()

    def to_dict(self) -> dict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> SourceRange:
        return SourceRange(SourceLocation.from_dict(d["start"]), SourceLocation.from_dict(d["end"]))


class SourceFile:
    """Represents a loaded source file with line-indexed content for snippet extraction."""

    def __init__(self, path: str, content: Optional[str] = None,
                 language: LanguageOrigin = LanguageOrigin.UNKNOWN):
        self.path = path
        self.language = language
        self._content: Optional[str] = content
        self._lines: Optional[List[str]] = None

    @staticmethod
    def from_path(path: str, language: Optional[LanguageOrigin] = None) -> SourceFile:
        if language is None:
            ext = os.path.splitext(path)[1].lower()
            language = {".c": LanguageOrigin.C, ".h": LanguageOrigin.C,
                        ".rs": LanguageOrigin.RUST}.get(ext, LanguageOrigin.UNKNOWN)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return SourceFile(path, content, language)

    @staticmethod
    def from_string(content: str, name: str = "<string>",
                    language: LanguageOrigin = LanguageOrigin.UNKNOWN) -> SourceFile:
        return SourceFile(name, content, language)

    @property
    def content(self) -> str:
        if self._content is None:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                self._content = f.read()
        return self._content

    @property
    def lines(self) -> List[str]:
        if self._lines is None:
            self._lines = self.content.splitlines(keepends=True)
        return self._lines

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)

    def get_line(self, line_number: int) -> Optional[str]:
        """Get a single line (1-indexed). Returns None if out of range."""
        if 1 <= line_number <= self.num_lines:
            return self.lines[line_number - 1].rstrip("\n\r")
        return None

    def get_lines(self, start: int, end: int) -> List[str]:
        """Get lines from start to end (1-indexed, inclusive)."""
        start = max(1, start)
        end = min(self.num_lines, end)
        return [self.lines[i - 1].rstrip("\n\r") for i in range(start, end + 1)]

    def get_snippet(self, location: SourceLocation, context: int = 2) -> str:
        """Extract a source snippet around a location with context lines."""
        if location.file != self.path and location.file != self.basename:
            return ""
        start = max(1, location.line - context)
        end = min(self.num_lines, location.line + context)
        result_lines: List[str] = []
        for i in range(start, end + 1):
            line_text = self.get_line(i)
            if line_text is None:
                continue
            marker = ">>>" if i == location.line else "   "
            result_lines.append(f"{marker} {i:4d} | {line_text}")
            if i == location.line and location.column > 0:
                padding = " " * (len(f"{marker} {i:4d} | ") + location.column - 1)
                result_lines.append(f"{padding}^")
        return "\n".join(result_lines)

    def get_range_snippet(self, source_range: SourceRange, context: int = 1) -> str:
        """Extract a source snippet for a range with context lines."""
        start = max(1, source_range.start.line - context)
        end = min(self.num_lines, source_range.end.line + context)
        result_lines: List[str] = []
        for i in range(start, end + 1):
            line_text = self.get_line(i)
            if line_text is None:
                continue
            in_range = source_range.start.line <= i <= source_range.end.line
            marker = ">>>" if in_range else "   "
            result_lines.append(f"{marker} {i:4d} | {line_text}")
        return "\n".join(result_lines)

    def location_at(self, line: int, column: int) -> SourceLocation:
        return SourceLocation(self.path, line, column, self.language)

    def range_at(self, start_line: int, start_col: int,
                 end_line: int, end_col: int) -> SourceRange:
        return SourceRange(
            self.location_at(start_line, start_col),
            self.location_at(end_line, end_col),
        )

    def find_text(self, text: str, start_line: int = 1) -> Optional[SourceLocation]:
        """Find the first occurrence of text starting from start_line."""
        for i in range(start_line - 1, self.num_lines):
            line = self.lines[i]
            col = line.find(text)
            if col >= 0:
                return self.location_at(i + 1, col + 1)
        return None

    def find_all(self, text: str) -> List[SourceLocation]:
        """Find all occurrences of text in the file."""
        results: List[SourceLocation] = []
        for i, line in enumerate(self.lines):
            start = 0
            while True:
                col = line.find(text, start)
                if col < 0:
                    break
                results.append(self.location_at(i + 1, col + 1))
                start = col + 1
        return results

    def __repr__(self) -> str:
        return f"SourceFile({self.path!r}, {self.num_lines} lines)"


class SourceFileRegistry:
    """Registry that caches loaded source files for reuse."""

    def __init__(self):
        self._files: Dict[str, SourceFile] = {}

    def load(self, path: str, language: Optional[LanguageOrigin] = None) -> SourceFile:
        abs_path = os.path.abspath(path)
        if abs_path not in self._files:
            self._files[abs_path] = SourceFile.from_path(abs_path, language)
        return self._files[abs_path]

    def register(self, source_file: SourceFile) -> None:
        abs_path = os.path.abspath(source_file.path)
        self._files[abs_path] = source_file

    def get(self, path: str) -> Optional[SourceFile]:
        return self._files.get(os.path.abspath(path))

    def get_snippet(self, location: SourceLocation, context: int = 2) -> str:
        sf = self.get(location.file)
        if sf is None:
            return f"  (source file {location.file!r} not loaded)"
        return sf.get_snippet(location, context)

    @property
    def files(self) -> List[SourceFile]:
        return list(self._files.values())

    def clear(self) -> None:
        self._files.clear()
