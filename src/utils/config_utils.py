"""Configuration utilities: deep merge, env var expansion, path resolution, validation."""

from __future__ import annotations

import os
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any],
               merge_lists: bool = False) -> Dict[str, Any]:
    """Recursively merge override into base. Returns a new dict.

    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.
        merge_lists: If True, concatenate lists instead of replacing.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value, merge_lists)
        elif key in result and merge_lists and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = copy.deepcopy(value)
    return result


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """In-place recursive update of target with source."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict into dotted keys. {'a': {'b': 1}} -> {'a.b': 1}."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key, sep))
        else:
            result[full_key] = value
    return result


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten dotted keys back into nested dicts."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_env_vars(value: str, env: Optional[Dict[str, str]] = None,
                    default: str = "") -> str:
    """Expand environment variables in a string.

    Supports ${VAR} and $VAR syntax. Falls back to `default` for undefined vars.
    """
    lookup = env if env is not None else os.environ

    def replacer(m: re.Match) -> str:
        var_name = m.group(1) or m.group(2)
        # Support ${VAR:-default} syntax
        if ":-" in var_name:
            name, fallback = var_name.split(":-", 1)
            return lookup.get(name, fallback)
        return lookup.get(var_name, default)

    return _ENV_PATTERN.sub(replacer, value)


def expand_env_recursive(config: Any, env: Optional[Dict[str, str]] = None) -> Any:
    """Recursively expand environment variables in a config structure."""
    if isinstance(config, str):
        return expand_env_vars(config, env)
    if isinstance(config, dict):
        return {k: expand_env_recursive(v, env) for k, v in config.items()}
    if isinstance(config, list):
        return [expand_env_recursive(v, env) for v in config]
    return config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """Resolve a path, expanding ~ and making relative paths absolute.

    Args:
        path: The path to resolve.
        base_dir: Base directory for relative paths. Defaults to cwd.
    """
    path = os.path.expanduser(path)
    path = expand_env_vars(path)
    if not os.path.isabs(path):
        base = base_dir or os.getcwd()
        path = os.path.join(base, path)
    return os.path.normpath(path)


def resolve_paths(paths: List[str], base_dir: Optional[str] = None) -> List[str]:
    """Resolve a list of paths."""
    return [resolve_path(p, base_dir) for p in paths]


def find_config_file(names: List[str], search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Search for a config file by name in multiple directories.

    Searches in order: search_dirs, cwd, parent directories up to root, home.
    """
    dirs = list(search_dirs or [])
    cwd = os.getcwd()
    dirs.append(cwd)

    # Walk up from cwd
    current = cwd
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break
        dirs.append(parent)
        current = parent

    dirs.append(os.path.expanduser("~"))

    for d in dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {'; '.join(errors)}")


def validate_required(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Validate that required keys are present. Returns list of missing keys."""
    missing = [k for k in required_keys if k not in config]
    return [f"Missing required key: {k}" for k in missing]


def validate_types(config: Dict[str, Any],
                   type_spec: Dict[str, type]) -> List[str]:
    """Validate that config values match expected types."""
    errors: List[str] = []
    for key, expected in type_spec.items():
        if key in config and not isinstance(config[key], expected):
            actual = type(config[key]).__name__
            errors.append(f"Key '{key}': expected {expected.__name__}, got {actual}")
    return errors


def validate_range(config: Dict[str, Any],
                   range_spec: Dict[str, Tuple[Any, Any]]) -> List[str]:
    """Validate that numeric config values are within specified ranges."""
    errors: List[str] = []
    for key, (lo, hi) in range_spec.items():
        if key in config:
            val = config[key]
            if val < lo or val > hi:
                errors.append(f"Key '{key}': value {val} not in range [{lo}, {hi}]")
    return errors


def validate_choices(config: Dict[str, Any],
                     choice_spec: Dict[str, List[Any]]) -> List[str]:
    """Validate that config values are from allowed choices."""
    errors: List[str] = []
    for key, choices in choice_spec.items():
        if key in config and config[key] not in choices:
            errors.append(f"Key '{key}': value {config[key]!r} not in {choices}")
    return errors


def validate_config(config: Dict[str, Any],
                    required: Optional[List[str]] = None,
                    types: Optional[Dict[str, type]] = None,
                    ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
                    choices: Optional[Dict[str, List[Any]]] = None,
                    raise_on_error: bool = True) -> List[str]:
    """Run all validations on a config dict."""
    errors: List[str] = []
    if required:
        errors.extend(validate_required(config, required))
    if types:
        errors.extend(validate_types(config, types))
    if ranges:
        errors.extend(validate_range(config, ranges))
    if choices:
        errors.extend(validate_choices(config, choices))
    if errors and raise_on_error:
        raise ConfigValidationError(errors)
    return errors


# ---------------------------------------------------------------------------
# Config diffing
# ---------------------------------------------------------------------------

def diff_configs(a: Dict[str, Any], b: Dict[str, Any],
                 path: str = "") -> List[Tuple[str, Any, Any]]:
    """Compute differences between two config dicts.

    Returns list of (dotted_key, value_in_a, value_in_b).
    """
    diffs: List[Tuple[str, Any, Any]] = []
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{path}.{key}" if path else key
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(diff_configs(va, vb, full_key))
        elif va != vb:
            diffs.append((full_key, va, vb))
    return diffs


def format_diff(diffs: List[Tuple[str, Any, Any]]) -> str:
    """Format config diffs as human-readable text."""
    lines: List[str] = []
    for key, va, vb in diffs:
        if va is None:
            lines.append(f"  + {key}: {vb!r}")
        elif vb is None:
            lines.append(f"  - {key}: {va!r}")
        else:
            lines.append(f"  ~ {key}: {va!r} -> {vb!r}")
    return "\n".join(lines)
