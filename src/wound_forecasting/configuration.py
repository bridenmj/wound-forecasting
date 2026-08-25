"""Configuration loading with explicit environment substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml_config(
    path: str | Path, replacements: dict[str, str] | None = None
) -> dict:
    """Load YAML after resolving ``${NAME}`` placeholders.

    Values supplied through ``replacements`` take precedence over environment
    variables. Unresolved placeholders fail rather than silently becoming bad
    paths.
    """
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Install PyYAML to load experiment configurations"
        ) from error
    text = Path(path).read_text(encoding="utf-8")
    values = {**os.environ, **(replacements or {})}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"Configuration variable {name!r} is unresolved")
        return values[name]

    resolved = _ENVIRONMENT.sub(replace, text)
    config = yaml.safe_load(resolved)
    if not isinstance(config, dict):
        raise TypeError("Top-level configuration must be a mapping")
    return config
