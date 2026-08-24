"""Subject-level manifest helpers.

These functions make the held-out split explicit and avoid the unresolved
notebook globals present in the paper-era ``create_test.py`` fragment.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_PIGS = ("ID1323", "ID1324", "ID1326", "ID1327")
DEFAULT_TEST_PIGS = ("ID1325", "ID1328")


def pig_id(wound_id: str) -> str:
    """Return a pig identifier such as ``ID1325`` from a wound identifier."""
    return wound_id.split("_", 1)[0]


def clip_after_512(path: str) -> str:
    """Make an image path relative to its historical ``512x512`` root."""
    normalized = str(path).replace("\\", "/")
    marker = "512x512/"
    return normalized.split(marker, 1)[1] if marker in normalized else normalized


def select_pigs(
    manifest: Mapping[str, Any],
    pigs: Iterable[str],
    *,
    minimum_days: int = 5,
    normalize_paths: bool = True,
) -> dict[str, Any]:
    """Select eligible wounds belonging only to the requested pigs."""
    selected_pigs = set(pigs)
    selected: dict[str, Any] = {}

    for wound_id, days in manifest.items():
        if pig_id(wound_id) not in selected_pigs or len(days) < minimum_days:
            continue
        selected[wound_id] = {
            day: [clip_after_512(path) for path in paths]
            if normalize_paths
            else list(paths)
            for day, paths in days.items()
        }
    return selected


def load_test_manifest(
    manifest_path: str | Path,
    test_pigs: Iterable[str] = DEFAULT_TEST_PIGS,
    *,
    minimum_days: int = 5,
) -> dict[str, Any]:
    """Load a JSON manifest and return the held-out subject subset."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return select_pigs(manifest, test_pigs, minimum_days=minimum_days)

