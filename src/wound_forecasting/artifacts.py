"""Resolve local paths and Hugging Face release artifacts."""

from __future__ import annotations

from pathlib import Path


def resolve_artifact(
    value: str | Path | None,
    *,
    repository: str,
    filename: str,
    repo_type: str = "model",
    token: str | None = None,
) -> Path:
    """Use an explicit local file or download the named Hub artifact."""
    if value is not None:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install huggingface_hub or provide an explicit local artifact path"
        ) from error
    return Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            repo_type=repo_type,
            token=token,
        )
    )
