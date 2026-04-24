"""Helpers for loading top-level project packages from UI script entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path(file_path: str) -> None:
    """Insert the repository root ahead of sibling-package imports when needed."""
    repo_root = Path(file_path).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
