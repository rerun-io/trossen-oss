"""Filesystem locations for the local Trossen ingestion pipeline.

`data/` holds the source episode MCAPs. On a dev machine it is a git-ignored
symlink to an existing episode directory; for other users the `download` task
populates it from the Hugging Face Hub. Either way the resolved location is
overridable via the ``TROSSEN_DATA_ROOT`` environment variable. Generated RRDs
go to a separate git-ignored `outputs/` tree, never into the source.
"""

from __future__ import annotations

import os
from pathlib import Path

from trossen_oss import REPO_ROOT

DEFAULT_DATA_DIR: Path = REPO_ROOT / "data"
"""Default episode source directory (repo-local ``data/``)."""
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "outputs"
"""Default root for generated artifacts (repo-local ``outputs/``)."""

EPISODE_MCAP_GLOB: str = "episode_*_proto.mcap"
"""Glob matching source episode MCAP files."""


def episode_mcap_name(index: int) -> str:
    """Return the source MCAP filename for an episode index.

    Args:
        index: Zero-based episode number.

    Returns:
        The episode MCAP filename, e.g. ``"episode_0_proto.mcap"``.
    """
    return f"episode_{index}_proto.mcap"


def data_dir() -> Path:
    """Return the episode source directory.

    Resolves ``TROSSEN_DATA_ROOT`` if set, otherwise the repo-local ``data/``.
    """
    override: str | None = os.getenv("TROSSEN_DATA_ROOT")
    return Path(override) if override else DEFAULT_DATA_DIR


def output_dir() -> Path:
    """Return the root directory for generated artifacts.

    Resolves ``TROSSEN_OUTPUT_DIR`` if set, otherwise the repo-local ``outputs/``.
    """
    override: str | None = os.getenv("TROSSEN_OUTPUT_DIR")
    return Path(override) if override else DEFAULT_OUTPUT_DIR


def rrd_dir() -> Path:
    """Return the directory that holds generated RRD files (``outputs/rrds``)."""
    return output_dir() / "rrds"
