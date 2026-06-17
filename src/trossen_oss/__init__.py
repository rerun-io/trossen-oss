"""Local Rerun ingestion pipeline for Trossen bimanual episodes."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

# src-layout: src/trossen_oss/__init__.py -> repo root is three levels up.
REPO_ROOT: Path = Path(__file__).parent.parent.parent

# Runtime type + array-shape checking is enabled only in the dev environment,
# which sets PIXI_DEV_MODE=1. Default/production runs carry no overhead.
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
