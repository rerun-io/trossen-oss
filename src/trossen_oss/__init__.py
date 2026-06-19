"""Local Rerun ingestion pipeline for Trossen bimanual episodes."""

from __future__ import annotations

import os

__version__ = "0.1.0"

# Runtime type + array-shape checking is enabled only in the dev environment,
# which sets PIXI_DEV_MODE=1. Default/production runs carry no overhead.
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
