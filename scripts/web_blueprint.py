"""
Write a *reduced* blueprint (.rbl) tuned for the small embedded web viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun.blueprint as rrb
import tyro

from trossen_oss.preprocessing import APPLICATION_ID

# (origin, display name) for the four camera 2D views, in grid order.
CAMERAS: list[tuple[str, str]] = [
    ("/external/cam_high", "High"),
    ("/external/cam_low", "Low"),
    ("/robot_left/wrist_camera", "Left wrist"),
    ("/robot_right/wrist_camera", "Right wrist"),
]


def web_blueprint() -> rrb.Blueprint:
    """Reduced layout for the embedded web viewer: 3D scene + camera grid + text strip."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                # Show 3D scene in one half, the four cameras as grid in the other half. 
                rrb.Spatial3DView(
                    name="Scene",
                    spatial_information=rrb.SpatialInformation(target_frame="world"),
                    eye_controls=rrb.archetypes.EyeControls3D(spin_speed=0.25),
                ),
                rrb.Grid(
                    *(rrb.Spatial2DView(origin=origin, name=name) for origin, name in CAMERAS),
                    grid_columns=2,
                    name="Cameras",
                ),
                column_shares=[1, 1],
            ),
            # Show text log at the bottom.
            rrb.TextLogView(origin="/", name="Instruction"),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )


@dataclass
class WebBlueprintConfig:
    """Configuration for writing the reduced web-viewer blueprint."""

    output_path: Path = Path("outputs") / "web_blueprint.rbl"
    """Destination ``.rbl`` file."""


def main(cfg: WebBlueprintConfig) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    web_blueprint().save(APPLICATION_ID, cfg.output_path)
    print(f"Wrote reduced web-viewer blueprint to: {cfg.output_path}")


if __name__ == "__main__":
    main(tyro.cli(WebBlueprintConfig, description="Write the reduced embedded-web-viewer blueprint"))
