"""Register locally-generated RRDs into a Rerun data-platform catalog.

The local catalog is the in-memory ``rerun server`` (start it with
``rerun server --port 51234``). Each episode is one *segment* whose id is the
RRD ``recording_id``; the base recording and the URDF model are registered as
two *layers* of that single segment, and the saved blueprint becomes the
dataset default. This mirrors the cloud ingestion flow, with local file URIs
instead of an object-store prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51234"
"""gRPC URL of a locally-running ``rerun server`` data platform."""


@dataclass
class EpisodeLayers:
    """The RRD files and blueprint that make up one logical episode recording."""

    base_rrd: Path
    """Base recording RRD (signals, video, computed transforms)."""
    urdf_rrd: Path
    """URDF model-layer RRD (meshes + static transforms); shares the base recording_id."""
    blueprint: Path | None = None
    """Optional saved ``.rbl`` registered as the dataset default blueprint."""


def register_episode_dataset(
    catalog_url: str,
    dataset_name: str,
    layers: EpisodeLayers,
    *,
    recreate: bool = True,
) -> DatasetEntry:
    """Create (or replace) a catalog dataset and register one layered episode.

    Args:
        catalog_url: gRPC URL of the local Rerun data platform.
        dataset_name: Catalog dataset name to (re)create.
        layers: Base/URDF RRDs (and optional blueprint) for the episode.
        recreate: Delete an existing dataset of the same name before registering.

    Returns:
        The registered :class:`DatasetEntry`.
    """
    client: CatalogClient = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset: DatasetEntry = client.create_dataset(dataset_name, exist_ok=True)

    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE
    # Both RRDs share a recording_id, so they land on the same segment as two
    # distinct layers (base + urdf) rather than colliding as duplicate segments.
    dataset.register([layers.base_rrd.resolve().as_uri()], layer_name="base", on_duplicate=on_duplicate).wait()
    dataset.register([layers.urdf_rrd.resolve().as_uri()], layer_name="urdf", on_duplicate=on_duplicate).wait()
    if layers.blueprint is not None:
        dataset.register_blueprint(layers.blueprint.resolve().as_uri(), set_default=True)
    return dataset


@dataclass
class CatalogConfig:
    """Configuration for registering the local OSS outputs into a catalog."""

    output_dir: Path = Path("outputs")
    """Directory containing data.rrd, urdf.rrd, and robot_data_preprocessing.rbl."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun data platform (``rerun server``)."""
    dataset_name: str = "trossen_oss"
    """Catalog dataset name to (re)create."""


def main(cfg: CatalogConfig) -> None:
    """Register the OSS pipeline outputs into the local catalog."""
    layers: EpisodeLayers = EpisodeLayers(
        base_rrd=cfg.output_dir / "data.rrd",
        urdf_rrd=cfg.output_dir / "urdf.rrd",
        blueprint=cfg.output_dir / "robot_data_preprocessing.rbl",
    )
    dataset: DatasetEntry = register_episode_dataset(cfg.catalog_url, cfg.dataset_name, layers)
    print(f"Registered dataset '{dataset.name}' on {cfg.catalog_url}")
    print(f"  segments: {dataset.segment_ids()}")
