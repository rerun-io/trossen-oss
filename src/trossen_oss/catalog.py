"""Register locally-generated RRDs into a Rerun catalog.

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
"""gRPC URL of a locally-running ``rerun server`` catalog."""


def _episode_rrds(rrd_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return the (base, urdf) RRD paths for every episode in ``rrd_dir``, paired by id."""
    base_rrds: list[Path] = sorted(rrd_dir.glob("*_data.rrd"))
    urdf_rrds: list[Path] = [rrd_dir / base.name.replace("_data.rrd", "_urdf.rrd") for base in base_rrds]
    return base_rrds, urdf_rrds


def register_episodes(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path | None = None,
    *,
    recreate: bool = True,
) -> DatasetEntry:
    """Create (or replace) a catalog dataset and register every episode in ``rrd_dir``.

    Each ``<id>_data.rrd`` becomes a segment (``base`` layer) and the matching
    ``<id>_urdf.rrd`` is registered as that segment's ``urdf`` layer — paired by the
    shared ``recording_id``. The blueprint is registered once as the dataset default.

    Args:
        catalog_url: gRPC URL of the local Rerun catalog.
        dataset_name: Catalog dataset name to (re)create.
        rrd_dir: Directory holding the per-episode ``*_data.rrd`` / ``*_urdf.rrd`` files.
        blueprint: Optional saved ``.rbl`` registered as the dataset default blueprint.
        recreate: Delete an existing dataset of the same name before registering.

    Returns:
        The registered :class:`DatasetEntry`.
    """
    base_rrds, urdf_rrds = _episode_rrds(rrd_dir)
    if not base_rrds:
        raise FileNotFoundError(f"No *_data.rrd files found in {rrd_dir}")

    client: CatalogClient = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset: DatasetEntry = client.create_dataset(dataset_name, exist_ok=True)

    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE
    # Each RRD carries its own recording_id, so a single call registers every
    # episode as its own segment; the urdf layer attaches by matching recording_id.
    dataset.register(
        [base.resolve().as_uri() for base in base_rrds], layer_name="base", on_duplicate=on_duplicate
    ).wait()
    dataset.register(
        [urdf.resolve().as_uri() for urdf in urdf_rrds if urdf.exists()],
        layer_name="urdf",
        on_duplicate=on_duplicate,
    ).wait()
    if blueprint is not None and blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        print(f"  registered default blueprint: {blueprint.name}")
    elif blueprint is not None:
        print(f"  no blueprint registered (not found at {blueprint})")
    return dataset


@dataclass
class CatalogConfig:
    """Configuration for registering the local OSS outputs into a catalog."""

    output_dir: Path = Path("outputs")
    """Directory holding outputs/rrds/*.rrd and robot_data_preprocessing.rbl."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (``rerun server``)."""
    dataset_name: str = "trossen_oss"
    """Catalog dataset name to (re)create."""
    recreate: bool = True
    """Delete and recreate the dataset before registering. Pass ``--no-recreate`` to
    re-register onto the existing dataset (REPLACE per layer) — the experiment loop's
    "fix a bad pass by re-registering" idiom, without rebuilding the whole dataset."""


def main(cfg: CatalogConfig) -> None:
    """Register every preprocessed episode into the local catalog as one dataset."""
    dataset: DatasetEntry = register_episodes(
        cfg.catalog_url,
        cfg.dataset_name,
        cfg.output_dir / "rrds",
        cfg.output_dir / "robot_data_preprocessing.rbl",
        recreate=cfg.recreate,
    )
    segments: list[str] = dataset.segment_ids()
    print(f"Registered dataset '{dataset.name}' on {cfg.catalog_url} with {len(segments)} segment(s):")
    for segment in segments:
        print(f"  - {segment}")
