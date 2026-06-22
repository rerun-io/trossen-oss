"""Enrich registered episodes with a derived quality layer (the Refine *enrich* step).

After Collect registers each episode as a catalog segment, the experiment loop's Refine
stage *enriches* recordings: derived signals — model outputs, operator metadata, quality
verdicts — are attached as new **layers** that share the original recording id, so the raw
recording is never mutated and a bad pass is fixed by re-registering (not by editing data).

Here we derive a per-episode motion-quality verdict from a catalog query (right-arm joint
travel via :func:`arm_activity`), flag the low-motion episodes — the ones worth curating out
of a training set — and write each verdict as a tiny ``quality`` layer RRD that re-registers
onto its segment. The episodes' ``base`` and ``urdf`` layers are left untouched, and the
verdict is then visible in the viewer and queryable like any other component, so it can feed
back into the Train step's segment filter.

Needs a running catalog (``pixi run serve``) with the dataset registered (``pixi run register``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from trossen_oss.catalog import DEFAULT_CATALOG_URL
from trossen_oss.preprocessing import APPLICATION_ID
from trossen_oss.query import arm_activity


@dataclass
class QualityVerdict:
    """Per-episode motion-quality verdict derived during enrichment."""

    right_motion: float
    """Total right-arm joint travel for the episode (rad) — the task-richness proxy."""
    flagged: bool
    """True when ``right_motion`` is below the cutoff (a low-motion episode to curate out)."""


def motion_verdicts(client: CatalogClient, dataset: DatasetEntry, min_motion: float) -> dict[str, QualityVerdict]:
    """Derive a ``{segment_id: QualityVerdict}`` map from the cross-episode arm-activity query."""
    activity = arm_activity(client, dataset)
    verdicts: dict[str, QualityVerdict] = {}
    for row in activity.to_pylist():
        motion: float = float(row["right_motion"])
        segment_id: str = row["rerun_segment_id"]
        verdicts[segment_id] = QualityVerdict(right_motion=motion, flagged=motion < min_motion)
    return verdicts


def write_quality_layer(segment_id: str, verdict: QualityVerdict, min_motion: float, out_dir: Path) -> Path:
    """Write one segment's quality verdict to a tiny RRD that shares its recording id.

    The shared ``recording_id`` is what lets this RRD attach as a new layer on the existing
    segment; ``/quality`` is a fresh entity, so the schema stays additive.
    """
    path: Path = out_dir / f"{segment_id}_quality.rrd"
    label: str = "flagged-low-motion" if verdict.flagged else "ok"
    # send_properties=False is essential: a layer must NOT write /__properties (RecordingInfo).
    # If it does, that chunk collides with the base recording's recording-level properties when
    # the catalog merges the layers by recording_id, which clobbers the segment's identity and
    # stops the dataset's default blueprint from applying on open (cameras lose their world target).
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id=APPLICATION_ID, recording_id=segment_id, send_properties=False
    )
    recording.save(str(path))
    recording.log(
        "quality",
        rr.TextDocument(
            f"**verdict:** {label}\n\nright_motion: {verdict.right_motion:.2f} rad (cutoff {min_motion:.2f})",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    # Also log the metric as a static scalar so the verdict is queryable, not just readable.
    recording.log("quality/right_motion", rr.Scalars(verdict.right_motion), static=True)
    recording.flush(timeout_sec=30.0)
    return path


def register_quality_layer(dataset: DatasetEntry, rrds: list[Path], layer_name: str) -> None:
    """Re-register the per-episode quality RRDs as a new ``layer_name`` layer on their segments."""
    dataset.register(
        [rrd.resolve().as_uri() for rrd in rrds],
        layer_name=layer_name,
        on_duplicate=OnDuplicateSegmentLayer.REPLACE,
    ).wait()


@dataclass
class EnrichConfig:
    """Configuration for the post-registration enrichment step."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (``rerun server``)."""
    dataset_name: str = "trossen_oss"
    """Catalog dataset whose segments are enriched."""
    min_motion: float = 6.0
    """Right-arm joint-travel cutoff (rad): episodes below this are flagged low-motion."""
    layer_name: str = "quality"
    """Layer name for the derived quality verdict (registered with REPLACE, so it is re-runnable)."""
    output_dir: Path = Path("outputs")
    """Directory for the per-episode quality RRDs (written under ``outputs/quality/``)."""


def main(cfg: EnrichConfig) -> None:
    """Derive per-episode quality verdicts and attach them as a new catalog layer."""
    client: CatalogClient = CatalogClient(cfg.catalog_url)
    dataset: DatasetEntry = client.get_dataset(cfg.dataset_name)

    verdicts: dict[str, QualityVerdict] = motion_verdicts(client, dataset, cfg.min_motion)
    quality_dir: Path = cfg.output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    rrds: list[Path] = [
        write_quality_layer(segment_id, verdict, cfg.min_motion, quality_dir)
        for segment_id, verdict in verdicts.items()
    ]
    register_quality_layer(dataset, rrds, cfg.layer_name)

    flagged: int = sum(1 for verdict in verdicts.values() if verdict.flagged)
    print(f"enriched {len(verdicts)} segments with a '{cfg.layer_name}' layer (raw recordings untouched)")
    print(f"  flagged {flagged} low-motion episode(s) below {cfg.min_motion:.2f} rad of right-arm travel")
    layers: list[str] = (
        dataset.segment_table().select("rerun_layer_names").to_arrow_table().column("rerun_layer_names").to_pylist()[0]
    )
    print(f"  segment layers now: {sorted(layers)} — browse the 'quality' verdict in the catalog viewer")
