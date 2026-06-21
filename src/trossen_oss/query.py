"""Query the Trossen catalog: ask questions across every registered episode.

This is the *query* part of the **Refine** stage of the experiment loop
(Collect → Refine → Train → Deploy; Refine = register + enrich + query). With the
episodes registered as catalog segments, we ask cross-dataset
questions through the ``CatalogClient`` reader + DataFusion — the DataFrame API
for segment metadata and SQL for per-signal questions — without opening the
viewer. It mirrors the analysis notebooks but runs headless against the real
schema (timeline ``message_log_time``; per-joint ``Scalars`` at
``/robot_*/joints/<joint>``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyarrow as pa
from datafusion import col
from rerun.catalog import CatalogClient, DatasetEntry

from trossen_oss.catalog import DEFAULT_CATALOG_URL

TIMELINE: str = "message_log_time"
"""Index timeline used for per-signal reads."""
ARM_JOINTS: tuple[str, ...] = tuple(f"joint_{i}" for i in range(6))
"""The six revolute arm joints (the prismatic carriage joints are the gripper)."""
REVOLUTE_VELOCITY_LIMIT: float = math.pi
"""URDF velocity limit shared by every revolute arm joint (rad/s)."""


def _scalar(joint: str) -> str:
    """Quoted DataFusion column name for a joint's scalar value (a one-element list)."""
    return f'"{joint}:Scalars:scalars"[1]'


def segment_overview(dataset: DatasetEntry) -> pa.Table:
    """One row per episode: segment id, registered layers, chunk count, byte size.

    A single ``segment_table`` round-trip — the cheap way to inventory the dataset.
    """
    return (
        dataset.segment_table()
        .select("rerun_segment_id", "rerun_layer_names", "rerun_num_chunks", "rerun_size_bytes")
        .sort(col("rerun_segment_id"))
        .to_arrow_table()
    )


def arm_activity(client: CatalogClient, dataset: DatasetEntry) -> pa.Table:
    """Per-episode total joint travel (summed ``max − min``) for each arm.

    A cross-dataset comparison you can't see from one file: it reveals which arm
    follows a fixed script (identical every episode) versus which one is task-driven
    (varies episode to episode). One reader over both arms' joints; MAX/MIN skip the
    NULL rows that the per-entity reader layout produces.
    """
    left = [f"/robot_left/joints/{j}" for j in ARM_JOINTS]
    right = [f"/robot_right/joints/{j}" for j in ARM_JOINTS]
    left_sum = " + ".join(f"MAX({_scalar(j)}) - MIN({_scalar(j)})" for j in left)
    right_sum = " + ".join(f"MAX({_scalar(j)}) - MIN({_scalar(j)})" for j in right)
    ctx = client.ctx
    ctx.register_view("arms", dataset.filter_contents(left + right).reader(index=TIMELINE))
    return ctx.sql(
        f"""
        SELECT rerun_segment_id,
               {left_sum} AS left_motion,
               {right_sum} AS right_motion
        FROM arms
        GROUP BY rerun_segment_id
        ORDER BY right_motion DESC
        """
    ).to_arrow_table()


def velocity_limit_violations(
    client: CatalogClient,
    dataset: DatasetEntry,
    joint: str,
    velocity_limit: float = REVOLUTE_VELOCITY_LIMIT,
) -> pa.Table:
    """How often each episode exceeds ``joint``'s URDF velocity limit.

    The experiment-loop page's quality example ("across every episode, how often does
    this joint exceed its velocity limit?"). We derive velocity by finite-differencing
    the 1 kHz position signal (``LAG`` over the per-segment time order), then count the
    samples above ``velocity_limit`` and the peak speed per episode — a quality signal
    that turns the dataset into something that describes itself.
    """
    column: str = _scalar(joint)
    ctx = client.ctx
    ctx.register_view("vel_signal", dataset.filter_contents([joint]).reader(index=TIMELINE))
    return ctx.sql(
        f"""
        WITH velocity AS (
          SELECT rerun_segment_id,
                 ({column} - LAG({column}) OVER w)
                 / (NULLIF(CAST({TIMELINE} AS DOUBLE) - LAG(CAST({TIMELINE} AS DOUBLE)) OVER w, 0) / 1e9) AS v
          FROM vel_signal
          WINDOW w AS (PARTITION BY rerun_segment_id ORDER BY {TIMELINE})
        )
        SELECT rerun_segment_id,
               COUNT(*) FILTER (WHERE ABS(v) > {velocity_limit}) AS violations,
               MAX(ABS(v)) AS peak_velocity
        FROM velocity
        GROUP BY rerun_segment_id
        ORDER BY peak_velocity DESC
        """
    ).to_arrow_table()


@dataclass
class QueryConfig:
    """Configuration for the headless catalog query demo."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun data platform (``rerun server``)."""
    dataset_name: str = "trossen_oss"
    """Catalog dataset to query."""
    joint: str = "/robot_right/joints/joint_1"
    """Joint entity path for the velocity-limit query (defaults to the active right arm)."""


def main(cfg: QueryConfig) -> None:
    """Run a few cross-dataset quality queries against the catalog and print results."""
    client: CatalogClient = CatalogClient(cfg.catalog_url)
    dataset: DatasetEntry = client.get_dataset(cfg.dataset_name)

    overview: pa.Table = segment_overview(dataset)
    total_bytes: int = sum(overview.column("rerun_size_bytes").to_pylist())
    layers = overview.column("rerun_layer_names").to_pylist()[0]
    ids = overview.column("rerun_segment_id")
    print(f"Dataset '{cfg.dataset_name}': {overview.num_rows} episodes, layers {layers}, {total_bytes / 1e9:.1f} GB")
    print(f"  segments: {ids[0].as_py()} … {ids[-1].as_py()}")

    activity: pa.Table = arm_activity(client, dataset)
    left = activity.column("left_motion").to_pylist()
    right = activity.column("right_motion").to_pylist()
    print("\nPer-arm motion (summed joint travel per episode, rad):")
    print(f"  left arm : {min(left):.2f}–{max(left):.2f}  (≈constant — a fixed script)")
    print(f"  right arm: {min(right):.2f}–{max(right):.2f}  (varies — the task-driven arm)")
    print("  most active episodes (right arm):")
    for row in activity.slice(0, 3).to_pylist():
        print(f"    {row['rerun_segment_id']}: {row['right_motion']:.2f} rad")

    violations: pa.Table = velocity_limit_violations(client, dataset, cfg.joint)
    affected = sum(1 for v in violations.column("violations").to_pylist() if v > 0)
    print(f"\nHow often does {cfg.joint} exceed its {REVOLUTE_VELOCITY_LIMIT:.2f} rad/s URDF velocity limit?")
    print(f"  {affected} of {overview.num_rows} episodes exceed it; worst by peak velocity:")
    for row in violations.slice(0, 5).to_pylist():
        print(
            f"    {row['rerun_segment_id']}: {row['violations']} samples over limit, peak {row['peak_velocity']:.2f} rad/s"
        )
