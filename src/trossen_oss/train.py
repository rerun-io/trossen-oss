"""Toy "the training set is a query" example — a next-state model over a catalog query.

The Train stage of the experiment loop. We pick the training / holdout episodes
with a catalog query (rank by right-arm motion via :func:`arm_activity`), then
**stream** their per-joint ``Scalars`` straight out of the catalog through Rerun's
experimental PyTorch dataloader (:class:`RerunIterableDataset`) and fit a tiny MLP
that predicts the right arm's next joint vector from the current one. The training
run is logged back to Rerun and registered as a segment in a ``trossen_oss_runs``
catalog dataset, so runs are browsable alongside the episodes. Toy by design — the
goal is to exercise the register -> query -> dataloader -> train loop, not accuracy.

The dataloader usage mirrors Rerun's upstream example and how-to:
- https://rerun.io/docs/howto/train/dataloader
- https://github.com/rerun-io/rerun/tree/main/examples/python/dataloader

As there, the Rerun dataset is wrapped directly in a PyTorch ``DataLoader`` (a
:class:`NextStateCollate` assembles batches and drops short windows), shuffling is
driven by ``set_epoch`` for the iterable dataset (or the sampler for the map one),
and ``num_workers`` / ``fetch_size`` prefetch from the server — nothing is drained
into memory up front.

Needs a running catalog (``pixi run serve``) with the dataset registered
(``pixi run register``).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing as torch_mp
from jaxtyping import Float
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    NumericDecoder,
    RerunIterableDataset,
    RerunMapDataset,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trossen_oss.catalog import DEFAULT_CATALOG_URL
from trossen_oss.query import arm_activity

RIGHT_JOINTS: tuple[str, ...] = tuple(f"/robot_right/joints/joint_{i}" for i in range(6))
"""The six right-arm revolute joints — the task-driven arm, used as the model state."""
TIMELINE: str = "message_log_time"
"""Index timeline used for fixed-rate sampling."""
NUM_JOINTS: int = len(RIGHT_JOINTS)
"""State / prediction dimensionality (one scalar per right-arm joint)."""

type RerunDataset = RerunIterableDataset | RerunMapDataset
"""Either Rerun dataloader dataset flavour — iterable (streaming) or map (random-access)."""
type StateBatch = tuple[Float[Tensor, "b 6"], Float[Tensor, "b 6"]]
"""A collated ``(state, next_state)`` mini-batch."""


def select_segments(
    client: CatalogClient, dataset: DatasetEntry, num_train: int, num_val: int
) -> tuple[list[str], list[str]]:
    """Pick train / holdout segments by querying the catalog (the training set is a query).

    Ranks episodes by total right-arm joint travel (:func:`arm_activity`, already
    sorted by ``right_motion`` descending) and takes the most active ones — the
    episodes with the most motion make the richest next-state data.

    Args:
        client: Connected catalog client.
        dataset: The registered dataset to read from.
        num_train: Number of episodes for the training split.
        num_val: Number of episodes for the held-out validation split.

    Returns:
        A ``(train_segment_ids, val_segment_ids)`` pair.
    """
    ranked: list[str] = arm_activity(client, dataset).column("rerun_segment_id").to_pylist()
    chosen: list[str] = ranked[: num_train + num_val]
    return chosen[:num_train], chosen[num_train : num_train + num_val]


def build_dataset(
    dataset: DatasetEntry,
    segments: list[str],
    rate_hz: float,
    style: Literal["iterable", "map"] = "iterable",
    fetch_size: int = 256,
) -> RerunDataset:
    """Build a next-state dataset over ``segments``: one ``(now, next)`` pair per joint.

    Each joint field uses a one-period window ``(0, period_ns)`` so a sample carries
    ``[q_t, q_{t+1/rate}]`` for that joint (the window offsets are in nanoseconds on the
    timestamp timeline). :class:`NextStateCollate` splits each pair into the current
    state and the prediction target and drops the short windows at segment starts.

    Args:
        dataset: The registered dataset to stream from.
        segments: Segment ids selected by the query.
        rate_hz: Fixed sampling rate applied to the 1 kHz signal.
        style: ``"iterable"`` (:class:`RerunIterableDataset`, internal shuffling) or
            ``"map"`` (:class:`RerunMapDataset`, random access via DataLoader samplers).
        fetch_size: Samples fetched per server query (iterable dataset only).
    """
    period_ns: int = round(1e9 / rate_hz)
    fields: dict[str, Field] = {
        joint: Field(f"{joint}:Scalars:scalars", decode=NumericDecoder(), window=(0, period_ns))
        for joint in RIGHT_JOINTS
    }
    source: DataSource = DataSource(dataset=dataset, segments=segments)
    sampling: FixedRateSampling = FixedRateSampling(rate_hz=rate_hz)
    if style == "map":
        return RerunMapDataset(source=source, index=TIMELINE, fields=fields, timeline_sampling=sampling)
    return RerunIterableDataset(
        source=source, index=TIMELINE, fields=fields, timeline_sampling=sampling, fetch_size=fetch_size
    )


class NextStateCollate:
    """Picklable collate that assembles ``(state, next_state)`` batches from streamed windows.

    Each joint :class:`Field` carries a one-period window ``[q_t, q_{t+1}]``; we drop
    samples whose window is short — the first grid point of a segment has no value
    at-or-before it, which the dataloader contract leaves to the consumer to filter —
    and stack the present / next halves into ``(B, 6)`` tensors. A plain class (not a
    closure) so it pickles cleanly across ``DataLoader`` worker processes.
    """

    def __call__(self, samples: list[dict[str, Tensor | None]]) -> StateBatch:
        states: list[Tensor] = []
        next_states: list[Tensor] = []
        for sample in samples:
            pairs: list[Tensor | None] = [sample[joint] for joint in RIGHT_JOINTS]
            if any(pair is None or pair.numel() != 2 for pair in pairs):
                continue
            states.append(torch.stack([pair[0] for pair in pairs]))  # type: ignore[index]
            next_states.append(torch.stack([pair[1] for pair in pairs]))  # type: ignore[index]
        if not states:
            empty: Float[Tensor, "0 6"] = torch.empty(0, NUM_JOINTS)
            return empty, empty
        return torch.stack(states).float(), torch.stack(next_states).float()


def compute_stats(loader: DataLoader) -> tuple[Float[Tensor, "6"], Float[Tensor, "6"], int]:
    """Stream one pass over ``loader`` to get per-joint ``(mean, std, sample_count)``.

    A single streaming pass — like LeRobot precomputing dataset statistics — so state
    and next-state can be standardized into unit-variance space before the MSE.
    """
    total: Float[Tensor, "6"] = torch.zeros(NUM_JOINTS)
    total_sq: Float[Tensor, "6"] = torch.zeros(NUM_JOINTS)
    count: int = 0
    for state, _ in loader:
        if state.shape[0] == 0:
            continue
        total += state.sum(dim=0)
        total_sq += (state * state).sum(dim=0)
        count += state.shape[0]
    mean: Float[Tensor, "6"] = total / max(count, 1)
    variance: Float[Tensor, "6"] = (total_sq / max(count, 1)) - mean * mean
    std: Float[Tensor, "6"] = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean, std, count


class NextStatePolicy(nn.Module):
    """Tiny MLP that predicts the next right-arm joint vector from the current one."""

    def __init__(self, num_joints: int = NUM_JOINTS, hidden: int = 64) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(num_joints, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_joints),
        )

    def forward(self, state: Float[Tensor, "b 6"]) -> Float[Tensor, "b 6"]:
        out: Float[Tensor, "b 6"] = self.net(state)
        return out


def run_epoch(
    policy: NextStatePolicy,
    loader: DataLoader,
    mean: Float[Tensor, "6"],
    std: Float[Tensor, "6"],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one streamed pass over ``loader``; train when ``optimizer`` is given, else evaluate.

    State and next-state are standardized with the shared per-joint ``mean`` / ``std``
    (they follow the same distribution), so the MSE is in unit-variance space. Empty
    batches (every sample filtered as a short window) are skipped. Returns mean loss.
    """
    training: bool = optimizer is not None
    policy.train(training)
    loss_sum: float = 0.0
    sample_count: int = 0
    with torch.set_grad_enabled(training):
        for state, target in loader:
            if state.shape[0] == 0:
                continue
            state_norm: Float[Tensor, "b 6"] = (state - mean) / std
            target_norm: Float[Tensor, "b 6"] = (target - mean) / std
            prediction: Float[Tensor, "b 6"] = policy(state_norm)
            loss: Tensor = loss_fn(prediction, target_norm)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * state.shape[0]
            sample_count += state.shape[0]
    return loss_sum / max(sample_count, 1)


def make_loader(dataset: RerunDataset, batch_size: int, num_workers: int) -> DataLoader:
    """Wrap a Rerun dataset directly in a PyTorch ``DataLoader`` (the upstream idiom).

    Shuffling is the map dataset's sampler or — for the iterable dataset — internal,
    reseeded per epoch via :meth:`set_epoch`. Workers prefetch from the server.
    """
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "collate_fn": NextStateCollate(),
        "num_workers": num_workers,
        "shuffle": isinstance(dataset, RerunMapDataset),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)  # type: ignore[arg-type]


def runs_blueprint() -> rrb.Blueprint:
    """Loss-curve layout for the runs dataset: train / val loss over the ``epoch`` timeline."""
    return rrb.Blueprint(
        rrb.TimeSeriesView(origin="/loss", name="Training loss"),
        rrb.TimePanel(timeline="epoch", play_state=rrb.components.PlayState.Paused),
    )


def register_training_run(
    client: CatalogClient, run_path: Path, runs_dataset_name: str, blueprint: Path | None
) -> DatasetEntry:
    """Register a finished training-run RRD as a new segment in the runs dataset.

    The run is its own recording, so its ``recording_id`` becomes the segment id and
    successive runs accumulate as history. The dataset is created on first use and
    never recreated, so a re-run with the same id replaces just that segment.

    Args:
        client: Connected catalog client.
        run_path: The finalized ``train_run.rrd`` to register.
        runs_dataset_name: Catalog dataset that collects runs (created if absent).
        blueprint: Optional saved ``.rbl`` registered as the dataset default.

    Returns:
        The runs :class:`DatasetEntry`.
    """
    runs: DatasetEntry = client.create_dataset(runs_dataset_name, exist_ok=True)
    runs.register([run_path.resolve().as_uri()], layer_name="base", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    if blueprint is not None and blueprint.exists():
        runs.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
    return runs


@dataclass
class TrainConfig:
    """Configuration for the toy next-state training run."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun data platform (``rerun server``)."""
    dataset_name: str = "trossen_oss"
    """Catalog dataset to train on."""
    num_train_segments: int = 8
    """Number of (most-active) episodes used for training."""
    num_val_segments: int = 2
    """Number of held-out episodes used for validation."""
    rate_hz: float = 10.0
    """Fixed sampling rate (downsamples the 1 kHz signal so a step is meaningful)."""
    dataset_style: Literal["iterable", "map"] = "iterable"
    """Rerun dataset flavour: ``iterable`` (streaming) or ``map`` (random access)."""
    epochs: int = 10
    """Number of training epochs (the toy loss is essentially converged well before this)."""
    batch_size: int = 256
    """Mini-batch size."""
    num_workers: int = 0
    """DataLoader worker processes that prefetch from the catalog (0 = main process).

    The Rerun dataloader requires the ``spawn`` start method for workers (forked
    workers deadlock on their first catalog call); :func:`main` sets it, so any
    value > 0 prefetches safely. Defaults to 0 for a snappy, dependency-light run.
    """
    fetch_size: int = 256
    """Samples fetched per server query (iterable dataset only)."""
    hidden: int = 64
    """Hidden width of the MLP."""
    learning_rate: float = 1e-3
    """Adam learning rate."""
    output_dir: Path = Path("outputs")
    """Directory for the logged training-run RRD."""
    runs_dataset_name: str = "trossen_oss_runs"
    """Catalog dataset that collects training runs (one segment per run)."""
    register_run: bool = True
    """Register the finished run as a segment in ``runs_dataset_name``."""


def main(cfg: TrainConfig) -> None:
    """Curate a dataset by query, stream-train the toy next-state policy, and log the run to Rerun."""
    # The Rerun dataloader needs 'spawn' for DataLoader workers (forked workers deadlock on
    # their first catalog call). Set it before any dataset is built so num_workers > 0 is safe
    # and the construction-time warning stays quiet; harmless when num_workers == 0.
    torch_mp.set_start_method("spawn", force=True)

    client: CatalogClient = CatalogClient(cfg.catalog_url)
    dataset: DatasetEntry = client.get_dataset(cfg.dataset_name)

    train_segments: list[str]
    val_segments: list[str]
    train_segments, val_segments = select_segments(client, dataset, cfg.num_train_segments, cfg.num_val_segments)
    print(f"training set is a query: {len(train_segments)} train / {len(val_segments)} val (most-active episodes)")
    print(f"  train: {train_segments}")
    print(f"  val  : {val_segments}")

    train_ds: RerunDataset = build_dataset(dataset, train_segments, cfg.rate_hz, cfg.dataset_style, cfg.fetch_size)
    val_ds: RerunDataset = build_dataset(dataset, val_segments, cfg.rate_hz, cfg.dataset_style, cfg.fetch_size)
    train_loader: DataLoader = make_loader(train_ds, cfg.batch_size, cfg.num_workers)
    val_loader: DataLoader = make_loader(val_ds, cfg.batch_size, cfg.num_workers)
    print(f"streaming a {cfg.dataset_style} dataset from the catalog ({len(train_ds)} train samples before trimming)")

    mean: Float[Tensor, "6"]
    std: Float[Tensor, "6"]
    train_count: int
    mean, std, train_count = compute_stats(train_loader)
    print(f"computed per-joint stats over {train_count} streamed train transitions")

    policy: NextStatePolicy = NextStatePolicy(hidden=cfg.hidden)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    loss_fn: nn.Module = nn.MSELoss()

    run_id: str = f"run-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    rr.init("trossen_oss_train", recording_id=run_id)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_path: Path = cfg.output_dir / "train_run.rrd"
    rr.save(str(run_path))
    rr.log(
        "run_info",
        rr.TextDocument(
            f"run_id: {run_id}\n"
            f"train segments: {train_segments}\n"
            f"val segments: {val_segments}\n"
            f"dataset_style={cfg.dataset_style} rate_hz={cfg.rate_hz} epochs={cfg.epochs} "
            f"batch_size={cfg.batch_size} num_workers={cfg.num_workers} hidden={cfg.hidden} lr={cfg.learning_rate}"
        ),
        static=True,
    )

    num_params: int = sum(p.numel() for p in policy.parameters())
    print(f"\ntraining {num_params}-param MLP for {cfg.epochs} epochs (CPU):")
    first_val: float = run_epoch(policy, val_loader, mean, std, loss_fn, optimizer=None)
    print(f"  epoch  0  val {first_val:.4f}  (before training)")
    for epoch in range(1, cfg.epochs + 1):
        if isinstance(train_ds, RerunIterableDataset):
            train_ds.set_epoch(epoch)
        train_loss: float = run_epoch(policy, train_loader, mean, std, loss_fn, optimizer)
        val_loss: float = run_epoch(policy, val_loader, mean, std, loss_fn, optimizer=None)
        rr.set_time("epoch", sequence=epoch)
        rr.log("loss/train", rr.Scalars(train_loss))
        rr.log("loss/val", rr.Scalars(val_loss))
        print(f"  epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}")

    final_val: float = run_epoch(policy, val_loader, mean, std, loss_fn, optimizer=None)
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    if recording is not None:
        recording.flush(timeout_sec=30.0)  # finalize the RRD before registering
    print(f"\nval loss {first_val:.4f} -> {final_val:.4f} ({100 * (1 - final_val / first_val):.0f}% lower)")
    print(f"logged loss curves to {run_path}")

    if cfg.register_run:
        blueprint_path: Path = cfg.output_dir / "train_run_blueprint.rbl"
        runs_blueprint().save("trossen_oss_train", str(blueprint_path))
        runs: DatasetEntry = register_training_run(client, run_path, cfg.runs_dataset_name, blueprint_path)
        print(
            f"registered run '{run_id}' as a segment in dataset '{cfg.runs_dataset_name}' "
            f"({len(runs.segment_ids())} run(s) total) — browse it in the catalog viewer"
        )
