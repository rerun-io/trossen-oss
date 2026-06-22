"""Fast pure-tensor tests for the toy trainer (no catalog).

Guarded with ``importorskip`` so they skip in the torch-free dev env and run in the
``dataloader`` env (``pixi run -e dataloader test``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from trossen_oss.train import (  # noqa: E402
    NUM_JOINTS,
    RIGHT_JOINTS,
    NextStateCollate,
    NextStatePolicy,
    compute_stats,
    run_epoch,
)


def test_policy_preserves_joint_dim() -> None:
    policy = NextStatePolicy()
    out = policy(torch.zeros(4, NUM_JOINTS))
    assert out.shape == (4, NUM_JOINTS)


def test_collate_drops_short_and_missing_windows() -> None:
    collate = NextStateCollate()
    full: dict[str, object] = {joint: torch.tensor([float(i), float(i) + 1.0]) for i, joint in enumerate(RIGHT_JOINTS)}
    short = dict(full)
    short[RIGHT_JOINTS[0]] = torch.tensor([1.0])  # one-element window at a segment boundary
    missing = dict(full)
    missing[RIGHT_JOINTS[1]] = None  # decoder returned nothing
    state, next_state = collate([full, short, missing])  # type: ignore[arg-type]
    assert state.shape == (1, NUM_JOINTS)  # only the complete sample survives
    assert next_state.shape == (1, NUM_JOINTS)
    assert torch.allclose(state[0], torch.arange(NUM_JOINTS, dtype=torch.float32))
    assert torch.allclose(next_state[0], torch.arange(1, NUM_JOINTS + 1, dtype=torch.float32))


def test_compute_stats_matches_torch() -> None:
    torch.manual_seed(0)
    states = torch.randn(256, NUM_JOINTS)
    loader = DataLoader(TensorDataset(states, states.clone()), batch_size=32)
    mean, std, count = compute_stats(loader)
    assert count == 256
    assert torch.allclose(mean, states.mean(dim=0), atol=1e-4)
    assert torch.allclose(std, states.std(dim=0, unbiased=False), atol=1e-3)


def test_run_epoch_reduces_loss() -> None:
    torch.manual_seed(0)
    states = torch.randn(512, NUM_JOINTS)
    targets = states.roll(1, dims=1)  # a deterministic permutation the MLP can fit
    loader = DataLoader(TensorDataset(states, targets), batch_size=64, shuffle=True)
    mean = states.mean(dim=0)
    std = states.std(dim=0).clamp_min(1e-6)
    policy = NextStatePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    first: float = run_epoch(policy, loader, mean, std, loss_fn, optimizer=None)
    for _ in range(20):
        run_epoch(policy, loader, mean, std, loss_fn, optimizer)
    last: float = run_epoch(policy, loader, mean, std, loss_fn, optimizer=None)
    assert last < first * 0.5  # the toy policy at least halves its loss
