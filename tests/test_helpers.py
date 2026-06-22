"""Pure-Python unit tests that need neither a catalog nor torch (run in the dev env)."""

from __future__ import annotations

from pathlib import Path

from trossen_oss.preprocessing import _episode_number, discover_episode_mcaps, episode_recording_id
from trossen_oss.query import _scalar


def test_episode_recording_id_zero_pads() -> None:
    assert episode_recording_id(Path("episode_7_proto.mcap")) == "episode_007"
    assert episode_recording_id(Path("episode_42_proto.mcap")) == "episode_042"
    assert episode_recording_id(Path("episode_100_proto.mcap")) == "episode_100"


def test_episode_number_parses_digits() -> None:
    assert _episode_number(Path("episode_13_proto.mcap")) == 13
    assert _episode_number(Path("episode_0_proto.mcap")) == 0


def test_scalar_quotes_column() -> None:
    assert _scalar("/robot_right/joints/joint_1") == '"/robot_right/joints/joint_1:Scalars:scalars"[1]'


def test_discover_episode_mcaps_sorted_numerically(tmp_path: Path) -> None:
    for number in (2, 10, 1):
        (tmp_path / f"episode_{number}_proto.mcap").touch()
    found: list[Path] = discover_episode_mcaps(tmp_path)
    # episode_10 must sort after episode_2 (numeric, not lexicographic).
    assert [p.name for p in found] == ["episode_1_proto.mcap", "episode_2_proto.mcap", "episode_10_proto.mcap"]
    assert [p.name for p in discover_episode_mcaps(tmp_path, limit=2)] == [
        "episode_1_proto.mcap",
        "episode_2_proto.mcap",
    ]
