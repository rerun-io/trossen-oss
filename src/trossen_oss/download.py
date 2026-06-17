"""Download Trossen episode MCAPs from the Hugging Face Hub into ``data/``.

This mirrors the eventual user flow: instead of the dev-machine ``data/``
symlink, users fetch episodes from the Hub into the same ``data/`` directory the
rest of the pipeline reads from. The dataset is not published yet, so
``HF_DATASET_REPO_ID`` is a placeholder — pass ``--repo-id`` or update the
constant once the real dataset id is known.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trossen_oss.paths import EPISODE_MCAP_GLOB, data_dir, episode_mcap_name

# PLACEHOLDER: the Trossen episode dataset is not on the Hub yet. Update this (or
# pass --repo-id) when the real dataset identity is decided.
HF_DATASET_REPO_ID: str = "pablovela5620/trossen-mjwarp-episodes"
"""Placeholder Hugging Face dataset repo id for the episode MCAPs."""

DEFAULT_NUM_EPISODES: int = 100
"""Default episode subset size for a quick start (full set is 1024)."""


def download_episodes(
    *,
    repo_id: str = HF_DATASET_REPO_ID,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    download_all: bool = False,
    dest: Path | None = None,
) -> list[Path]:
    """Download episode MCAPs from the Hub into the local data directory.

    Args:
        repo_id: Hugging Face dataset repo id to fetch from.
        num_episodes: Number of episodes (``episode_0``..``episode_{n-1}``) to
            fetch when ``download_all`` is False.
        download_all: When True, fetch every episode MCAP in the repo.
        dest: Destination directory; defaults to the resolved ``data/`` dir.

    Returns:
        Sorted list of downloaded episode MCAP paths.
    """
    # Accelerated transfers via hf-transfer (declared as a project dependency).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    destination: Path = dest if dest is not None else data_dir()
    destination.mkdir(parents=True, exist_ok=True)

    allow_patterns: list[str] = (
        [EPISODE_MCAP_GLOB] if download_all else [episode_mcap_name(index) for index in range(num_episodes)]
    )
    local_dir: str = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )
    return sorted(Path(local_dir).glob(EPISODE_MCAP_GLOB))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Trossen episode MCAPs from the Hugging Face Hub.")
    parser.add_argument("--repo-id", default=HF_DATASET_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument(
        "--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of episodes to fetch (from episode_0)"
    )
    parser.add_argument("--all", dest="download_all", action="store_true", help="Fetch every episode in the repo")
    parser.add_argument("--dest", type=Path, default=None, help="Destination directory (defaults to data/)")
    args = parser.parse_args()

    paths: list[Path] = download_episodes(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        download_all=args.download_all,
        dest=args.dest,
    )
    print(f"Downloaded {len(paths)} episode MCAP(s) into {args.dest or data_dir()}")


if __name__ == "__main__":
    main()
