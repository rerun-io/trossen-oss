from __future__ import annotations

import tyro

from trossen_oss.train import TrainConfig, main

if __name__ == "__main__":
    main(tyro.cli(TrainConfig, description="Toy next-state training over a catalog query"))
