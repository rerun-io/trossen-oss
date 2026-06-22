from __future__ import annotations

import tyro

from trossen_oss.enrich import EnrichConfig, main

if __name__ == "__main__":
    main(tyro.cli(EnrichConfig, description="Enrich registered episodes with a derived quality layer"))
