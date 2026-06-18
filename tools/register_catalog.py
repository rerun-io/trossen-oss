from __future__ import annotations

import tyro

from trossen_oss.catalog import CatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(CatalogConfig, description="Register local RRD outputs into the Rerun catalog"))
