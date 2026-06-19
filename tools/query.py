from __future__ import annotations

import tyro

from trossen_oss.query import QueryConfig, main

if __name__ == "__main__":
    main(tyro.cli(QueryConfig, description="Query the local Rerun catalog across all episodes"))
