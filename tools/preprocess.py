from __future__ import annotations

import tyro

from trossen_oss.preprocessing import PreprocessingConfig, main

if __name__ == "__main__":
    main(tyro.cli(PreprocessingConfig, description="Generate RRD files"))
