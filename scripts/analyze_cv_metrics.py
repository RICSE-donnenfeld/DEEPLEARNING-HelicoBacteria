from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    root_script = Path(__file__).resolve().parent.parent / "analyze_cv_metrics.py"
    runpy.run_path(str(root_script), run_name="__main__")

