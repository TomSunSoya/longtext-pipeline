r"""Repository-local launcher for longtext-pipeline.

Run from the repo root without requiring PYTHONPATH or editable installation:

    python .\longtext.py run input.txt
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Avoid Windows code-page crashes when CLI output contains unicode symbols.
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from longtext_pipeline.cli import app  # noqa: E402


if __name__ == "__main__":
    app()
