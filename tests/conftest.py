from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _current_branch_name() -> str:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        branch = "unknown"

    if branch == "HEAD":
        branch = "detached"

    # Keep directory names filesystem-safe and stable across machines.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", branch)


@pytest.fixture(scope="session")
def outdir() -> Path:
    output_dir = ROOT / "tests" / "test_out" / f"branch_{_current_branch_name()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir




