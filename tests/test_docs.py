from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_CONFIG = ROOT / "docs" / "mkdocs.yml"
DOCS_CONFIG_REL = Path("docs") / "mkdocs.yml"
WALKTHROUGH = ROOT / "docs" / "examples" / "wdm_walkthrough.py"


@pytest.mark.skipif(find_spec("mkdocs") is None, reason="mkdocs is not installed")
def test_mkdocs_builds_walkthrough_page(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_path, existing_pythonpath])

    mkdocs_bin = Path(sys.executable).with_name("mkdocs")
    command = [str(mkdocs_bin), "build", "-f", str(DOCS_CONFIG_REL)]
    if not mkdocs_bin.exists():
        command = [sys.executable, "-m", "mkdocs", "build", "-f", str(DOCS_CONFIG_REL)]

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    site_dir = ROOT / "site"
    walkthrough_page = site_dir / "examples" / "wdm_walkthrough" / "index.html"
    assert walkthrough_page.exists()

    html = walkthrough_page.read_text(encoding="utf-8")
    assert "WDM Walkthrough" in html
    assert "Max abs reconstruction error" in html

    api_page = site_dir / "reference" / "api" / "index.html"
    assert api_page.exists()

    api_html = api_page.read_text(encoding="utf-8")
    assert "Core Objects" in api_html
    assert "Backend Utilities" in api_html
    assert "Plotting Helpers" in api_html
    assert "TimeSeries dataclass" not in api_html
    assert "FrequencySeries dataclass" not in api_html
    assert "WDM dataclass" not in api_html
    assert "Implementation:" not in api_html
    assert "Source code in" in api_html
    assert "<img" in html or "jp-RenderedImage" in html


def test_walkthrough_script_executes() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_path, existing_pythonpath])

    subprocess.run(
        [sys.executable, str(WALKTHROUGH)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
