from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_CONFIG = ROOT / "docs" / "mkdocs.yml"
WALKTHROUGH = ROOT / "docs" / "examples" / "wdm_walkthrough.py"


@pytest.mark.skipif(find_spec("mkdocs") is None, reason="mkdocs is not installed")
def test_mkdocs_builds_walkthrough_page(tmp_path: Path) -> None:
    site_dir = tmp_path / "site"
    config_text = DOCS_CONFIG.read_text(encoding="utf-8")
    config_text = re.sub(
        r"^docs_dir:\s*.+$",
        f"docs_dir: '{ROOT / 'docs'}'",
        config_text,
        flags=re.MULTILINE,
    )
    config_text = re.sub(
        r"^site_dir:\s*.+$",
        f"site_dir: '{site_dir}'",
        config_text,
        flags=re.MULTILINE,
    )
    temp_config = tmp_path / "mkdocs.yml"
    temp_config.write_text(config_text, encoding="utf-8")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_path, existing_pythonpath])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "-f",
            str(temp_config),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    walkthrough_page = site_dir / "examples" / "wdm_walkthrough" / "index.html"
    assert walkthrough_page.exists()

    html = walkthrough_page.read_text(encoding="utf-8")
    assert "WDM Walkthrough" in html
    assert "Max abs reconstruction error" in html


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
