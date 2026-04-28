from __future__ import annotations
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_CONFIG = ROOT / "mkdocs.yml"
DOCS_CONFIG_REL = Path("mkdocs.yml")
WALKTHROUGH = ROOT / "docs" / "examples" / "wdm_walkthrough.py"
BENCHMARK_SCRIPT = ROOT / "docs" / "examples" / "generate_benchmark_plot.py"


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

    benchmark_page = site_dir / "benchmarks" / "index.html"
    assert benchmark_page.exists()

    benchmark_html = benchmark_page.read_text(encoding="utf-8")
    assert "checked-in benchmark snapshot" in benchmark_html
    assert "benchmark_runtime.png" in benchmark_html

    study_page = site_dir / "studies" / "toymodels" / "monochrome_stationary_psd" / "index.html"
    assert study_page.exists()

    study_html = study_page.read_text(encoding="utf-8")
    assert "Sinusoid in Colored Noise" in study_html
    assert "outdir_monochrome_stationary_psd/posterior_comparison.png" in study_html
    assert "jupyter-wrapper" not in study_html

    lisa_page = site_dir / "studies" / "lisa" / "lisa_demo" / "index.html"
    assert lisa_page.exists()

    lisa_html = lisa_page.read_text(encoding="utf-8")
    assert "LISA Galactic-Binary Study" in lisa_html
    assert (
        "lisa/outdir_lisa/galactic_background/seed_0/posterior_marginals_compare.png"
        in lisa_html
    )
    assert (
        "lisa/outdir_lisa/galactic_background/seed_0/posterior_interval_compare.png"
        in lisa_html
    )
    assert "data_generation.py" in lisa_html
    assert "jupyter-wrapper" not in lisa_html

    api_page = site_dir / "reference" / "api" / "index.html"
    assert api_page.exists()

    api_html = api_page.read_text(encoding="utf-8")
    assert "The generated API reference now lives at the bottom of" in api_html
    assert "API Overview" in api_html
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


def test_benchmark_artifact_script_executes(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_path, existing_pythonpath])

    json_path = tmp_path / "benchmark_results.json"
    plot_path = tmp_path / "benchmark_runtime.png"

    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--backends",
            "numpy",
            "--n",
            "512",
            "1024",
            "--runs",
            "1",
            "--output-json",
            str(json_path),
            "--output-plot",
            str(plot_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json_path.exists()
    assert plot_path.exists()
