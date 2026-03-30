# Releasing and Dev Notes

This repository is already wired to publish to PyPI from GitHub Actions.

The publish workflow lives in `.github/workflows/pypi.yml` and runs when a GitHub Release is
published. It builds the source distribution and wheel, checks the metadata with `twine`, and then
publishes to PyPI using GitHub trusted publishing.

## Current release flow

The package version is currently set manually in `pyproject.toml`.

Before cutting a release:

1. Update `project.version` in `pyproject.toml`.
2. Keep the `tool.commitizen.version` value in sync.
3. Run the local verification steps below.
4. Commit and push the release changes.
5. Create and publish a GitHub Release with a matching tag such as `v0.1.0`.
6. Watch the `pypi` workflow in GitHub Actions and confirm the publish succeeds.

## Local verification

Use these commands from the repository root:

```bash
uv sync --frozen --extra dev
uv run pytest
uv build
uvx twine check dist/*
```

What these do:

- `uv run pytest` checks the package and test suite.
- `uv build` creates `dist/*.tar.gz` and `dist/*.whl`.
- `uvx twine check dist/*` validates the built package metadata and long description rendering.

## Build artifacts

Do not commit generated release artifacts.

Typical generated files and directories:

- `dist/`
- `build/`
- `src/*.egg-info/`

## README and PyPI

PyPI renders the package long description from `README.md`.

Keep these constraints in mind:

- Use absolute web URLs for images.
- Avoid links that only work inside the local repository checkout.
- Re-run `uvx twine check dist/*` after changing the README.

## Optional future change: SCM-based versioning

If this project switches to `setuptools-scm` later, the release process should change slightly:

- remove the manual `project.version` field
- derive the package version from Git tags
- publish only from version tags such as `v0.1.0`

If that change is made, update this file and the `Commitizen` configuration so there is only one
source of truth for the package version.

## Developer notes

- Core objects live under `src/wdm_transform/datatypes/`.
- Transform kernels live under `src/wdm_transform/transforms/`.
- Shared window helpers live under `src/wdm_transform/windows.py`.
- Keep the public API ergonomic: `from wdm_transform import TimeSeries, FrequencySeries, WDM`.
- Prefer small release commits that only touch packaging, docs, and versioning when preparing PyPI
  publishes.
