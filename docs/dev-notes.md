# Releasing and Dev Notes

This repository is wired to publish to PyPI from GitHub Actions.

The publish workflow lives in `.github/workflows/pypi.yml` and runs when a version tag such as
`v0.1.0` is pushed. It builds the source distribution and wheel, checks the metadata with `twine`,
and then publishes to PyPI using GitHub trusted publishing.

## Current release flow

The package version is derived from Git tags via `setuptools-scm`.

Before cutting a release:

1. Make sure the release commit is already on `main`.
2. Run the local verification steps below.
3. Ask Commitizen what the next version should be.
4. Let Commitizen create the release tag.
5. Push `main` and the new tag to GitHub.
6. Watch the `pypi` workflow in GitHub Actions and confirm the publish succeeds.
7. Optionally create a GitHub Release that points at the same tag.

## Choosing the next version

Do not pick the next tag from memory. Use Commitizen:

```bash
cz bump --dry-run
```

That shows the current version and the next version Commitizen would create from the commits since
the last tag.

Then create the real release tag with:

```bash
cz bump
git push origin main --follow-tags
```

With the current commit rules:

- `fix:` creates a patch bump such as `0.1.0` -> `0.1.1`
- `feat:` creates a minor bump such as `0.1.0` -> `0.2.0`
- `add:` creates a minor bump such as `0.1.0` -> `0.2.0`

If you are doing a fix update right now, the expected path is:

```bash
cz bump --dry-run
cz bump
git push origin main --follow-tags
```

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

This project now uses `setuptools-scm`, so:

- remove the manual `project.version` field
- derive the package version from Git tags
- publish only from version tags such as `v0.1.0`

`Commitizen` is configured to read the version from SCM as well, so Git tags are the single source
of truth.

## Developer notes

- Core objects live under `src/wdm_transform/datatypes/`.
- Transform kernels live under `src/wdm_transform/transforms/`.
- Shared window helpers live under `src/wdm_transform/windows.py`.
- Keep the public API ergonomic: `from wdm_transform import TimeSeries, FrequencySeries, WDM`.
- Prefer small release commits that only touch packaging, docs, and versioning when preparing PyPI
  publishes.
