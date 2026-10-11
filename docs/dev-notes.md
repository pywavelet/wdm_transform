# Releasing and Dev Notes

This repository is wired to publish to PyPI from GitHub Actions.

The publish workflow lives in `.github/workflows/pypi.yml` and runs when a qualifying commit lands
on `main`. A qualifying commit starts with `fix:`, `feat:`, or `add:`.

The workflow runs the tests, asks Commitizen to bump the version, creates the release commit and Git
tag, builds the source distribution and wheel, checks the metadata with `twine`, and then publishes
to PyPI using GitHub trusted publishing.

## Current release flow

The package version is derived from Git tags via `setuptools-scm`.

Before cutting a release:

1. Make sure the user-facing change is committed with a `fix:`, `feat:`, or `add:` message.
2. Run the local verification steps below when practical.
3. Push the commit to `main`.
4. Watch the `pypi` workflow in GitHub Actions and confirm the publish succeeds.
5. Optionally create a GitHub Release that points at the generated tag.

## Choosing the next version

Do not pick the next tag from memory. Commitizen chooses the version in CI from the commits since
the last release tag.

To preview what CI will do locally, use:

```bash
cz bump --dry-run
```

That shows the current version and the next version Commitizen would create from the commits since
the last tag.

With the current commit rules:

- `fix:` creates a patch bump such as `0.1.0` -> `0.1.1`
- `feat:` creates a minor bump such as `0.1.0` -> `0.2.0`
- `add:` creates a minor bump such as `0.1.0` -> `0.2.0`

To release, push a qualifying commit to `main`:

```bash
git push origin main
```

Do not run `cz bump` manually for the normal release path. The workflow does that after tests pass.

If the workflow needs to be rerun without a new commit, use the manual `workflow_dispatch` button
for the `pypi` workflow. If the current `main` commit already has a `v*` release tag, the workflow
reuses that tag instead of bumping again.

## Manual fallback

If CI cannot push the release commit or tag, the equivalent manual commands are:

```bash
cz bump --dry-run
cz bump --annotated-tag
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

## Developer notes

- Core objects live under `src/wdm_transform/datatypes/`.
- Transform kernels live under `src/wdm_transform/transforms/`.
- Shared window helpers live under `src/wdm_transform/windows.py`.
- Keep the public API ergonomic: `from wdm_transform import TimeSeries, FrequencySeries, WDM`.

## Gaussian Addon Baseline And Conventions

Work started from `f5ca7d1624cefac1d8d59fd98e1527dafd2e559a` on `WDG_addon`.
The unchanged test suite passed **91 tests with one CuPy skip** in 35.37 seconds
after development and optional JAX dependencies were installed. The environment
was Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1, JAX/JAXlib 0.9.2 on macOS arm64.
JAX x64 is enabled by the existing backend. No lockfile update was needed.

Ignored `test_output/wdg_baseline/` retains environment, original CLI results,
default-window matched timings, and deterministic noise/chirp/impulse/sine
coefficient and round-trip arrays. `tests/test_meyer_regression.py` contains
compact complete pre-change coefficient fixtures. The original CLI fixes
`nt=1024,a=0.4,batch=3`; additional baseline measurements use actual default
`a=1/3,d=1`, `nt=64/256`, N=4096/65536 and batch 1/3.

The normalized Gaussian coordinate is `u=2*offset/nt`, with the same
`sqrt(2/nt)` scaling as Meyer. Channel spacing is `nt/2` FFT bins. Gaussian
support q retains offsets `[-q*nt/2,q*nt/2)`, independent of the parity lattice
used for orthogonalization. Scientific validation checks cross-band shifts,
the complete basis Gram structure and independent reference agreement.
The Wilson phase is `Cnm=1` when `n+m` is even and `Cnm=i` when it is odd,
as given by the existing formula and implementation.

`wdm_transform_window_benchmark` saves machine-readable results and a report
under ignored `test_output/wdg_benchmarks/`. Use `--baseline-source PATH/src`
with an archived starting source to run a matched before/after Meyer comparison
without checking out another branch. Timings include synchronized first calls
and warm repetitions; host allocation probes run separately. JAX device allocator
peaks are not measured. The detailed validation record is retained locally in
`test_output/wdg_benchmarks/validation_report.md`; the published
[window selection guide](guide/choosing-a-window.md) explains the practical
tradeoffs with a matched comparison figure.
