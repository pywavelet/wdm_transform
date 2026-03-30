# Package Layout

The current source tree is designed to keep the WDM logic separate from data containers and backend
selection.

## Source modules

- `src/wdm_transform/datatypes/series.py`
  Defines `TimeSeries` and `FrequencySeries`.
- `src/wdm_transform/datatypes/wdm.py`
  Defines `WDM` as the packed coefficient object and thin wrapper API.
- `src/wdm_transform/transforms/__init__.py`
  Selects the backend-specific transform module lazily.
- `src/wdm_transform/transforms/xp_backend.py`
  Holds the shared NumPy/CuPy forward and inverse kernels.
- `src/wdm_transform/transforms/jax_backend.py`
  Reserves the JAX transform implementation surface.
- `src/wdm_transform/windows.py`
  Holds `phi`, `Cnm`, and related shared transform helpers.
- `src/wdm_transform/backends/`
  Holds the backend abstraction and the NumPy backend registration.
- `src/wdm_transform/plotting.py`
  Holds small plotting helpers without making plotting part of the core types.

## Why this layout

This keeps a clean line between:

- domain objects
- transform implementation
- backend-specific array and FFT access
- optional documentation and plotting utilities

That separation matters once JAX and CuPy are added, because the public API can stay stable while
backend-specific execution details move behind the backend interface.
