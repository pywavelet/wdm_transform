# wdm-transform

`wdm-transform` provides an object-oriented interface for WDM transforms on sampled
one-dimensional signals.

The package centers on three core objects:

- `TimeSeries` for 1D sampled time-domain data.
- `FrequencySeries` for raw FFT-domain data with spacing metadata.
- `WDM` for packed WDM coefficients plus forward and inverse transforms.

The default implementation is NumPy-based, and the public API routes array operations through a
backend registry so JAX and CuPy backends can fit into the same object model.

## Layout

- `src/wdm_transform/datatypes/series.py`: time and frequency domain containers.
- `src/wdm_transform/datatypes/wdm.py`: WDM coefficient container and wrapper methods.
- `src/wdm_transform/transforms/__init__.py`: lazy dispatch for backend-specific transforms.
- `src/wdm_transform/transforms/xp_backend.py`: shared NumPy/CuPy forward and inverse kernels.
- `src/wdm_transform/transforms/jax_backend.py`: JAX transform entry points.
- `src/wdm_transform/windows.py`: shared window and synthesis helpers.
- `src/wdm_transform/backends/`: backend registry and NumPy backend.
- `src/wdm_transform/plotting.py`: plotting helpers.
- `tests/`: end-to-end round-trip tests.

## Start Here

- Read the package layout for the current source tree.
- Read the API overview for the intended object model.
- Open the WDM walkthrough for executed examples, demo plots, and simple timing output.
- Use the API reference for signatures and docstrings extracted with `mkdocstrings`.
