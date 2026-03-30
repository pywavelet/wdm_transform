# wdm-transform

Initial package scaffold for experimenting with a roll-based WDM transform derived from an
exploratory reference notebook.

The package currently provides three core objects:

- `TimeSeries` for 1D sampled time-domain data.
- `FrequencySeries` for raw FFT-domain data with spacing metadata.
- `WDM` for packed WDM coefficients plus forward and inverse transforms.

The implementation is NumPy-only for now, but the public API routes array operations through a
backend registry so JAX and CuPy backends can be added later without reworking the object model.

## Layout

- `src/wdm_transform/datatypes/series.py`: time and frequency domain containers.
- `src/wdm_transform/datatypes/wdm.py`: WDM coefficient container and wrapper methods.
- `src/wdm_transform/transforms/__init__.py`: lazy dispatch for backend-specific transforms.
- `src/wdm_transform/transforms/xp_backend.py`: shared NumPy/CuPy forward and inverse kernels.
- `src/wdm_transform/transforms/jax_backend.py`: JAX transform entry points.
- `src/wdm_transform/windows.py`: shared window and synthesis helpers.
- `src/wdm_transform/backends/`: backend registry and NumPy backend.
- `src/wdm_transform/plotting.py`: plotting helpers.
- `tests/`: pytest coverage for end-to-end round trips.

## Quick start

```python
import numpy as np
from wdm_transform import TimeSeries, WDM

nt = 32
dt = 0.1
x = np.sin(2 * np.pi * np.arange(nt * 16) * dt * 0.08)

series = TimeSeries(x, dt=dt)
coeffs = WDM.from_time_series(series, nt=nt)
reconstructed = coeffs.to_time_series()

np.testing.assert_allclose(reconstructed.data, series.data)
```
