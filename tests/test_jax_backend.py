from __future__ import annotations

import numpy as np
import pytest

from wdm_transform import WDM, TimeSeries, get_backend

jax = pytest.importorskip("jax")


def _example_series(
    nt: int = 32,
    nf: int = 32,
    dt: float = 1.1,
) -> tuple[TimeSeries, int, int]:
    n_total = nt * nf
    times = np.arange(n_total) * dt
    duration = n_total * dt
    envelope = np.exp(-((times - duration / 2.0) ** 2) / (duration / 4.0) ** 2)
    carrier = np.cos(2.0 * np.pi * times * (0.1 + times * 0.153 / duration) + 0.3)
    data = envelope * carrier + 0.2 * np.sin(2.0 * np.pi * times * 0.08)
    return TimeSeries(data, dt=dt), nt, nf


def test_jax_backend_forward_and_inverse_match_numpy() -> None:
    series, nt, _ = _example_series()
    numpy_coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)

    jax_backend = get_backend("jax")
    jax_series = TimeSeries(series.data, dt=series.dt, backend=jax_backend)
    jax_coeffs = WDM.from_time_series(
        jax_series,
        nt=nt,
        a=1.0 / 3.0,
        backend=jax_backend,
    )
    jax_recovered = jax_coeffs.to_time_series()

    np.testing.assert_allclose(
        np.asarray(jax_coeffs.coeffs),
        np.asarray(numpy_coeffs.coeffs),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(jax_recovered.data),
        np.asarray(series.data),
        atol=1e-10,
        rtol=1e-10,
    )
    assert isinstance(jax_coeffs.coeffs, jax.Array)


def test_jax_backend_frequency_reconstruction_matches_numpy() -> None:
    series, nt, _ = _example_series()
    numpy_coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)
    numpy_frequency = numpy_coeffs.to_frequency_series()

    jax_backend = get_backend("jax")
    jax_series = TimeSeries(series.data, dt=series.dt, backend=jax_backend)
    jax_coeffs = WDM.from_time_series(
        jax_series,
        nt=nt,
        a=1.0 / 3.0,
        backend=jax_backend,
    )
    jax_frequency = jax_coeffs.to_frequency_series()

    np.testing.assert_allclose(
        np.asarray(jax_frequency.data),
        np.asarray(numpy_frequency.data),
        atol=1e-10,
        rtol=1e-10,
    )
