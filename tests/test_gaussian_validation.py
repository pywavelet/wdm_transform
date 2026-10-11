"""Gaussian edge validation and literal atom checks at odd support lengths."""

from __future__ import annotations

import numpy as np
import pytest
from _gaussian_reference import direct_basis

from wdm_transform import WDM, TimeSeries
from wdm_transform.backends import Backend
from wdm_transform.transforms import (
    from_freq_to_wdm,
    from_time_to_wdm,
    from_wdm_to_time,
)
from wdm_transform.windows_gaussian import gaussian_unit


def _options(backend: str | Backend, support: int = 16) -> dict:
    return dict(
        a=1.0 / 3.0,
        d=1.0,
        dt=0.2,
        backend=backend,
        window="gaussian",
        nu=0.5,
        support=support,
    )


def test_gaussian_time_input_projects_to_real_before_fft(backend: Backend) -> None:
    rng = np.random.default_rng(298)
    real = rng.normal(size=(2, 128))
    complex_input = real + 1j * 1e15 * rng.normal(size=(2, 128))
    options = _options(backend)
    actual = from_time_to_wdm(complex_input, nt=8, nf=16, **options)
    expected = from_time_to_wdm(real, nt=8, nf=16, **options)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("parameter", ["nt", "nf"])
@pytest.mark.parametrize("invalid", [0, -2, True, 8.0, "8", None, np.inf])
def test_gaussian_full_forward_requires_integer_dimensions(
    parameter: str, invalid: object
) -> None:
    dimensions = dict(nt=8, nf=16)
    dimensions[parameter] = invalid
    options = _options("numpy")
    for function in (from_time_to_wdm, from_freq_to_wdm):
        with pytest.raises(ValueError, match="positive even integer|positive nf"):
            function(np.zeros(128), **dimensions, **options)


def test_direct_gaussian_constructor_rejects_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="positive even integer"):
        WDM(np.zeros((0, 3)), dt=1.0, window="gaussian")


@pytest.mark.parametrize("invalid", [0, -2, 3, True, 8.0, "8", None, np.inf])
def test_gaussian_factories_require_integer_time_bins(invalid: object) -> None:
    ts = TimeSeries(np.zeros(128), dt=1.0)
    fs = ts.to_frequency_series()
    factories = (
        lambda: ts.to_wdm(nt=invalid, window="gaussian"),
        lambda: fs.to_wdm(nt=invalid, window="gaussian"),
        lambda: WDM.from_time_series(ts, nt=invalid, window="gaussian"),
        lambda: WDM.from_frequency_series(fs, nt=invalid, window="gaussian"),
    )
    for factory in factories:
        with pytest.raises(ValueError, match="positive even integer"):
            factory()


def test_gaussian_empty_batch_keeps_all_forward_and_inverse_shapes(
    backend: Backend,
) -> None:
    nt, nf = 8, 16
    series = TimeSeries(np.empty((0, nt * nf)), dt=0.25, backend=backend)
    for source in (series, series.to_frequency_series()):
        transformed = source.to_wdm(nt=nt, window="gaussian")
        assert transformed.coeffs.shape == (0, nt, nf + 1)
        assert transformed.to_time_series().data.shape == (0, nt * nf)
        assert transformed.to_frequency_series().data.shape == (0, nt * nf)


@pytest.mark.parametrize("support", [1, 3, 7, 15])
def test_odd_support_analysis_and_synthesis_match_literal_atoms(
    backend: Backend, support: int
) -> None:
    nt, nf = 8, 16
    rng = np.random.default_rng(76 + support)
    samples = rng.normal(size=(2, nt * nf))
    test_coeffs = rng.normal(size=(2, nt, nf + 1))
    basis = direct_basis(nt, nf, support=support, unit_window=gaussian_unit)
    options = _options(backend, support=support)
    actual_analysis = from_time_to_wdm(samples, nt=nt, nf=nf, **options)
    expected_analysis = (samples @ basis.T).reshape(2, nt, nf + 1)
    np.testing.assert_allclose(actual_analysis, expected_analysis, atol=2e-12)
    actual_synthesis = from_wdm_to_time(test_coeffs, **options)
    expected_synthesis = test_coeffs.reshape(2, -1) @ basis / (nt * nf)
    np.testing.assert_allclose(actual_synthesis, expected_synthesis, atol=2e-13)
    np.testing.assert_allclose(
        np.vdot(np.asarray(actual_analysis), test_coeffs) / (nt * nf),
        np.vdot(samples, np.asarray(actual_synthesis)),
        atol=2e-13,
    )
