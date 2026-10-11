"""Gaussian full-transform tests against independent atoms and synthesis."""

import numpy as np
import pytest
from _gaussian_reference import direct_analysis, direct_basis

from wdm_transform import WDM, FrequencySeries, TimeSeries, get_backend
from wdm_transform.transforms import (
    from_freq_to_wdm,
    from_time_to_wdm,
    from_wdm_to_freq,
    from_wdm_to_time,
)


def _options(backend_name, *, nu=0.5, support=16):
    return dict(
        a=1 / 3,
        d=1,
        dt=0.25,
        backend=backend_name,
        window="gaussian",
        nu=nu,
        support=support,
    )


def _signal(family, count):
    samples = np.arange(count)
    fraction = samples / count
    if family == "noise":
        return np.random.default_rng(2026).normal(size=count)
    if family.startswith("impulse"):
        data = np.zeros(count)
        location = {
            "impulse_start": 0,
            "impulse_middle": count // 3,
            "impulse_end": count - 1,
        }[family]
        data[location] = 1
        return data
    if family == "sine_on_bin":
        return np.sin(2 * np.pi * (count // 16) * fraction)
    if family == "sine_off_bin":
        return np.sin(2 * np.pi * (count // 16 + 0.37) * fraction)
    if family == "chirp":
        return np.sin(2 * np.pi * count * (0.03 * fraction + 0.12 * fraction**2))
    if family == "sine_gaussian":
        return np.exp(-0.5 * ((fraction - 0.53) / 0.02) ** 2) * np.cos(
            2 * np.pi * 0.17 * samples
        )
    if family == "near_dc":
        return 0.4 + np.sin(2 * np.pi * fraction)
    if family == "near_nyquist":
        return 0.4 * (-1.0) ** samples + np.cos(2 * np.pi * (count // 2 - 2) * fraction)
    raise AssertionError(f"Unknown test signal {family}")


@pytest.mark.parametrize("nt,nf", [(32, 16), (64, 32), (128, 64)])
@pytest.mark.parametrize(
    "family",
    [
        "noise",
        "impulse_start",
        "impulse_middle",
        "impulse_end",
        "sine_on_bin",
        "sine_off_bin",
        "chirp",
        "sine_gaussian",
        "near_dc",
        "near_nyquist",
    ],
)
def test_signal_family_roundtrips_and_full_frequency_consistency(
    backend_name, nt, nf, family
):
    data = _signal(family, nt * nf)
    options = _options(backend_name)
    coeffs = from_time_to_wdm(data, nt=nt, nf=nf, **options)
    from_frequency = from_freq_to_wdm(np.fft.fft(data), nt=nt, nf=nf, **options)
    assert coeffs.shape == (1, nt, nf + 1)
    assert np.isrealobj(np.asarray(coeffs))
    np.testing.assert_allclose(from_frequency, coeffs, rtol=1e-13, atol=2e-12)
    recovered = np.asarray(from_wdm_to_time(coeffs, **options))[0]
    relative = np.linalg.norm(recovered - data) / np.linalg.norm(data)
    assert relative < 1e-10
    recovered_frequency = from_wdm_to_freq(coeffs, **options)
    np.testing.assert_allclose(
        recovered_frequency, np.fft.fft(recovered)[None, :], rtol=2e-13, atol=1e-11
    )
    # The full row Gram test establishes edge duplication and half-norm atoms.
    # With this actual convention the redundant coefficient frame has bound N.
    energy = np.sum(np.asarray(coeffs) ** 2) / (nt * nf)
    assert energy == pytest.approx(np.sum(data**2), rel=2e-11)


@pytest.mark.parametrize("support", [8, 16])
def test_optimized_analysis_matches_direct_paper_atoms(backend_name, support):
    nt, nf = 8, 16
    data = np.stack([_signal("noise", nt * nf), _signal("near_nyquist", nt * nf)])
    reference = direct_analysis(data, nt, nf, support=support)
    result = from_time_to_wdm(
        data, nt=nt, nf=nf, **_options(backend_name, support=support)
    )
    np.testing.assert_allclose(result, reference, rtol=1e-12, atol=2e-12)


@pytest.mark.parametrize("support", [8, 16])
def test_synthesis_is_independent_atom_adjoint_for_arbitrary_pixels(
    backend_name, support
):
    nt, nf = 8, 16
    rng = np.random.default_rng(43)
    coeffs = rng.normal(size=(2, nt, nf + 1))
    # Independent dense atoms include DC/Nyquist rows and their redundancy.
    atoms = direct_basis(nt, nf, support=support)
    expected = coeffs.reshape(2, -1) @ atoms / (nt * nf)
    options = _options(backend_name, support=support)
    actual = from_wdm_to_time(coeffs, **options)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-13)
    np.testing.assert_allclose(
        from_wdm_to_freq(coeffs, **options),
        np.fft.fft(expected, axis=-1),
        rtol=1e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("channel,time", [(0, 0), (1, 1), (15, 2), (16, 7)])
def test_single_pixel_atoms_include_correct_boundary_synthesis(
    backend_name, channel, time
):
    nt, nf = 8, 16
    atoms = direct_basis(nt, nf, support=16)
    coeffs = np.zeros((nt, nf + 1))
    coeffs[time, channel] = 1
    result = from_wdm_to_time(coeffs, **_options(backend_name))[0]
    np.testing.assert_allclose(
        result,
        atoms[time * (nf + 1) + channel] / (nt * nf),
        rtol=1e-12,
        atol=2e-13,
    )


def test_nonhermitian_frequency_input_projects_to_real_signal(backend_name):
    nt, nf = 8, 16
    rng = np.random.default_rng(45)
    spectrum = rng.normal(size=nt * nf) + 1j * rng.normal(size=nt * nf)
    spectrum[0] += 100j
    spectrum[nt * nf // 2] -= 100j
    real_signal = np.fft.ifft(spectrum).real
    expected = direct_analysis(real_signal, nt, nf, support=16)
    actual = from_freq_to_wdm(spectrum, nt=nt, nf=nf, **_options(backend_name))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=2e-13)
    recovered = from_wdm_to_freq(actual, **_options(backend_name))
    projected = np.fft.fft(real_signal)
    assert (
        np.linalg.norm(np.asarray(recovered)[0] - projected) / np.linalg.norm(projected)
        < 1e-10
    )


@pytest.mark.parametrize("family", ["noise", "impulse_middle", "chirp"])
def test_support_convergence_at_standard_shape(backend_name, family):
    nt, nf = 64, 32
    data = _signal(family, nt * nf)
    errors = []
    for support in (4, 8, 16):
        options = _options(backend_name, support=support)
        coeffs = from_time_to_wdm(data, nt=nt, nf=nf, **options)
        recovered = from_wdm_to_time(coeffs, **options)[0]
        errors.append(
            np.linalg.norm(np.asarray(recovered) - data) / np.linalg.norm(data)
        )
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 10000
    assert errors[2] < 1e-10


@pytest.mark.parametrize("nu,support", [(0.3, 32), (0.5, 16), (0.8, 16)])
def test_shape_parameter_roundtrip_with_adequate_support(backend_name, nu, support):
    nt, nf = 64, 32
    data = _signal("noise", nt * nf)
    options = _options(backend_name, nu=nu, support=support)
    coeffs = from_time_to_wdm(data, nt=nt, nf=nf, **options)
    recovered = from_wdm_to_time(coeffs, **options)[0]
    assert np.linalg.norm(np.asarray(recovered) - data) / np.linalg.norm(data) < 1e-10


def test_broad_frequency_window_requires_more_than_default_support(backend_name):
    nt, nf = 64, 32
    data = _signal("noise", nt * nf)
    errors = []
    for support in (16, 32):
        options = _options(backend_name, nu=0.3, support=support)
        coeffs = from_time_to_wdm(data, nt=nt, nf=nf, **options)
        recovered = from_wdm_to_time(coeffs, **options)[0]
        errors.append(
            np.linalg.norm(np.asarray(recovered) - data) / np.linalg.norm(data)
        )
    assert 1e-9 < errors[0] < 1e-6
    assert errors[1] < 1e-11


def test_batch_equals_stacking_individual_analysis_and_synthesis(backend_name):
    nt, nf = 32, 16
    data = np.stack([_signal(name, nt * nf) for name in ("noise", "chirp", "near_dc")])
    options = _options(backend_name)
    together = from_time_to_wdm(data, nt=nt, nf=nf, **options)
    separately = np.concatenate(
        [np.asarray(from_time_to_wdm(row, nt=nt, nf=nf, **options)) for row in data]
    )
    np.testing.assert_allclose(together, separately, rtol=0, atol=2e-12)
    recovered = from_wdm_to_time(together, **options)
    separate_inverse = np.concatenate(
        [np.asarray(from_wdm_to_time(row, **options)) for row in together]
    )
    np.testing.assert_allclose(recovered, separate_inverse, rtol=0, atol=2e-13)


def test_tiny_grid_exposes_controlled_support_limitation(backend_name):
    data = np.array([0.3, -1.2, 0.7, 2.1])
    errors = []
    for support in (1, 2):
        options = _options(backend_name, support=support)
        coeffs = from_time_to_wdm(data, nt=2, nf=2, **options)
        assert coeffs.shape == (1, 2, 3)
        assert np.isfinite(np.asarray(coeffs)).all()
        recovered = from_wdm_to_time(coeffs, **options)[0]
        errors.append(
            np.linalg.norm(np.asarray(recovered) - data) / np.linalg.norm(data)
        )
    assert 0 < errors[1] < errors[0] < 0.1


def test_public_series_paths_retain_inverse_window_metadata(backend_name):
    nt, nf = 32, 32
    data = _signal("chirp", nt * nf)
    time_series = TimeSeries(data, dt=0.25, backend=backend_name)
    frequency_series = time_series.to_frequency_series()
    time_based = time_series.to_wdm(nt=nt, window="gaussian", nu=0.3, support=32)
    frequency_based = frequency_series.to_wdm(
        nt=nt, window="gaussian", nu=0.3, support=32
    )
    factory_time = WDM.from_time_series(
        time_series, nt=nt, window="gaussian", nu=0.3, support=32
    )
    factory_frequency = WDM.from_frequency_series(
        frequency_series, nt=nt, window="gaussian", nu=0.3, support=32
    )
    direct_object = WDM(
        time_based.coeffs,
        time_based.dt,
        1 / 3,
        1,
        time_based.backend,
        window="gaussian",
        nu=0.3,
        support=32,
    )
    for result in (
        time_based,
        frequency_based,
        factory_time,
        factory_frequency,
        direct_object,
    ):
        assert result.window == "gaussian"
        assert result.nu == 0.3
        assert result.support == 32
        np.testing.assert_allclose(
            result.coeffs, time_based.coeffs, rtol=1e-13, atol=2e-12
        )
        np.testing.assert_allclose(
            result.to_time_series().data, data[None, :], rtol=0, atol=3e-12
        )
        np.testing.assert_allclose(
            result.to_frequency_series().data,
            frequency_series.data,
            rtol=1e-11,
            atol=5e-11,
        )


def test_frequency_series_nonhermitian_projection_uses_full_grid(backend_name):
    nt, nf = 8, 16
    rng = np.random.default_rng(47)
    spectrum = rng.normal(size=nt * nf) + 1j * rng.normal(size=nt * nf)
    frequency_series = FrequencySeries(spectrum, df=0.1, backend=backend_name)
    result = frequency_series.to_wdm(nt=nt, window="gaussian", support=16)
    target = np.fft.ifft(spectrum).real
    np.testing.assert_allclose(
        result.to_time_series().data, target[None, :], rtol=0, atol=3e-12
    )


def test_gaussian_plot_smoke():
    import matplotlib.pyplot as plt

    data = _signal("chirp", 32 * 16)
    transformed = TimeSeries(data, dt=0.25).to_wdm(nt=32, window="gaussian")
    figure, axes = transformed.plot()
    figure.canvas.draw()
    assert axes is not None
    plt.close(figure)


def test_optional_cupy_gaussian_matches_numpy():
    try:
        backend = get_backend("cupy")
        if backend.xp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CuPy-compatible GPU is available")
    except (ImportError, RuntimeError):
        pytest.skip("CuPy or a compatible GPU is unavailable")
    nt, nf = 32, 16
    data = _signal("noise", nt * nf)
    reference = from_time_to_wdm(data, nt=nt, nf=nf, **_options("numpy"))
    actual = from_time_to_wdm(data, nt=nt, nf=nf, **_options(backend))
    np.testing.assert_allclose(
        backend.xp.asnumpy(actual), reference, rtol=1e-12, atol=2e-12
    )
    recovered = from_wdm_to_time(actual, **_options(backend))
    np.testing.assert_allclose(
        backend.xp.asnumpy(recovered), data[None, :], rtol=0, atol=3e-11
    )
