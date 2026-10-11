"""Public window selection, metadata retention, and compatibility checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import Parameter, signature

import numpy as np
import pytest

from wdm_transform import WDM, FrequencySeries, TimeSeries, transforms
from wdm_transform.backends import Backend, get_backend


def test_wdm_positional_construction_remains_compatible() -> None:
    backend = get_backend("numpy")
    coeffs = np.zeros((8, 9))
    wdm = WDM(coeffs, 0.25, 0.2, 1.0, backend)
    assert wdm.shape == (1, 8, 9)
    assert (wdm.dt, wdm.a, wdm.d, wdm.backend) == (0.25, 0.2, 1.0, backend)
    assert (wdm.window, wdm.nu, wdm.support) == ("meyer", None, None)
    for name in ("window", "nu", "support"):
        assert signature(WDM).parameters[name].kind is Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        WDM(coeffs, 0.25, 0.2, 1.0, backend, "gaussian")


def test_constructor_resolves_gaussian_defaults_and_is_frozen() -> None:
    wdm = WDM(np.zeros((8, 9)), 0.25, window="gaussian")
    assert (wdm.window, wdm.nu, wdm.support) == ("gaussian", 0.5, 8)
    assert "window='gaussian'" in repr(wdm)
    assert "nu=0.5" in repr(wdm)
    assert "support=8" in repr(wdm)
    with pytest.raises(FrozenInstanceError):
        wdm.window = "meyer"


def test_implicit_and_explicit_meyer_are_identical(backend: Backend) -> None:
    rng = np.random.default_rng(813)
    ts = TimeSeries(rng.normal(size=(2, 512)), dt=0.25, backend=backend)
    implicit = ts.to_wdm(nt=32, a=0.2)
    explicit = ts.to_wdm(nt=32, a=0.2, window="meyer")
    via_frequency = ts.to_frequency_series().to_wdm(nt=32, a=0.2, window="meyer")
    np.testing.assert_array_equal(implicit.coeffs, explicit.coeffs)
    np.testing.assert_allclose(implicit.coeffs, via_frequency.coeffs, atol=1e-12)
    np.testing.assert_array_equal(
        implicit.to_time_series().data, explicit.to_time_series().data
    )
    assert (implicit.window, implicit.nu, implicit.support) == ("meyer", None, None)


def test_gaussian_all_full_forward_pathways_agree(backend: Backend) -> None:
    nt, nf = 32, 32
    ts = TimeSeries(
        np.random.default_rng(471).normal(size=(2, nt * nf)),
        dt=0.25,
        backend=backend,
    )
    fs = ts.to_frequency_series()
    options = dict(window="gaussian", nu=0.5, support=16)
    outputs = [
        ts.to_wdm(nt=nt, **options),
        fs.to_wdm(nt=nt, **options),
        WDM.from_time_series(ts, nt=nt, **options),
        WDM.from_frequency_series(fs, nt=nt, **options),
    ]
    for wdm in outputs:
        assert (wdm.window, wdm.nu, wdm.support) == ("gaussian", 0.5, 16)
        assert wdm.shape == (2, nt, nf + 1)
        assert wdm.backend is backend
        np.testing.assert_allclose(wdm.coeffs, outputs[0].coeffs, atol=2e-12)

    full_options = dict(
        nt=nt, nf=nf, a=1.0 / 3.0, d=1.0, dt=ts.dt, backend=backend, **options
    )
    from_time = transforms.from_time_to_wdm(ts.data, **full_options)
    from_freq = transforms.from_freq_to_wdm(fs.data, **full_options)
    np.testing.assert_allclose(from_time, outputs[0].coeffs, atol=2e-12)
    np.testing.assert_allclose(from_freq, outputs[0].coeffs, atol=2e-12)
    # A common layout does not imply interchangeable basis coefficients.
    meyer = ts.to_wdm(nt=nt)
    assert np.linalg.norm(np.asarray(meyer.coeffs) - np.asarray(from_time)) > 1e-3


def test_gaussian_inverse_retains_custom_parameters(backend: Backend) -> None:
    nt, nf = 32, 32
    ts = TimeSeries(np.random.default_rng(312).normal(size=nt * nf), dt=0.4)
    wdm = ts.to_wdm(nt=nt, window="gaussian", nu=0.8, support=16, backend=backend)
    manual = WDM(
        wdm.coeffs,
        wdm.dt,
        wdm.a,
        wdm.d,
        wdm.backend,
        window=wdm.window,
        nu=wdm.nu,
        support=wdm.support,
    )
    options = dict(
        a=wdm.a,
        d=wdm.d,
        dt=wdm.dt,
        backend=backend,
        window="gaussian",
        nu=0.8,
        support=16,
    )
    expected_time = transforms.from_wdm_to_time(wdm.coeffs, **options)
    expected_freq = transforms.from_wdm_to_freq(wdm.coeffs, **options)
    for output in (wdm, manual):
        np.testing.assert_array_equal(output.to_time_series().data, expected_time)
        np.testing.assert_array_equal(output.to_frequency_series().data, expected_freq)
        assert output.to_time_series().backend is backend
        assert output.to_frequency_series().backend is backend
    np.testing.assert_allclose(
        expected_freq, np.fft.fft(np.asarray(expected_time), axis=-1), atol=3e-12
    )
    wrong_basis = transforms.from_wdm_to_time(
        wdm.coeffs, a=wdm.a, d=wdm.d, dt=wdm.dt, backend=backend
    )
    assert np.linalg.norm(np.asarray(wrong_basis) - np.asarray(expected_time)) > 1e-3


@pytest.mark.parametrize("nf", [4, 8, 16, 32])
def test_factory_records_effective_default_support(nf: int) -> None:
    ts = TimeSeries(np.zeros(8 * nf), dt=1.0)
    wdm = ts.to_wdm(nt=8, window="gaussian")
    from_freq = ts.to_frequency_series().to_wdm(nt=8, window="gaussian")
    for output in (wdm, from_freq):
        assert output.nu == 0.5
        assert output.support == min(16, nf)
        assert output.shape == (1, 8, nf + 1)


@pytest.mark.parametrize(
    "options",
    [
        {"window": "Gaussian"},
        {"window": "raw_gaussian"},
        {"window": "meyer", "nu": 0.5},
        {"window": "meyer", "support": 4},
        {"window": "gaussian", "nu": 0.0},
        {"window": "gaussian", "nu": np.nan},
        {"window": "gaussian", "nu": np.inf},
        {"window": "gaussian", "support": 0},
        {"window": "gaussian", "support": 33},
        {"window": "gaussian", "support": 4.0},
        {"window": "gaussian", "support": True},
        {"window": "gaussian", "a": 0.2},
        {"window": "gaussian", "d": 2.0},
    ],
)
def test_invalid_metadata_rejected_across_public_pathways(options: dict) -> None:
    nt, nf = 8, 32
    ts = TimeSeries(np.zeros(nt * nf), dt=1.0)
    fs = FrequencySeries(np.zeros(nt * nf, dtype=complex), df=1.0 / (nt * nf))
    coeffs = np.zeros((nt, nf + 1))
    factories = (
        lambda: WDM(coeffs, 1.0, **options),
        lambda: ts.to_wdm(nt=nt, **options),
        lambda: fs.to_wdm(nt=nt, **options),
        lambda: WDM.from_time_series(ts, nt=nt, **options),
        lambda: WDM.from_frequency_series(fs, nt=nt, **options),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()

    full_options = dict(a=1.0 / 3.0, d=1.0, dt=1.0, backend="numpy")
    full_options.update(options)
    for forward, data in (
        (transforms.from_time_to_wdm, ts.data),
        (transforms.from_freq_to_wdm, fs.data),
    ):
        with pytest.raises(ValueError):
            forward(data, nt=nt, nf=nf, **full_options)
    for inverse in (transforms.from_wdm_to_time, transforms.from_wdm_to_freq):
        with pytest.raises(ValueError):
            inverse(coeffs, **full_options)


@pytest.mark.parametrize(
    "function_name",
    [
        "from_freq_to_wdm_subband",
        "forward_wdm_subband",
        "from_freq_to_wdm_band",
        "forward_wdm_band",
        "from_wdm_to_freq_subband",
        "inverse_wdm_subband",
    ],
)
@pytest.mark.parametrize("window", ["unknown", "Gaussian"])
def test_subband_invalid_window_includes_aliases(
    function_name: str, window: str
) -> None:
    options = dict(
        df=1.0 / 256.0,
        nfreqs_fourier=129,
        nfreqs_wdm=32,
        ntimes_wdm=8,
        window=window,
    )
    if "inverse" in function_name or "from_wdm" in function_name:
        options["mmin"] = 0
        data = np.zeros((8, 2))
    else:
        options["kmin"] = 0
        data = np.zeros(2, dtype=complex)
        if function_name in ("from_freq_to_wdm_band", "forward_wdm_band"):
            options.update(mmin=0, nf_sub_wdm=1)
    with pytest.raises(ValueError, match="[Ww]indow|Gaussian"):
        getattr(transforms, function_name)(data, **options)


def test_gaussian_plot_uses_common_grid() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    wdm = TimeSeries(np.zeros(512), dt=0.25).to_wdm(nt=32, window="gaussian")
    fig, axes = wdm.plot()
    assert np.ravel(axes)[0].get_ylabel() == "Frequency [Hz]"
    plt.close(fig)
