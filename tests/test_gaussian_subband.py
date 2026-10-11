"""Gaussian compact subbands match full transforms and literal tap geometry."""

import numpy as np
import pytest

from wdm_transform import get_backend
from wdm_transform.transforms import (
    forward_wdm_band,
    forward_wdm_subband,
    fourier_span_from_wdm_span,
    from_freq_to_wdm,
    from_freq_to_wdm_band,
    from_freq_to_wdm_subband,
    from_wdm_to_freq,
    from_wdm_to_freq_subband,
    inverse_wdm_subband,
    wdm_span_from_fourier_span,
)


def _grid(nt=16, nf=32):
    return dict(
        nfreqs_fourier=nt * nf // 2 + 1,
        nfreqs_wdm=nf,
        ntimes_wdm=nt,
    )


def _literal_one_sided_taps(nt, nf, support):
    offsets = np.arange(-support * nt // 2, support * nt // 2)
    periodic = (np.arange(nf + 1)[:, None] * (nt // 2) + offsets) % (nt * nf)
    return np.minimum(periodic, nt * nf - periodic)


@pytest.mark.parametrize("nt,nf", [(2, 2), (4, 8), (8, 16)])
def test_span_geometry_exhaustively_matches_literal_modulo_indices(nt, nf):
    nyquist = nt * nf // 2
    for support in sorted({1, 2, 3, nf} & set(range(1, nf + 1))):
        taps = _literal_one_sided_taps(nt, nf, support)
        options = dict(_grid(nt, nf), window="gaussian", support=support)
        for first in range(nyquist + 1):
            for last in range(first, nyquist + 1):
                touched = np.flatnonzero(
                    np.any((taps >= first) & (taps <= last), axis=1)
                )
                expected = int(touched.min()), int(touched.max() - touched.min() + 1)
                assert (
                    wdm_span_from_fourier_span(
                        kmin=first, lendata=last - first + 1, **options
                    )
                    == expected
                )
        for first in range(nf + 1):
            for last in range(first, nf + 1):
                selected = taps[first : last + 1]
                expected = int(selected.min()), int(selected.max() - selected.min() + 1)
                assert (
                    fourier_span_from_wdm_span(
                        mmin=first, nf_sub_wdm=last - first + 1, **options
                    )
                    == expected
                )


@pytest.mark.parametrize(
    "support,kmin,lendata",
    [
        (1, 0, 1),
        (1, 256, 1),
        (1, 8, 1),
        (1, 15, 2),
        (1, 120, 8),
        (1, 0, 257),
        (3, 0, 6),
        (3, 252, 5),
        (3, 24, 9),
        (3, 96, 1),
        (4, 0, 1),
        (4, 256, 1),
        (4, 112, 10),
        (8, 0, 5),
        (8, 254, 3),
        (8, 112, 9),
        (16, 0, 1),
        (16, 128, 3),
        (32, 0, 1),
        (32, 256, 1),
        (32, 112, 10),
    ],
)
def test_compact_analysis_matches_full_zero_padded_spectrum(
    backend_name, support, kmin, lendata
):
    nt, nf = 16, 32
    rng = np.random.default_rng(711)
    crop = rng.normal(size=lendata) + 1j * rng.normal(size=lendata)
    one_sided = np.zeros(nt * nf // 2 + 1, dtype=complex)
    one_sided[kmin : kmin + lendata] = crop
    # Self-conjugate bins are projected real, including deliberately complex input.
    full_spectrum = np.concatenate([one_sided, one_sided[-2:0:-1].conj()])
    full_spectrum[0] = full_spectrum[0].real
    full_spectrum[nt * nf // 2] = full_spectrum[nt * nf // 2].real
    options = dict(
        _grid(nt, nf),
        df=0.125,
        backend=backend_name,
        window="gaussian",
        support=support,
    )
    compact, first_channel = from_freq_to_wdm_subband(crop, kmin=kmin, **options)
    expected_start, expected_count = wdm_span_from_fourier_span(
        **_grid(nt, nf),
        kmin=kmin,
        lendata=lendata,
        window="gaussian",
        support=support,
    )
    assert first_channel == expected_start
    assert compact.shape == (nt, expected_count)
    full_coefficients = from_freq_to_wdm(
        full_spectrum,
        nt=nt,
        nf=nf,
        a=1 / 3,
        d=1,
        dt=1 / (nt * nf * options["df"]),
        backend=backend_name,
        window="gaussian",
        support=support,
    )[0]
    np.testing.assert_allclose(
        compact,
        full_coefficients[:, expected_start : expected_start + expected_count],
        rtol=1e-12,
        atol=2e-12,
    )
    outside = np.asarray(full_coefficients).copy()
    outside[:, expected_start : expected_start + expected_count] = 0
    np.testing.assert_allclose(outside, 0, rtol=0, atol=2e-13)


@pytest.mark.parametrize(
    "support,mmin,nf_sub_wdm",
    [
        (1, 0, 1),
        (1, 32, 1),
        (3, 0, 3),
        (3, 30, 3),
        (3, 12, 2),
        (4, 0, 1),
        (4, 32, 1),
        (8, 12, 3),
        (16, 14, 1),
        (32, 0, 1),
        (32, 32, 1),
        (32, 9, 3),
    ],
)
def test_arbitrary_compact_synthesis_matches_zero_padded_full_coefficients(
    backend_name, support, mmin, nf_sub_wdm
):
    nt, nf = 16, 32
    crop = np.random.default_rng(713).normal(size=(nt, nf_sub_wdm))
    full_coefficients = np.zeros((nt, nf + 1))
    full_coefficients[:, mmin : mmin + nf_sub_wdm] = crop
    compact, first_bin = from_wdm_to_freq_subband(
        crop,
        **_grid(nt, nf),
        df=0.125,
        mmin=mmin,
        backend=backend_name,
        window="gaussian",
        support=support,
    )
    expected_start, expected_count = fourier_span_from_wdm_span(
        **_grid(nt, nf),
        mmin=mmin,
        nf_sub_wdm=nf_sub_wdm,
        window="gaussian",
        support=support,
    )
    assert first_bin == expected_start
    assert compact.shape == (expected_count,)
    full_spectrum = from_wdm_to_freq(
        full_coefficients,
        a=1 / 3,
        d=1,
        dt=1 / (nt * nf * 0.125),
        backend=backend_name,
        window="gaussian",
        support=support,
    )[0, : nt * nf // 2 + 1]
    np.testing.assert_allclose(
        compact,
        full_spectrum[expected_start : expected_start + expected_count],
        rtol=1e-12,
        atol=2e-12,
    )
    outside = np.asarray(full_spectrum).copy()
    outside[expected_start : expected_start + expected_count] = 0
    np.testing.assert_allclose(outside, 0, rtol=0, atol=2e-13)
    if expected_start == 0:
        assert np.asarray(compact)[0].imag == 0
    if expected_start + expected_count == nt * nf // 2 + 1:
        assert np.asarray(compact)[-1].imag == 0


def test_requested_band_aliases_and_gaussian_defaults(backend_name):
    nt, nf = 16, 32
    data = np.random.default_rng(715).normal(size=7).astype(complex)
    options = dict(
        _grid(nt, nf), df=0.125, kmin=120, backend=backend_name, window="gaussian"
    )
    compact, first_channel = from_freq_to_wdm_subband(data, **options)
    alias, alias_start = forward_wdm_subband(data, **options)
    np.testing.assert_array_equal(alias, compact)
    assert alias_start == first_channel
    explicit, explicit_start = from_freq_to_wdm_subband(
        data, **options, nu=0.5, support=16
    )
    np.testing.assert_array_equal(explicit, compact)
    assert explicit_start == first_channel
    requested_start = first_channel + 1
    requested_count = compact.shape[1] - 2
    requested = from_freq_to_wdm_band(
        data, **options, mmin=requested_start, nf_sub_wdm=requested_count
    )
    alias_band = forward_wdm_band(
        data, **options, mmin=requested_start, nf_sub_wdm=requested_count
    )
    np.testing.assert_array_equal(requested, compact[:, 1:-1])
    np.testing.assert_array_equal(alias_band, requested)
    inverse_options = dict(
        _grid(nt, nf),
        df=0.125,
        mmin=first_channel,
        backend=backend_name,
        window="gaussian",
    )
    inverse, first_bin = from_wdm_to_freq_subband(compact, **inverse_options)
    inverse_alias, alias_bin = inverse_wdm_subband(compact, **inverse_options)
    np.testing.assert_array_equal(inverse_alias, inverse)
    assert alias_bin == first_bin
    with pytest.raises(ValueError, match="outside"):
        from_freq_to_wdm_band(data, **options, mmin=0, nf_sub_wdm=nf + 1)


@pytest.mark.parametrize("nu,support", [(0.3, 32), (0.5, 16), (0.8, 16)])
def test_subband_roundtrip_with_adequate_support(backend_name, nu, support):
    nt, nf = 32, 32
    kmin = 219
    crop = np.random.default_rng(717).normal(size=9) + 1j * np.random.default_rng(
        718
    ).normal(size=9)
    options = dict(
        _grid(nt, nf),
        df=0.125,
        backend=backend_name,
        window="gaussian",
        nu=nu,
        support=support,
    )
    coefficients, first_channel = from_freq_to_wdm_subband(crop, kmin=kmin, **options)
    recovered, first_bin = from_wdm_to_freq_subband(
        coefficients, mmin=first_channel, **options
    )
    expected = np.zeros(recovered.shape, dtype=complex)
    expected[kmin - first_bin : kmin - first_bin + len(crop)] = crop
    assert (
        np.linalg.norm(np.asarray(recovered) - expected) / np.linalg.norm(expected)
        < 1e-10
    )


def test_subband_support_convergence(backend_name):
    nt, nf = 32, 32
    kmin = 219
    crop = np.random.default_rng(719).normal(size=9) + 1j * np.random.default_rng(
        720
    ).normal(size=9)
    errors = []
    for support in (4, 8, 16):
        options = dict(
            _grid(nt, nf),
            df=0.125,
            backend=backend_name,
            window="gaussian",
            support=support,
        )
        coefficients, first_channel = from_freq_to_wdm_subband(
            crop, kmin=kmin, **options
        )
        recovered, first_bin = from_wdm_to_freq_subband(
            coefficients, mmin=first_channel, **options
        )
        expected = np.zeros(recovered.shape, dtype=complex)
        expected[kmin - first_bin : kmin - first_bin + len(crop)] = crop
        errors.append(
            np.linalg.norm(np.asarray(recovered) - expected) / np.linalg.norm(expected)
        )
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 10000
    assert errors[2] < 1e-10


@pytest.mark.parametrize(
    "helper", [wdm_span_from_fourier_span, fourier_span_from_wdm_span]
)
@pytest.mark.parametrize(
    "changed",
    [
        dict(ntimes_wdm=0),
        dict(ntimes_wdm=True),
        dict(ntimes_wdm=4.0),
        dict(nfreqs_wdm=0),
        dict(nfreqs_wdm=False),
        dict(nfreqs_wdm=8.0),
        dict(nfreqs_fourier=17.0),
        dict(nfreqs_fourier=True),
        dict(ntimes_wdm=3),
        dict(nfreqs_wdm=7),
        dict(nfreqs_fourier=16),
    ],
)
def test_gaussian_span_rejects_invalid_grid_dimensions(helper, changed):
    grid = _grid(4, 8)
    grid.update(changed)
    crop = (
        dict(kmin=0, lendata=1)
        if helper is wdm_span_from_fourier_span
        else dict(mmin=0, nf_sub_wdm=1)
    )
    with pytest.raises(ValueError):
        helper(**grid, **crop, window="gaussian")


@pytest.mark.parametrize(
    "helper,crop",
    [
        (wdm_span_from_fourier_span, dict(kmin=True, lendata=1)),
        (wdm_span_from_fourier_span, dict(kmin=0.0, lendata=1)),
        (wdm_span_from_fourier_span, dict(kmin=0, lendata=1.0)),
        (fourier_span_from_wdm_span, dict(mmin=False, nf_sub_wdm=1)),
        (fourier_span_from_wdm_span, dict(mmin=0.0, nf_sub_wdm=1)),
        (fourier_span_from_wdm_span, dict(mmin=0, nf_sub_wdm=1.0)),
    ],
)
def test_gaussian_span_rejects_noninteger_crop_coordinates(helper, crop):
    with pytest.raises(ValueError, match="integer"):
        helper(**_grid(4, 8), **crop, window="gaussian")


def test_gaussian_span_defaults_and_shape_parameter_do_not_change_geometry():
    options = dict(_grid(), kmin=120, lendata=7, window="gaussian")
    expected = wdm_span_from_fourier_span(**options)
    assert wdm_span_from_fourier_span(**options, nu=0.3, support=16) == expected
    assert wdm_span_from_fourier_span(**options, nu=0.8, support=16) == expected
    with pytest.raises(ValueError, match="meaningful"):
        wdm_span_from_fourier_span(**_grid(), kmin=0, lendata=1, support=1)


def test_large_grid_metadata_keeps_actual_subband_arrays_compact():
    import tracemalloc

    nt, nf = 256, 65536
    kmin = 2_796_160
    rng = np.random.default_rng(721)
    crop = rng.normal(size=64) + 1j * rng.normal(size=64)
    options = dict(_grid(nt, nf), df=0.125, window="gaussian", support=16)
    tracemalloc.start()
    numpy_coefficients, first_channel = from_freq_to_wdm_subband(
        crop, kmin=kmin, backend="numpy", **options
    )
    numpy_recovered, first_bin = from_wdm_to_freq_subband(
        numpy_coefficients, mmin=first_channel, backend="numpy", **options
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # A full complex FFT here is 256 MiB; compact host work remains below 32 MiB.
    assert peak_bytes < 32 * 1024**2
    assert numpy_coefficients.shape == (nt, 32)
    assert numpy_recovered.size < 10000
    expected = np.zeros(numpy_recovered.shape, dtype=complex)
    expected[kmin - first_bin : kmin - first_bin + crop.size] = crop
    assert np.linalg.norm(numpy_recovered - expected) / np.linalg.norm(expected) < 1e-10
    pytest.importorskip("jax")
    jax_coefficients, jax_channel = from_freq_to_wdm_subband(
        crop, kmin=kmin, backend="jax", **options
    )
    jax_recovered, jax_bin = from_wdm_to_freq_subband(
        jax_coefficients, mmin=jax_channel, backend="jax", **options
    )
    assert jax_channel == first_channel
    assert jax_bin == first_bin
    np.testing.assert_allclose(
        jax_coefficients, numpy_coefficients, rtol=1e-12, atol=2e-12
    )
    np.testing.assert_allclose(jax_recovered, numpy_recovered, rtol=1e-12, atol=2e-12)


def test_jax_subband_jit_jvp_vjp_and_grad_match_full_transform():
    jax = pytest.importorskip("jax")
    get_backend("jax")
    jnp = pytest.importorskip("jax.numpy")
    nt, nf, support = 8, 16, 3
    kmin, lendata = 0, 9
    first_channel, channel_count = wdm_span_from_fourier_span(
        **_grid(nt, nf),
        kmin=kmin,
        lendata=lendata,
        window="gaussian",
        support=support,
    )
    first_bin, fourier_count = fourier_span_from_wdm_span(
        **_grid(nt, nf),
        mmin=first_channel,
        nf_sub_wdm=channel_count,
        window="gaussian",
        support=support,
    )
    options = dict(
        _grid(nt, nf), df=0.125, backend="jax", window="gaussian", support=support
    )
    rng = np.random.default_rng(723)
    crop = jnp.asarray(rng.normal(size=lendata) + 1j * rng.normal(size=lendata))
    tangent = jnp.asarray(rng.normal(size=lendata) + 1j * rng.normal(size=lendata))

    def compact_analysis(values):
        return from_freq_to_wdm_subband(values, kmin=kmin, **options)[0]

    def full_analysis(values):
        one_sided = jnp.zeros(nt * nf // 2 + 1, dtype=jnp.complex128)
        one_sided = one_sided.at[kmin : kmin + lendata].set(values)
        full_spectrum = jnp.concatenate([one_sided, jnp.conj(one_sided[-2:0:-1])])
        coefficients = from_freq_to_wdm(
            full_spectrum,
            nt=nt,
            nf=nf,
            a=1 / 3,
            d=1,
            dt=1,
            backend="jax",
            window="gaussian",
            support=support,
        )[0]
        return coefficients[:, first_channel : first_channel + channel_count]

    np.testing.assert_allclose(
        jax.jit(compact_analysis)(crop), full_analysis(crop), rtol=1e-12, atol=2e-12
    )
    _, compact_tangent = jax.jvp(compact_analysis, (crop,), (tangent,))
    _, full_tangent = jax.jvp(full_analysis, (crop,), (tangent,))
    np.testing.assert_allclose(compact_tangent, full_tangent, rtol=1e-12, atol=2e-12)
    cotangent = jnp.asarray(rng.normal(size=(nt, channel_count)))
    _, compact_pullback = jax.vjp(compact_analysis, crop)
    _, full_pullback = jax.vjp(full_analysis, crop)
    np.testing.assert_allclose(
        compact_pullback(cotangent)[0],
        full_pullback(cotangent)[0],
        rtol=1e-12,
        atol=2e-12,
    )
    compact_gradient = jax.grad(lambda values: jnp.sum(compact_analysis(values) ** 2))(
        crop
    )
    full_gradient = jax.grad(lambda values: jnp.sum(full_analysis(values) ** 2))(crop)
    np.testing.assert_allclose(compact_gradient, full_gradient, rtol=1e-12, atol=2e-12)

    coefficients = jnp.asarray(rng.normal(size=(nt, channel_count)))
    coefficient_tangent = jnp.asarray(rng.normal(size=coefficients.shape))

    def compact_synthesis(values):
        return from_wdm_to_freq_subband(values, mmin=first_channel, **options)[0]

    def full_synthesis(values):
        full_coefficients = jnp.zeros((nt, nf + 1))
        full_coefficients = full_coefficients.at[
            :, first_channel : first_channel + channel_count
        ].set(values)
        spectrum = from_wdm_to_freq(
            full_coefficients,
            a=1 / 3,
            d=1,
            dt=1,
            backend="jax",
            window="gaussian",
            support=support,
        )[0]
        return spectrum[first_bin : first_bin + fourier_count]

    np.testing.assert_allclose(
        jax.jit(compact_synthesis)(coefficients),
        full_synthesis(coefficients),
        rtol=1e-12,
        atol=2e-12,
    )
    _, compact_tangent = jax.jvp(
        compact_synthesis, (coefficients,), (coefficient_tangent,)
    )
    _, full_tangent = jax.jvp(full_synthesis, (coefficients,), (coefficient_tangent,))
    np.testing.assert_allclose(compact_tangent, full_tangent, rtol=1e-12, atol=2e-12)
    spectrum_cotangent = jnp.asarray(
        rng.normal(size=fourier_count) + 1j * rng.normal(size=fourier_count)
    )
    _, compact_pullback = jax.vjp(compact_synthesis, coefficients)
    _, full_pullback = jax.vjp(full_synthesis, coefficients)
    np.testing.assert_allclose(
        compact_pullback(spectrum_cotangent)[0],
        full_pullback(spectrum_cotangent)[0],
        rtol=1e-12,
        atol=2e-12,
    )


def test_first_jitted_gaussian_subband_call_does_not_leak_cached_tracers():
    jax = pytest.importorskip("jax")
    get_backend("jax")
    jnp = pytest.importorskip("jax.numpy")
    nt, nf, support, nu = 14, 24, 5, 0.47
    kmin = 31
    crop = jnp.asarray(np.random.default_rng(724).normal(size=13) + 1j)
    options = dict(
        _grid(nt, nf),
        df=0.125,
        backend="jax",
        window="gaussian",
        support=support,
        nu=nu,
    )

    def analyze(values):
        return from_freq_to_wdm_subband(values, kmin=kmin, **options)[0]

    # This window key is unique in the suite: its first device transfer is traced.
    with jax.checking_leaks():
        first_trace = jax.jit(analyze)(crop)
        independent_trace = jax.jit(lambda values: analyze(2 * values) / 2)(crop)
        eager = analyze(crop)
        first_trace.block_until_ready()
        independent_trace.block_until_ready()
        eager.block_until_ready()
    assert first_trace.dtype == jnp.float64
    np.testing.assert_allclose(first_trace, independent_trace, rtol=0, atol=2e-13)
    np.testing.assert_allclose(first_trace, eager, rtol=0, atol=2e-13)
    first_channel, channel_count = wdm_span_from_fourier_span(
        **_grid(nt, nf),
        kmin=kmin,
        lendata=crop.size,
        window="gaussian",
        support=support,
        nu=nu,
    )
    one_sided = np.zeros(nt * nf // 2 + 1, dtype=complex)
    one_sided[kmin : kmin + crop.size] = np.asarray(crop)
    full_spectrum = np.concatenate([one_sided, one_sided[-2:0:-1].conj()])
    full_coefficients = from_freq_to_wdm(
        full_spectrum,
        nt=nt,
        nf=nf,
        a=1 / 3,
        d=1,
        dt=1,
        backend="numpy",
        window="gaussian",
        support=support,
        nu=nu,
    )[0]
    np.testing.assert_allclose(
        first_trace,
        full_coefficients[:, first_channel : first_channel + channel_count],
        rtol=1e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("kmin,lendata", [(0, 9), (59, 6), (23, 7)])
def test_compact_analysis_synthesis_adjoint_uses_one_sided_weights(
    backend_name, kmin, lendata
):
    nt, nf, support = 8, 16, 3
    rng = np.random.default_rng(725)
    crop = rng.normal(size=lendata) + 1j * rng.normal(size=lendata)
    if kmin == 0:
        crop[0] = crop[0].real
    if kmin + lendata == nt * nf // 2 + 1:
        crop[-1] = crop[-1].real
    options = dict(
        _grid(nt, nf),
        df=0.125,
        backend=backend_name,
        window="gaussian",
        support=support,
    )
    analysis, first_channel = from_freq_to_wdm_subband(crop, kmin=kmin, **options)
    pixels = rng.normal(size=analysis.shape)
    synthesis, first_bin = from_wdm_to_freq_subband(
        pixels, mmin=first_channel, **options
    )
    selected = np.asarray(synthesis)[kmin - first_bin : kmin - first_bin + lendata]
    weights = np.full(lendata, 2.0)
    if kmin == 0:
        weights[0] = 1
    if kmin + lendata == nt * nf // 2 + 1:
        weights[-1] = 1
    lhs = np.sum(np.asarray(analysis) * pixels)
    rhs = np.vdot(crop, selected * weights).real
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=2e-12)


@pytest.mark.parametrize("df", [0, -1, np.nan, np.inf, True])
def test_gaussian_public_subbands_reject_invalid_frequency_spacing(backend_name, df):
    options = dict(_grid(8, 16), df=df, backend=backend_name, window="gaussian")
    with pytest.raises(ValueError, match="df"):
        from_freq_to_wdm_subband(np.ones(4, dtype=complex), kmin=3, **options)
    with pytest.raises(ValueError, match="df"):
        from_wdm_to_freq_subband(np.ones((8, 2)), mmin=3, **options)


@pytest.mark.parametrize(
    "changed",
    [
        dict(mmin=True),
        dict(mmin=3.0),
        dict(nf_sub_wdm=True),
        dict(nf_sub_wdm=2.0),
        dict(nf_sub_wdm=0),
        dict(nf_sub_wdm=-1),
    ],
)
def test_requested_gaussian_band_rejects_invalid_channel_coordinates(
    backend_name, changed
):
    options = dict(
        _grid(8, 16),
        df=0.125,
        kmin=3,
        mmin=3,
        nf_sub_wdm=2,
        backend=backend_name,
        window="gaussian",
    )
    options.update(changed)
    with pytest.raises(ValueError):
        from_freq_to_wdm_band(np.ones(4, dtype=complex), **options)


@pytest.mark.parametrize("support", [0, -1, 17, True, 1.0])
def test_gaussian_public_subbands_reject_invalid_support(backend_name, support):
    options = dict(
        _grid(8, 16), df=0.125, backend=backend_name, window="gaussian", support=support
    )
    with pytest.raises(ValueError, match="support"):
        from_freq_to_wdm_subband(np.ones(4, dtype=complex), kmin=3, **options)
    with pytest.raises(ValueError, match="support"):
        from_wdm_to_freq_subband(np.ones((8, 2)), mmin=3, **options)


def test_gaussian_public_subbands_reject_malformed_or_outside_spans(backend_name):
    options = dict(_grid(8, 16), df=0.125, backend=backend_name, window="gaussian")
    for data, kmin in (
        (np.empty(0, dtype=complex), 3),
        (np.ones((2, 2)), 3),
        (np.ones(4, dtype=complex), -1),
        (np.ones(4, dtype=complex), 63),
        (np.ones(4, dtype=complex), True),
        (np.ones(4, dtype=complex), 3.0),
    ):
        with pytest.raises(ValueError):
            from_freq_to_wdm_subband(data, kmin=kmin, **options)
    for data, mmin in (
        (np.empty((8, 0)), 3),
        (np.ones((7, 2)), 3),
        (np.ones((1, 8, 2)), 3),
        (np.ones((8, 2)), -1),
        (np.ones((8, 2)), 16),
        (np.ones((8, 2)), True),
        (np.ones((8, 2)), 3.0),
    ):
        with pytest.raises(ValueError):
            from_wdm_to_freq_subband(data, mmin=mmin, **options)


def test_optional_cupy_gaussian_subbands_match_numpy():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CuPy-compatible GPU is available")
        cupy.zeros(1).sum().item()
    except Exception as error:
        pytest.skip(f"CuPy GPU runtime is unavailable: {error}")
    options = dict(_grid(16, 32), df=0.125, window="gaussian", support=3)
    for kmin in (0, 120, 253):
        crop = np.array([1 + 2j, 2 - 3j, -1 + 4j, -3 + 1j])
        expected_coefficients, expected_channel = from_freq_to_wdm_subband(
            crop, kmin=kmin, backend="numpy", **options
        )
        actual_coefficients, actual_channel = from_freq_to_wdm_subband(
            cupy.asarray(crop), kmin=kmin, backend="cupy", **options
        )
        assert actual_channel == expected_channel
        np.testing.assert_allclose(
            cupy.asnumpy(actual_coefficients),
            expected_coefficients,
            rtol=1e-12,
            atol=2e-12,
        )
        expected_spectrum, expected_bin = from_wdm_to_freq_subband(
            expected_coefficients, mmin=expected_channel, backend="numpy", **options
        )
        actual_spectrum, actual_bin = from_wdm_to_freq_subband(
            actual_coefficients, mmin=actual_channel, backend="cupy", **options
        )
        assert actual_bin == expected_bin
        np.testing.assert_allclose(
            cupy.asnumpy(actual_spectrum),
            expected_spectrum,
            rtol=1e-12,
            atol=2e-12,
        )
