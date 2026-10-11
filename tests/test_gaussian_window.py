"""Independent scientific validation of the modified Gaussian Wilson window."""

import numpy as np
import pytest
from _gaussian_reference import (
    AUTHORS_SAMPLE_POSITIONS,
    AUTHORS_SAMPLES,
    direct_basis,
    paper_window,
)

from wdm_transform.windows_gaussian import gaussian_unit, gaussian_window


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8])
def test_window_matches_independent_paper_recursion(nu):
    positions = np.linspace(-8, 8, 513)
    reference = paper_window(positions, nu)
    np.testing.assert_allclose(reference.imag, 0, atol=3e-14)
    # nu=.8 has a 2e-11 finite lattice residual at M=N=24.
    tolerance = 3e-11 if nu == 0.8 else 3e-14
    np.testing.assert_allclose(
        gaussian_unit(positions, nu), reference.real, rtol=0, atol=tolerance
    )


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8])
def test_window_matches_authors_offline_samples(nu):
    np.testing.assert_allclose(
        gaussian_unit(AUTHORS_SAMPLE_POSITIONS, nu),
        AUTHORS_SAMPLES[nu],
        rtol=0,
        atol=3e-14,
    )


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8])
def test_recursion_lattice_truncation_converges(nu):
    positions = AUTHORS_SAMPLE_POSITIONS
    errors = [
        np.max(
            np.abs(paper_window(positions, nu, lattice=size).real - AUTHORS_SAMPLES[nu])
        )
        for size in (8, 16, 24)
    ]
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 100


def test_recursion_series_truncation_is_independent_of_spectral_support():
    positions = np.linspace(-4, 4, 33)
    target = gaussian_unit(positions, 0.3)
    errors = [
        np.max(np.abs(paper_window(positions, 0.3, lattice=20, terms=n).real - target))
        for n in (4, 12, 40)
    ]
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 100


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8])
def test_real_even_cross_band_orthogonality(nu):
    positions = np.linspace(-0.5, 0.5, 37)
    translations = np.arange(-32, 33)
    window = gaussian_unit(positions[:, None] + translations, nu)
    assert np.isrealobj(window)
    assert np.all(np.isfinite(window))
    np.testing.assert_allclose(
        gaussian_unit(-positions[:, None] - translations, nu),
        window,
        rtol=0,
        atol=2e-15,
    )
    for shift in (-4, -2, -1, 0, 1, 2, 4):
        translated = gaussian_unit(positions[:, None] + translations + 2 * shift, nu)
        inner = np.sum(window * translated, axis=1)
        np.testing.assert_allclose(inner, float(shift == 0), rtol=0, atol=3e-14)


@pytest.mark.parametrize("nt", [2, 8, 32, 64])
def test_sampled_normalization_and_signed_frequency_indices(nt):
    support = 16
    offsets = np.arange(-support * nt // 2, support * nt // 2)
    sampled = gaussian_window(nt, nu=0.5, support=support)
    np.testing.assert_allclose(
        sampled,
        np.sqrt(2 / nt) * gaussian_unit(2 * offsets / nt, 0.5),
        rtol=0,
        atol=1e-16,
    )
    # The normalization follows df=2/nt in the paper's unit-frequency grid.
    # Sampling can be coarse at nt=2; the discrete identity still fixes the norm.
    assert np.isfinite(sampled).all()
    assert abs(np.sum(sampled**2) - 1) < 3e-12
    assert sampled[support * nt // 2] > 0


def test_window_cache_cannot_be_corrupted():
    first = gaussian_window(32, nu=0.5, support=16)
    expected = first.copy()
    if first.flags.writeable:
        first[:] = 0
    else:
        with pytest.raises(ValueError):
            first[:] = 0
    np.testing.assert_array_equal(gaussian_window(32, nu=0.5, support=16), expected)


def test_frequency_and_time_moments_reproduce_uncertainty_product():
    # x=f/DeltaF and t conjugate to x; DeltaT*DeltaF=1/2.
    count = 8192
    dx = 32 / count
    frequency = (np.arange(count) - count // 2) * dx
    phi = gaussian_unit(frequency, 0.5)
    energy = np.sum(phi**2) * dx
    frequency_variance = np.sum(frequency**2 * phi**2) * dx / energy
    time = np.fft.fftfreq(count, d=dx)
    fourier = np.fft.fft(phi)
    time_variance = np.sum(time**2 * abs(fourier) ** 2) / np.sum(abs(fourier) ** 2)
    assert abs(energy - 1) < 3e-14
    assert np.sqrt(frequency_variance) == pytest.approx(0.40362210653, abs=2e-10)
    assert 4 * time_variance == pytest.approx(frequency_variance, abs=3e-12)
    ratio = 4 * np.pi * np.sqrt(time_variance * frequency_variance)
    assert ratio == pytest.approx(1.0235987753, abs=2e-9)


def test_direct_basis_gram_and_edge_redundancy():
    nt, nf = 8, 32
    atoms = direct_basis(nt, nf, support=16, unit_window=gaussian_unit)
    gram = atoms.T @ atoms / (nt * nf)
    assert np.max(np.abs(gram - np.eye(nt * nf))) < 1e-11
    norms = np.sum(atoms.reshape(nt, nf + 1, -1) ** 2, axis=-1) / (nt * nf)
    np.testing.assert_allclose(norms[:, 1:-1], 1, rtol=0, atol=1e-11)
    np.testing.assert_allclose(norms[:, [0, -1]], 0.5, rtol=0, atol=1e-11)
    # Each edge has duplicated nt/2 independent atoms, unlike an interior band.
    reshaped = atoms.reshape(nt, nf + 1, -1)
    for channel in (0, nf):
        np.testing.assert_allclose(
            reshaped[: nt // 2, channel],
            reshaped[nt // 2 :, channel],
            rtol=0,
            atol=2e-12,
        )


def test_spectral_truncation_converges_without_window_renormalization():
    nt, nf = 8, 32
    errors = []
    for support in (4, 8, 16):
        atoms = direct_basis(nt, nf, support=support, unit_window=gaussian_unit)
        errors.append(np.max(np.abs(atoms.T @ atoms / (nt * nf) - np.eye(nt * nf))))
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 10000
    assert errors[2] < 1e-11


@pytest.mark.parametrize("nu,support", [(0.3, 32), (0.5, 16), (0.8, 16)])
def test_white_noise_covariance_and_complete_row_gram(nu, support):
    # White input with covariance I has exact output covariance A A.T.
    # Divide by N to expose unit interior atoms and duplicated half-norm edges.
    nt, nf = 8, 32
    atoms = direct_basis(nt, nf, nu=nu, support=support, unit_window=gaussian_unit)
    covariance = atoms @ atoms.T / (nt * nf)
    expected = np.zeros_like(covariance)
    for time in range(nt):
        for channel in range(nf + 1):
            row = time * (nf + 1) + channel
            if channel in (0, nf):
                expected[row, row] = 0.5
                duplicate = ((time + nt // 2) % nt) * (nf + 1) + channel
                expected[row, duplicate] = 0.5
            else:
                expected[row, row] = 1
    np.testing.assert_allclose(covariance, expected, rtol=0, atol=1e-11)


def test_distant_samples_do_not_wrap_the_finite_zak_grid():
    np.testing.assert_allclose(
        gaussian_unit(np.array([-1024, -128, 128, 1024]), 0.5), 0, atol=1e-24
    )
