"""Independent, offline references for the Gaussian Wilson construction.

The lattice recurrence below is a direct implementation of Pearson et al.,
arXiv:2609.05601, Eqs. (1)--(5).  It deliberately uses the dense overlap matrix
instead of the production Zak/polyphase normalization.  No external package,
download, or production window routine is needed by this oracle.

The compact numerical fixture was evaluated with the authors' reference
implementation in XGI-MSU/WDMWaveletTransforms PR #5, head
e6493d716a5a3532a528bbef3d2f83d0bbe20bad (verified open on 2026-10-11).
Its evaluation used M=N=32, Kmax=80, L=12, a 32x32 frame-bound grid, FFT
recursion, and null_mode=0.  The pinned source declares GPL-2.0-or-later;
no implementation code is copied here.  The
authors' nu parametrizes their time-domain Gaussian, so their frequency
evaluation phihat(x/2, nu=1/(4*nu)) / sqrt(2) matches this paper convention.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

AUTHORS_SAMPLE_POSITIONS = np.array(
    [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8], dtype=float
)
AUTHORS_SAMPLES = {
    0.3: np.array(
        [
            0.8881926083321093,
            0.8762041396865402,
            0.8403782551543537,
            0.7031946819665372,
            0.5113190869589361,
            0.32026260604702467,
            0.05263151469482072,
            -0.0487307948542995,
            -0.023818964244727615,
            0.006237681777564952,
            0.00266191929979749,
            -0.0008092416623491575,
            -0.00033409567922345034,
            0.00010862166527726224,
        ]
    ),
    0.5: np.array(
        [
            0.9606680992514263,
            0.9484701141182637,
            0.9075842438477778,
            0.7067777743826877,
            0.4120231526272319,
            0.1953950044470629,
            0.015253524452520161,
            -0.019007078538547256,
            -0.004183632914292437,
            0.0006349600698603335,
            0.00013502107983063004,
            -2.3004313492128678e-05,
            -4.852864699299592e-06,
            8.720833444406083e-07,
        ]
    ),
    0.8: np.array(
        [
            0.9935342601222517,
            0.9877987520140263,
            0.9615970001122549,
            0.707099171149152,
            0.27360743048278124,
            0.08021483257235147,
            0.0023197248816663935,
            -0.0032168819774001265,
            -0.0002627411359190441,
            1.5900440934759964e-05,
            1.292239678955687e-06,
            -8.701779304210543e-08,
            -7.063816566120764e-09,
            4.99778703556782e-10,
        ]
    ),
}


def _paper_frame_bounds(nu: float) -> tuple[float, float]:
    """Evaluate paper Eq. (5), including its sqrt(2) and period-two lattice."""
    a = np.arange(32) / 32
    b = np.arange(32) / 32
    k = np.arange(-12, 13)
    phase = np.exp(2j * np.pi * a[:, None, None] * k[None, None, :])
    gaussian = (2 * nu) ** 0.25 * np.exp(
        -np.pi * nu * (2 * (b[None, :, None] - k[None, None, :])) ** 2
    )
    zak = np.sqrt(2) * np.sum(phase * gaussian, axis=-1)
    frame = abs(zak) ** 2 + abs(np.roll(zak, 16, axis=1)) ** 2
    return float(frame.min()), float(frame.max())


@lru_cache(maxsize=12)
def _paper_coefficients(
    nu: float, lattice: int, terms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Dense Eq. (3), without production FFT normalization or null projection."""
    m, n = np.meshgrid(
        np.arange(-lattice, lattice + 1),
        np.arange(-lattice, lattice + 1),
        indexing="ij",
    )
    m, n = m.ravel(), n.ravel()
    dm = m[None, :] - m[:, None]
    dn = n[None, :] - n[:, None]
    # Integral phases are evaluated exactly instead of a large-angle exponential.
    phase = 1j ** ((dm * (n[None, :] + n[:, None])) % 4)
    overlap = phase * np.exp(-np.pi * nu / 2 * dn**2 - np.pi / (8 * nu) * dm**2)
    lower, upper = _paper_frame_bounds(nu)
    b = np.zeros(m.size, dtype=complex)
    b[lattice * (2 * lattice + 1) + lattice] = 1
    coefficients = b.copy()
    central_binomial = 1.0
    for k in range(terms):
        b = b - (2 / (lower + upper)) * (overlap @ b)
        central_binomial *= (2 * k + 1) / (2 * k + 2)
        coefficients += central_binomial * b
    return m, n, coefficients, 2 / np.sqrt(lower + upper)


def paper_window(
    f: np.ndarray, nu: float = 0.5, *, lattice: int = 24, terms: int = 80
) -> np.ndarray:
    """Evaluate the independent lattice series in units of a full band spacing."""
    f = np.asarray(f, dtype=float)
    m, n, coefficients, prefactor = _paper_coefficients(nu, lattice, terms)
    out = np.zeros(f.size, dtype=complex)
    flat = f.ravel()
    for modulation in np.unique(m):
        select = m == modulation
        gaussians = (2 * nu) ** 0.25 * np.exp(
            -np.pi * nu * (flat[None, :] - n[select, None]) ** 2
        )
        translated = np.sum(coefficients[select, None] * gaussians, axis=0)
        out += np.exp(1j * np.pi * modulation * flat) * translated
    return (prefactor * out).reshape(f.shape)


def direct_analysis(
    data: np.ndarray,
    nt: int,
    nf: int,
    *,
    nu: float = 0.5,
    support: int = 16,
    unit_window=None,
) -> np.ndarray:
    """Literal spectral atom sums; no folding or small-FFT production kernels.

    Window samples are from the independent recurrence unless an evaluator is
    explicitly supplied for testing the finite-support basis separately.
    """
    data = np.asarray(data)
    offsets = np.arange(-support * nt // 2, support * nt // 2)
    evaluate = paper_window if unit_window is None else unit_window
    phi = np.sqrt(2 / nt) * np.asarray(evaluate(2 * offsets / nt, nu)).real
    spectrum = np.fft.fft(data, axis=-1)
    rows = np.arange(nt)
    oscillation = np.exp(2j * np.pi * offsets[:, None] * rows / nt)
    edge_oscillation = np.exp(4j * np.pi * offsets[:, None] * rows / nt)
    result = np.empty(data.shape[:-1] + (nt, nf + 1))
    for channel in range(nf + 1):
        shifted = spectrum[..., (channel * nt // 2 + offsets) % (nt * nf)]
        weighted = shifted * phi
        if channel in (0, nf):
            result[..., :, channel] = (weighted @ edge_oscillation).real / np.sqrt(2)
        else:
            phase = np.where((rows + channel) % 2, 1j, 1)
            sign = (-1.0) ** (channel * rows)
            result[..., :, channel] = (
                np.sqrt(2) * sign * (phase.conj() * (weighted @ oscillation)).real
            )
    return result


def direct_basis(
    nt: int, nf: int, *, nu: float = 0.5, support: int = 16, unit_window=None
) -> np.ndarray:
    """Analysis rows on the real time basis, including both redundant edges."""
    return (
        direct_analysis(
            np.eye(nt * nf),
            nt,
            nf,
            nu=nu,
            support=support,
            unit_window=unit_window,
        )
        .reshape(nt * nf, -1)
        .T
    )
