import pytest
from hypothesis import given, example, note, strategies as st
from inline_snapshot import snapshot

import numpy as np

from wdm_transform.backends import NUMPY_BACKEND
from wdm_transform.backends.jax_backend import load_jax_backend
from wdm_transform.windows import gnmf


@st.composite
def grid(draw, min=2):
    # all that we claim in the paper: our equations should work
    # for all even Nt and even Nf.
    assert min >= 2
    halfmin = min // 2
    Nt = 2 * draw(st.integers(halfmin, halfmin + 5))
    Nf = 2 * draw(st.integers(halfmin, halfmin + 5))
    return (Nt, Nf)


@st.composite
def one_n_two_ms(draw, min=2):
    Nt, Nf = draw(grid(min))
    n = draw(st.integers(0, Nt - 1))
    m1 = draw(st.integers(0, Nf))
    m2 = draw(st.integers(0, Nf))
    return (Nt, Nf, n, m1, m2)


@st.composite
def two_ns_one_m(draw, min=2):
    Nt, Nf = draw(grid(min))
    n1 = draw(st.integers(0, Nt - 1))
    n2 = draw(st.integers(0, Nt - 1))
    m = draw(st.integers(0, Nf))
    return (Nt, Nf, n1, n2, m)


# Desired level of precision on inner products.
atol = 1e-8

pos_real = st.floats(
    min_value=1e-200,
    max_value=1e200,
    exclude_min=True,
    allow_subnormal=False,
    allow_nan=False,
    allow_infinity=False,
)

# Using max_value=0.5, exclude_max=True isn't enough.
shape_param_a = st.floats(0, 0.5 - 1e-7)

# d != 1 not implemented
shape_param_d = st.just(1)
# shape_param_d = st.floats(min_value=1)

# JAX disabled because very slow, maybe JIT-in-a-loop overhead?
JAX_BACKEND = load_jax_backend()
backend_gen = st.just(NUMPY_BACKEND) #| st.just(JAX_BACKEND)

def ortho_condition(Nf, Nt, n1, n2, m1, m2):
    if m1 == m2:
        if m1 in (0, Nf):
            if n1 == n2 or abs(n1 - n2) == (Nt / 2):
                return 0.5
        elif n1 == n2:
            return 1
    return 0


@given(one_n_two_ms(min=100), pos_real, shape_param_a, shape_param_d, backend_gen)
def test_ortho_twom(args, dT, a, d, backend):
    Nt, Nf, n, m1, m2 = args
    note(f"{Nt=}, {Nf=}")
    note(f"{n=}, {m1=}, {m2=}")
    xp = backend.xp

    freqs = xp.fft.fftfreq(Nt * Nf, dT / Nf)
    w1 = gnmf(backend, n, m1, freqs, dT, Nf, a, d)
    w2 = gnmf(backend, n, m2, freqs, dT, Nf, a, d)

    norm_squared = xp.sum(w1.conj() * w1).real

    note(f"{norm_squared = }")
    note(f"{w1.shape=}")
    note(f"{w2.shape=}")

    inner = xp.sum(w1.conj() * w2)
    note(f"{inner = }")
    note(f"{abs(inner) = }")
    assert xp.isclose(inner, ortho_condition(Nf, Nt, n, n, m1, m2), atol=atol)


@given(two_ns_one_m(min=10), pos_real, shape_param_a, shape_param_d, backend_gen)
def test_ortho_twon(args, dT, a, d, backend):
    Nt, Nf, n1, n2, m = args
    note(f"{Nt=}, {Nf=}")
    note(f"{n1=}, {n2=}, {m=}")
    xp = backend.xp

    freqs = xp.fft.fftfreq(Nt * Nf, dT / Nf)
    w1 = gnmf(backend, n1, m, freqs, dT, Nf, a, d)
    w2 = gnmf(backend, n2, m, freqs, dT, Nf, a, d)

    norm_squared = xp.sum(w1.conj() * w1).real

    note(f"{norm_squared = }")
    note(f"{w1.shape=}")
    note(f"{w2.shape=}")

    inner = xp.sum(w1.conj() * w2)
    note(f"{inner = }")
    note(f"{abs(inner) = }")
    assert xp.isclose(inner, ortho_condition(Nf, Nt, n1, n2, m, m), atol=atol)


def test_normalization():
    # The point of this was originally to dig deeper into normalization issues.
    # After updating to latest code those issues are gone.
    # This test just tells us that the normalization works fine.

    def norm(Nf, Nt, n, m):
        dt = 1
        freqs = np.fft.fftfreq(Nt * Nf, dt)
        xp = np
        nt = int(xp.asarray(freqs).shape[-1]) // int(Nf)
        assert nt == Nt
        g = gnmf(NUMPY_BACKEND, n, m, freqs, Nf * dt, Nf, 0.49, 1)
        return np.sum(g.conj() * g).real

    def comp(Nf, Nt, n, m, expected=1):
        n1 = norm(Nf, Nt, n, m)
        close = np.isclose(n1, expected)
        return f"{n1:.10f} {close}"

    # Reproducing I saw in the property test
    assert comp(10, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(14, 14, 0, 1) == snapshot("1.0000000000 True")

    # It seems unaffected by n
    assert comp(10, 10, 1, 1) == snapshot("1.0000000000 True")
    assert comp(12, 10, 1, 1) == snapshot("1.0000000000 True")
    assert comp(10, 12, 1, 1) == snapshot("1.0000000000 True")
    assert comp(12, 12, 1, 1) == snapshot("1.0000000000 True")
    assert comp(14, 14, 1, 1) == snapshot("1.0000000000 True")

    # fixed Nf=10, changing Nt
    assert comp(10, 8, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 14, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 16, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 18, 0, 1) == snapshot("1.0000000000 True")

    # fixed Nt=10, changing Nf
    assert comp(8, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(14, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(16, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(18, 10, 0, 1) == snapshot("1.0000000000 True")

    # fixed Nf=12, changing Nt
    assert comp(12, 8, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 10, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 14, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 16, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 18, 0, 1) == snapshot("1.0000000000 True")

    # fixed Nt=12, changing Nf
    assert comp(8, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(10, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(12, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(14, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(16, 12, 0, 1) == snapshot("1.0000000000 True")
    assert comp(18, 12, 0, 1) == snapshot("1.0000000000 True")

    # norm at edge channels is 0.5
    assert comp(10, 10, 0, 0, 0.5) == snapshot("0.5000000000 True")
    assert comp(10, 10, 0, 10, 0.5) == snapshot("0.5000000000 True")
