"""Input/waveform derivatives, checked against independent real basis atoms."""

from __future__ import annotations

import numpy as np
import pytest
from _gaussian_reference import direct_basis

from wdm_transform import TimeSeries
from wdm_transform.backends import get_backend
from wdm_transform.transforms import from_time_to_wdm, from_wdm_to_time

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
get_backend("jax")  # Match the package's float64 configuration before arrays.


def _forward(x, nt=8, nf=16, nu=0.5, support=16):
    return from_time_to_wdm(
        x,
        nt=nt,
        nf=nf,
        a=1 / 3,
        d=1,
        dt=1,
        backend="jax",
        window="gaussian",
        nu=nu,
        support=support,
    )


def _inverse(w, nu=0.5, support=16):
    return from_wdm_to_time(
        w,
        a=1 / 3,
        d=1,
        dt=1,
        backend="jax",
        window="gaussian",
        nu=nu,
        support=support,
    )


def test_jit_jvp_vjp_and_synthesis_adjoint():
    nt, nf = 8, 16
    rng = np.random.default_rng(95)
    x, direction = rng.normal(size=(2, nt * nf))
    cotangent = rng.normal(size=(1, nt, nf + 1))
    forward = jax.jit(_forward)
    inverse = jax.jit(_inverse)
    coefficients, tangent = jax.jvp(
        forward, (jnp.asarray(x),), (jnp.asarray(direction),)
    )
    assert isinstance(coefficients, jax.Array)
    np.testing.assert_allclose(tangent, forward(direction), atol=2e-13)
    basis = direct_basis(nt, nf)
    np.testing.assert_allclose(coefficients.ravel(), basis @ x, atol=2e-12)
    _, pullback = jax.vjp(forward, jnp.asarray(x))
    actual_vjp = pullback(jnp.asarray(cotangent))[0]
    expected_vjp = basis.T @ cotangent.ravel()
    np.testing.assert_allclose(actual_vjp, expected_vjp, atol=2e-11)
    synthesis = np.asarray(inverse(cotangent))[0]
    np.testing.assert_allclose(synthesis, expected_vjp / (nt * nf), atol=2e-13)
    np.testing.assert_allclose(
        np.vdot(np.asarray(coefficients), cotangent) / (nt * nf),
        np.vdot(x, synthesis),
        atol=2e-13,
    )


def test_gradient_of_energy_and_waveform_parameter():
    nt, nf = 32, 32
    rng = np.random.default_rng(11)
    x = rng.normal(size=nt * nf)

    def forward(y):
        return _forward(y, nt=nt, nf=nf)

    def energy(y):
        return jnp.sum(forward(y) ** 2) / (2 * nt * nf)

    np.testing.assert_allclose(jax.jit(jax.grad(energy))(x), x, atol=3e-11)
    times = jnp.arange(nt * nf) / (nt * nf)

    def objective(theta):
        waveform = theta * jnp.sin(71 * times) + jnp.cos(theta) * jnp.cos(123 * times)
        return energy(waveform)

    theta, step = 0.7, 1e-5
    actual = float(jax.jit(jax.grad(objective))(theta))
    finite_difference = float(
        (objective(theta + step) - objective(theta - step)) / (2 * step)
    )
    np.testing.assert_allclose(actual, finite_difference, atol=3e-8, rtol=2e-9)


def test_first_window_use_inside_jit_does_not_cache_a_tracer():
    # This shape/nu/support key is intentionally unused by the other tests.
    transform = jax.jit(
        lambda x: TimeSeries(x, dt=0.2, backend="jax")
        .to_wdm(nt=10, window="gaussian", nu=0.37, support=7)
        .coeffs
    )
    x = np.random.default_rng(59).normal(size=200)
    first = transform(x)
    # A different trace reuses the same cached constant window.
    second = jax.jit(lambda y: _forward(y, nt=10, nf=20, nu=0.37, support=7))(x)
    np.testing.assert_array_equal(first, second)
    with jax.checking_leaks():
        jax.jit(lambda y: _forward(y, nt=12, nf=20, nu=0.43, support=9))(np.zeros(240))
