from __future__ import annotations

import numpy as np

import wdm_transform.plotting as plotting
from wdm_transform import WDM, FrequencySeries, TimeSeries


def test_time_series_to_wdm_matches_classmethod() -> None:
    nt, nf, dt = 32, 32, 0.125
    samples = np.sin(2.0 * np.pi * np.arange(nt * nf) * dt * 0.15)
    series = TimeSeries(samples, dt=dt)

    from_instance = series.to_wdm(nt=nt, a=0.25)
    from_class = WDM.from_time_series(series, nt=nt, a=0.25)

    np.testing.assert_allclose(from_instance.coeffs, from_class.coeffs)
    assert from_instance.dt == from_class.dt
    assert from_instance.a == from_class.a


def test_frequency_series_to_wdm_matches_classmethod() -> None:
    nt, nf, dt = 32, 32, 0.125
    samples = np.sin(2.0 * np.pi * np.arange(nt * nf) * dt * 0.15)
    series = TimeSeries(samples, dt=dt).to_frequency_series()

    from_instance = series.to_wdm(nt=nt, a=0.25)
    from_class = WDM.from_frequency_series(series, nt=nt, a=0.25)

    np.testing.assert_allclose(from_instance.coeffs, from_class.coeffs)
    assert from_instance.dt == from_class.dt
    assert from_instance.a == from_class.a


def test_plot_methods_delegate_to_plotting_helpers(monkeypatch) -> None:
    time_series = TimeSeries(np.arange(8, dtype=float), dt=0.25)
    frequency_series = FrequencySeries(np.arange(8, dtype=complex), df=0.5)
    wdm = WDM.from_time_series(time_series, nt=2)
    sentinel = object(), object()
    calls: list[tuple[str, object, dict[str, object]]] = []

    def _record(name: str):
        def _inner(obj: object, **kwargs: object) -> tuple[object, object]:
            calls.append((name, obj, kwargs))
            return sentinel

        return _inner

    monkeypatch.setattr(plotting, "plot_time_series", _record("time"))
    monkeypatch.setattr(plotting, "plot_frequency_series", _record("freq"))
    monkeypatch.setattr(plotting, "plot_wdm_grid", _record("wdm"))

    assert time_series.plot(color="k") == sentinel
    assert frequency_series.plot(magnitude=False) == sentinel
    assert wdm.plot(show_colorbar=False) == sentinel

    assert calls == [
        ("time", time_series, {"color": "k"}),
        ("freq", frequency_series, {"magnitude": False}),
        ("wdm", wdm, {"show_colorbar": False}),
    ]
