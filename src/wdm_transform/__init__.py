"""Public package exports for the WDM transform scaffold."""

from .backends import Backend, get_backend, register_backend
from .datatypes import FrequencySeries, TimeSeries, WDM
from .signal_processing import (
    matched_filter_snr_rfft,
    matched_filter_snr_wdm,
    noise_characteristic_strain,
    rfft_characteristic_strain,
    wdm_noise_variance,
)

__all__ = [
    "Backend",
    "FrequencySeries",
    "TimeSeries",
    "WDM",
    "get_backend",
    "matched_filter_snr_rfft",
    "matched_filter_snr_wdm",
    "noise_characteristic_strain",
    "register_backend",
    "rfft_characteristic_strain",
    "wdm_noise_variance",
]
