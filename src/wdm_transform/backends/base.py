from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Backend:
    """Minimal backend wrapper around an array namespace and FFT namespace."""

    name: str
    xp: Any
    fft: Any

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        """Convert a value into an array for this backend."""
        return self.xp.asarray(value, dtype=dtype)
