from __future__ import annotations

import numpy as np

from .base import Backend


NUMPY_BACKEND = Backend(name="numpy", xp=np, fft=np.fft)
