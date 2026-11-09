from __future__ import annotations

import numpy as np


def clamp(v, lo, hi):
    return np.clip(v, lo, hi)


def ang_diff(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(d)


def ang_lerp(a: float, b: float, t: float) -> float:
    return a + ang_diff(b, a) * float(np.clip(t, 0.0, 1.0))


def norm2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-8 else x


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.2f}" for x in np.ravel(v)) + "]"
