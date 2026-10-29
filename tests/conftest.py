import numpy as np
import pytest
from scipy.special import erf


def render_spots(shape, spots, sigma, background=100.0, noise_sd=2.0, seed=0):
    """Integrated-Gaussian spots (y, x, amplitude) on a flat, noisy background."""
    rng = np.random.default_rng(seed)
    H, W = shape
    img = np.full(shape, background, dtype=np.float64)
    s2 = sigma * np.sqrt(2.0)
    y_edges = np.arange(H + 1) - 0.5
    x_edges = np.arange(W + 1) - 0.5
    for y, x, amp in spots:
        py = np.diff(0.5 * (1 + erf((y_edges - y) / s2)))
        px = np.diff(0.5 * (1 + erf((x_edges - x) / s2)))
        img += amp * np.outer(py, px)
    img += rng.normal(0.0, noise_sd, shape)
    return img.astype(np.float32)


@pytest.fixture
def grid_spots():
    """A 4×4 grid of well-separated spots with sub-pixel offsets."""
    rng = np.random.default_rng(1)
    spots = []
    for gy in range(4):
        for gx in range(4):
            y = 15 + 20 * gy + rng.uniform(-0.5, 0.5)
            x = 15 + 20 * gx + rng.uniform(-0.5, 0.5)
            spots.append((y, x, 2000.0))
    return np.array(spots)
