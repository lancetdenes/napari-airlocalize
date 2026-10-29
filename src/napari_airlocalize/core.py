"""
AIRLOCALIZE algorithm core — pure Python, no GUI dependencies.

Imported by both the napari plugin and the headless batch CLI. Keeping this
module free of napari/qt/magicgui imports lets it run on HPC nodes that have
only numpy/scipy/pandas installed.

Pipeline (per 2D frame):
    0. (Optional) flat-field — img -= gaussian(img, σ_flat) before DoG only
    1. DoG smooth            — gaussian(σ_small) − gaussian(σ_large), mean-centered
    2. Candidates            — dilation local maxima above threshold, ranked by raw
    3. Per-spot fit          — plane-fit local background + Gaussian-mask centroid
    4. Post-fit dedup        — re-run dilation local-max test on fitted centers
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, grey_dilation
from scipy.optimize import curve_fit
from scipy.special import erf


# Keys consumed by detect_frame. Anything else in a params dict is ignored.
PARAM_KEYS = (
    "psfSigma", "threshUnits", "threshLevel", "maxSpots", "minDistBetweenSpots",
    "filterHi", "filterLo", "flatFieldSigma",
    "fittedRegionSize", "bgCorrectionMode", "bgRegionThickness",
    "psfType", "tol", "maxIterations",
)

DEFAULT_PARAMS = {
    "psfSigma": 1.3,
    "threshUnits": "SD",
    "threshLevel": 1.8,
    "maxSpots": 20000,
    "minDistBetweenSpots": 2,
    "filterHi": 1.0,
    "filterLo": 2.15,
    "flatFieldSigma": 0.0,
    "fittedRegionSize": 3.0,
    "bgCorrectionMode": "localPlane",
    "bgRegionThickness": 1,
    "psfType": "integratedGaussian",
    "tol": 0.01,
    "maxIterations": 100,
}


# ── algorithm primitives ────────────────────────────────────────────────────


def apply_flatfield(img: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract a heavily-blurred copy of the image to flatten large-scale BG."""
    img = img.astype(np.float32)
    if sigma <= 0:
        return img
    return img - gaussian_filter(img, sigma)


def dog_filter(img: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    """DoG with the AIRLOCALIZE legacy 1.5× scale factor and per-frame mean removal."""
    img = img.astype(np.float32)
    smooth = gaussian_filter(img, sigma_small) - gaussian_filter(img, sigma_large)
    smooth = 1.5 * smooth
    return smooth - smooth.mean()


def compute_threshold(raw: np.ndarray, smooth: np.ndarray, units: str, level: float) -> float:
    if units == "absolute":
        return float(level)
    if units == "SD":
        return float(level * raw.std())
    if units == "legacySD":
        return float(level * smooth.std())
    raise ValueError(f"unknown threshUnits: {units}")


def find_candidates(
    smooth: np.ndarray,
    raw: np.ndarray,
    threshold: float,
    min_dist: int,
    max_spots: int,
) -> np.ndarray:
    """Local maxima above threshold, ranked by raw intensity. Returns (N,3): y,x,raw."""
    size = 2 * min_dist + 1
    footprint = np.ones((size, size), dtype=bool)
    footprint[min_dist, min_dist] = False
    is_max = smooth >= grey_dilation(smooth, footprint=footprint)
    is_max &= smooth > threshold
    ys, xs = np.nonzero(is_max)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    intensities = raw[ys, xs].astype(np.float64)
    order = np.argsort(-intensities)
    if len(order) > max_spots:
        order = order[:max_spots]
    return np.column_stack([ys[order], xs[order], intensities[order]])


def _shell_mask(shape: tuple, thickness: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    t = thickness
    m[:t, :] = True
    m[-t:, :] = True
    m[:, :t] = True
    m[:, -t:] = True
    return m


def _plane_fit_bg(roi: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """Fit I = a*x + b*y + c on shell pixels; evaluate over ROI."""
    yy, xx = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]].astype(np.float32)
    A = np.column_stack([xx[shell], yy[shell], np.ones(shell.sum(), dtype=np.float32)])
    b = roi[shell]
    coefs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coefs[0] * xx + coefs[1] * yy + coefs[2]


def _integrated_gaussian_kernel(yc: float, xc: float, sigma: float, H: int, W: int) -> np.ndarray:
    """Per-pixel integrated 2D Gaussian via erf. Returns (H, W), sums to ≤1."""
    s2 = sigma * np.sqrt(2.0)
    y_edges = np.arange(H + 1, dtype=np.float32) - 0.5
    x_edges = np.arange(W + 1, dtype=np.float32) - 0.5
    cdf_y = 0.5 * (1 + erf((y_edges - yc) / s2))
    cdf_x = 0.5 * (1 + erf((x_edges - xc) / s2))
    py = np.diff(cdf_y)
    px = np.diff(cdf_x)
    return np.outer(py, px).astype(np.float32)


def _sampled_gaussian_kernel(yc: float, xc: float, sigma: float, H: int, W: int) -> np.ndarray:
    """Continuous 2D Gaussian sampled at pixel centers."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.exp(-((yy - yc) ** 2 + (xx - xc) ** 2) / (2.0 * sigma ** 2))


def _kernel(psf_type: str, yc: float, xc: float, sigma: float, H: int, W: int) -> np.ndarray:
    if psf_type == "integratedGaussian":
        return _integrated_gaussian_kernel(yc, xc, sigma, H, W)
    if psf_type == "gaussian":
        return _sampled_gaussian_kernel(yc, xc, sigma, H, W)
    raise ValueError(f"unknown psfType: {psf_type}")


def fit_spot(
    raw: np.ndarray,
    y: int,
    x: int,
    psf_sigma: float,
    fitted_region_size: float,
    bg_mode: str,
    bg_thickness: int,
    psf_type: str,
    tol: float,
    max_iterations: int,
    global_median: float,
):
    """Plane-fit local background + Gaussian-mask localization on one candidate."""
    cutwidth = int(np.ceil(fitted_region_size * psf_sigma))
    H, W = raw.shape
    y0, y1 = y - cutwidth, y + cutwidth + 1
    x0, x1 = x - cutwidth, x + cutwidth + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    roi = raw[y0:y1, x0:x1].astype(np.float32)
    Hr, Wr = roi.shape

    if bg_mode == "localPlane":
        shell = _shell_mask(roi.shape, bg_thickness)
        roi_bg = _plane_fit_bg(roi, shell)
        I = roi - roi_bg
    elif bg_mode == "localMedian":
        shell = _shell_mask(roi.shape, bg_thickness)
        I = roi - float(np.median(roi[shell]))
    elif bg_mode == "globalMedian":
        I = roi - global_median
    else:
        raise ValueError(f"unknown bgCorrectionMode: {bg_mode}")

    yc, xc = float(cutwidth), float(cutwidth)
    yy, xx = np.mgrid[0:Hr, 0:Wr].astype(np.float32)
    last_g = None
    last_N0 = 0.0
    for _ in range(max_iterations):
        g = _kernel(psf_type, yc, xc, psf_sigma, Hr, Wr)
        s_g_I = float((g * I).sum())
        s_g_g = float((g * g).sum())
        if s_g_g <= 0 or s_g_I <= 0:
            return None
        N0 = s_g_I / s_g_g
        new_yc = float((yy * g * I).sum() / s_g_I)
        new_xc = float((xx * g * I).sum() / s_g_I)
        last_g = g
        last_N0 = N0
        if max(abs(new_yc - yc), abs(new_xc - xc)) < tol:
            yc, xc = new_yc, new_xc
            break
        yc, xc = new_yc, new_xc
        if not (0.0 <= yc < Hr and 0.0 <= xc < Wr):
            return None

    if last_g is None or last_N0 <= 0:
        return None

    if psf_type == "integratedGaussian":
        intensity = 4.0 * last_N0
    else:
        intensity = float(last_N0 * last_g.sum())
    residual = float(np.sqrt(((last_N0 * last_g - I) ** 2).sum()))

    return {
        "y": y0 + yc,
        "x": x0 + xc,
        "sigma": float(psf_sigma),
        "intensity": float(intensity),
        "residual": residual,
    }


def dedupe(spots_df: pd.DataFrame, min_dist: int) -> pd.DataFrame:
    """Greedy dedup: keep highest-intensity spot first; drop any later spot
    within Chebyshev distance < min_dist of an already-kept one."""
    if spots_df.empty:
        return spots_df
    df = spots_df.sort_values("intensity", ascending=False).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    ys = df["y"].values
    xs = df["x"].values
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            if abs(ys[j] - ys[i]) < min_dist and abs(xs[j] - xs[i]) < min_dist:
                keep[j] = False
    return df[keep].reset_index(drop=True)


# ── PSF measurement ─────────────────────────────────────────────────────────


def _gaussian_1d(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def fit_gaussian_1d(profile: np.ndarray, min_sigma: float = 0.5) -> float:
    """Fit a 1D Gaussian + offset to a profile; return σ. Raises on bad input."""
    y = np.asarray(profile, dtype=float)
    if y.size < 5 or not np.isfinite(y).all():
        raise ValueError("profile too small or non-finite")
    x = np.arange(y.size, dtype=float)
    y_min, y_max = float(y.min()), float(y.max())
    p0 = [max(1e-12, y_max - y_min), float(np.argmax(y)), max(min_sigma, y.size / 8.0), y_min]
    bounds = ([0.0, 0.0, min_sigma, -np.inf], [np.inf, float(y.size - 1), np.inf, np.inf])
    popt, _ = curve_fit(_gaussian_1d, x, y, p0=p0, bounds=bounds, maxfev=20000)
    return float(popt[2])


def measure_psf_at_point(img: np.ndarray, y: int, x: int, half_window: int = 8) -> tuple[float, float]:
    """Fit 1D Gaussians to the row and column through (y, x). Returns (σ_y, σ_x)."""
    H, W = img.shape
    y0, y1 = max(0, y - half_window), min(H, y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(W, x + half_window + 1)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        raise ValueError("empty ROI")
    cy_local = int(np.clip(y - y0, 0, roi.shape[0] - 1))
    cx_local = int(np.clip(x - x0, 0, roi.shape[1] - 1))
    sigma_x = fit_gaussian_1d(roi[cy_local, :])
    sigma_y = fit_gaussian_1d(roi[:, cx_local])
    return sigma_y, sigma_x


def auto_measure_psf(
    img: np.ndarray,
    params: dict,
    n_spots: int = 20,
    half_window: int = 8,
    edge_margin: int = 10,
) -> tuple[float, list[tuple[float, float]]]:
    """Run candidate detection, fit top-N well-separated bright spots, return (median σ, per-spot σ)."""
    src = apply_flatfield(img, params.get("flatFieldSigma", 0.0))
    sigma_small = params["filterHi"]
    sigma_large = params["filterLo"] * params["psfSigma"]
    smooth = dog_filter(src, sigma_small, sigma_large)
    threshold = compute_threshold(src, smooth, params["threshUnits"], params["threshLevel"])
    sep = max(params["minDistBetweenSpots"], int(round(2 * half_window)))
    cands = find_candidates(smooth, img, threshold, sep, max_spots=n_spots * 5)

    H, W = img.shape
    sigmas: list[tuple[float, float]] = []
    for cy, cx, _ in cands:
        if cy < edge_margin or cy >= H - edge_margin:
            continue
        if cx < edge_margin or cx >= W - edge_margin:
            continue
        try:
            sy, sx = measure_psf_at_point(img, int(round(cy)), int(round(cx)), half_window)
        except Exception:
            continue
        if not (0.5 <= sx <= half_window) or not (0.5 <= sy <= half_window):
            continue
        sigmas.append((sy, sx))
        if len(sigmas) >= n_spots:
            break

    if not sigmas:
        raise RuntimeError("auto-PSF: no usable spots; lower threshold or check candidates")
    arr = np.asarray(sigmas, dtype=float)
    sigma_med = float(np.median(0.5 * (arr[:, 0] + arr[:, 1])))
    return sigma_med, sigmas


# ── pipeline ────────────────────────────────────────────────────────────────


def detect_frame(img: np.ndarray, params: dict) -> pd.DataFrame:
    """Run the full AIRLOCALIZE pipeline on a single 2D frame."""
    raw = img.astype(np.float32)
    src = apply_flatfield(raw, params.get("flatFieldSigma", 0.0))
    sigma_small = params["filterHi"]
    sigma_large = params["filterLo"] * params["psfSigma"]
    smooth = dog_filter(src, sigma_small, sigma_large)
    threshold = compute_threshold(src, smooth, params["threshUnits"], params["threshLevel"])
    cands = find_candidates(
        smooth, raw, threshold, params["minDistBetweenSpots"], params["maxSpots"]
    )
    if len(cands) == 0:
        return pd.DataFrame(columns=["y", "x", "sigma", "intensity", "residual"])

    global_median = float(np.median(raw))
    rows = []
    for cy, cx, _ in cands:
        spot = fit_spot(
            raw,
            int(round(cy)),
            int(round(cx)),
            psf_sigma=params["psfSigma"],
            fitted_region_size=params["fittedRegionSize"],
            bg_mode=params["bgCorrectionMode"],
            bg_thickness=params["bgRegionThickness"],
            psf_type=params["psfType"],
            tol=params["tol"],
            max_iterations=params["maxIterations"],
            global_median=global_median,
        )
        if spot is not None:
            rows.append(spot)
    if not rows:
        return pd.DataFrame(columns=["y", "x", "sigma", "intensity", "residual"])
    df = pd.DataFrame(rows)
    df = dedupe(df, params["minDistBetweenSpots"])
    return df


def detect_stack(stack: np.ndarray, params: dict) -> pd.DataFrame:
    """Run detect_frame across a (T, H, W) or (H, W) array. Adds a 'frame' column."""
    if stack.ndim == 2:
        df = detect_frame(stack, params)
        df["frame"] = 0
        return df
    if stack.ndim != 3:
        raise ValueError(f"expected 2D or 3D stack, got shape {stack.shape}")
    parts = []
    for i in range(stack.shape[0]):
        df_i = detect_frame(stack[i], params)
        df_i["frame"] = i
        parts.append(df_i)
    return pd.concat(parts, ignore_index=True)


def compute_smooth(img: np.ndarray, params: dict) -> np.ndarray:
    """Same pre-threshold DoG that detect_frame uses; exposed for live preview."""
    raw = img.astype(np.float32)
    src = apply_flatfield(raw, params.get("flatFieldSigma", 0.0))
    return dog_filter(src, params["filterHi"], params["filterLo"] * params["psfSigma"])


# ── params I/O ──────────────────────────────────────────────────────────────


def load_params(path) -> dict:
    """Load a params JSON, fill in any missing keys from DEFAULT_PARAMS, ignore extras."""
    with open(path) as f:
        payload = json.load(f)
    params = dict(DEFAULT_PARAMS)
    for k in PARAM_KEYS:
        if k in payload:
            params[k] = payload[k]
    return params


def save_params(path, params: dict, extra: dict | None = None) -> None:
    """Write params JSON. `extra` (e.g. psfSamples) is merged in alongside."""
    payload = {k: params[k] for k in PARAM_KEYS if k in params}
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2))
