import numpy as np
import pytest

from napari_airlocalize import DEFAULT_PARAMS, auto_measure_psf, detect_frame, detect_stack
from napari_airlocalize.core import dedupe, load_params, save_params

from conftest import render_spots

SIGMA = 1.3


def _match(found, truth):
    """For each true spot, distance to the nearest detection."""
    d = np.hypot(found[:, None, 0] - truth[None, :, 0], found[:, None, 1] - truth[None, :, 1])
    return d.min(axis=0)


def test_recovers_subpixel_positions(grid_spots):
    img = render_spots((90, 90), grid_spots, SIGMA)
    df = detect_frame(img, dict(DEFAULT_PARAMS, psfSigma=SIGMA))
    assert len(df) == len(grid_spots)
    err = _match(df[["y", "x"]].to_numpy(), grid_spots)
    assert err.max() < 0.15


@pytest.mark.parametrize("bg_mode", ["localPlane", "localMedian", "globalMedian"])
@pytest.mark.parametrize("psf_type", ["integratedGaussian", "gaussian"])
def test_all_modes_run(grid_spots, bg_mode, psf_type):
    img = render_spots((90, 90), grid_spots, SIGMA)
    params = dict(DEFAULT_PARAMS, psfSigma=SIGMA, bgCorrectionMode=bg_mode, psfType=psf_type)
    df = detect_frame(img, params)
    assert len(df) == len(grid_spots)
    assert _match(df[["y", "x"]].to_numpy(), grid_spots).max() < 0.3


def test_blank_frame_gives_no_spots():
    img = render_spots((64, 64), [], SIGMA)
    df = detect_frame(img, dict(DEFAULT_PARAMS, threshUnits="absolute", threshLevel=50.0))
    assert df.empty
    assert list(df.columns) == ["y", "x", "sigma", "intensity", "residual"]


def test_detect_stack_adds_frame_column(grid_spots):
    stack = np.stack([render_spots((90, 90), grid_spots, SIGMA, seed=s) for s in range(3)])
    df = detect_stack(stack, dict(DEFAULT_PARAMS, psfSigma=SIGMA))
    assert sorted(df["frame"].unique()) == [0, 1, 2]
    assert (df.groupby("frame").size() == len(grid_spots)).all()


def test_auto_psf_recovers_sigma(grid_spots):
    img = render_spots((90, 90), grid_spots, 1.6, noise_sd=1.0)
    sigma, samples = auto_measure_psf(img, dict(DEFAULT_PARAMS, psfSigma=1.6), half_window=6)
    assert len(samples) > 5
    assert sigma == pytest.approx(1.6, abs=0.15)


def test_dedupe_keeps_brightest():
    import pandas as pd

    df = pd.DataFrame({"y": [10.0, 10.5, 30.0], "x": [10.0, 10.5, 30.0],
                       "intensity": [5.0, 9.0, 1.0], "sigma": 1.0, "residual": 0.0})
    out = dedupe(df, min_dist=2)
    assert len(out) == 2
    assert out["intensity"].max() == 9.0


def test_params_roundtrip(tmp_path):
    params = dict(DEFAULT_PARAMS, psfSigma=2.2, threshLevel=4.0)
    path = tmp_path / "p.json"
    save_params(path, params, extra={"psfSamples": [[1.0, 1.1]]})
    loaded = load_params(path)
    assert loaded == params
