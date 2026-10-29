import numpy as np
import pandas as pd
import tifffile

from napari_airlocalize import DEFAULT_PARAMS, save_params
from napari_airlocalize.batch import main

from conftest import render_spots


def test_batch_cli(tmp_path, grid_spots):
    stack = np.stack([render_spots((90, 90), grid_spots, 1.3, seed=s) for s in range(4)])
    tif = tmp_path / "movie.tif"
    tifffile.imwrite(tif, stack.astype(np.uint16), photometric="minisblack")
    params = tmp_path / "params.json"
    save_params(params, DEFAULT_PARAMS)

    out = tmp_path / "out"
    rc = main(["--params", str(params), "--out", str(out), "--frames", "1:3",
               "--combined", str(tmp_path / "all.csv"), "--quiet", str(tif)])
    assert rc == 0

    df = pd.read_csv(out / "movie_airlocalize.csv")
    assert list(df.columns) == ["frame", "y", "x", "sigma", "intensity", "residual"]
    assert sorted(df["frame"].unique()) == [1, 2]
    combined = pd.read_csv(tmp_path / "all.csv")
    assert (combined["file"] == "movie.tif").all()
