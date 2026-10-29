"""Widget tests against a canvas-free napari ViewerModel (no OpenGL needed)."""

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from napari.components import ViewerModel

from conftest import render_spots


@pytest.fixture
def tuner(grid_spots):
    from napari_airlocalize._widget import AIRLocalizeTuner, _wrap_in_scroll

    viewer = ViewerModel()
    stack = np.stack([render_spots((90, 90), grid_spots, 1.3, seed=s) for s in range(3)])
    viewer.add_image(stack, name="demo")
    t = AIRLocalizeTuner(viewer)
    t._scroll = _wrap_in_scroll(t)
    t.image_selector.value = "demo"
    return t


def test_manifest_discovered():
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()
    mf = pm.get_manifest("napari-airlocalize")
    assert [w.display_name for w in mf.contributions.widgets] == ["AIRLOCALIZE Tuner"]


def test_detect_current_and_all(tuner, grid_spots):
    tuner._detect_current_frame()
    frame = tuner.viewer.dims.current_step[0]
    assert tuner.status.value == f"Found {len(grid_spots)} spots in frame {frame}"
    tuner._detect_all_frames()
    pts = tuner.viewer.layers[tuner.POINTS_LAYER]
    assert pts.data.shape == (3 * len(grid_spots), 3)


def test_preview_and_auto_psf(tuner):
    tuner.preview_enable.value = True
    assert tuner.PREVIEW_LAYER in tuner.viewer.layers
    tuner._auto_measure_psf()
    assert tuner.status.value.startswith("Auto-measured PSF")
    tuner._apply_psf_mean()
    assert 1.0 < tuner.psf_sigma.value < 1.6


def test_params_save_load(tuner, tmp_path):
    path = tmp_path / "p.json"
    tuner.params_path.value = path
    tuner.thresh_level.value = 3.3
    tuner._save_params()
    tuner.thresh_level.value = 1.0
    tuner._load_params()
    assert tuner.thresh_level.value == pytest.approx(3.3)


def test_export_csv(tuner, tmp_path):
    tuner._source_path = str(tmp_path / "movie.tif")
    tuner._detect_all_frames()
    tuner._export_csv()
    assert (tmp_path / "movie_airlocalize.csv").exists()
