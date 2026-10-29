"""
AIRLOCALIZE napari plugin — interactive tuner for the algorithm in
`core.py`. The core is GUI-free and runs the same on HPC via the
`airlocalize-batch` command.

Launch:
    napari  →  Plugins ▸ AIRLOCALIZE Tuner
    or:  airlocalize-napari [optional_tif_path ...]
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from magicgui import widgets
import napari

from .core import (
    PARAM_KEYS,
    apply_flatfield, dog_filter, compute_threshold, find_candidates,
    fit_spot, dedupe,
    fit_gaussian_1d, measure_psf_at_point, auto_measure_psf,
    detect_frame, compute_smooth,
    load_params as core_load_params, save_params as core_save_params,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── napari widget ───────────────────────────────────────────────────────────


class AIRLocalizeTuner:
    POINTS_LAYER = "AIRLOCALIZE Spots"

    PREVIEW_LAYER = "AIRLOCALIZE Threshold Preview"

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self._features_df: pd.DataFrame | None = None
        self._source_path: str | None = None
        self._psf_samples: list[tuple[float, float]] = []
        self._click_mode_active = False
        self._smooth_cache: dict[int, np.ndarray] = {}
        self._smooth_cache_key: tuple | None = None
        self._suppress_callbacks = False

        # Image
        self.image_selector = widgets.ComboBox(
            label="Image Layer", choices=self._image_choices
        )
        viewer.layers.events.inserted.connect(lambda _: self._refresh_image_choices())
        viewer.layers.events.removed.connect(lambda _: self._refresh_image_choices())

        # Detection
        self.psf_sigma = widgets.FloatSpinBox(
            label="psfSigma (px)", value=1.3, min=0.3, max=10.0, step=0.1,
            tooltip="Estimated PSF σ in pixels (xy)",
        )
        self.thresh_units = widgets.ComboBox(
            label="threshUnits", choices=["absolute", "SD", "legacySD"], value="SD",
        )
        self.thresh_level = widgets.FloatSpinBox(
            label="threshLevel", value=1.8, min=-1e6, max=1e9, step=0.1,
            tooltip="Threshold in chosen units (n×σ for SD/legacySD, raw value for absolute)",
        )
        self.max_spots = widgets.SpinBox(
            label="maxSpots", value=20000, min=1, max=10_000_000, step=100,
        )
        self.min_dist = widgets.SpinBox(
            label="minDistBetweenSpots (px)", value=2, min=1, max=50, step=1,
            tooltip="Min Chebyshev distance between spots; controls close-pair resolution",
        )

        # Filter
        self.filter_hi = widgets.FloatSpinBox(
            label="filterHi (small σ, px)", value=1.0, min=0.1, max=10.0, step=0.1,
        )
        self.filter_lo = widgets.FloatSpinBox(
            label="filterLo (× psfSigma)", value=2.15, min=1.05, max=10.0, step=0.05,
        )

        # ROI / fit
        self.fitted_region_size = widgets.FloatSpinBox(
            label="fittedRegionSize (× σ)", value=3.0, min=1.0, max=10.0, step=0.5,
        )
        self.bg_mode = widgets.ComboBox(
            label="bgCorrectionMode",
            choices=["localPlane", "localMedian", "globalMedian"], value="localPlane",
        )
        self.bg_thickness = widgets.SpinBox(
            label="bgRegionThickness (px)", value=1, min=1, max=10, step=1,
        )
        self.psf_type = widgets.ComboBox(
            label="psfType", choices=["integratedGaussian", "gaussian"], value="integratedGaussian",
        )
        self.tol = widgets.FloatSpinBox(
            label="tol (px)", value=0.01, min=1e-4, max=1.0, step=0.01,
        )
        self.max_iter = widgets.SpinBox(
            label="maxIterations", value=100, min=1, max=10000, step=10,
        )

        # Pre-bandpass flat-field
        self.flatfield_enable = widgets.CheckBox(
            label="flat-field BG subtract", value=False,
            tooltip="Subtract a heavily blurred copy of the image before DoG. Helpful on uneven/noisy data.",
        )
        self.flatfield_sigma = widgets.FloatSpinBox(
            label="flatFieldSigma (px)", value=5.0, min=0.5, max=100.0, step=0.5,
        )

        # PSF measurement
        self.psf_click_mode = widgets.CheckBox(
            label="PSF click mode (click bright spots)", value=False,
            tooltip="When on, left-click on a spot to fit X/Y Gaussians and add σ sample.",
        )
        self.psf_click_mode.changed.connect(self._on_click_mode_toggle)
        self.psf_half_window = widgets.SpinBox(
            label="PSF fit half-window (px)", value=8, min=3, max=30, step=1,
        )
        self.psf_auto_btn = widgets.PushButton(text="Auto-Measure PSF (top spots)")
        self.psf_auto_btn.clicked.connect(self._auto_measure_psf)
        self.psf_status = widgets.Label(value="PSF samples: 0")
        self.psf_apply_btn = widgets.PushButton(text="Apply mean σ → psfSigma")
        self.psf_apply_btn.clicked.connect(self._apply_psf_mean)
        self.psf_reset_btn = widgets.PushButton(text="Reset PSF samples")
        self.psf_reset_btn.clicked.connect(self._reset_psf_samples)

        # Threshold preview
        self.preview_enable = widgets.CheckBox(
            label="live threshold preview", value=False,
            tooltip="Show red overlay of pixels above DoG threshold on the current frame.",
        )
        self.preview_enable.changed.connect(self._on_preview_toggle)
        self.cdf_btn = widgets.PushButton(text="Show DoG CDF")
        self.cdf_btn.clicked.connect(self._show_cdf)

        # Live-preview triggers (debounce-ish: only update if preview is on)
        for w in (self.psf_sigma, self.thresh_units, self.thresh_level,
                  self.filter_hi, self.filter_lo,
                  self.flatfield_enable, self.flatfield_sigma):
            w.changed.connect(lambda _=None: self._maybe_refresh_preview())

        # Param save/load
        self.params_path = widgets.FileEdit(
            label="params file", mode="w",
            filter="*.json",
            tooltip="Path used by Save/Load Params buttons",
        )
        self.save_params_btn = widgets.PushButton(text="Save Params (JSON)")
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn = widgets.PushButton(text="Load Params (JSON)")
        self.load_params_btn.clicked.connect(self._load_params)

        # Buttons
        self.detect_current_btn = widgets.PushButton(text="Detect (Current Frame)")
        self.detect_current_btn.clicked.connect(self._detect_current_frame)
        self.detect_all_btn = widgets.PushButton(text="Detect (All Frames)")
        self.detect_all_btn.clicked.connect(self._detect_all_frames)
        self.hist_btn = widgets.PushButton(text="Show Mass Histogram")
        self.hist_btn.clicked.connect(self._show_mass_histogram)
        self.export_btn = widgets.PushButton(text="Export CSV")
        self.export_btn.clicked.connect(self._export_csv)

        self.status = widgets.Label(value="Ready")

        self.container = widgets.Container(
            widgets=[
                widgets.Label(value="── Image ──"),
                self.image_selector,
                widgets.Label(value="── Detection ──"),
                self.psf_sigma,
                self.thresh_units,
                self.thresh_level,
                self.max_spots,
                self.min_dist,
                widgets.Label(value="── DoG filter ──"),
                self.filter_hi,
                self.filter_lo,
                widgets.Label(value="── Pre-bandpass ──"),
                self.flatfield_enable,
                self.flatfield_sigma,
                widgets.Label(value="── PSF measurement ──"),
                self.psf_click_mode,
                self.psf_half_window,
                self.psf_auto_btn,
                self.psf_status,
                self.psf_apply_btn,
                self.psf_reset_btn,
                widgets.Label(value="── Threshold preview ──"),
                self.preview_enable,
                self.cdf_btn,
                widgets.Label(value="── ROI / fit ──"),
                self.fitted_region_size,
                self.bg_mode,
                self.bg_thickness,
                self.psf_type,
                self.tol,
                self.max_iter,
                widgets.Label(value="── Params ──"),
                self.params_path,
                self.save_params_btn,
                self.load_params_btn,
                widgets.Label(value="── Run ──"),
                self.detect_current_btn,
                self.detect_all_btn,
                self.hist_btn,
                self.export_btn,
                widgets.Label(value="── Status ──"),
                self.status,
            ]
        )

        # mouse handler for click-mode PSF measurement
        self.viewer.mouse_drag_callbacks.append(self._on_viewer_click)
        self.viewer.dims.events.current_step.connect(lambda _=None: self._maybe_refresh_preview())

    # ── helpers ─────────────────────────────────────────────────────────────

    def _image_choices(self, combo=None):
        return [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]

    def _refresh_image_choices(self):
        self.image_selector.choices = self._image_choices()

    def _get_image_data(self):
        name = self.image_selector.value
        if name is None:
            raise ValueError("No image layer selected")
        return self.viewer.layers[name].data

    def _params(self) -> dict:
        return dict(
            psfSigma=self.psf_sigma.value,
            threshUnits=self.thresh_units.value,
            threshLevel=self.thresh_level.value,
            maxSpots=self.max_spots.value,
            minDistBetweenSpots=self.min_dist.value,
            filterHi=self.filter_hi.value,
            filterLo=self.filter_lo.value,
            flatFieldSigma=(self.flatfield_sigma.value if self.flatfield_enable.value else 0.0),
            fittedRegionSize=self.fitted_region_size.value,
            bgCorrectionMode=self.bg_mode.value,
            bgRegionThickness=self.bg_thickness.value,
            psfType=self.psf_type.value,
            tol=self.tol.value,
            maxIterations=self.max_iter.value,
        )

    # ── current frame helpers ──────────────────────────────────────────────

    def _current_frame_index(self, data) -> int:
        if data.ndim < 3:
            return 0
        return int(self.viewer.dims.current_step[0])

    def _current_frame_image(self) -> tuple[np.ndarray, int]:
        data = self._get_image_data()
        idx = self._current_frame_index(data)
        img = data[idx] if data.ndim >= 3 else data
        return np.asarray(img), idx

    # ── PSF measurement ────────────────────────────────────────────────────

    def _on_click_mode_toggle(self, value):
        self._click_mode_active = bool(value)
        self.status.value = (
            "Click bright, isolated spots to record σ samples."
            if self._click_mode_active else "PSF click mode off."
        )

    def _on_viewer_click(self, viewer, event):
        if not self._click_mode_active:
            return
        if event.type != "mouse_press" or event.button != 1:
            return
        try:
            data = self._get_image_data()
            pos = event.position
            if pos is None:
                return
            if data.ndim >= 3:
                _, y, x = (int(round(p)) for p in pos[-3:])
            else:
                y, x = (int(round(p)) for p in pos[-2:])
            img, _ = self._current_frame_image()
            sy, sx = measure_psf_at_point(img, y, x, half_window=int(self.psf_half_window.value))
            if not (0.4 <= sx <= self.psf_half_window.value) or not (0.4 <= sy <= self.psf_half_window.value):
                self.status.value = f"Rejected fit (σx={sx:.2f}, σy={sy:.2f})"
                return
            self._psf_samples.append((sy, sx))
            self._refresh_psf_status()
        except Exception as e:
            self.status.value = f"PSF fit failed: {e}"

    def _auto_measure_psf(self):
        try:
            img, _ = self._current_frame_image()
            sigma_med, samples = auto_measure_psf(
                img,
                self._params(),
                n_spots=20,
                half_window=int(self.psf_half_window.value),
            )
            self._psf_samples.extend(samples)
            self._refresh_psf_status()
            self.status.value = f"Auto-measured PSF: median σ = {sigma_med:.3f} from {len(samples)} spots"
        except Exception as e:
            self.status.value = f"Auto-PSF failed: {e}"

    def _refresh_psf_status(self):
        n = len(self._psf_samples)
        if n == 0:
            self.psf_status.value = "PSF samples: 0"
            return
        arr = np.asarray(self._psf_samples, dtype=float)
        mean_sigma = float(np.mean(0.5 * (arr[:, 0] + arr[:, 1])))
        self.psf_status.value = (
            f"PSF samples: {n} | mean σ = {mean_sigma:.3f}  "
            f"(σ_y={arr[:, 0].mean():.3f}, σ_x={arr[:, 1].mean():.3f})"
        )

    def _apply_psf_mean(self):
        if not self._psf_samples:
            self.status.value = "No PSF samples — click spots or run Auto-Measure first."
            return
        arr = np.asarray(self._psf_samples, dtype=float)
        mean_sigma = float(np.mean(0.5 * (arr[:, 0] + arr[:, 1])))
        self.psf_sigma.value = round(mean_sigma, 3)
        self.status.value = f"psfSigma ← {mean_sigma:.3f} (n={len(self._psf_samples)})"

    def _reset_psf_samples(self):
        self._psf_samples.clear()
        self._refresh_psf_status()
        self.status.value = "PSF samples cleared."

    # ── threshold preview ─────────────────────────────────────────────────

    def _smooth_for_current_frame(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (raw_frame, smooth, frame_idx). Caches per-frame smooth."""
        img, idx = self._current_frame_image()
        params = self._params()
        cache_key = (
            id(self._get_image_data()),
            params["psfSigma"], params["filterHi"], params["filterLo"],
            params["flatFieldSigma"],
        )
        if self._smooth_cache_key != cache_key:
            self._smooth_cache.clear()
            self._smooth_cache_key = cache_key
        if idx not in self._smooth_cache:
            self._smooth_cache[idx] = compute_smooth(img, params)
        return img, self._smooth_cache[idx], idx

    def _on_preview_toggle(self, value):
        if value:
            self._refresh_preview()
        else:
            if self.PREVIEW_LAYER in self.viewer.layers:
                self.viewer.layers.remove(self.PREVIEW_LAYER)

    def _maybe_refresh_preview(self):
        if self._suppress_callbacks:
            return
        if self.preview_enable.value:
            self._refresh_preview()

    def _refresh_preview(self):
        try:
            img, smooth, _ = self._smooth_for_current_frame()
            params = self._params()
            threshold = compute_threshold(img, smooth, params["threshUnits"], params["threshLevel"])
            mask = (smooth > threshold).astype(np.uint8)
            if self.PREVIEW_LAYER in self.viewer.layers:
                layer = self.viewer.layers[self.PREVIEW_LAYER]
                layer.data = mask
                layer.contrast_limits = (0, 1)
            else:
                self.viewer.add_image(
                    mask, name=self.PREVIEW_LAYER, colormap="red",
                    opacity=0.45, blending="additive", contrast_limits=(0, 1),
                )
            n_pix = int(mask.sum())
            self.status.value = f"Preview: threshold={threshold:.4g}, {n_pix} pixels above"
        except Exception as e:
            self.status.value = f"Preview failed: {e}"

    def _show_cdf(self):
        try:
            import matplotlib.pyplot as plt
            img, smooth, idx = self._smooth_for_current_frame()
            params = self._params()
            threshold = compute_threshold(img, smooth, params["threshUnits"], params["threshLevel"])
            flat = smooth.ravel()
            flat = flat[np.isfinite(flat)]
            sorted_vals = np.sort(flat)
            frac_ge = 1.0 - np.arange(sorted_vals.size) / sorted_vals.size
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sorted_vals, frac_ge, lw=1.2)
            ax.axvline(threshold, color="red", linestyle="--", lw=1.2, label=f"threshold={threshold:.4g}")
            ax.set_xscale("symlog", linthresh=max(1e-6, np.nanpercentile(np.abs(flat), 5)))
            ax.set_xlabel("DoG value")
            ax.set_ylabel("fraction ≥ threshold")
            ax.set_title(f"DoG CDF — frame {idx}")
            ax.legend()
            fig.tight_layout()
            fig.show()
        except Exception as e:
            self.status.value = f"CDF failed: {e}"

    # ── params save/load ──────────────────────────────────────────────────

    def _params_file_path(self) -> Path:
        path = self.params_path.value
        if path is None or str(path) == ".":
            base = self._source_path or self.image_selector.value or "airlocalize_params"
            stem = Path(str(base)).stem if base else "airlocalize_params"
            return Path(stem + "_params.json")
        return Path(str(path))

    def _save_params(self):
        try:
            path = self._params_file_path()
            extra = {
                "psfSamples": self._psf_samples,
                "flatFieldEnabled": bool(self.flatfield_enable.value),
            }
            core_save_params(path, self._params(), extra=extra)
            self.params_path.value = path
            self.status.value = f"Saved params → {path}"
        except Exception as e:
            self.status.value = f"Save failed: {e}"

    def _load_params(self):
        try:
            path = self._params_file_path()
            params = core_load_params(path)
            with open(path) as f:
                payload = json.load(f)
            self._suppress_callbacks = True
            mapping = {
                "psfSigma": self.psf_sigma, "threshUnits": self.thresh_units,
                "threshLevel": self.thresh_level, "maxSpots": self.max_spots,
                "minDistBetweenSpots": self.min_dist,
                "filterHi": self.filter_hi, "filterLo": self.filter_lo,
                "fittedRegionSize": self.fitted_region_size,
                "bgCorrectionMode": self.bg_mode, "bgRegionThickness": self.bg_thickness,
                "psfType": self.psf_type, "tol": self.tol, "maxIterations": self.max_iter,
            }
            for key, w in mapping.items():
                if key in params:
                    w.value = params[key]
            # flat-field needs the toggle inferred from the saved sigma
            ff_sigma = float(params.get("flatFieldSigma", 0.0))
            self.flatfield_enable.value = bool(payload.get("flatFieldEnabled", ff_sigma > 0))
            self.flatfield_sigma.value = ff_sigma if ff_sigma > 0 else self.flatfield_sigma.value
            if "psfSamples" in payload:
                self._psf_samples = [tuple(s) for s in payload["psfSamples"]]
                self._refresh_psf_status()
            self._suppress_callbacks = False
            self._maybe_refresh_preview()
            self.status.value = f"Loaded params ← {path}"
        except Exception as e:
            self._suppress_callbacks = False
            self.status.value = f"Load failed: {e}"

    # ── detection ───────────────────────────────────────────────────────────

    def _detect_current_frame(self):
        try:
            data = self._get_image_data()
            frame_idx = self.viewer.dims.current_step[0] if data.ndim >= 3 else 0
            img = data[frame_idx] if data.ndim >= 3 else data
            self.status.value = f"Detecting frame {frame_idx}…"
            df = detect_frame(np.asarray(img), self._params())
            df["frame"] = frame_idx
            self._features_df = df
            self.status.value = f"Found {len(df)} spots in frame {frame_idx}"
            self._show_points(df, single_frame=True)
        except Exception as e:
            self.status.value = f"Error: {e}"
            raise

    def _detect_all_frames(self):
        try:
            data = self._get_image_data()
            if data.ndim < 3:
                self.status.value = "Need a time-series (≥3D) image"
                return
            n_frames = data.shape[0]
            self.status.value = f"Detecting across {n_frames} frames…"
            params = self._params()
            all_rows = []
            for i in range(n_frames):
                df = detect_frame(np.asarray(data[i]), params)
                df["frame"] = i
                all_rows.append(df)
            features = pd.concat(all_rows, ignore_index=True)
            self._features_df = features
            self.status.value = f"Found {len(features)} spots across {n_frames} frames"
            self._show_points(features, single_frame=False)
        except Exception as e:
            self.status.value = f"Error: {e}"
            raise

    # ── visualization ───────────────────────────────────────────────────────

    def _show_points(self, df: pd.DataFrame, single_frame: bool):
        if df.empty:
            if self.POINTS_LAYER in self.viewer.layers:
                self.viewer.layers.remove(self.POINTS_LAYER)
            return
        if single_frame:
            coords = df[["y", "x"]].values
        else:
            coords = df[["frame", "y", "x"]].values
        props = {
            "intensity": df["intensity"].values,
            "sigma": df["sigma"].values,
            "residual": df["residual"].values,
        }
        size = 2.0 * self.fitted_region_size.value * self.psf_sigma.value + 1.0
        # Recreate the layer if dimensionality changed (e.g. switched between
        # current-frame and all-frames detection).
        if self.POINTS_LAYER in self.viewer.layers:
            existing = self.viewer.layers[self.POINTS_LAYER]
            if existing.ndim != coords.shape[1]:
                self.viewer.layers.remove(self.POINTS_LAYER)
        if self.POINTS_LAYER in self.viewer.layers:
            layer = self.viewer.layers[self.POINTS_LAYER]
            layer.data = coords
            layer.properties = props
            layer.size = size
        else:
            self.viewer.add_points(
                coords,
                name=self.POINTS_LAYER,
                size=size,
                face_color="transparent",
                border_color="cyan",
                border_width=0.1,
                opacity=0.8,
                properties=props,
            )

    def _show_mass_histogram(self):
        if self._features_df is None or self._features_df.empty:
            self.status.value = "Run detection first"
            return
        try:
            import matplotlib.pyplot as plt
            df = self._features_df
            masses = df["intensity"].values
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for ax, yscale in zip(axes[0], ("linear", "log")):
                ax.hist(masses, bins=50)
                ax.set_xlabel("intensity (N0 × kernel norm)")
                ax.set_ylabel("count")
                ax.set_yscale(yscale)
                ax.set_title(f"intensity ({yscale} y)")
            for ax, col in zip(axes[1], ("x", "y")):
                frac = df[col].values % 1.0
                ax.hist(frac, bins=20, range=(0, 1))
                ax.axhline(len(frac) / 20, color="gray", linestyle=":", label="flat reference")
                ax.set_xlabel(f"{col} mod 1")
                ax.set_ylabel("count")
                ax.set_title(f"subpixel bias ({col})")
                ax.legend()
            fig.suptitle(f"AIRLOCALIZE diagnostics — {len(masses)} spots")
            fig.tight_layout()
            fig.show()
        except Exception as e:
            self.status.value = f"Histogram error: {e}"

    def _export_csv(self):
        if self._features_df is None or self._features_df.empty:
            self.status.value = "Nothing to export"
            return
        import os
        layer_path = None
        name = self.image_selector.value
        if name is not None and name in self.viewer.layers:
            layer_path = getattr(self.viewer.layers[name].source, "path", None)
        base = self._source_path or layer_path or name or "airlocalize"
        out = os.path.splitext(base)[0] + "_airlocalize.csv"
        self._features_df.to_csv(out, index=False)
        self.status.value = f"Saved → {out}"


def _wrap_in_scroll(tuner: AIRLocalizeTuner):
    # Wrap the panel in a QScrollArea so napari doesn't expand the window to fit it.
    from qtpy.QtWidgets import QScrollArea

    scroll = QScrollArea()
    scroll.setWidget(tuner.container.native)
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(360)
    scroll._airlocalize_tuner = tuner  # keep the Python object alive with the widget
    return scroll


def make_tuner_widget(viewer: napari.viewer.Viewer):
    """npe2 widget contribution: Plugins ▸ AIRLOCALIZE Tuner."""
    return _wrap_in_scroll(AIRLocalizeTuner(viewer))


def main():
    import sys
    import tifffile

    viewer = napari.Viewer()
    source_paths = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            data = tifffile.imread(path)
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            viewer.add_image(data, name=name)
            source_paths.append(path)

    tuner = AIRLocalizeTuner(viewer)
    if source_paths:
        tuner._source_path = source_paths[0]

    viewer.window.add_dock_widget(_wrap_in_scroll(tuner), name="AIRLOCALIZE Tuner", area="right")
    napari.run()


if __name__ == "__main__":
    main()
