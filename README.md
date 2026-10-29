# napari-airlocalize

[![tests](https://github.com/lancetdenes/napari-airlocalize/actions/workflows/tests.yml/badge.svg)](https://github.com/lancetdenes/napari-airlocalize/actions/workflows/tests.yml)

A Python port of [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE)
(Lionnet et al., 2011), a sub-pixel spot detection algorithm for fluorescence
microscopy. It comes in three parts:

- **`napari_airlocalize.core`** is the algorithm itself. It is pure
  numpy/scipy/pandas with no GUI dependencies.
- **AIRLOCALIZE Tuner** is a napari dock widget for picking parameters
  interactively on your own images. It includes live threshold preview, PSF
  measurement and diagnostics.
- **`airlocalize-batch`** is a headless command-line tool. It applies a saved
  parameter file to many TIFF stacks, for example on an HPC cluster.

The usual workflow is to tune parameters on a few movies in napari, save them
to JSON, and then run `airlocalize-batch` on everything.

## Installation

```bash
pip install "napari-airlocalize[napari] @ git+https://github.com/lancetdenes/napari-airlocalize"
```

For batch-only use, such as a cluster node with no display, leave out the
`[napari]` extra:

```bash
pip install "git+https://github.com/lancetdenes/napari-airlocalize"
```

## Usage

### Interactive tuning in napari

Open napari, load a 2D image or a (T, Y, X) stack, and choose
**Plugins ▸ AIRLOCALIZE Tuner**. You can also launch it directly:

```bash
airlocalize-napari movie.tif
```

1. Choose the image layer.
2. Measure the PSF. Click **Auto-Measure PSF**, or turn on *PSF click mode*
   and click a few bright, isolated spots. Then click **Apply mean σ →
   psfSigma**.
3. Turn on **live threshold preview**. Pixels above the threshold show as a
   red overlay. Adjust `threshLevel` until the overlay covers the real spots
   and leaves the background clear.
4. Click **Detect (Current Frame)** to check the result, then
   **Detect (All Frames)**.
5. Click **Save Params (JSON)** to keep the settings, and **Export CSV** to
   save the spot table.

### Batch processing

```bash
airlocalize-batch --params params.json --out results/ movie1.tif movie2.tif
```

| Option | Meaning |
|---|---|
| `--frames START:END` | process only frames `[START, END)` |
| `--combined all.csv` | also write one CSV with a `file` column |
| `--suffix STR` | output suffix (default `_airlocalize.csv`) |
| `--quiet` | suppress progress output |

[`examples/slurm_example.sh`](examples/slurm_example.sh) is a SLURM array-job
template that runs one task per TIFF.

### From Python

```python
import tifffile
from napari_airlocalize import DEFAULT_PARAMS, detect_stack, load_params

stack = tifffile.imread("movie.tif")              # (T, Y, X) or (Y, X)
params = load_params("params.json")               # or dict(DEFAULT_PARAMS, psfSigma=1.5)
spots = detect_stack(stack, params)               # pandas DataFrame
```

### Output

Each detected spot is one row:

| column | description |
|---|---|
| `frame` | frame index |
| `y`, `x` | sub-pixel position (pixels; integer = pixel centre) |
| `sigma` | PSF σ used for the fit (fixed, = `psfSigma`) |
| `intensity` | integrated spot intensity above local background |
| `residual` | RMS of the fit residual in the fit ROI |

## Algorithm

The algorithm runs on each 2D frame:

1. *(optional)* Flat-field the image by subtracting a Gaussian blur with
   σ = `flatFieldSigma`. This copy is used only for candidate detection.
2. Apply a difference-of-Gaussians band-pass with σ = `filterHi` and
   σ = `filterLo × psfSigma`, then subtract the mean.
3. Find candidates: local maxima of the filtered image that lie above the
   threshold, ranked by raw intensity, keeping at most `maxSpots`.
4. Fit each candidate in a (2·⌈`fittedRegionSize`·σ⌉+1)² region of the raw
   image. The local background is subtracted (a plane fit to the border
   pixels by default). The spot centre is then found by iterative
   Gaussian-mask localization (Thompson et al., 2002) with σ held fixed at
   `psfSigma`.
5. Deduplicate: when fitted spots are closer than `minDistBetweenSpots`,
   only the brightest one is kept.

### Parameters

| parameter | default | meaning |
|---|---|---|
| `psfSigma` | 1.3 | PSF σ in pixels |
| `threshUnits` | `SD` | `SD`: × raw-image SD; `legacySD`: × filtered-image SD; `absolute`: raw value on the filtered image |
| `threshLevel` | 1.8 | threshold in `threshUnits` |
| `maxSpots` | 20000 | maximum candidates per frame |
| `minDistBetweenSpots` | 2 | minimum spot separation (px, Chebyshev) |
| `filterHi` | 1.0 | small DoG σ (px) |
| `filterLo` | 2.15 | large DoG σ, as a multiple of `psfSigma` |
| `flatFieldSigma` | 0 (off) | pre-filter background subtraction σ (px) |
| `fittedRegionSize` | 3.0 | fit ROI half-width, as a multiple of `psfSigma` |
| `bgCorrectionMode` | `localPlane` | `localPlane`, `localMedian`, or `globalMedian` |
| `bgRegionThickness` | 1 | ROI border width used for the background (px) |
| `psfType` | `integratedGaussian` | `integratedGaussian` or `gaussian` |
| `tol` | 0.01 | convergence tolerance (px) |
| `maxIterations` | 100 | Gaussian-mask iteration cap |

### Differences from the MATLAB original

This port covers the 2D / time-series path with the Gaussian-mask fit engine,
which is the AIRLOCALIZE default. It does not include 3D detection, the full
nonlinear least-squares fit, or ROI-mask adaptive thresholding. It adds an
optional flat-field pre-filter and the interactive napari tools.

## Citation

If you use this package, please cite the original AIRLOCALIZE paper:

> Lionnet T, Czaplinski K, Darzacq X, Shav-Tal Y, Wells AL, Chao JA, Park HY,
> de Turris V, Lopez-Jones M, Singer RH. A transgenic mouse for in vivo
> detection of endogenous labeled mRNA. *Nature Methods* 8, 165–170 (2011).
> doi:[10.1038/nmeth.1551](https://doi.org/10.1038/nmeth.1551)

and this repository (see [`CITATION.cff`](CITATION.cff)).

## License

BSD 3-Clause. See [LICENSE](LICENSE), which also reproduces the original
AIRLOCALIZE copyright notice.
