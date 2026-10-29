"""
AIRLOCALIZE batch CLI — headless spot detection for HPC.

Loads detection parameters from a JSON file (saved from the napari plugin)
and runs `detect_frame` over every frame of every input TIFF, writing one
CSV per input file.

Usage:
    airlocalize-batch --params params.json --out results/ file1.tif [file2.tif ...]

Optional:
    --frames START:END    Process only frames [START, END). Default: all.
    --suffix STR          Output suffix. Default: "_airlocalize.csv".
    --quiet               Suppress per-file progress lines.
    --combined PATH       Also write one combined CSV with a `file` column.

Output schema (per file):
    frame, y, x, sigma, intensity, residual

For SLURM array jobs see examples/slurm_example.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .core import detect_stack, load_params


def _parse_frame_range(spec: str | None, n_frames: int) -> tuple[int, int]:
    if not spec:
        return 0, n_frames
    if ":" not in spec:
        raise ValueError(f"--frames expects START:END, got {spec!r}")
    a, b = spec.split(":", 1)
    start = int(a) if a else 0
    end = int(b) if b else n_frames
    if not (0 <= start < end <= n_frames):
        raise ValueError(f"frame range {start}:{end} out of bounds [0, {n_frames}]")
    return start, end


def process_file(
    in_path: Path,
    out_path: Path,
    params: dict,
    frames_spec: str | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    stack = tifffile.imread(in_path)
    if stack.ndim == 2:
        stack = stack[None, ...]
    elif stack.ndim != 3:
        raise ValueError(f"{in_path}: expected 2D or 3D TIFF, got shape {stack.shape}")

    start, end = _parse_frame_range(frames_spec, stack.shape[0])
    sub = stack[start:end]
    df = detect_stack(sub, params)
    if not df.empty and start != 0:
        df["frame"] = df["frame"] + start

    cols = ["frame", "y", "x", "sigma", "intensity", "residual"]
    df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if not quiet:
        print(f"  {in_path.name} → {out_path.name}  ({len(df)} spots, frames {start}:{end})",
              flush=True)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIRLOCALIZE batch CLI (HPC-friendly)")
    p.add_argument("--params", required=True, type=Path,
                   help="JSON params file saved from the napari plugin")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--frames", default=None,
                   help="Optional frame range START:END (e.g. 0:50)")
    p.add_argument("--suffix", default="_airlocalize.csv",
                   help='Output suffix appended to TIFF stem (default: "_airlocalize.csv")')
    p.add_argument("--quiet", action="store_true", help="Suppress per-file progress")
    p.add_argument("--combined", type=Path, default=None,
                   help="Also write one combined CSV with a `file` column")
    p.add_argument("inputs", nargs="+", type=Path, help="Input TIFF files")
    args = p.parse_args(argv)

    params = load_params(args.params)
    if not args.quiet:
        print(f"Loaded params from {args.params}", flush=True)
        print(f"  psfSigma={params['psfSigma']}, threshold={params['threshUnits']}/{params['threshLevel']},"
              f" flatFieldSigma={params['flatFieldSigma']}", flush=True)
        print(f"Processing {len(args.inputs)} file(s) → {args.out}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Path, Exception]] = []
    combined_parts: list[pd.DataFrame] = []
    for in_path in args.inputs:
        out_path = args.out / (in_path.stem + args.suffix)
        try:
            df = process_file(in_path, out_path, params, args.frames, args.quiet)
            if args.combined is not None:
                df = df.copy()
                df.insert(0, "file", in_path.name)
                combined_parts.append(df)
        except Exception as e:
            failures.append((in_path, e))
            print(f"FAILED {in_path}: {e}", file=sys.stderr, flush=True)

    if args.combined is not None and combined_parts:
        args.combined.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(combined_parts, ignore_index=True).to_csv(args.combined, index=False)
        if not args.quiet:
            print(f"Combined CSV → {args.combined}", flush=True)

    if failures:
        print(f"\n{len(failures)}/{len(args.inputs)} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
