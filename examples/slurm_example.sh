#!/bin/bash
# AIRLOCALIZE batch — SLURM array job example.
#
# Submits one task per input TIFF. Each task processes one file end-to-end
# (no intra-file parallelism), which keeps the job graph simple and lets
# SLURM handle scheduling.
#
# Usage:
#   1. Put all your TIFFs in $INPUT_DIR.
#   2. Edit ACCOUNT, PARTITION, INPUT_DIR, OUTPUT_DIR, PARAMS, ENV.
#   3. sbatch slurm_example.sh
#
# This script assumes a `find` over INPUT_DIR returns the TIFF list in a
# stable order. Each array task picks the file at index $SLURM_ARRAY_TASK_ID.

#SBATCH --job-name=airlocalize
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/airlocalize_%A_%a.out
#SBATCH --error=logs/airlocalize_%A_%a.err
# --- size the array to match the number of input files ---
# (set N_FILES = number of TIFFs and use:  --array=0-(N_FILES-1)%MAX_PARALLEL)
#SBATCH --array=0-99%20

set -euo pipefail

# ── EDIT ME ────────────────────────────────────────────────────────────────
INPUT_DIR="/path/to/tifs"
OUTPUT_DIR="/path/to/results"
PARAMS="/path/to/params.json"
ENV_ACTIVATE="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate airlocalize-env"  # any env with napari-airlocalize installed
TIF_GLOB="*.tif"                            # or "*.ome.tif" etc.
# ───────────────────────────────────────────────────────────────────────────

mkdir -p logs "$OUTPUT_DIR"

eval "$ENV_ACTIVATE"

# Build a deterministic list of inputs and pick this task's file.
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "$TIF_GLOB" | sort)
N_FILES=${#FILES[@]}
if (( SLURM_ARRAY_TASK_ID >= N_FILES )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= file count ${N_FILES}; nothing to do."
    exit 0
fi
INPUT="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Host:        $(hostname)"
echo "Task:        $SLURM_ARRAY_TASK_ID / $N_FILES"
echo "Input:       $INPUT"
echo "Output dir:  $OUTPUT_DIR"
echo "Params:      $PARAMS"

airlocalize-batch \
    --params "$PARAMS" \
    --out    "$OUTPUT_DIR" \
    "$INPUT"

echo "Done."
