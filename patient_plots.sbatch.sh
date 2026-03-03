#!/bin/bash
#SBATCH -J helico_patient_plots
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/deeplearning
#SBATCH -t 0-00:30
#SBATCH -p dcca40
#SBATCH --mem 8000
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

echo "[start] $(date) host=$(hostname)"
cd /hhome/ricse04/deeplearning
source .venv/bin/activate

python -u analyze_patient_level_metrics.py \
  --patient-root output/patient_level \
  --out-dir output/patient_level/plots

echo "[end] $(date)"

