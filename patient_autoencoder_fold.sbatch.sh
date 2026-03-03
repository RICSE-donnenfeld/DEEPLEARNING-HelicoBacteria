#!/bin/bash
#SBATCH -J helico_patient_ae
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/deeplearning
#SBATCH -t 0-03:00
#SBATCH -p dcca40
#SBATCH --mem 24000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

echo "[start] $(date) host=$(hostname)"
nvidia-smi

cd /hhome/ricse04/deeplearning
source .venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

python -u patient_level_pipeline.py \
  --model ae \
  --use-cv-fold-checkpoints \
  --patient-k-folds 5 \
  --seed 42 \
  --log-every-patients 10 \
  --out-dir output/patient_level/ae

echo "[end] $(date)"

