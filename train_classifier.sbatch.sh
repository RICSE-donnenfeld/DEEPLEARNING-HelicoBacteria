#!/bin/bash
#SBATCH -J helico_cnn
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/autoencoder
#SBATCH -t 0-02:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

nvidia-smi

cd /hhome/ricse04/autoencoder

source .venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
python model_classifier.py --epochs 150 --k-folds 5 --seed 42
