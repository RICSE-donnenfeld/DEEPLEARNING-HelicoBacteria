# Helico Patch Training

Minimal commands to train locally and run training on the cluster.

## 1) Local training

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset location expected by scripts

The code expects this folder in the project root:

`HelicoDataSet/`

and uses:

- `HelicoDataSet/CoordAnnotatedAllPatches_stripped.csv` (or falls back to `CoordAnnotatedAllPatches.csv`)
- `HelicoDataSet/CrossValidation/Annotated/...` images

### Train CNN classifier

```bash
python model.py --epochs 150
```

Saved checkpoint: `cnn_model.pth`

### Train autoencoder

```bash
python model_autoencoder.py --epochs 150
```

Saved checkpoint: `autoencoder_model.pth`  
Saved threshold file: `autoencoder_threshold.txt`

## 2) Run on cluster (Slurm)

Two job scripts are provided:

- `train.sbatch.sh` (CNN)
- `train_autoencoder.sbatch.sh` (autoencoder)

### Submit jobs

```bash
sbatch train.sbatch.sh
sbatch train_autoencoder.sbatch.sh
```

### Monitor and logs

```bash
squeue -u $USER
```

Logs are written to:

- `slurm_io/%x_%u_%j.out`
- `slurm_io/%x_%u_%j.err`

### Important

The sbatch scripts currently use:

- working directory: `/hhome/ricse04/autoencoder`
- virtualenv activation: `source .venv/bin/activate`
