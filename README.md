# Helico Patch Training + Report

Minimal runbook for model training, CV diagnostics, cluster jobs, and LaTeX report build.

## Project layout

- `model_classifier.py`, `model_autoencoder.py`: canonical training/eval CLIs
- `src/helico/`: shared CV utilities used by both models
- `scripts/`: utility launchers (diagnostics wrappers)
- `scripts/cluster/`: transfer helpers (`rsync_pull.sh`, `rsync_push.sh`)
- `latex/`: report source (with Mermaid source in `latex/figures/methodology.mmd`)
- `cv/`, `output/`, `slurm_io/`: generated artifacts

Compatibility notes:
- `model.py` is a wrapper to `model_classifier.py` (legacy command support).
- `train.sbatch.sh` is kept, but `train_classifier.sbatch.sh` is preferred.
- root `rsync_pull.sh` / `rsync_push.sh` are wrappers to `scripts/cluster/`.

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
python model_classifier.py --epochs 150
```

Saved checkpoint: `cnn_model.pth`

### Train autoencoder

```bash
python model_autoencoder.py --epochs 150
```

Saved checkpoint: `autoencoder_model.pth`  
Saved threshold file: `autoencoder_threshold.txt`

### Run patient-level k-fold cross-validation

```bash
python model_classifier.py --epochs 150 --k-folds 5 --seed 42
python model_autoencoder.py --epochs 150 --k-folds 5 --seed 42
```

Outputs are saved under:

- `cv/cnn/fold_*/` and `cv/cnn/summary.json`
- `cv/autoencoder/fold_*/` and `cv/autoencoder/summary.json`

### Generate CV diagnostics and plots

Using root scripts:

```bash
python analyze_cv_metrics.py
python analyze_ae_fold_diagnostics.py
```

Using script wrappers:

```bash
python scripts/analyze_cv_metrics.py
python scripts/analyze_ae_fold_diagnostics.py
```

Outputs are written to `cv/plots/`.

### Patient-level pipeline

Run patient-level aggregation and evaluation (CrossValidation patient-CV + HoldOut):

```bash
python patient_level_pipeline.py --model cnn
python patient_level_pipeline.py --model ae
```

Fold-consistent mode (uses `cv/<model>/fold_i` checkpoints from k-fold training):

```bash
python patient_level_pipeline.py --model cnn --use-cv-fold-checkpoints --patient-k-folds 5 --seed 42
python patient_level_pipeline.py --model ae --use-cv-fold-checkpoints --patient-k-folds 5 --seed 42
```

Optional examples:

```bash
python patient_level_pipeline.py --model ae --patch-threshold 0.07 --patient-k-folds 5 --seed 42
python patient_level_pipeline.py --model cnn --patch-threshold 0.5
```

Outputs are saved under:

- `output/patient_level/cnn/`
- `output/patient_level/ae/`
- Includes:
  - `patient_cv_folds_from_cv_models.json` (when fold-consistent mode is used)
  - `holdout_metrics_per_fold_from_cv_models.json`
  - `patient_summary.json`

Patient-level plots and summary:

```bash
python analyze_patient_level_metrics.py
```

Generates:

- `output/patient_level/plots/patient_cnn_fold_lines.png`
- `output/patient_level/plots/patient_autoencoder_fold_lines.png`
- `output/patient_level/plots/patient_models_cv_mean_std_comparison.png`
- `output/patient_level/plots/patient_metrics_summary.txt`

### Optional task shortcuts

```bash
make train-cnn
make train-ae
make cv-cnn
make cv-ae
make diag-cv
make diag-ae
make patient-cnn
make patient-ae
make patient-cnn-folds
make patient-ae-folds
make diag-patient
make latex
```

## 2) Run on cluster (Slurm)

Two job scripts are provided:

- `train_classifier.sbatch.sh` (CNN)
- `train_autoencoder.sbatch.sh` (autoencoder)
- `patient_cnn_fold.sbatch.sh` (patient-level pipeline, CNN)
- `patient_autoencoder_fold.sbatch.sh` (patient-level pipeline, AE)
- `patient_plots.sbatch.sh` (patient-level plots/summary)

### Submit jobs

```bash
sbatch train_classifier.sbatch.sh
sbatch train_autoencoder.sbatch.sh
sbatch patient_cnn_fold.sbatch.sh
sbatch patient_autoencoder_fold.sbatch.sh
sbatch patient_plots.sbatch.sh
```

Legacy:

```bash
sbatch train.sbatch.sh
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

## 3) Build LaTeX report

Local (from repo root):

```bash
cd latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

GitHub Actions:

- Workflow: `.github/workflows/build-latex.yml`
- Generates Mermaid figure and compiles `latex/main.tex`
- Uploads artifacts: PDF and generated figures
