#!/bin/bash
set -euo pipefail

rsync -avz  \
--exclude='HelicoDataSet/' \
--exclude ".git" \
--exclude ".venv" \
--exclude ".mypy_cache" \
--exclude ".Trash-1000" \
--exclude "__pycache__" \
--exclude ".ruff_cache" \
--exclude ".pytest_cache" \
--exclude ".mypy_cache" \
--exclude ".gitignore" \
-e "ssh -p 55022" \
ricse04@158.109.75.52:/hhome/ricse04/autoencoder/ ~/projets/deeplearning/

