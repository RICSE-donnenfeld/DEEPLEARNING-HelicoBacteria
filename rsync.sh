rsync -avz  \
--exclude='HelicoDataSet/' \
--exclude ".git" \
--exclude ".venv" \
--exclude ".mypy_cache" \
-e "ssh -p 55022" \
ricse04@158.109.75.52:/hhome/ricse04/autoencoder/ ~/projets/deeplearning/
