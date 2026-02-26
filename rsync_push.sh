rsync -avz  \
--exclude='HelicoDataSet/' \
--exclude ".git" \
--exclude ".venv" \
--exclude ".mypy_cache" \
--exclude ".Trash-1000" \
-e "ssh -p 55022" \
~/projets/deeplearning/ ricse04@158.109.75.52:/hhome/ricse04/deeplearning/
