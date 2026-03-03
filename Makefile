.PHONY: train-cnn train-ae cv-cnn cv-ae diag-cv diag-ae patient-cnn patient-ae patient-cnn-folds patient-ae-folds diag-patient latex

train-cnn:
	python model_classifier.py --epochs 150

train-ae:
	python model_autoencoder.py --epochs 150

cv-cnn:
	python model_classifier.py --epochs 150 --k-folds 5 --seed 42

cv-ae:
	python model_autoencoder.py --epochs 150 --k-folds 5 --seed 42

diag-cv:
	python analyze_cv_metrics.py

diag-ae:
	python analyze_ae_fold_diagnostics.py

patient-cnn:
	python patient_level_pipeline.py --model cnn

patient-ae:
	python patient_level_pipeline.py --model ae

patient-cnn-folds:
	python patient_level_pipeline.py --model cnn --use-cv-fold-checkpoints --patient-k-folds 5 --seed 42

patient-ae-folds:
	python patient_level_pipeline.py --model ae --use-cv-fold-checkpoints --patient-k-folds 5 --seed 42

diag-patient:
	python analyze_patient_level_metrics.py

latex:
	cd latex && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

