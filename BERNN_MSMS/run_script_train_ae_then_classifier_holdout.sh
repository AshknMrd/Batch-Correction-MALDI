#!/bin/bash
set -e

# Run training on the built-in Alzheimer dataset --- dloss: inverseTriplet DANN revTriplet normae no 
python bernn/dl/train/train_ae_then_classifier_holdout.py \
  --device=cuda:0 \
  --dataset=alzheimer \
  --n_trials=1 \
  --n_repeats=1 \
  --n_epochs=10 \
  --early_stop=5 \
  --early_warmup_stop=5 \
  --exp_id=test_alzheimer_then \
  --path=data/Alzheimer_Test \
  --csv_file=unique_genes.csv \
  --groupkfold=1 \
  --embeddings_meta=2 \
  --n_meta=2 \
  --dloss=DANN \
  --variational=1 \
  --kan=0 \
  --tied_weights=0 \
  --rec_loss=l1 \
  --pool=1 \
  --use_l1=0 \
  --prune_network=0 \
  --update_grid=0 \
  --log_mlflow=0 \
  --log_neptune=0 \
  --log_tb=1 \
  --log_metrics=1 \
  --log_plots=1 \
  --keep_models=1
