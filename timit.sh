#! /usr/bin/env bash

# variables
TIMIT=~/Databases/TIMIT
SEEDS=20

# prep the dataset
python prep/timit.py data/timit preamble $TIMIT
python prep/timit.py data/timit init_phn --lm
python prep/timit.py data/timit torch_dir
cat data/timit/local/phn48/lm_train.trn.gz | \
  gunzip -c | \
  trn-to-torch-token-data-dir - data/timit/ext/token2id.txt data/timit/lm

# pretrain RNN-LMs
# The weights of these LMs will be fed into all AMs tested. It's important that
# the configurations remain the same
