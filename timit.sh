#! /usr/bin/env bash

# variables
TIMIT=~/Databases/TIMIT
SEEDS=20
device=cuda

# prep the dataset
python prep/timit.py data/timit preamble $TIMIT
python prep/timit.py data/timit init_phn --lm
# 40mel+1energy fbank features every 10ms, stacked 3 at a time for 123-dim
# feature vectors every 30ms
python prep/timit.py data/timit torch_dir \
  --computer-json prep/conf/feats/fbank_41.json \
  --postprocess prep/conf/post/stack_3.json \
  --seed 0
cat data/timit/local/phn48/lm_train.trn.gz | \
  gunzip -c | \
  trn-to-torch-token-data-dir - data/timit/ext/token2id.txt data/timit/lm

python asr.py data/timit \
  --read-yaml conf/dummy.yml \
  --device cuda \
  --model-dir exp/timit/dummy/lm \
  train_lm exp/timit/dummy/lm/final.pt

python asr.py data/timit \
  --read-yaml conf/dummy.yml \
  --device cuda \
  eval_lm exp/timit/dummy/lm/final.pt

python asr.py data/timit \
  --read-yaml conf/dummy.yml \
  --device cuda \
  --model-dir exp/timit/dummy/am \
  train_am exp/timit/dummy/am/final.pt \
    --pretrained-lm-path exp/timit/dummy/lm/final.pt

# python asr.py data/timit \
#   --read-yaml conf/dummy.yml \
#   --device cuda \
#   train_am /dev/null

python asr.py data/timit \
  --read-yaml conf/dummy.yml \
  --device cuda \
  decode_am exp/timit/dummy/am/final.pt exp/timit/dummy/hyp

torch-token-data-dir-to-trn \
  exp/timit/dummy/hyp \
  data/timit/ext/id2token.txt \
  exp/timit/dummy/test.hyp.trn

python prep/timit.py data filter exp/timit/dummy/test.hyp{,.filt}.trn

~/kaldi/tools/sctk/bin/sclite \
  -r data/timit/ext/test.ref.trn \
  -h exp/timit/dummy/test.hyp.filt.trn \
  -i swb