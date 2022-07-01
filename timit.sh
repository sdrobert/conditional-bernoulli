#! /usr/bin/env bash

set -e

stage="${1:-1}"
timit_dir=~/Databases/TIMIT
SEEDS=20
device=cuda
# models=( full indep partial )
models=( indep )
# estimators=( direct marginal partial-indep srswor ais-c ais-g sf-biased sf-is )
estimators=( marginal srswor )
# lms=( lm nolm )
lms=( lm )
invalids=( full_marginal full_partial-indep partial_full )

# prep the dataset
if [ $stage -le 1 ]; then
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
fi

# construct the experiment matrix and delete any old experiment stuff
if [ $stage -le 2 ]; then
  rm -rf conf/matrix/* exp/timit/lm/* exp/timit/am/*
  mkdir -p conf/matrix
  for model in "${models[@]}"; do
    for estimator in "${estimators[@]}"; do
      for lm in "${lms[@]}"; do
        mname="${model}_${estimator}_${lm}"
        is_invalid=0
        for invalid in "${invalids[@]}"; do
          if [[ "$mname" =~ "$invalid" ]]; then
            is_invalid=1
            break
          fi
        done
        [ "$is_invalid" = 1 ] && continue
        combine-yaml-files \
          --nested \
          conf/proto/{base,model_$model,estimator_$estimator,lm_$lm}.yaml \
          conf/matrix/$mname.yaml
      done
    done
  done
fi

# train the language models
if [ $stage -le 3 ] && [[ "lm " =~ "${lms[*]}" ]]; then
  mkdir -p exp/timit/lm
  python asr.py data/timit eval_lm > exp/timit/lm/results.txt
  for seed in $(seq 1 $SEEDS); do
    python asr.py data/timit \
      --read-yaml conf/proto/lm_lm.yaml \
      --device $device \
      --model-dir exp/timit/lm/$seed \
      --seed $seed \
      train_lm exp/timit/lm/$seed/final.pt

    python asr.py data/timit \
      --read-yaml conf/proto/lm_lm.yaml \
      --device $device \
      eval_lm exp/timit/lm/$seed/final.pt >> exp/timit/lm/results.txt
  done
fi

if [ $stage -le 4 ]; then
  for yml in conf/matrix/*; do
    mname="$(echo $yml | cut -d / -f 3 | cut -d . -f 1)"
    for seed in $(seq 1 $SEEDS); do
      mdir="exp/timit/am/$mdir/$seed"
      mkdir -p "$mdir"
      xtra_args=""
      if [[ "$mname" =~ "_lm" ]]; then
        xtra_args="--pretrained-lm-path exp/timit/lm/$seed/final.pt"
      fi
      python asr.py data/timit \
        --read-yaml $yml \
        --device $device \
        --model-dir $mdir \
        --seed $seed \
        train_am $xtra_args $mdir/final.pt
    done
  done
fi

exit 0
# scratch

python asr.py data/timit \
  --read-yaml conf/matrix/full_ais-g_lm.yaml \
  --device cuda \
  train_am /dev/null

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