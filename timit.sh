#! /usr/bin/env bash

set -e

usage () {
  cat << EOF 1>&2
Usage: $0
  -s N:           Run from stage N.
                  Value: '$stage'
  -i PTH:         Location of TIMIT data directory (unprocessed).
                  Value: '$timit'
  -d PTH:         Location of processed data directory.
                  Value: '$data'
  -o PTH:         Location to store experiment artifacts.
                  Value: '$exp'
  -b 'A [B ...]'  The beam widths to test for decoding
  -n N:           Number of repeated trials to perform.
                  Value: '$seeds'
  -k N:           Offset (inclusive) of the seed to start from.
                  Value: '$offset'
  -c DEVICE:      Device to run experiments on.
                  Value: '$device'
  -m 'A [B ...]': Model configurations to experiment with.
                  Value: '${models[*]}'
  -e 'A [B ...]'  Estimators to experment with.
                  Value: '${estimators[*]}'
  -l 'A [B ...]'  LM combinations to experiment with.
                  Value: '${lms[*]}'
  -x              Run only the current stage.
  -h              Display usage info and exit.
EOF
  exit ${1:-1}
}

argcheck_nat() {
  arg="$1"
  shift
  while (( $# )); do
    if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
      echo "$0: '-$arg' argument '$1' is not a natural number" 1>&2
      usage
    fi
    shift
  done
}

argcheck_rdir() {
  if ! [ -d "$2" ]; then
    echo "$0: '-$1' argument '$2' is not a readable directory." 1>&2
    usage
  fi
}

argcheck_writable() {
  if ! [ -w "$2" ]; then
    echo "$0: '-$1' argument '$2' is not writable." 1>&2
    usage
  fi
}

argcheck_choices() {
  if ! [[ " $3 " =~ " $2 " ]]; then
    echo "$0: '-$1' argument '$2' not one of '$3'." 1>&2
    usage
  fi
}

check_config() {
  if [[ " ${INVALIDS[*]} " =~ " $1_$2 " ]] || [[ " ${INVALIDS[*]} " =~ " $2_$3" ]]; then
    return 1
  fi
  mkdir -p "$confdir"
  yml="$confdir/$1_$2_$3.yaml"
  combine-yaml-files \
    --nested \
    conf/proto/{base,model_$1,estimator_$2,lm_$3}.yaml "$yml"
}

# constants
ALL_MODELS=( full indep partial )
ALL_ESTIMATORS=( direct marginal partial srswor ais-c ais-g sf-biased sf-is )
ALL_LMS=( lm-flatstart lm-pretrained nolm )
INVALIDS=( full_marginal full_partial partial_full )

# variables
stage=1
timit=
data=data/timit
exp=exp/timit
seeds=20
offset=1
device=cuda
models=( "${ALL_MODELS[@]}" )
estimators=( "${ALL_ESTIMATORS[@]}" )
lms=( "${ALL_LMS[@]}" )
beam_widths=( 1 2 4 8 16 32 )
only=0

while getopts "xhs:i:d:o:b:n:k:c:m:e:l:" opt; do
  case $opt in
    s)
      argcheck_nat $opt "$OPTARG"
      stage=$OPTARG
      ;;
    k)
      argcheck_nat $opt "$OPTARG"
      offset=$OPTARG
      ;;
    i)
      argcheck_rdir $opt "$OPTARG"
      timit="$OPTARG"
      ;;
    d)
      argcheck_writable $opt "$OPTARG"
      data="$OPTARG"
      ;;
    o)
      argcheck_writable $opt "$OPTARG"
      exp="$OPTARG"
      ;;
    b)
      argcheck_nat $opt $OPTARG
      beam_widths=( $OPTARG )
      ;;
    n)
      argcheck_nat $opt "$OPTARG"
      seeds=$OPTARG
      ;;
    c)
      device="$OPTARG"
      ;;
    m)
      argcheck_choices $opt "$OPTARG" "${ALL_MODELS[*]}"
      models=( $OPTARG )
      ;;
    e)
      argcheck_choices $opt "$OPTARG" "${ALL_ESTIMATORS[*]}"
      estimators=( $OPTARG )
      ;;
    l)
      argcheck_choices $opt "$OPTARG" "${ALL_LMS[*]}"
      lms=( $OPTARG )
      ;;
    x)
      only=1
      ;;
    h)
      echo "Shell recipe to perform experiments on TIMIT." 1>&2
      usage 0
      ;;
  esac
done

if [ $# -ne $(($OPTIND - 1)) ]; then
  echo "Expected no positional arguments but found one: '${@:$OPTIND}'" 1>&2
  usage
fi

confdir="$exp/conf"
lmdir="$exp/lm"
amdir="$exp/am"

# prep the dataset
if [ $stage -le 1 ]; then
  if [ ! -f "$data/.complete" ]; then 
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -t)" 1>&2
      exit 1
    fi
    python prep/timit.py "$data" preamble "$timit"
    python prep/timit.py "$data" init_phn --lm
    # 40mel+1energy fbank features every 10ms, stacked 3 at a time for 123-dim
    # feature vectors every 30ms
    python prep/timit.py "$data" torch_dir \
      --computer-json prep/conf/feats/fbank_41.json \
      --seed 0
    cat "$data/local/phn48/lm_train.trn.gz" | \
      gunzip -c | \
      trn-to-torch-token-data-dir - "$data/ext/token2id.txt" "$data/lm"
    touch "$data/.complete"
  fi
  ((only)) && exit 0
fi

# pretrain the language models (unnecessary if not doing LM pretraining)
if [ $stage -le 2 ]; then
  if [[ " ${lms[*]} " =~ " lm-pretrained " ]]; then
    mkdir -p "$lmdir"
    [ -f "$lmdir/results.ngram.txt" ] || \
      python asr.py "$data" eval_lm > "$lmdir/results.ngram.txt"
    for seed in $(seq $offset $seeds); do
      [ -f "$lmdir/$seed/final.pt" ] || \
        python asr.py "$data" \
          --read-yaml conf/proto/lm_lm-pretrained.yaml \
          --device "$device" \
          --model-dir "$lmdir/$seed" \
          --seed $seed \
          train_lm "$lmdir/$seed/final.pt"

      [ -f "$lmdir/results.$seed.txt" ] || \
        python asr.py "$data" \
          --read-yaml conf/proto/lm_lm-pretrained.yaml \
          --device "$device" \
          eval_lm "$lmdir/$seed/final.pt" > "$lmdir/results.$seed.txt"
    done
  fi
  ((only)) && exit 0
fi

# train the acoustic models
if [ $stage -le 3 ]; then
  for model in "${models[@]}"; do
    for estimator in "${estimators[@]}"; do
      for lm in "${lms[@]}"; do
        check_config $model $estimator $lm || continue
        mname="${model}_${estimator}_${lm}"
        yml="$confdir/$mname.yaml"
        for seed in $(seq $offset $seeds); do
          mdir="$amdir/$mname/$seed"
          mkdir -p "$mdir"
          xtra_args=( )
          if [[ "$mname" =~ "_lm-pretrained" ]]; then
            xtra_args=( "--pretrained-lm-path" "$lmdir/$seed/final.pt" )
          fi
          [ -f "$mdir/final.pt" ] || \
            python asr.py "$data" \
              --read-yaml "$yml" \
              --device "$device" \
              --model-dir "$mdir" \
              --seed $seed \
              train_am "${xtra_args[@]}" "$mdir/final.pt"
        done
      done
    done
  done
  ((only)) && exit 0
fi

# decode and compute error rates
if [ $stage -le 4 ]; then
  for model in "${models[@]}"; do
    for estimator in "${estimators[@]}"; do
      for lm in "${lms[@]}"; do
        check_config $model $estimator $lm || continue
        mname="${model}_${estimator}_${lm}"
        yml="$confdir/$mname.yaml"
        for seed in $(seq $offset $seeds); do
          mdir="$amdir/$mname/$seed"
          if [ ! -f "$mdir/final.pt" ]; then
            echo "$mdir/final.pt doesn't exist (did stage 3 finish?)" 1>&2
            exit 1
          fi
          for part in dev test; do
            hdir="$mdir/hyp/$part"
            if [ "$part" = dev ]; then
              xtra_args=( "--dev" )
              active_widths=( "${beam_widths[@]}" )
            else
              xtra_args=( )
              active_widths=( "$(awk '
$1 ~ /^best/ {a=gensub(/.*\/dev\.hyp\.([^.]*).*$/, "\\1", 1, $3); print a}
' "$amdir/$mname/results.dev.$seed.txt")" )
            fi
            for beam_width in "${active_widths[@]}"; do
              bdir="$hdir/$beam_width"
              mkdir -p "$bdir"
              python asr.py "$data" \
                --read-yaml "$yml" \
                --device "$device" \
                decode_am \
                  "${xtra_args[@]}" --beam-width "$beam_width" \
                  "$mdir/final.pt" "$bdir"
              torch-token-data-dir-to-trn \
                "$bdir" "$data/ext/id2token.txt" \
                "$mdir/$part.hyp.$beam_width.utrn"
              python prep/timit.py "$data" filter \
                "$mdir/$part.hyp.$beam_width."{u,}trn
            done
            python prep/error-rates-from-trn.py \
              "$data/ext/$part.ref.trn" "$mdir/$part.hyp."*.trn \
              --suppress-warning > "$amdir/$mname/results.$part.$seed.txt"
          done
        done
      done
    done
  done
  ((only)) && exit 0
fi

# compute descriptives for all the models
for part in dev test; do
  for mdir in $(find "$amdir" -maxdepth 1 -mindepth 1 -type d); do
    results=( $(find "$mdir" -name "results.$part.*.txt" -print) )
    if [ "${#results[@]}" -gt 0 ]; then
      echo -n "$part ${mdir##*/}: "
      awk '
BEGIN {n=0; s=0; min=1000; max=0}
$1 ~ /best/ {
  x=substr($NF, 1, length($NF) - 1);
  a[n++]=x; s+=x; if (x < min) min=x; if (x > max) max=x;
}
END {
  mean=s/n; med=a[int(n/2)];
  var=0; for (i=0;i<n;i++) var+=(a[i] - mean) * (a[i] - mean) / n; std=sqrt(var);
  printf "n=%d, mean=%.1f%%, std=%.1f%%, med=%.1f%%, min=%.1f%%, max=%.1f%%\n", n, mean, std, med, min, max;
}' "${results[@]}"
    fi
  done
done

exit 0

