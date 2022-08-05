#! /usr/bin/env bash

set -e

source scripts/utils.sh

usage () {
  cat << EOF 1>&2
Usage: $0
  -s N            Run from stage N.
                  Value: '$stage'
  -i PTH          Location of TIMIT data directory (unprocessed).
                  Value: '$timit'
  -d PTH          Location of processed data directory.
                  Value: '$data'
  -o PTH          Location to store experiment artifacts.
                  Value: '$exp'
  -b 'A [B ...]'  The beam widths to test for decoding
  -n N            Number of repeated trials to perform.
                  Value: '$seeds'
  -k N            Offset (inclusive) of the seed to start from.
                  Value: '$offset'
  -c DEVICE       Device to run experiments on.
                  Value: '$device'
  -m 'A [B ...]'  Model configurations to experiment with.
                  Value: '${models[*]}'
  -z 'A [B ...]'  Dependency structures to experiment with.
                  Value: '${dependencies[*]}'
  -e 'A [B ...]'  Estimators to experment with.
                  Value: '${estimators[*]}'
  -l 'A [B ...]'  LM combinations to experiment with.
                  Value: '${lms[*]}'
  -q              Add one to quiet level.
  -x              Run only the current stage.
  -h              Display usage info and exit.
EOF
  exit ${1:-1}
}


# constants
ALL_MODELS=( $(find conf/proto -mindepth 1 -type d -exec basename {} \;) )
ALL_DEPENDENCIES=( full indep partial )
ALL_ESTIMATORS=( direct marginal cb srswor ais-c ais-g sf-biased sf-is ctc )
ALL_LMS=( lm-flatstart lm-pretrained lm-embedding nolm )
INVALIDS=( 
  'full_marginal' 'full_cb' 'full_ctc' 'partial_marginal' 'partial_ctc'
  'ctc_lm-pretrained' 'ctc_lm-flatstart' 'ctc_lm-embedding'
)
OFFSET="${TIMIT_OFFSET:-0}"
STRIDE="${TIMIT_STRIDE:-1}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# variables
quiet=""
stage=1
timit=
data=data/timit
exp=exp/timit
seeds=20
offset=1
device=cuda
models=( loa-small )
dependencies=( "${ALL_DEPENDENCIES[@]}" )
estimators=( "${ALL_ESTIMATORS[@]}" )
lms=( "${ALL_LMS[@]}" )
beam_widths=( 1 2 4 8 16 32 )
only=0


while getopts "xqhs:i:d:o:b:n:k:c:m:z:e:l:" opt; do
  case $opt in
    s)
      argcheck_is_nat $opt "$OPTARG"
      stage=$OPTARG
      ;;
    k)
      argcheck_is_nat $opt "$OPTARG"
      offset=$OPTARG
      ;;
    i)
      argcheck_is_readable $opt "$OPTARG"
      timit="$OPTARG"
      ;;
    d)
      data="$OPTARG"
      ;;
    o)
      exp="$OPTARG"
      ;;
    b)
      argcheck_all_nat $opt "$OPTARG"
      beam_widths=( $OPTARG )
      ;;
    n)
      argcheck_is_nat $opt "$OPTARG"
      seeds=$OPTARG
      ;;
    c)
      device="$OPTARG"
      ;;
    m)
      argcheck_all_a_choice $opt "${ALL_MODELS[@]}" "$OPTARG"
      models=( $OPTARG )
      ;;
    z)
      argcheck_all_a_choice $opt "${ALL_DEPENDENCIES[@]}" "$OPTARG"
      dependencies=( $OPTARG )
      ;;
    e)
      argcheck_all_a_choice $opt "${ALL_ESTIMATORS[@]}" "$OPTARG"
      estimators=( $OPTARG )
      ;;
    l)
      argcheck_all_a_choice $opt "${ALL_LMS[@]}" "$OPTARG"
      lms=( $OPTARG )
      ;;
    x)
      only=1
      ;;
    q)
      quiet="$quiet -q"
      ;;
    h)
      echo "Shell recipe to perform experiments on TIMIT." 1>&2
      usage 0
      ;;
  esac
done

if [ $# -ne $(($OPTIND - 1)) ]; then
  echo "Expected no positional arguments but found one" 1>&2
  usage
fi

confdir="$exp/conf"
lmdir="$exp/lm"
amdir="$exp/am"
mel_combos=( $(
  echo "" |
  prod "" "${models[@]}" |
  prod _ "${dependencies[@]}" |
  prod _ "${estimators[@]}" |
  prod _ "${lms[@]}" |
  filter is_not is_a_match "${INVALIDS[@]}"
) )
ncombos="${#mel_combos[@]}"
ncs=$((ncombos * seeds))
ms=$((seeds * ${#models[@]}))
ncsb=$((ncs * ${#beam_widths[@]}))
model=
dependency=
estimator=
lm=
seed=

unpack_nc() {
  local combo
  IFS='_' read -ra combos <<< "${mel_combos[$1]}" 
  model=${combos[0]}
  dependency=${combos[1]}
  estimator=${combos[2]}
  lm=${combos[3]}
}

unpack_m() {
  model="${models[(( $1 % ${#models[@]} ))]}"
}

unpack_s() {
  seed=$(printf '%02d' $(( $1 % seeds + 1)))
}

unpack_ncs() {
  unpack_s $1
  unpack_nc $(( $1 / seeds ))
}

unpack_ms() {
  unpack_s $1
  unpack_m $(( $1 / seeds ))
}

combine() {
  yml="$(mktemp)"
  combine-yaml-files \
    --nested --quiet \
    conf/proto/${model}/{base,dep_${dependency},estimator_${estimator},lm_${lm}}.yaml \
    "$yml"
  echo "$yml"
}

# prep the dataset
if [ $stage -le 1 ]; then
  if [ ! -f "$data/.complete" ]; then 
    echo "Beginning stage 1"
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -i)" 1>&2
      exit 1
    fi
    argcheck_is_writable d "$data"
    argcheck_is_readable i "$timit"
    # python prep/timit.py "$data" preamble "$timit"
    # python prep/timit.py "$data" init_phn --lm --vocab-size 61
    # 40mel+1energy fbank features every 10ms
    # python prep/timit.py "$data" torch_dir \
    #   --computer-json prep/conf/feats/fbank_41.json \
    #   --seed 0
    cat "$data/local/phn61/lm_train.trn.gz" | \
      gunzip -c | \
      trn-to-torch-token-data-dir - "$data/ext/token2id.txt" "$data/lm"
    touch "$data/.complete"
    echo "Finished stage 1"
  else
    echo "$data/.complete exists already. Skipping stage 1."
  fi
  ((only)) && exit 0
fi

# pretrain the language dependencies (unnecessary if not doing LM pretraining)
if [ $stage -le 2 ]; then
  if [[ " ${lms[*]} " =~ " lm-pretrained " ]]; then
    mkdir -p "$lmdir/$model"
    if [ ! -f "$lmdir/results.ngram.txt" ]; then
      echo "Beginning stage 2 - results.ngram.txt"
      python asr.py "$data" $quiet eval_lm > "$lmdir/results.ngram.txt"
      echo "Ending stage 2 - results.ngram.txt"
    fi
    for (( i = OFFSET; i < ms ; i += STRIDE )); do
      unpack_ms $i
      if [ ! -f "$lmdir/$model/$seed/final.pt" ]; then
        echo "Beginning stage 2 - training LM for model $model and seed $seed"
        python asr.py "$data" $quiet \
          --read-yaml conf/proto/${model}/lm_lm-pretrained.yaml \
          --device "$device" \
          --model-dir "$lmdir/$model/$seed" \
          --seed $seed \
          train_lm "$lmdir/$model/$seed/final.pt"
        echo "Ending stage 2 - training LM for model $model and seed $seed"
      else
        echo "Stage 2 - $lmdir/$seed/final.pt exists. Skipping"
      fi

      if [ ! -f "$lmdir/results.$model.$seed.txt" ]; then
        echo "Beginning stage 2 - computing LM perplexity for seed $seed"
        python asr.py "$data" $quiet \
          --read-yaml conf/proto/${model}/lm_lm-pretrained.yaml \
          --device "$device" \
          eval_lm "$lmdir/$model/$seed/final.pt" \
            > "$lmdir/results.$model.$seed.txt"
        echo "Ending stage 2 - computing LM perplexity for seed $seed"
      fi
    done
  fi
  ((only)) && exit 0
fi

# train the acoustic models
if [ $stage -le 3 ]; then
  for (( i = OFFSET; i < ncs; i += STRIDE )); do
    unpack_ncs $i
    mname="${model}_${dependency}_${estimator}_${lm}"
    yml="$(combine)"
    mdir="$amdir/$mname/$seed"
    mkdir -p "$mdir"
    xtra_args=( )
    if [ "$lm" = "lm-pretrained" ]; then
      xtra_args=( "--pretrained-lm-path" "$lmdir/$model/$seed/final.pt" )
    fi
    if [ ! -f "$mdir/final.pt" ]; then
      echo "Beginning stage 3 - training $mname with seed $seed"
      python asr.py "$data" $quiet \
        --read-yaml "$yml" \
        --device "$device" \
        --model-dir "$mdir" \
        --seed $seed \
        train_am "${xtra_args[@]}" "$mdir/final.pt"
      echo "Ending stage 3 - training $mname with seed $seed"
    else
      echo "Stage 3 - $mdir/final.pt exists. Skipping"
    fi
  done
  ((only)) && exit 0
fi

# decode and compute error rates
if [ $stage -le 4 ]; then
  for (( i = $OFFSET; i < ncs; i += $STRIDE )); do
    unpack_ncs $i
    mname="${model}_${dependency}_${estimator}_${lm}"
    yml="$(combine)"
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
        beam_width="$(printf '%02d' $((10#$beam_width + 0)))"
        bdir="$hdir/$beam_width"
        mkdir -p "$bdir"
        if [ ! -f "$bdir/.complete" ]; then
          echo "Beginning stage 4 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
          python asr.py "$data" $quiet \
            --read-yaml "$yml" \
            --device "$device" \
            decode_am \
              "${xtra_args[@]}" --beam-width "$beam_width" \
              "$mdir/final.pt" "$bdir"
          touch "$bdir/.complete"
          echo "Ending stage 4 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
        else
          echo "'$bdir/.complete' exists. Skipping decoding $part using" \
            "$mname with seed $seed and beam width $beam_width"
        fi
        if [ ! -f "$mdir/$part.hyp.$beam_width.trn" ]; then
          echo "Beginning stage 4 - gathering hyps for $part using $mname" \
            "with $seed and beam with $beam_width"
          torch-token-data-dir-to-trn \
            "$bdir" "$data/ext/id2token.txt" \
            "$mdir/$part.hyp.$beam_width.utrn"
          python prep/timit.py "$data" filter \
            "$mdir/$part.hyp.$beam_width."{u,}trn
          echo "Ending stage 4 - gathering hyps for $part using $mname" \
            "with seed $seed and beam with $beam_width"
        fi
      done
      active_files=( "$mdir/$part.hyp."*.trn )
      if [ ${#active_files[@]} -ne ${#active_widths[@]} ]; then
        echo "The number of evaluated beam widths does not equal the number" \
          "of hypothesis files for partition '$part' in '$mdir'. This could" \
          "mean you changed the -b parameter after running once or you reran" \
          "experiments with different parameters and the partition is" \
          "'test'. Delete all hyp files in '$amdir' and try runing this step" \
          "again" 1>&2
        exit 1
      fi
      [ -f "$amdir/$mname/results.$part.$seed.txt" ] || \
        python prep/error-rates-from-trn.py \
          "$data/ext/$part.ref.trn" "$mdir/$part.hyp."*.trn \
          --suppress-warning > "$amdir/$mname/results.$part.$seed.txt"
    done
  done
  ((only)) && exit 0
fi

# compute descriptives for all the dependencies
echo "Phone Error Rates:"
for part in dev test; do
  for mdir in $(find "$amdir" -maxdepth 1 -mindepth 1 -type d); do
    results=( $(find "$mdir" -name "results.$part.*.txt" -print) )
    if [ "${#results[@]}" -gt 0 ]; then
      echo -n "$part ${mdir##*/}: "
      awk '
BEGIN {n=0; s=0; min=1000; max=0}
$1 ~ /best/ {
  x=substr($NF, 1, length($NF) - 1) + 0;
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

