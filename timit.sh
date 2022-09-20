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
                  Value: '$nseeds'
  -k N            Positive offset (inclusive) of the seed to start from.
                  Value: '$kseeds'
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
  -p 'A [B]'      Training regimes to experiment with.
                  Value: '${regimes[*]}'
  -f N            Run distributed over N nodes.
                  Value: '$world_size'
  -q              Add one to quiet level.
  -x              Run only the current stage.
  -j              Exclude recommended combinations.
  -h              Display usage info and exit.
EOF
  exit ${1:-1}
}


# constants
# ALL_MODELS=( $(find conf/proto -mindepth 1 -type d -exec basename {} \;) )
ALL_MODELS=( deeprnn-uni )
ALL_DEPENDENCIES=( full indep partial )
ALL_ESTIMATORS=( direct marginal cb srswor ais-c ais-g sf-biased sf-is ctc )
ALL_LMS=( lm-rnn lm-embedding nolm )
ALL_REGIMES=( pretrained flatstart )
CORE_INVALIDS=(
  'full_marginal' 'full_cb' 'full_ctc' 'partial_marginal' 'partial_ctc'
  'ctc_lm-rnn' 'ctc_lm-embedding' '^[^_]*_nolm_pretrained'
)
RECOMMENDED_INVALIDS=(
  'indep_direct' 'partial_direct' 'indep_cb' 'indep_srswor' 'partial_srswor'
  'indep_ais-c' 'partial_ais-c' 'indep_ais-g' 'partial_ais-g' 'indep_sf-biased'
  'partial_sf-biased' 'indep_sf-is' 'partial_sf-is'
)
OFFSET="${TIMIT_OFFSET:-0}"
STRIDE="${TIMIT_STRIDE:-1}"
TMPDIR="$(mktemp -d)"
VOCAB_SIZE=60
trap 'rm -rf "$TMPDIR"' EXIT

# variables
quiet=""
stage=1
timit=
data=data/timit
exp=exp/timit
nseeds=20
kseeds=1
device=cuda
world_size=0
invalids=( "${CORE_INVALIDS[@]}" )
models=( "${ALL_MODELS[@]}" )
dependencies=( "${ALL_DEPENDENCIES[@]}" )
estimators=( "${ALL_ESTIMATORS[@]}" )
lms=( "${ALL_LMS[@]}" )
regimes=( "${ALL_REGIMES[@]}" )
# beam_widths=( 1 2 4 8 16 32 )
beam_widths=( 4 )
only=0

# for determinism
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

while getopts "xqhjs:i:d:o:b:n:k:c:m:z:e:l:p:f:" opt; do
  case $opt in
    s)
      argcheck_is_nat $opt "$OPTARG"
      stage=$OPTARG
      ;;
    k)
      argcheck_is_nat $opt "$OPTARG"
      kseeds=$OPTARG
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
      nseeds=$OPTARG
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
    p)
      argcheck_all_a_choice $opt "${ALL_REGIMES[@]}" "$OPTARG"
      regimes=( $OPTARG )
      ;;
    f)
      argcheck_is_nat $opt "$OPTARG"
      world_size=$OPTARG
      ;;
    x)
      only=1
      ;;
    q)
      quiet="$quiet -q"
      ;;
    j)
      invalids+=( "${RECOMMENDED_INVALIDS[@]}" )
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
encdir="$exp/enc"
model=
dependency=
estimator=
lm=
seed=
pretrain=

get_combos() {
  local x=""
  local del=""
  local name
  local array
  while (($#)); do
    name="$1"
    shift
    if [ "$name" = "model" ]; then
      array=( "${models[@]}" )
    elif [ "$name" = "dependency" ]; then
      array=( "${dependencies[@]}" )
    elif [ "$name" = "estimator" ]; then
      array=( "${estimators[@]}" )
    elif [ "$name" = "lm" ]; then
      array=( "${lms[@]}" )
    elif [ "$name" = "regime" ]; then
      array=( "${regimes[@]}" )
    elif [ "$name" = "seed" ]; then
      array=( $(seq $kseeds $nseeds) )
    else
      array=( "$name" )
    fi
    x="$(echo "$x" | prod "$del" "${array[@]}")"
    del="_"
  done
  echo "$x" | filter is_not is_a_match "${invalids[@]}"
}


unpack_combo() {
  local i=$1
  shift
  local combo
  local value
  IFS='_' read -ra combo <<< "$(get_combos "$@" | sed $((i + 1))'q;d')" 
  for ((i = 0; i < $#; i++)); do
    name="${@:i+1:1}"
    value="${combo[i]}"
    [ "$name" = "$value" ] || eval "$name=$value"
  done
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
  if [ ! -f "$data/.complete" ] && [ $OFFSET -eq 0 ]; then 
    echo "Beginning stage 1"
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -i)" 1>&2
      exit 1
    fi
    argcheck_is_writable d "$data"
    argcheck_is_readable i "$timit"
    python prep/timit.py "$data" preamble "$timit"
    python prep/timit.py "$data" init_phn --lm --vocab-size ${VOCAB_SIZE}
    # 40mel+1energy fbank features every 10ms
    python prep/timit.py "$data" torch_dir \
      --computer-json prep/conf/feats/fbank_41.json \
      --seed 0
    compute-mvn-stats-for-torch-feat-data-dir \
      "$data/train/feat" "$data/ext/train.mvn.pt"
    cat "$data/local/phn${VOCAB_SIZE}/lm_train.trn.gz" | \
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
  if [[ "${regimes[*]}" =~ "pretrained" ]]; then
    N=$(get_combos model lm pretrained seed | wc -l)
    if [ $N -gt 0 ]; then
      mkdir -p "$lmdir"
      if [ ! -f "$lmdir/results.ngram.txt" ] && [ $OFFSET -eq 0 ]; then
        echo "Beginning stage 2 - results.ngram.txt"
        python asr.py "$data" $quiet eval_lm > "$lmdir/results.ngram.txt" \
          || rm -f "$lmdir/results.ngram.txt"
        echo "Ending stage 2 - results.ngram.txt"
      fi
    fi
    for (( i = OFFSET; i < N ; i += STRIDE )); do
      # the only settings that matter are the model, lm, and pretrained; we put
      # dummy configs in the rest
      unpack_combo $i model lm pretrained seed
      dependency=indep
      estimator=marginal
      yml="$(combine)"
      mdir="$lmdir/${model}_${lm}/$seed"
      mkdir -p "$mdir"
      if [ ! -f "$mdir/final.pt" ]; then
        echo "Beginning stage 2 - training LM for $model, $lm and seed $seed"
        python asr.py "$data" $quiet \
          --read-yaml "$yml" \
          --device "$device" \
          --model-dir "$mdir" \
          --seed $seed \
          train_lm "$mdir/final.pt"
        echo "Ending stage 2 - training LM for $model, $lm, and seed $seed"
      else
        echo "Stage 2 - $mdir/final.pt exists. Skipping"
      fi

      if [ ! -f "$lmdir/results.${model}_${lm}.$seed.txt" ]; then
        echo "Beginning stage 2 - computing LM perplexity $model, $lm, and seed $seed"
        python asr.py "$data" $quiet \
          --read-yaml "$yml" \
          --device "$device" \
          eval_lm "$mdir/final.pt" \
            > "$lmdir/results.${model}_${lm}.$seed.txt"
        echo "Ending stage 2 - computing LM perplexity $model, $lm, and seed $seed"
      fi
    done
  else
    echo "'pretrained' not in selected regimes - skipping stage 2"
  fi
  ((only)) && exit 0
fi

# pretrain up to the latent part of the model (the "encoder")
if [ $stage -le 3 ]; then
  if [[ "${regimes[*]}" =~ "pretrained" ]]; then
    N=$(get_combos model dependency estimator pretrained seed | wc -l)
    for (( i = OFFSET; i < N; i += STRIDE )); do
      unpack_combo $i model dependency estimator pretrained seed
      lm=nolm
      yml="$(combine)"
      mname="${model}_${dependency}_${estimator}"
      mdir="$encdir/$mname/$seed"
      mkdir -p "$mdir"
      if [ ! -f "$mdir/final.pt" ]; then
        echo "Beginning stage 3 - pretraining $mname with seed $seed"
        python asr.py "$data" $quiet \
          --read-yaml "$yml" \
          --device "$device" \
          --model-dir "$mdir" \
          --seed $seed \
          train_am "$mdir/final.pt" --pretraining --ddp-world-size $world_size
        echo "Ending stage 3 - pretraining $mname with seed $seed"
      else
        echo "Stage 3 - $mdir/final.pt exists. Skipping"
      fi
    done
  else
    echo "'pretrained' not in selected regimes - skipping stage 3"
  fi
  ((only)) && exit 0
fi


# train everything
if [ $stage -le 4 ]; then
  N=$(get_combos model dependency estimator lm regime seed | wc -l)
  for (( i = OFFSET; i < N; i += STRIDE )); do
    unpack_combo $i model dependency estimator lm regime seed
    mname="${model}_${dependency}_${estimator}_${lm}_${regime}"
    yml="$(combine)"
    mdir="$amdir/$mname/$seed"
    mkdir -p "$mdir"
    xtra_args=( )
    if [ "$regime" = "pretrained" ]; then
      if [ "$lm" != "nolm" ]; then
        lmpth="$lmdir/${model}_${lm}/$seed/final.pt"
        if [ ! -f "$lmpth" ]; then
          echo "Cannot train $mname with seed $seed: '$lmpth' does not exist" \
            "(did you finish stage 2?)" 1>&2
          exit 1
        fi
        xtra_args+=( "--pretrained-lm-path" "$lmpth" )
      fi
      encpth="$encdir/${model}_${dependency}_${estimator}/$seed/final.pt"
      if [ ! -f "$encpth" ]; then
          echo "Cannot train $mname with seed $seed: '$encpth' does not exist"\
            "(did you finish stage 3?)" 1>&2
          exit 1
      fi
      xtra_args+=( "--pretrained-enc-path" "$encpth" )
    fi
    if [ ! -f "$mdir/final.pt" ]; then
      echo "Beginning stage 4 - training $mname with seed $seed"
      python asr.py "$data" $quiet \
        --read-yaml "$yml" \
        --device "$device" \
        --model-dir "$mdir" \
        --seed $seed \
        train_am "${xtra_args[@]}" "$mdir/final.pt" \
          --ddp-world-size $world_size
      echo "Ending stage 4 - training $mname with seed $seed"
    else
      echo "Stage 4 - $mdir/final.pt exists. Skipping"
    fi
  done
  ((only)) && exit 0
fi

# decode and compute error rates
if [ $stage -le 5 ]; then
  N=$(get_combos model dependency estimator lm regime seed | wc -l)
  for (( i = $OFFSET; i < ncs; i += $STRIDE )); do
    unpack_combo $i model dependency estimator lm regime seed
    mname="${model}_${dependency}_${estimator}_${lm}_${regime}"
    yml="$(combine)"
    mdir="$amdir/$mname/$seed"
    mpth="$mdir/final.pt"
    if [ ! -f "$mpth" ]; then
          echo "Cannot decode $mname with seed $seed: '$mpth' does not exist" \
            "(did you finish stage 4?)" 1>&2
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
          echo "Beginning stage 5 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
          python asr.py "$data" $quiet \
            --read-yaml "$yml" \
            --device "$device" \
            decode_am \
              "${xtra_args[@]}" --beam-width "$beam_width" \
              "$mpth" "$bdir"
          touch "$bdir/.complete"
          echo "Ending stage 5 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
        else
          echo "'$bdir/.complete' exists. Skipping decoding $part using" \
            "$mname with seed $seed and beam width $beam_width"
        fi
        if [ ! -f "$mdir/$part.hyp.$beam_width.trn" ]; then
          echo "Beginning stage 5 - gathering hyps for $part using $mname" \
            "with $seed and beam with $beam_width"
          torch-token-data-dir-to-trn \
            "$bdir" "$data/ext/id2token.txt" \
            "$mdir/$part.hyp.$beam_width.utrn"
          python prep/timit.py "$data" filter \
            "$mdir/$part.hyp.$beam_width."{u,}trn
          echo "Ending stage 5 - gathering hyps for $part using $mname" \
            "with seed $seed and beam with $beam_width"
        fi
      done
      active_files=( "$mdir/$part.hyp."*.trn )
      if [ ${#active_files[@]} -ne ${#active_widths[@]} ]; then
        echo "The number of evaluated beam widths does not equal the number" \
          "of hypothesis files for partition '$part' in '$mdir'. This could" \
          "mean you changed the -b parameter after running once or you reran" \
          "experiments with different parameters and the partition is" \
          "'test'. Delete all hyp files in '$amdir' and try running this step"\
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

