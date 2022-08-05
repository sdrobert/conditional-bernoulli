#! /usr/bin/env bash

gpu_numbers=( $(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\):.*/\1/p') )
# IFS=' ' read -ra gpu_numbers <<< "${1:-"$(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\):.*/\1/p')"}"
ngpus="${#gpu_numbers[@]}"
timit_dir="${2:-~/Databases/TIMIT}"

set -e

python -c 'import asr'

mkdir -p "logs/timit"

if [ "${ngpus}" -eq 0 ]; then
  echo "No GPUs specified!" 2>&1
  exit 1
fi

global_args=( "$@" )

export TIMIT_STRIDE="$ngpus"

run_stage() {
  stage="$(printf '%02d' $1)"
  echo "Beginning stage $stage"
  shift
  pids=( )
  for i in "${!gpu_numbers[@]}"; do
    gpu_number="${gpu_numbers[$i]}"
    (
      export CUDA_VISIBLE_DEVICES=$gpu_number;
      export TIMIT_OFFSET=$i;
      ./timit.sh -s $stage -x -q "${global_args[@]}" > "logs/timit/stage-$stage-$gpu_number.log" 2>&1;
    ) & 
    pids+=( $! )
  done
  wait "${pids[@]}"
  echo "Done stage $stage"
}

for stage in {1..4}; do
  run_stage $stage
done

./timit.sh -s 10000