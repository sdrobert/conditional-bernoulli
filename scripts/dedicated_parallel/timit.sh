#! /usr/bin/env bash

if [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
  IFS=',' read -ra gpu_numbers <<< "${CUDA_VISIBLE_DEVICES}"
else
  gpu_numbers=( $(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\):.*/\1/p') )
fi
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
  rm -f logs/timit/stage-$stage-*.log
  for i in "${!gpu_numbers[@]}"; do
    gpu_number="${gpu_numbers[$i]}"
    (
      export CUDA_VISIBLE_DEVICES=$gpu_number;
      export TIMIT_OFFSET=$i;
      ./timit.sh -s $stage -x -q "${global_args[@]}" \
        > "logs/timit/stage-$stage-$i.log" 2>&1;
    ) & 
    pids+=( $! )
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
      echo "Process ${gpu_numbers[i]} failed" 1>&2
      return 1
    fi
  done
  echo "Done stage $stage"
}

for stage in {1..5}; do
  run_stage $stage
done

# get results
./timit.sh -s 999