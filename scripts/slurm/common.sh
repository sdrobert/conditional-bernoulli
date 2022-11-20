# source this

set -e

mkdir -p logs/timit

timit_dir="${1:-"$HOME/Databases/TIMIT"}"
cpu_opts="${2:-"-p cpu"}"
gpu_opts="${3:-"-p p100 --gpus-per-task=1"}"

if [ ! -d "$timit_dir" ]; then
  echo "$timit_dir is not a directory"
  exit 1
fi

check_and_do() {
  local opts="$1"
  shift
  local leftover="$(./timit.sh "$@" -xxj)"
  if ((leftover>0)); then
    echo "Starting: ./timit.sh $* -xqj ($leftover tasks)"
    sbatch $opts --ntasks=1 -a 1-$leftover -W scripts/slurm/timit_wrapper.sh "$@" -xqj
    v=$?
    if [ $? -eq 0 ]; then
      echo "./timit.sh $* -xqj succeeded"
    else
      echo "./timit.sh $* -xqj failed"
    fi
    return $?
  else
    echo "./timit.sh $* -xqj already done"
  fi
}