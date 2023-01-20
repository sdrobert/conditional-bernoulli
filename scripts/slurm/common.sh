# source this

mkdir -p logs/timit

extra_flags="${1:-"-j"}"
timit_dir="${2:-"$HOME/Databases/TIMIT"}"
cpu_opts="${3:-"-p cpu"}"
gpu_opts="${4:-"-p t4v2 --gpus-per-task=1"}"

if [ ! -d "$timit_dir" ]; then
  echo "$timit_dir is not a directory"
  exit 1
fi

check_and_do() {
  local opts="$1"
  shift
  local leftover="$(./timit.sh "$@" $extra_flags -xx)"
  if [ ! -z "$leftover" ]; then
    echo "Starting: ./timit.sh $* -xq $extra_flags ($leftover)"
    sbatch $opts --ntasks=1 -a $leftover -W scripts/slurm/timit_wrapper.sh "$@" -xq $extra_flags
    v=$?
    if [ $v -eq 0 ]; then
      echo "./timit.sh $* -xq $extra_flags succeeded"
    else
      echo "./timit.sh $* -xq $extra_flags failed"
    fi
    return $v
  else
    echo "./timit.sh $* -xq $extra_flags already done"
  fi
}